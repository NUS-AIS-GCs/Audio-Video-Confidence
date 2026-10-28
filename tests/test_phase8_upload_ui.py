"""Upload section: readable status card around the unchanged DeploymentUI.submit callback."""
from types import SimpleNamespace

import pytest

from deployment.ui.upload_section import EMPTY_CARD, UPLOAD_MESSAGES, upload_card


@pytest.mark.parametrize("code", sorted(UPLOAD_MESSAGES))
def test_every_known_code_has_a_fixed_message(code):
    status = "MODEL_NOT_READY" if code in ("RAW_INFERENCE_GATE_CLOSED", "DECODER_DEPENDENCY_MISSING") else "REJECTED"
    html = upload_card({"status": status, "predictions": None, "error_code": code})
    title, detail = UPLOAD_MESSAGES[code]
    assert title in html.replace("&#x27;", "'") and code in html and "no prediction is shown" in html


def test_model_not_ready_and_rejected_are_rendered_differently():
    ok = upload_card({"status": "MODEL_NOT_READY", "error_code": "RAW_INFERENCE_GATE_CLOSED", "predictions": None,
                      "issues": [{"category": "audio", "code": "MISSING_ARTIFACT"}]})
    bad = upload_card({"status": "REJECTED", "error_code": "INVALID_DURATION", "predictions": None})
    assert 'data-kind="not_ready"' in ok and "MODEL_NOT_READY" in ok
    assert 'data-kind="rejected"' in bad and "REJECTED" in bad


def test_unknown_codes_and_odd_results_never_echo_server_text():
    html = upload_card({"status": "REJECTED", "error_code": "<script>alert(1)</script>/srv/secret/path"})
    assert "<script>" not in html and "/srv/secret" not in html and "UPLOAD_NOT_ACCEPTED" in html
    assert "UPLOAD_NOT_ACCEPTED" in upload_card(None)
    assert "UPLOAD_NOT_ACCEPTED" in upload_card({"status": "REJECTED", "error_code": 7})


def test_card_never_shows_scores_even_if_a_result_had_them():
    html = upload_card({"status": "MODEL_NOT_READY", "error_code": "RAW_INFERENCE_GATE_CLOSED",
                        "predictions": {"match": 0.97}})
    assert "0.97" not in html and "match" not in html.lower()


def test_real_gradio_wiring_keeps_private_submit_and_clears_stale_results():
    gr = pytest.importorskip("gradio")
    from deployment.ui.upload_section import build_upload_section
    calls = []
    ui = SimpleNamespace(config=SimpleNamespace(max_upload_bytes=50 * 1024 * 1024),
                         submit_owned=lambda path, request: calls.append(path) or {"status": "REJECTED", "predictions": None,
                                                                    "error_code": "UPLOAD_REQUIRED"})
    with gr.Blocks() as demo:
        build_upload_section(gr, ui)
    deps = demo.fns.values()
    click = next(d for d in deps if getattr(d.fn, '__name__', '') == "check")
    change = next(d for d in deps if getattr(d.fn, '__name__', '') == "clear")
    assert click.api_visibility == "private" and click.concurrency_limit == 1
    assert change.api_visibility == "private"
    assert click.trigger_mode == "once"
    assert any(d.cancels for d in deps)  # file changes cancel in-flight generator results
    assert isinstance(click.inputs[0], gr.Textbox) and change.inputs == []
    loading, (card, raw, button) = list(click.fn(None, None))
    assert 'aria-busy="true"' in loading[0] and loading[1] is None
    assert not loading[2]['interactive'] and not button['interactive']
    assert calls == [None] and "No file selected" in card and raw["error_code"] == "UPLOAD_REQUIRED"
    card, raw, button = change.fn()
    assert (card, raw) == (EMPTY_CARD, None) and button['interactive']


def test_loading_errors_and_technical_details_are_sanitized():
    import gradio as gr
    from deployment.ui.upload_section import build_upload_section
    def fail(_, request):
        raise RuntimeError('/private/checkpoint secret')
    ui = SimpleNamespace(config=SimpleNamespace(max_upload_bytes=1024), submit_owned=fail)
    with gr.Blocks() as demo:
        build_upload_section(gr, ui)
    fn = next(f.fn for f in demo.fns.values() if getattr(f.fn, '__name__', '') == 'check')
    loading, final = list(fn('unit.mp4', None))
    assert loading[1] is None
    assert final[1] == {'status': 'REJECTED', 'predictions': None, 'error_code': 'UPLOAD_VALIDATION_FAILED'}
    assert '/private' not in str(final) and 'secret' not in str(final)

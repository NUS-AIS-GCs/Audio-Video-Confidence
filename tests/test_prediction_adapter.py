"""Contract and authorization tests; unit guards never claim model performance."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from deployment.internal_backend import InternalTrainBackend
from deployment.prediction_adapter import DEMO_CASES, InternalPredictionAdapter, PairingPrediction
from deployment.ui.internal_section import prediction_card
from deployment.ui.upload_section import public_admission_result, upload_card
from fusion.contracts import ContractError


def genuine_prediction():
    path = Path("_local/phase8_stage2d/profile-01/native-volume/run/profile.json")
    if not path.is_file():
        pytest.skip("authentic prior TRAIN response unavailable")
    p = json.loads(path.read_text())["requests"][0]["response"]["predictions"]["E3"]
    return PairingPrediction("original-1", DEMO_CASES[0].audio_id, DEMO_CASES[0].video_id,
                             p["score"], p["logit"], 1.0)


def test_genuine_response_can_only_render_in_internal_result_card():
    p = genuine_prediction()
    html = prediction_card(p.to_dict())
    assert p.classification in html and "INTERNAL TRAIN ONLY" in html
    assert str(p.score) in html and str(p.threshold) in html
    assert "not a probability" in html
    public = public_admission_result(p.to_dict())
    assert public["predictions"] is None and public["status"] == "MODEL_NOT_READY"
    assert str(p.score) not in upload_card(public)


@pytest.mark.parametrize("change", [
    {"threshold": .5}, {"score": float("nan")}, {"score": True},
    {"production_predictions_enabled": True}, {"model_sha256": "0"*64},
    {"case_id": "TEST-1"}, {"audio_source": "unapproved"}, {"processing_seconds": -1.0},
    {"model_version": "substitute"}, {"status": "READY"}, {"schema": "unknown"},
])
def test_typed_response_rejects_promotion_and_invalid_values(change):
    with pytest.raises(ContractError):
        replace(genuine_prediction(), **change)


def test_renderer_refuses_extra_unvalidated_fields_and_classification_tampering():
    r = genuine_prediction().to_dict()
    r["classification"] = "100% authentic"
    assert "100% authentic" not in prediction_card(r)
    assert "unavailable" in prediction_card(r)
    assert "unavailable" in prediction_card({**genuine_prediction().to_dict(), "event_predictions": ["label"]})


def adapter_without_models():
    return InternalPredictionAdapter(object.__new__(InternalTrainBackend))


def test_anonymous_and_spoofed_requests_never_reach_models():
    adapter = adapter_without_models()
    with mock.patch.object(adapter, "predict_case", side_effect=AssertionError("must authenticate first")):
        for request in (None, SimpleNamespace(username="alice"), SimpleNamespace(request=object())):
            assert adapter.submit("original-1", request)["error_code"] == "INTERNAL_AUTHENTICATION_REQUIRED"


def test_unapproved_cases_and_resource_contention_fail_before_loading():
    adapter = adapter_without_models()
    for value in (None, {}, "../models", "TEST-1", "original-99"):
        assert adapter.predict_case(value)["error_code"] == "UNAPPROVED_TRAIN_CASE"
    adapter._lock.acquire()
    try:
        assert adapter.predict_case("original-1")["error_code"] == "RESOURCE_BUSY"
    finally:
        adapter._lock.release()
    assert adapter.predict_case("original-1")["error_code"] == "INTERNAL_INFERENCE_FAILED"
    assert not adapter._lock.locked()


def test_prespecified_negatives_and_positives_use_only_permitted_train_sources():
    from deployment.train_manifest import PERMITTED_IDS
    assert len(DEMO_CASES) == 10
    for c in DEMO_CASES:
        assert c.audio_id in PERMITTED_IDS and c.video_id in PERMITTED_IDS
        assert (c.audio_id == c.video_id) == (c.kind == "original")


def test_frozen_decision_rule_on_adjacent_values_and_exact_boundary():
    # Pure decision-rule unit test; no synthetic model response is constructed.
    import math
    from fusion.inference import CANDIDATES
    threshold = CANDIDATES["E3"].threshold
    classify = PairingPrediction.classification.fget
    assert classify(SimpleNamespace(score=math.nextafter(threshold, 0), threshold=threshold)) == "Audio-substituted-like pairing"
    assert classify(SimpleNamespace(score=threshold, threshold=threshold)) == "Original-like pairing"
    assert classify(SimpleNamespace(score=math.nextafter(threshold, 1), threshold=threshold)) == "Original-like pairing"


def test_default_app_never_constructs_internal_inference(monkeypatch, tmp_path):
    from deployment.gradio_app import create_app
    monkeypatch.setenv("AVAL_ACCESS_MODE", "gradio_auth")
    monkeypatch.setenv("AVAL_GRADIO_USERNAME", "test")
    monkeypatch.setenv("AVAL_GRADIO_PASSWORD", "unit-only")
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    with mock.patch.object(InternalTrainBackend, "__init__", side_effect=AssertionError("public load forbidden")):
        demo, _ = create_app()
        try:
            assert not any("Internal TRAIN" in str(getattr(b, "label", "")) for b in demo.blocks.values())
        finally:
            demo.aval_live_service.close()
            demo.aval_upload_store.close()


def test_internal_callback_retains_gradio_login_registry_boundary(monkeypatch, tmp_path):
    import secrets
    import gradio as gr
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from deployment.gradio_app import create_app
    from deployment.live.security import live_app_kwargs
    monkeypatch.setenv("AVAL_ACCESS_MODE", "gradio_auth")
    monkeypatch.setenv("AVAL_GRADIO_USERNAME", "alice")
    password = secrets.token_urlsafe(24)
    monkeypatch.setenv("AVAL_GRADIO_PASSWORD", password)
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    adapter = adapter_without_models()
    demo, config = create_app(internal_adapter=adapter)
    app = gr.mount_gradio_app(FastAPI(), demo, path="/", auth=config.gradio_auth,
                             app_kwargs=live_app_kwargs(demo.aval_live_service))
    fn = next(f for f in demo.fns.values() if getattr(f.fn, "__name__", "") == "infer")
    assert fn.api_visibility == "private" and fn.concurrency_limit == 1
    with TestClient(app) as client:
        assert client.get("/config").status_code == 401
        assert client.post("/gradio_api/queue/join", json={"fn_index": fn._id, "data": ["original-1"],
                                                          "session_hash": "spoof"}).status_code == 401
        assert client.post("/login", data={"username": "alice", "password": password}).status_code == 200
        assert client.get("/config").status_code == 200
        # Resolving the actual registry request authorizes selection, not arbitrary paths.
        raw = SimpleNamespace(request=None)
        with mock.patch("deployment.prediction_adapter.login_owner", return_value="registry-owner"), \
             mock.patch.object(adapter, "predict_case", wraps=adapter.predict_case) as called:
            assert adapter.submit("TEST-1", raw)["error_code"] == "UNAPPROVED_TRAIN_CASE"
            called.assert_called_once_with("TEST-1")

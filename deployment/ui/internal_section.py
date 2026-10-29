"""Internal-only result presentation. Never imported by the public app by default."""
from html import escape

from deployment.prediction_adapter import DEMO_CASES, PairingPrediction, internal_error


def prediction_card(result) -> str:
    try:
        fields = PairingPrediction.__dataclass_fields__
        prediction = PairingPrediction(**{k: result[k] for k in fields})
        if result != prediction.to_dict():
            raise ValueError
    except Exception:
        return ('<div class="av-result" role="status"><h4>Internal prediction unavailable</h4>'
                '<p>No score is shown. Check the authenticated TRAIN setup.</p></div>')
    p = prediction
    kind = "original" if p.score >= p.threshold else "substituted"
    return (f'<section class="av-panel" data-kind="{kind}" role="status" aria-live="polite">'
            '<p class="av-eyebrow">EXPERIMENTAL / INTERNAL TRAIN ONLY</p>'
            f'<h3 class="av-section-title">{p.classification}</h3>'
            f'<p class="av-note">Raw E3 score <strong>{p.score}</strong> · frozen threshold {p.threshold}</p>'
            f'<div class="av-score-track" aria-hidden="true"><b style="width:{100*p.score:.4f}%"></b>'
            f'<i style="left:{100*p.threshold:.4f}%"></i></div>'
            f'<p class="av-note">{escape(p.model_version)} · {p.processing_seconds:.2f} s<br>'
            f'Model SHA-256: <code>{p.model_sha256}</code></p>'
            '<p class="av-note">This is an uncalibrated correspondence score, not a probability of authenticity '
            'or proof of physical sound causation. Public predictions remain disabled. '
            'Historical parity and general shared-clock approval are unresolved.</p></section>')


def build_internal_section(gr, adapter):
    gr.Markdown("### Internal TRAIN demonstration\n**Authenticated, experimental, prespecified TRAIN cases only.** "
                "Original cases retain the source pairing; swapped cases use audio from the next permitted "
                "TRAIN source with the selected source's video. These cases do not establish accuracy or release readiness.")
    case = gr.Dropdown(choices=[(f"{c.key} · audio {c.audio_id} / video {c.video_id}", c.key)
                               for c in DEMO_CASES], value=DEMO_CASES[0].key, label="Authorized TRAIN case")
    run = gr.Button("Run genuine E3 · internal TRAIN only", variant="primary")
    card = gr.HTML('<p class="av-note">No inference has run. Select an authorized case.</p>')
    raw = gr.JSON(label="Validated internal response; public predictions disabled")

    def infer(case_id, request: gr.Request):
        try:
            result = adapter.submit(case_id, request)
        except Exception:
            result = internal_error("INTERNAL_INFERENCE_FAILED")
        return prediction_card(result), result

    event = run.click(infer, inputs=case, outputs=[card, raw], api_name=None, api_visibility="private",
              concurrency_limit=1, trigger_mode="once", show_progress="full")
    case.change(fn=lambda: ("<p class='av-note'>Case changed. Run inference to obtain a result.</p>", None),
                outputs=[card, raw], api_name=None, api_visibility="private", queue=False, cancels=[event])

    with gr.Accordion("Upload an authorized TRAIN original clip", open=False):
        gr.Markdown("**Internal TRAIN only.** Only the original filenames and SHA-bound bytes of the five "
                    "permitted MP4s are accepted. Other uploads stay unapproved. The existing ownership, "
                    "size, expiry and cleanup controls apply. No model weights or source clips are downloadable.")
        upload = gr.File(label="Authorized TRAIN MP4", file_types=[".mp4"], file_count="single",
                         type="filepath", elem_id="av-internal-upload-source")
        handle = gr.Textbox(visible=False)
        upload.change(fn=None, inputs=upload, outputs=handle, queue=False,
                      js="(file) => file && typeof file.path === 'string' ? file.path : ''")
        submit = gr.Button("Verify uploaded TRAIN clip", variant="primary")
        upload_card = gr.HTML("<p class='av-note'>No uploaded TRAIN inference has run.</p>")
        upload_raw = gr.JSON(label="Internal uploaded TRAIN response; public predictions disabled")

        def infer_upload(upload_handle, request: gr.Request):
            yield "<p class='av-note' role='status'>Checking authorized TRAIN upload...</p>", None, gr.update(interactive=False)
            try:
                result = adapter.submit_owned_upload(upload_handle, request)
            except Exception:
                result = internal_error("INTERNAL_INFERENCE_FAILED")
            yield prediction_card(result), result, gr.update(interactive=False)

        uploaded_event = submit.click(infer_upload, inputs=handle, outputs=[upload_card, upload_raw, submit],
            api_name=None, api_visibility="private", concurrency_limit=1, trigger_mode="once", show_progress="full")
        handle.change(fn=lambda: ("<p class='av-note'>Upload changed. Run verification for a fresh result.</p>",
                                 None, gr.update(interactive=True)),
            outputs=[upload_card, upload_raw, submit], api_name=None, api_visibility="private", queue=False,
            cancels=[uploaded_event])

"""Upload preview and admission results on the existing authenticated boundary.

The private callback resolves only a session-owned handle before admission. No internal
TRAIN result is a public result; Gradio must never preprocess user file paths here.

The card never shows a prediction or score. It shows the status and a fixed message per error code;
unknown codes get a generic message, so no raw server text reaches the page.
"""
from html import escape
from pathlib import Path
import re

# error_code -> (title, what the user can do). Messages are fixed text, never server output.
UPLOAD_MESSAGES = {
    "UPLOAD_AUTHENTICATION_REQUIRED": ("Sign in again", "Your upload session is not available. Sign in and select the file again."),
    "UPLOAD_NOT_FOUND": ("Upload expired or unavailable", "Select the file again, then press Verify."),
    "UPLOAD_REQUIRED": ("No file selected", "Choose one 10-second video file, then press Verify."),
    "UPLOAD_TOO_LARGE": ("File too large", "The file is over the upload limit. Export a smaller 10-second clip."),
    "UNSUPPORTED_FILENAME": ("File name not accepted", "Rename the file using letters, numbers, dashes or underscores."),
    "UNSUPPORTED_EXTENSION": ("File type not supported", "Use MP4, MOV or MKV (H.264 video + AAC audio), or WebM (VP9 + Opus)."),
    "UNSUPPORTED_UPLOAD": ("File could not be read", "Check that the file is a complete video and try again."),
    "MALFORMED_MEDIA_METADATA": ("File could not be read", "The file's media information is damaged. Re-export the clip."),
    "MISSING_OR_AMBIGUOUS_STREAMS": ("Needs exactly one video and one audio track",
                                     "Make sure the clip has sound, and only one video and one audio track."),
    "UNSUPPORTED_CONTAINER": ("Container not supported", "Re-export as a standard MP4, MOV, MKV or WebM file."),
    "UNSUPPORTED_CODEC": ("Codec not supported", "Use H.264 video with AAC audio (MP4, MOV, MKV), or VP9 with Opus (WebM)."),
    "INVALID_DURATION": ("Clip must be 10 seconds", "Trim the clip so both audio and video are 10 seconds long."),
    "MODALITY_DURATION_MISMATCH": ("Audio and video lengths differ", "Re-export so audio and video are both 10 seconds."),
    "PROBE_TIMEOUT": ("File took too long to check", "Try a simpler re-export of the clip."),
    "PROBE_OUTPUT_TOO_LARGE": ("File could not be checked", "Re-export the clip and try again."),
    "PROBE_OUTPUT_READ_FAILED": ("File could not be checked", "Re-export the clip and try again."),
    "UPLOAD_VALIDATION_FAILED": ("File could not be checked", "Try again. If it keeps failing, re-export the clip."),
    "AUTHENTICATION_CONFIGURATION_REQUIRED": ("Uploads are switched off", "The server's sign-in is not configured."),
    "DECODER_DEPENDENCY_MISSING": ("Checker unavailable", "The server's media decoder is not installed."),
    "RAW_INFERENCE_GATE_CLOSED": ("Upload accepted, verification not available yet",
                                  "The file passed every upload check. Audio-video verification stays off until "
                                  "the trained models are connected and verified, so no result is shown."),
}
_FALLBACK = ("Upload not accepted", "The file did not pass the upload checks.")

UPLOAD_CSS = """
.av-result { border-radius: 12px; padding: 14px 16px; border: 1px solid var(--av-line); background: var(--av-card); }
.av-result h4 { margin: 0 0 4px; font-size: 15px; color: var(--av-ink); }
.av-result p { margin: 0; font-size: 13px; color: var(--av-muted); line-height: 1.5; }
.av-result code { font-size: 12px; }
.av-result[data-kind="rejected"] { border-color: var(--av-bad); }
.av-result[data-kind="rejected"] h4 { color: var(--av-bad); }
.av-result[data-kind="not_ready"] { border-color: var(--av-warn); }
.av-result[data-kind="empty"] p { font-style: italic; }
"""

EMPTY_CARD = ('<div class="av-result" data-kind="empty"><p>No check yet. Choose a file and press '
              'Verify.</p></div>')
LOADING_CARD = ('<div class="av-result" role="status" aria-live="polite" aria-busy="true">'
                '<h4>Checking upload…</h4><p>Validating the file. Predictions remain disabled.</p></div>')
E3_PENDING = ('<div class="av-result" data-kind="not_ready"><h4>E3 correspondence result</h4>'
              '<p>Unavailable · predictions disabled. An approved correspondence result will appear here '
              'after the model release gates pass.</p>'
              '<dl><dt>Pairing decision</dt><dd>Unavailable</dd>'
              '<dt>Uncalibrated correspondence score</dt><dd>Unavailable</dd></dl></div>')


def public_admission_result(result) -> dict:
    """Technical details obey the public admission schema too; never expose internal results."""
    if not isinstance(result, dict):
        result = {}
    status = result.get("status")
    code = result.get("error_code")
    if not isinstance(status, str) or status not in {"MODEL_NOT_READY", "REJECTED"}:
        return {"status": "MODEL_NOT_READY", "predictions": None,
                "error_code": "RAW_INFERENCE_GATE_CLOSED"}
    clean = {"status": status, "predictions": None,
             "error_code": code if isinstance(code, str) and code in UPLOAD_MESSAGES else "UPLOAD_VALIDATION_FAILED"}
    issues = result.get("issues")
    if isinstance(issues, list):
        safe = [row for row in issues[:64] if type(row) is dict and set(row) == {"category", "code"}
                and all(type(v) is str and re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", v) for v in row.values())]
        if safe:
            clean["issues"] = safe
    return clean


def upload_card(result) -> str:
    """Readable HTML for one DeploymentUI.submit result. Never renders predictions."""
    if not isinstance(result, dict):
        result = {"status": "REJECTED"}
    status = result.get("status")
    code = result.get("error_code")
    title, detail = UPLOAD_MESSAGES.get(code, _FALLBACK) if isinstance(code, str) else _FALLBACK
    kind = "not_ready" if status == "MODEL_NOT_READY" else "rejected"
    label = "MODEL_NOT_READY" if kind == "not_ready" else "REJECTED"
    shown_code = escape(code) if isinstance(code, str) and code in UPLOAD_MESSAGES else "UPLOAD_NOT_ACCEPTED"
    return (f'<div class="av-result" data-kind="{kind}" role="status">'
            f'<h4>{escape(title)}</h4><p>{escape(detail)}</p>'
            f'<p><code>{label} · {shown_code}</code> · no prediction is shown</p></div>')


def build_upload_section(gr, ui) -> None:
    def check(upload_handle, request: gr.Request):
        yield LOADING_CARD, None, gr.update(interactive=False)
        try:
            result = public_admission_result(ui.submit_owned(upload_handle, request))
        except Exception:
            result = {"status": "REJECTED", "predictions": None, "error_code": "UPLOAD_VALIDATION_FAILED"}
        # DeploymentUI consumes/removes its managed upload. Do not offer another
        # check against that deleted path; selection/change re-enables Verify.
        yield upload_card(result), result, gr.update(interactive=False)

    def clear():
        return EMPTY_CARD, None, gr.update(interactive=True)

    gr.HTML(f"<style>{UPLOAD_CSS}</style>", padding=False)
    limit_mb = f"{ui.config.max_upload_bytes / (1024 * 1024):.3g}"
    gr.Markdown("### Upload verification\nChoose one recorded **10-second clip with sound** "
                f"(MP4, MOV or MKV with H.264 + AAC, or WebM with VP9 + Opus; up to {limit_mb} MB). "
                "Preview the clip, then press **Verify**. This checks admission only: "
                "**MODEL_NOT_READY — predictions remain disabled**. "
                "Uploads are removed after checking; select the file again to run another check.")
    gr.Markdown("Uploads not checked expire within five minutes, with cleanup checked every second. "
                "The preview uses your local file; server-side media downloads are disabled.")
    uploaded = gr.File(
        label="One 10-second audio-video file (MP4, MOV, MKV, or WebM)",
        file_count="single",
        file_types=[".mp4", ".mov", ".mkv", ".webm"],
        type="filepath",
        elem_id="av-upload-source",
    )
    # Never pass a FileData object through backend preprocessing: Gradio moves
    # file inputs into a shared cache even when preprocess=False. The frontend
    # forwards only our opaque handle into a scalar input. Server ownership
    # checks remain authoritative if this hidden value is forged.
    handle = gr.Textbox(visible=False)
    uploaded.change(fn=None, inputs=uploaded, outputs=handle,
                    js="(file) => file && typeof file.path === 'string' ? file.path : ''", queue=False)
    gr.HTML(value="", html_template=(
        f'<div class="av-upload-preview" data-limit="{ui.config.max_upload_bytes}">'
        '<video controls playsinline preload="metadata" hidden aria-label="Uploaded video preview"></video>'
        '<p class="av-preview-note av-note" role="status"></p></div>'),
        css_template=".av-upload-preview video { width: 100%; max-height: 360px; border-radius: 12px; background: #0b161b; }",
        js_on_load=(Path(__file__).resolve().parent.parent / "web/upload_preview.js").read_text(encoding="utf-8"),
        padding=False)
    submit = gr.Button("Verify (predictions disabled)", variant="primary")
    card = gr.HTML(EMPTY_CARD, padding=False)
    with gr.Accordion("Technical details", open=False):
        output = gr.JSON(label="Admission status; predictions are disabled")
    gr.HTML(E3_PENDING, padding=False)
    verify = submit.click(
        fn=check,
        inputs=handle,
        outputs=[card, output, submit],
        api_name=None,
        api_visibility="private",
        concurrency_limit=1,
        show_progress="full",
        trigger_mode="once",
    )
    uploaded.change(fn=clear, inputs=None, outputs=[card, output, submit], api_name=None,
                    api_visibility="private", queue=False, cancels=[verify])

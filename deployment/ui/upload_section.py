"""Upload Verification section (owner: Swaraj, upload interface / preview / results dashboard).

The secure callback (`DeploymentUI.submit`) and its private endpoint options are unchanged: this module
only turns its sanitised result into a readable status card. Swaraj's results dashboard can replace the
body of `build_upload_section`; keep the signature so the layout does not change.

The card never shows a prediction or score. It shows the status and a fixed message per error code;
unknown codes get a generic message, so no raw server text reaches the page.
"""
from html import escape

# error_code -> (title, what the user can do). Messages are fixed text, never server output.
UPLOAD_MESSAGES = {
    "UPLOAD_REQUIRED": ("No file selected", "Choose one 10-second video file, then press Check upload."),
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
              'Check upload.</p></div>')


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
    def check(upload_path):
        result = ui.submit(upload_path)
        return upload_card(result), result

    def clear(_upload_path):
        return EMPTY_CARD, None            # a new or removed file must not keep the old result

    gr.HTML(f"<style>{UPLOAD_CSS}</style>", padding=False)
    limit_mb = f"{ui.config.max_upload_bytes / (1024 * 1024):.3g}"
    gr.Markdown("### Upload verification\nCheck one recorded 10-second clip with sound "
                f"(MP4, MOV or MKV with H.264 + AAC, or WebM with VP9 + Opus; up to {limit_mb} MB).")
    uploaded = gr.File(
        label="One 10-second audio-video file (MP4, MOV, MKV, or WebM)",
        file_count="single",
        file_types=[".mp4", ".mov", ".mkv", ".webm"],
        type="filepath",
    )
    submit = gr.Button("Check upload (predictions disabled)", variant="primary")
    card = gr.HTML(EMPTY_CARD, padding=False)
    with gr.Accordion("Technical details", open=False):
        output = gr.JSON(label="Admission status; predictions are disabled")
    submit.click(
        fn=check,
        inputs=uploaded,
        outputs=[card, output],
        api_name=None,
        api_visibility="private",
        concurrency_limit=1,
    )
    uploaded.change(fn=clear, inputs=uploaded, outputs=[card, output], api_name=None,
                    api_visibility="private", queue=False)

"""Live Verification section: webcam + microphone capture into the rolling 10-second buffer.

The widget is a gr.HTML component. Its JavaScript (deployment/web/) captures and timestamps both
modalities; its server functions call deployment.live.service.LiveCaptureService. Those calls go through
Gradio's /component_server route, which applies the same login check as the rest of the app.
The section shows buffer readiness and synchronisation diagnostics only; it never shows a prediction.
"""
from pathlib import Path

from deployment.live.service import LiveCaptureService
from deployment.live.buffer import LiveBufferError
from deployment.live.security import request_owner

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

LIVE_TEMPLATE = """
<div class="av-live">
  <div class="av-live-media">
    <video class="av-preview" muted playsinline aria-label="Camera preview"></video>
    <p class="av-note">Microphone activity</p>
    <div class="av-meter" aria-hidden="true"><div class="av-meter-fill"></div></div>
    <div class="av-capture-time"><span class="av-timer">0.0 s / 10 s</span><span>Capture / diagnostics only</span></div>
    <progress class="av-progress" max="10" value="0" aria-label="Ten-second capture progress"></progress>
  </div>
  <div class="av-live-side">
    <div class="av-controls">
      <button class="av-btn av-primary av-record" type="button">Record 10 seconds</button>
      <button class="av-btn av-start" type="button">Start rolling capture</button>
      <button class="av-btn av-stop" type="button" disabled>Stop</button>
      <button class="av-btn av-reset" type="button">Reset</button>
    </div>
    <p class="av-state" data-kind="idle" role="status" aria-live="polite">Idle</p>
    <p class="av-msg" hidden role="alert"></p>
    <div class="av-grid-wrap">
      <div class="av-grid-label">Rolling buffer · last 10 seconds</div>
      <div class="av-grid">
        <i>1</i><i>2</i><i>3</i><i>4</i><i>5</i><i>6</i><i>7</i><i>8</i><i>9</i><i>10</i>
      </div>
      <div class="av-legend"><span data-s="ok">aligned</span><span data-s="missing_audio">missing audio</span><span data-s="dropped_frame">dropped frame</span><span data-s="muted_audio">muted</span></div>
    </div>
    <dl class="av-diag"></dl>
    <button class="av-btn av-check" type="button" disabled>Check window against the backend contract</button>
    <p class="av-contract av-note"></p>
  </div>
</div>
"""

LIVE_CSS = """
.av-live { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 18px; }
@media (max-width: 860px) { .av-live { grid-template-columns: 1fr; } }
.av-live-media { position: relative; }
.av-capture-time { display:flex; justify-content:space-between; gap:12px; margin-top:14px;
  color:var(--av-muted); font-size:12px; font-variant-numeric:tabular-nums; }
.av-progress { width:100%; height:6px; margin-top:8px; accent-color:var(--av-teal); }
.av-progress::-webkit-progress-bar { background:var(--av-pending); border-radius:4px; }
.av-progress::-webkit-progress-value { background:var(--av-teal); border-radius:4px; }
.av-btn:focus-visible { outline:3px solid #73d7ff; outline-offset:4px; }
.av-preview { width: 100%; aspect-ratio: 4 / 3; background: #0b161b; border-radius: 14px; object-fit: cover;
  transform: scaleX(-1); display: block; }
.av-meter { height: 6px; border-radius: 3px; background: var(--av-pending, #c9d4d8); margin-top: 8px; overflow: hidden; }
.av-meter-fill { height: 100%; width: 0%; background: var(--av-gradient, #0e9f9a); transition: width 80ms linear; }
.av-controls { display: flex; gap: 10px; flex-wrap: wrap; }
.av-btn { font: inherit; font-weight: 600; font-size: 14px; padding: 10px 16px; border-radius: 10px; cursor: pointer;
  border: 1px solid var(--av-line, #dbe5e8); background: var(--av-card, #fff); color: var(--av-ink, #10222b); }
.av-btn:disabled { opacity: 0.45; cursor: not-allowed; }
.av-primary { background: var(--av-gradient, #0e9f9a); color: #071724; border: none; }
.av-check { margin-top: 12px; width: 100%; }
.av-state { margin: 12px 0 4px; font-weight: 700; color: var(--av-ink, #10222b); }
.av-state[data-kind="ok"] { color: var(--av-ok, #2fbf71); }
.av-state[data-kind="error"] { color: var(--av-bad, #e0484f); }
.av-msg { margin: 4px 0; font-size: 13px; padding: 8px 10px; border-radius: 8px; background: rgba(224,72,79,.1); color: var(--av-bad, #e0484f); }
.av-msg[data-kind="warn"] { background: rgba(227,160,8,.12); color: #f4cc83; }
.av-msg[data-kind="info"] { background: rgba(99,230,212,.08); color: #bcece8; }
.av-grid-wrap { margin-top: 10px; }
.av-grid-label { font-size: 12px; font-weight: 600; color: var(--av-muted, #5b6b73); text-transform: uppercase; letter-spacing: .06em; }
.av-grid { display: grid; grid-template-columns: repeat(10, 1fr); gap: 4px; margin-top: 6px; }
.av-grid i { font-style: normal; text-align: center; font-size: 11px; font-weight: 700; padding: 10px 0; border-radius: 6px;
  color: #fff; background: var(--av-pending, #c9d4d8); transition: background 150ms; }
.av-grid i[data-status="ok"], .av-legend [data-s="ok"]::before { background: var(--av-ok, #2fbf71); }
.av-grid i[data-status="missing_audio"], .av-grid i[data-status="missing_audio_and_frame"],
.av-legend [data-s="missing_audio"]::before { background: var(--av-bad, #e0484f); }
.av-grid i[data-status="dropped_frame"], .av-legend [data-s="dropped_frame"]::before { background: var(--av-warn, #e3a008); }
.av-grid i[data-status="muted_audio"], .av-legend [data-s="muted_audio"]::before { background: #7c5cd6; }
.av-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; font-size: 12px; color: var(--av-muted, #5b6b73); }
.av-legend span::before { content: ""; display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }
.av-diag { display: grid; grid-template-columns: max-content 1fr; gap: 4px 12px; margin: 14px 0 0; font-size: 13px; }
.av-diag dt { color: var(--av-muted, #5b6b73); }
.av-diag dd { margin: 0; color: var(--av-ink, #10222b); font-variant-numeric: tabular-nums; }
@media (prefers-reduced-motion: reduce) { .av-meter-fill, .av-grid i { transition: none; } }
"""


def live_js() -> str:
    return (WEB_DIR / "live_timing.js").read_text(encoding="utf-8") + "\n" + \
        (WEB_DIR / "live_capture.js").read_text(encoding="utf-8")


def build_live_section(gr, service: LiveCaptureService) -> None:
    def dispatch(method, payload, request):
        try:
            owner = request_owner(request)
        except LiveBufferError:
            return {"status": "MODEL_NOT_READY", "predictions": None,
                    "error_code": "LIVE_AUTHENTICATION_REQUIRED"}
        return method(owner, payload)

    def live_start(payload, request: gr.Request):
        return dispatch(service.start, payload, request)

    def live_push(payload, request: gr.Request):
        return dispatch(service.push, payload, request)

    def live_check(payload, request: gr.Request):
        return dispatch(service.check, payload, request)

    def live_stop(payload, request: gr.Request):
        return dispatch(service.stop, payload, request)

    gr.Markdown(
        "### Live verification\n"
        "Record from your webcam and microphone. AValidate aligns audio and camera frames second by second "
        "on one shared clock and shows whether each second has both a camera frame and complete audio. "
        "**Record 10 seconds** stops by itself once ten aligned seconds are captured and checked; "
        "**Start rolling capture** keeps the latest ten seconds until you press Stop. "
        "**Predictions are disabled**: this section shows capture and sync diagnostics only.")
    gr.HTML(
        value="",
        html_template=LIVE_TEMPLATE,
        css_template=LIVE_CSS,
        js_on_load=live_js(),
        server_functions=[live_start, live_push, live_check, live_stop],
        elem_classes=["av-card"],
        padding=True,
    )
    gr.Markdown(
        "Your browser asks for camera and microphone access when you press Start. Audio and frames are "
        "held in memory for the ten-second window and the segments still filling. Stop or closing the tab "
        "releases local devices and requests server cleanup. If that request cannot arrive, a cleanup "
        "worker checks every second for sessions with 20 seconds without a successful update and clears them.",
        elem_classes=["av-note"])

# Phase 8 frontend: branded layout and live capture

Scope: Gradio layout, branding, navigation, responsive styling, and the **Live verification** tab
(webcam and microphone capture into a rolling ten-second buffer). Predictions stay disabled: the
live tab shows buffer readiness and sync diagnostics only, and nothing in it produces a score.

Not changed: `deployment/runtime.py`, Gradio authentication, upload validation,
`DeploymentUI.submit` and its private endpoint, launch options, and `audio/`, `vision/`, `fusion/`.

## Layout

`deployment/gradio_app.py::create_app()` builds a branded header, a collapsed startup-status
accordion and three tabs:

| Tab | Module | Owner |
| --- | --- | --- |
| Upload verification | `deployment/ui/upload_section.py` | Swaraj |
| Live verification | `deployment/ui/live_section.py` | frontend |
| About | `deployment/gradio_app.py` | frontend |

Theme, CSS and header live in `deployment/ui/theme.py`. The page has `width: 100%` and
`box-sizing: border-box` on `.gradio-container`; at 320, 375 and 1280 px,
`document.documentElement.scrollWidth` equals `clientWidth` on both the Upload and Live tabs. On
narrow screens Gradio folds the Live and About tabs into its "…" overflow menu.

## Live capture architecture

```
browser                                              server (login-protected /component_server)
getUserMedia ─► AudioWorklet ─► PCM blocks ─┐
             └► requestVideoFrameCallback ──┤  push every 250 ms ─► LiveCaptureService ─► LiveBuffer
                (rAF fallback)              │   live_start / live_push / live_check / live_stop
shared clock: AudioContext.getOutputTimestamp()
```

- **One timeline.** The audio sample counter is gapless, so it is the timeline. Each camera frame's
  capture time (`performance.now()` domain) is mapped onto it with the `AudioContext` output
  timestamp (`makeClock` in `deployment/web/live_timing.js`).
- **Ten one-second segments.** Second *k* covers audio samples `[k·sr, (k+1)·sr)` and takes the
  camera frame closest to *k* seconds, within 0.1 s. A second is valid only if its audio is
  complete and not muted and a frame exists. Nothing is repaired: no silence padding, no frame
  reuse.
- **Rolling window.** The buffer is ready when the latest ten finalised seconds are all valid.
  `LiveBuffer.snapshot()` gives 160000 samples at 16 kHz and a `(10, 224, 224, 3)` uint8 array;
  `to_raw_media_input()` validates against `fusion.inference_contracts.RawMediaInput`.
- **Server side.** `deployment/live/service.py` binds every live session to the Gradio session,
  caps sessions (4), push size and frame size, and frees a session on stop or after 20 s idle.
  Errors are stable codes and media bytes are never logged. The live module does not call any model.
- **Cleanup.** Stop, `pagehide`, a track `ended` event and server errors all release tracks, the
  audio graph, timers, canvases and the server session. Stop also resets the segment grid and
  diagnostics, because the server discards the buffer.

### Diagnostics

| Row | Meaning |
| --- | --- |
| Buffer | valid seconds out of 10 |
| Timeline | seconds of timeline, audio and video |
| Frame delivery lag | audio end minus the newest delivered frame. Frames are sent after their second closes, so this is a sawtooth between roughly 0 and 1 s. Seen with a fake device: 180–760 ms. Expected, not a sync error |
| Worst frame offset | largest gap between a second's start and its chosen frame (limit 100 ms) |
| Audio | capture rate, gap count, output latency |
| Video | measured fps, frames, off-grid, late and browser-dropped frames |

The first second of a capture is usually marked missing audio: the audio worklet starts a moment
after the timeline origin. It rolls out of the window within ten seconds.

### Error handling

Permission denied, no device, device in use, a missing audio or video track, an insecure page, a
device unplugged mid-capture, a stalled server and a browser without an audio clock each give a
specific message and re-enable Start. Muted tracks are reported per second as `muted_audio`.

## Run locally

```
AVAL_ACCESS_MODE=gradio_auth AVAL_GRADIO_USERNAME=space-user \
AVAL_GRADIO_PASSWORD=synthetic-test-password AVAL_MODEL_ROOT=_local/models \
GRADIO_TEMP_DIR=$PWD/_local/gradio_tmp PORT=7861 python -m deployment.gradio_app
```

The credentials above are test values from `tests/test_phase8_deployment.py`. Camera access needs a
secure context: `localhost` or https.

## Tests

```
python -m pytest -q tests/test_phase8_live.py tests/test_phase8_deployment.py
```

`tests/test_phase8_live.py` covers the buffer, service limits, session binding and the Node tests
in `tests/js/test_live_timing.js` (clock mapping, frame picker, PCM encoding, error messages).

## Verification status

Checked in headless Chromium with its fake camera and microphone: the buffer reaches 10/10 aligned
seconds, Stop releases the session and resets the display, and the error messages above appear for
`NotAllowedError`, `NotFoundError`, `NotReadableError`, `OverconstrainedError`, `SecurityError` and
`AbortError` (injected by overriding `getUserMedia`).

**Not yet checked** (needs real hardware or other browsers):

- [ ] Chrome with a real webcam and microphone: permission denied, no device, device in use by
      another app, unplug mid-capture, hardware mute, background tab.
- [ ] Firefox: the 16 kHz `AudioContext` request is expected to fail and fall back to the device
      rate (the server resamples); confirm.
- [ ] Safari: `AudioContext` resume after the permission prompt, `getOutputTimestamp` support,
      `requestVideoFrameCallback` (the code falls back to `requestAnimationFrame`).

## Integration notes for the upload tab

`build_upload_section(gr, ui)` keeps its signature and is called inside the Upload tab of
`create_app()`. Replace its body freely: add the video preview and results dashboard there.
`DeploymentUI.submit` and its private endpoint must stay as they are. The live tab only needs the
`LiveCaptureService` created in `create_app()`.

When a model is ready, the hook is `LiveBuffer.to_raw_media_input()`: it returns a validated
`RawMediaInput` for the current window. Wiring it to inference belongs to the backend owner.

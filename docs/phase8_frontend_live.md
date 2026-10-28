# Phase 8 frontend: branded layout and live capture

Scope: Gradio layout, branding, navigation, responsive styling, a readable status card for the
**Upload verification** tab, and the **Live verification** tab (webcam and microphone capture of
ten aligned seconds). Predictions stay disabled: nothing in either tab produces or shows a score.

Not changed: `deployment/runtime.py`, Gradio authentication, upload validation,
`DeploymentUI.submit` and its private endpoint, launch options, and `audio/`, `vision/`, `fusion/`.

## Layout

`deployment/gradio_app.py::create_app()` builds a branded header, a collapsed startup-status
accordion and three tabs:

| Tab | Module | Owner |
| --- | --- | --- |
| Upload verification | `deployment/ui/upload_section.py` | Swaraj (status card added here) |
| Live verification | `deployment/ui/live_section.py` | frontend |
| About | `deployment/gradio_app.py` | frontend |

Theme, CSS and header live in `deployment/ui/theme.py`. The page has `width: 100%` and
`box-sizing: border-box` on `.gradio-container`; at 320, 375 and 1280 px,
`document.documentElement.scrollWidth` equals `clientWidth` on both the Upload and Live tabs. On
narrow screens Gradio folds the Live and About tabs into its "…" overflow menu.

## Upload tab

The upload callback is still `DeploymentUI.submit` with the same private endpoint, queue and
`concurrency_limit=1`. `upload_section.py` wraps its result for display only:

- `upload_card(result)` turns the sanitised result into a status card. Each known `error_code` has
  fixed text (`UPLOAD_MESSAGES`); unknown codes show `UPLOAD_NOT_ACCEPTED`, so no server text, path
  or raw error reaches the page. `MODEL_NOT_READY` and `REJECTED` are styled differently. The card
  never renders predictions, even if a result contained them.
- Changing or removing the file clears the card and the raw JSON, so an old result is never shown
  next to a new file.
- The raw result stays available under a collapsed "Technical details" accordion.

| Upload | Card |
| --- | --- |
| none | No file selected · `REJECTED · UPLOAD_REQUIRED` |
| valid 10 s H.264/AAC MP4 | Upload accepted, verification not available yet · `MODEL_NOT_READY · RAW_INFERENCE_GATE_CLOSED` |
| 4 s MP4 | Clip must be 10 seconds · `REJECTED · INVALID_DURATION` |
| text file named `.mp4` | File could not be read · `REJECTED · UNSUPPORTED_UPLOAD` |

## Live capture modes

- **Record 10 seconds** (primary): starts capture, waits for the first window of ten aligned
  seconds, runs the contract check, then releases the camera and microphone by itself. The grid,
  diagnostics and check result stay on screen. Takes about 12 s (ten seconds plus the start-up
  second and the 0.5 s finalise lag).
- **Start rolling capture**: keeps the latest ten seconds until Stop; "Check window" is enabled
  while the window is ready.

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
specific message and re-enable both start buttons. Muted tracks are reported per second as
`muted_audio`.

### Review findings from the Stage 2E checklist

`docs/phase8_frontend_integration_checklist.md` listed four open limits on this branch:

| Finding | Change |
| --- | --- |
| Audio base64 decoded before the aggregate bound | `push` checks every field's type and the total encoded length (≤ 2 s of int16 at the session rate) before any `b64decode`; frame strings are length-checked first too |
| `cv2.imdecode` ran before dimensions were rejected | `image_size()` reads width and height from the JPEG SOF / PNG IHDR header; anything over 1920 px per side (or unreadable) is rejected before pixel decoding, and the decoded shape must match the header |
| Bin labels 0..9 do not prove achieved timestamps | `live_check` now returns `timing`: each frame's capture time and offset from its second, the worst offset, the tolerance, source seconds, capture rate, whether resampling ran and how many samples were clipped. `frame_timestamps_are_bin_labels: true` says so explicitly |
| Browser preprocessing differs from the TRAIN MP4/FFmpeg path; asymmetric PCM scaling | PCM is now `round(x·32768)` clipped to int16 for both signs, the same scale as FFmpeg's flt→s16 and soundfile's /32768 read (tested bit-exact). `live_check` returns `capture` (`aval.live-capture.v0`) describing the audio, video and clock path with `encoder_compatibility: NOT_VERIFIED`, and `inference: NOT_CONNECTED`. Browser resampling, JPEG and canvas scaling remain different from the TRAIN path and are not claimed equivalent |

`CONTRACT_VALID` therefore means only that the shapes, dtypes and masks fit `RawMediaInput`. It is
separate from `MODEL_NOT_READY` and from any future prediction status, and live tensors must not be
routed into Phase 6 provenance or the five-MP4 TRAIN whitelist.

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
python -m pytest -q tests/test_phase8_live.py tests/test_phase8_upload_ui.py tests/test_phase8_deployment.py
node --test tests/js/test_live_timing.js
```

- `tests/test_phase8_live.py`: buffer, service limits, session binding, the review fixes (bounds
  before decoding, header-size check, achieved timing, clipping count, FFmpeg-scale PCM) and the
  Node tests in `tests/js/test_live_timing.js`.
- `tests/test_phase8_upload_ui.py`: a message for every known code, no echo of unknown server text,
  no scores, and the real Gradio wiring (private submit, concurrency 1, clear on change).

The full suite has 8 failures in `tests/test_fusion_contracts.py`; the same 8 fail on clean `main`.

## How to test by hand

1. Start the app with the command above and open http://localhost:7861 (Chrome or Edge).
2. Log in with the test credentials.
3. **Upload tab:** press Check upload with no file, then try a valid 10 s MP4 (H.264 + AAC), a
   shorter clip and a non-video file renamed `.mp4`. Expect the cards in the table above. Remove
   the file and check that the card resets.
4. **Live tab, Record 10 seconds:** allow camera and microphone. After about 12 s the state reads
   "Recorded: 10 aligned seconds", all ten cells are green, the check line reports the worst frame
   offset, and the camera light goes off.
5. **Live tab, rolling:** press Start rolling capture, wait for "Buffer ready", press Check window,
   then Stop. The grid and diagnostics reset to Idle.
6. **Errors:** block the camera in site settings and press Record (permission message); open the
   camera in another app first (in-use message); unplug the camera mid-capture (disconnected
   message); mute the microphone (purple muted cells); switch tabs (background warning).

## Verification status

Checked in Chromium with a synthetic camera and microphone stream (canvas + oscillator through an
overridden `getUserMedia`):

- Record 10 seconds finished in 12 s with 10/10 aligned seconds, worst frame offset 13.9 ms at
  28 fps, contract check passed, and the preview stream was released.
- Rolling capture reached ready in 12 s, the check passed (worst offset 15.2 ms), and Stop reset
  the display to Idle.
- `NotAllowedError`, `NotFoundError`, `NotReadableError` (plus `OverconstrainedError`,
  `SecurityError`, `AbortError` in the earlier cloud run) show their messages and re-enable both
  start buttons.
- Upload tab: the four cases in the table above, and the card reset on file change.
- No horizontal scroll at 375 px on the Upload and Live tabs (`scrollWidth == clientWidth`).

A hidden or background browser tab throttles `requestVideoFrameCallback` and marks seconds as
dropped frames; that is the intended behaviour, and the background warning covers it.

**Not yet checked** (needs real hardware or other browsers):

- [ ] Chrome with a real webcam and microphone: permission denied, no device, device in use by
      another app, unplug mid-capture, hardware mute, background tab.
- [ ] Firefox: the 16 kHz `AudioContext` request is expected to fail and fall back to the device
      rate (the server resamples); confirm.
- [ ] Safari: `AudioContext` resume after the permission prompt, `getOutputTimestamp` support,
      `requestVideoFrameCallback` (the code falls back to `requestAnimationFrame`).

## Integration notes for the upload tab

`build_upload_section(gr, ui)` keeps its signature and is called inside the Upload tab of
`create_app()`. Swaraj can replace or extend its body with the video preview and results dashboard;
`upload_card` and `UPLOAD_MESSAGES` can be reused for the status line.
`DeploymentUI.submit` and its private endpoint must stay as they are. The live tab only needs the
`LiveCaptureService` created in `create_app()`.

When a model is ready, the hook is `LiveBuffer.to_raw_media_input()`: it returns a validated
`RawMediaInput` for the current window. Wiring it to inference belongs to the backend owner.

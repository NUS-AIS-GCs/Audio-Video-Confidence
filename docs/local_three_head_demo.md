# AValidate local three-head demonstration

Run from the GC2 repository root with Docker Desktop running. The corrected app
uses the existing authenticated broker, signed media jobs and isolated worker.
Original encoder/E3 weights, Head 2 checkpoint and frozen thresholds are unchanged.

```powershell
& './_local/phase8_stage1/venv/Scripts/python.exe' -B -m deployment.run_product_demo --models './_local/temporal_presence/product-models-host' --private-dir './_local/three_head_demo/private' --port 7864 --image avalidate:local-oneshot --no-build
```

URL: **http://127.0.0.1:7864**. Do not start another supervisor while this instance
is running. Existing login credentials remain in
`_local/three_head_demo/private/login.json`; view them locally without sharing them.
If rebuilding is needed, omit `--no-build`. Stop with Ctrl+C in the launch terminal,
or run:

```powershell
& './_local/phase8_stage1/venv/Scripts/python.exe' -B -m deployment.run_product_demo --private-dir './_local/three_head_demo/private' --stop
```

## Five-step manual demonstration

1. Open the URL in Chrome or Edge and sign in with the existing local credentials.
2. Choose **Upload Video**, select a complete ten-second H.264/AAC MP4 or MOV,
   and click **Verify**. Inspect all three prediction sections. The original TRAIN
   fixture is `_local/phase8_stage2c/models/video/parity_samples/c---zaDCTaE.mp4`.
3. Clear the upload and verify `_local/product_demo/fixtures/cyclic-negative.mp4`
   to demonstrate substituted audio and the genuine E3 MISMATCH decision.
4. For your physical-device test, choose **Live Verification**, explicitly consent,
   click **Record 10 Seconds** once and allow camera/microphone access. Confirm
   the preview and microphone activity; keep this tab visible and provide audible
   sound throughout the ten seconds. Recording stops and results appear automatically.
5. Inspect all three heads, then click **Reset**. Confirm results clear and the
   browser camera/microphone indicators turn off. Cancel also stops an active recording.

Only simulated browser devices are authorized for automated tests. A successful
simulated recording does not establish physical sensor synchronization or guarantee
that a particular device/browser will satisfy the strict capture contract.

## Capture behavior

The audio clock is primed before acquisition, then a worklet captures exactly ten
seconds of native audio samples. Both modalities use the existing shared monotonic
clock mapping. Acquisition stops independently of server readiness. Pending bounded
packets drain before one authenticated end-of-input seal. This seal finalizes the
last real segment without waiting for the rolling buffer's delivery grace; it
preserves all existing validity, sample, frame, mute and clock checks. Nothing is
padded, repaired or timestamped with invented source times.

Previously server readiness controlled device stop, allowing invalid windows to run
into the 13-second hard cap. Startup audio blocks could also be discarded while
waiting for a clock, and preview-startup drops were included in recording diagnostics.
Exact digital silence still fails as muted audio. The advanced frame-lag diagnostic
measures audio end minus the last sampled frame's start time; about one second at
the end of a window is expected for frames sampled at seconds 0–9. It is distinct
from the per-frame clock offsets, whose 100 ms limit remains unchanged.

Both input modes use `ThreeHeadResults`: original 29-class event evidence, ten
H2-AVE-presence-v1 values at threshold **0.25**, and E3 MATCH/MISMATCH at threshold
**0.009579099009671126**. Scores are uncalibrated. Physical causation and ground
truth are not inferred from predictions. Model limitations and raw diagnostics
are expandable. Failed inputs produce no model scores.

## Files changed for this correction

- `deployment/gradio_app.py`
- `deployment/live/buffer.py`
- `deployment/live/research_capture.py`
- `deployment/live/service.py`
- `deployment/ui/general_research_section.py`
- `deployment/ui/live_section.py`
- `deployment/ui/result_sections.py`
- `deployment/ui/theme.py`
- `deployment/web/live_capture.js`
- `deployment/web/live_research_result.js`
- `tests/integration_product_demo.cjs`
- `tests/js/test_event_results.js`
- `tests/js/test_live_security.js`
- `tests/test_live_research.py`
- `docs/product_demo.md`
- `docs/local_three_head_demo.md`

Existing unrelated working-tree changes were preserved. No commit, push, merge,
public endpoint enablement, model retraining or remote deployment was performed.

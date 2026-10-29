# Final authentic upload and live integration handoff

2026-10-29, Asia/Singapore. This extends the existing premium UI work; the preceding
[UI release review](prediction_ui_release.md) and its evidence remain intact.

## A. Repository and review status

Fetched `origin/main` at the start: `53ede8f5b824c19355acf1f9f06ff33ca6601315`.
`feature/prediction-adapter-premium-ui` still has that HEAD with local uncommitted changes.
The premium implementation is therefore not part of main. No matching remote feature-branch head
was returned by `git ls-remote`. Anonymous GitHub PR lookup returned 404, so PR/review records could
not be verified through that API. The user explicitly confirmed **no independent release approval yet**.
No existing work was reset, discarded, committed, pushed, merged or deployed.

Incremental changes in this pass:

- New: `deployment/artifact_catalog.json`, `deployment/provision_artifacts.py`,
  `deployment/provisioned_app.py`, `deployment/release_gate.py`, `tests/test_final_integration.py`, this report.
- Extended: `deployment/prediction_adapter.py`, `deployment/ui/internal_section.py`,
  `deployment/gradio_app.py`, `deployment/live/service.py`, `deployment/web/live_capture.js`,
  `deployment/ui/live_section.py`, `deployment/prepare_release.py`, `tests/js/test_live_security.js`, `README.md`.
- `deployment/uploads.py`: documentation only; the Stage 2H transport and lease implementation are unchanged.

## B. Artifact identity and private provisioning

All original runtime artifacts are available locally and match the acquired SHA-256 bindings.
The new [catalog](../deployment/artifact_catalog.json) binds exact hashes **and byte sizes**, including
model cards, normalization, class order, checkpoint metadata, integrity records and operating points.
The catalog itself is SHA-bound in the provisioner. E3/A2 hashes also agree with `fusion.inference.CANDIDATES`.
This verifies acquired runtime identity; it does not reconstruct the missing historical export provenance.

An empty deployment model directory is missing these 14 required E3-profile files:

```text
audio/encoder.pt                 # BEATs iter3 base
audio/weights.pt                 # trained LoRA/head
audio/norm_stats.json
audio/class_names.json
audio/environment.json
audio/model_card.json
video/weights.pt                 # trained ResNet-18
video/class_names.json
video/environment.json
video/model_card.json
E3/best.pt
E3/checkpoint_metadata.json
E3/checkpoint_integrity.json
E3/frozen_operating_point.json
```

Optional A2 adds `A2/best.pt`, `checkpoint_metadata.json`, `checkpoint_integrity.json`, and
`frozen_operating_point.json`. E3-only payload is 415,103,976 bytes; all 18 files total 423,836,556 bytes.
The actual private hosted filesystem was not inspected remotely; the hosted missing-artifact state
is the supplied observation. No private artifact repository/revision or read credential was provided.

The provisioner supports verified offline copying or a **private Hugging Face model repository at a
full 40-character commit SHA**, using the already-pinned `huggingface_hub==2.3.0` SDK. It checks repository
privacy, revision and expected file sizes before retrieval, then verifies every copied byte against
the catalog. No token is accepted in a URL or command-line flag; use the server-side Secret
`AVAL_ARTIFACT_READ_TOKEN`. SDK errors are sanitized. Staging stays private, failures clean their
temporary directory, and the final directory is published atomically. Modified existing assets fail
closed instead of being silently repaired. Preflight requires twice the payload plus 32 MiB free disk.

Example operator commands, with actual reviewed paths/repository commit supplied separately:

```sh
python -m deployment.provision_artifacts --destination /private/models --verify-only --include-a2
python -m deployment.provision_artifacts --destination /private/new-models --source-root /source/models --include-a2
python -m deployment.provision_artifacts --destination /private/new-models --repo OWNER/PRIVATE_REPO --revision FULL_COMMIT_SHA --include-a2
```

The optional `python -m deployment.provisioned_app` entry point first requires existing Gradio
authentication, provisions/verifies artifacts, maps their actual producer-bundle paths into the existing
runtime configuration, and starts the **gated public UI**. Its settings are `AVAL_ARTIFACT_REPO`,
`AVAL_ARTIFACT_COMMIT`, the read-token Secret, and optionally `AVAL_PRIVATE_ARTIFACT_ROOT` (default
`/tmp/aval-private-models`). `AVAL_ARTIFACT_SOURCE_ROOT` selects offline copying instead of remote retrieval.
The default Docker command remains `deployment.gradio_app`; no download is automatic in that mode.
The provisioning wrapper currently includes A2 to satisfy the existing readiness inventory.

Use private runtime storage outside the application source/cache download surfaces. Mount verified
models read-only for inference where the hosting setup permits it. The existing middleware disables
browser model/media file routes, and the configured model root remains blocked. Keep checkpoints out
of Git and image layers. No remote private-repository download was claimed or performed; that still
needs the owner's repository, commit, Secret and hosted transfer/startup measurement.

## C. Upload path implemented and tested

The separate internal app now includes **Upload an authorized TRAIN original clip**. It reuses the
Stage 2H opaque handle and login-owned lease, checks exact permitted filenames/content hashes, then
calls the existing `InternalTrainBackend.predict_internal` on the uploaded bytes. That backend performs
existing admission, clock checks, bounded audio decoding, original TRAIN video extraction, aligned
embeddings and frozen E3 inference. Models remain initialized once per worker. The validated typed
response preserves threshold **0.009579099009671126**, full-precision raw score, model identity,
processing time and experimental status. No probability calibration is implied.

Only the five already-authorized TRAIN originals are accepted here. Unknown bytes never reach the
decoder. Another login cannot consume a lease; it is single-use and cleaned on success/failure.
Authentication is checked again before releasing a result, so login revocation drops it. Selecting
another file clears/cancels the old result. This is not a general upload release.

Browser evidence: an authenticated MP4 upload produced **0.9999999996978804**, `Original-like pairing`;
clearing/replacing it removed the result, and an unapproved upload returned `UNAPPROVED_TRAIN_SOURCE`.
See [genuine uploaded result](../_local/final_integration/screens/internal-upload-genuine.png) and
`_local/final_integration/internal-browser-results.json`. Public uploads retain the exact admission
contract: `MODEL_NOT_READY`, null predictions and `RAW_INFERENCE_GATE_CLOSED` after successful admission.

## D. Live path and pending requests

The original warning came from counting the normal asynchronous `live_stop` cleanup request as an
outstanding operation immediately after successful recording. Normal cleanup now says **Releasing
capture resources...**; the stalled warning appears only if requests remain outstanding after the
one-second presentation grace period. Restart stays disabled until the actual requests settle, even
if their logical timeout has expired. A late start's cleanup request cannot briefly re-enable controls.

Backgrounding stops capture and requires a fresh foreground recording. An old device event cannot
stop a newer capture. Current device disconnection and an interrupted audio clock release both
modalities. Reset discards diagnostics; a late completed-window response cannot restore them.
Stop/Reset and normal completion clear their cleanup notice when the server request settles.

The completed window still passes through the existing `RawMediaInput` validation. Its response now
also contains an explicit closed `prediction_readiness` scope, with `LIVE_INFERENCE_GATE_CLOSED`,
null predictions and no approved release ID. Existing `CONTRACT_VALID` / `inference: NOT_CONNECTED`
fields are preserved. **No approved live E3 forward path is claimed or enabled.**

Current browser audio is float32 -> clipped PCM16 -> float32, with whole-window torchaudio resampling
when needed. Resampling overshoot is clipped and counted for diagnostics. Historical audio used an
FFmpeg mono/16k PCM path. Current video uses browser decode/canvas scaling/JPEG quality 0.9 followed
by OpenCV RGB/INTER_LINEAR resizing; historical TRAIN video used OpenCV frame-index extraction and
resizing. Equal shapes, bin labels and <=100 ms frame offsets do not establish identical waveforms,
pixels, preprocessing or historical encoder outputs. Neither resampled/clipped audio nor browser JPEG
transport has independent model-input approval. The unchanged capture profile reports `NOT_VERIFIED`.

## E/F. Numerical, temporal and independent-release assessment

Conclusive: acquired artifact hashes, model identities, class order, feature/probability dimensions,
unchanged frozen decision rule, authorized TRAIN computation, typed responses, and tested ownership/
cleanup boundaries. Earlier five-source/ten-pair repeated raw inference evidence is retained in the
[preceding report](prediction_ui_release.md); it was not retuned or broadly repeated here.

Not reconstructible from current evidence: original float32 pre-cast export values from float16 alone,
the unknown historical batching/layout/runtime, missing independently attested producer trace bindings,
and general shared-clock/decoder behavior for new media. See the preserved
[audio trace assessment](phase8_final_audio_trace_verification.md) and
[video parity assessment](phase8_final_video_frame_parity_verification.md).

`TRAIN_FLOAT16_CELL_PLUS_RUNTIME_V1` remains inactive. Exact historical parity tests and preprocessing
contracts are untouched. New provisioned canonical TRAIN scores below differ slightly from the raw
decode profile; those differences are reported, not accepted as a tolerance. The prior closest swapped
case's distance to the frozen threshold was 0.0022858280315968963; this does not establish independent
near-threshold coverage. No TEST-driven tuning, retraining or checkpoint substitution occurred.

Additional evidence can support only a **bounded, independently reviewed compatibility claim**:
producer-attested manifests and hashes, original float32/intermediate outputs and runtime controls,
separate decoder and same-input encoder studies, prespecified TRAIN positives/negatives/near-threshold
confirmation, and explicit supported input/runtime scope. Saturated positive scores or observed maxima
from five clips cannot select universal budgets. A separate release decision must bind all of these.

**Independent decision: absent. Approved upload scope: none. Approved live scope: none.**
`deployment.release_gate` cannot be promoted by environment flags or caller-supplied approval values.
Artifact provisioning and successful temporal diagnostics grant no release authority.

## G/H. Genuine provisioning evidence and hardware

A network-isolated Docker smoke copied all 18 original files into private runtime storage, verified them,
loaded the authentic encoders and E3, and evaluated two prespecified canonical TRAIN pairs:

| Canonical TRAIN case | Actual E3 score | Decision |
| --- | ---: | --- |
| Original source 1 | 0.9999999996978661 | Original-like pairing |
| Source 2 audio + source 1 video | 0.0000038527716556650005 | Audio-substituted-like pairing |

Both used threshold 0.009579099009671126 and the unchanged E3 model fingerprint. Audio/video embeddings
were `(10,256)` / `(10,512)`; both event-probability arrays were `(10,29)`. No event labels were promoted
to validated output. The scores are genuine model computations, not UI examples or live predictions.

Observed under UID 1000, 2 CPU quota, 4 GiB memory limit, four intra-op/one inter-op threads:
cached local provisioning 1.830 s, encoder/E3 initialization 2.745 s, aggregate two-audio/one-video encoding
plus two fusion decisions 2.361 s, peak **process** RSS 1,185,714,176 bytes. Storage used a task-private
1.5 GiB tmpfs and was removed with the container. These figures exclude remote transfer and are not
HTTP latency or hosted CPU Basic capacity guarantees. Prior full internal runtime measurements remain
applicable context. Use one worker/replica and measure hosted startup, cache space, container memory,
warm latency and transport limits before approving hardware. UI hosting success alone is insufficient.

## I. Validation and reviewed bundle

- Full regression: **1,022 passed, 1,346 passing subtests, two CUDA skips**, exit 0; one existing
  weight-norm deprecation warning. Focused adapter/security tests and all **14 Node-reported tests** passed.
- New tests cover catalog tampering, sizes/hashes, atomic failure cleanup, disk refusal, modified-cache
  refusal, private/pinned repository requirements, SDK error sanitization, immutable release state,
  unknown upload rejection, login-specific leases, revocation and completed-window closed readiness.
- Browser tests against the exact new public image passed login, local preview/admission, mobile layouts,
  synthetic-device ten-second capture, cleanup-message clearing and Reset, with no page errors. The
  separate internal app passed genuine uploaded TRAIN inference, stale-result clearing and unknown-file
  rejection. Physical devices and other browser engines were not tested.
- A new **87-file** allowlisted bundle built successfully as `avalidate:final-integration`. Its manifest
  and source hashes match the worktree. Optional provisioning code is included; internal experimental
  entry points stay separate. The default remains authenticated, gated UI on port 7860 as UID 1000.
  Missing artifact-source configuration safely refuses the optional provisioning startup.
- Server-side Stage 2F `deployment/live/security.py`, live buffer numerics, runtime/parity gates, encoder construction and internal backend
  remain byte-identical. All 40 acquired model-bundle files still match their original SHA-256 hashes.
  Prior full five-minute expiry/restart/shutdown and image-layer audits were reused; unchanged controls
  were not subjected to another broad audit. New lease failure/cleanup and actual browser flows were tested.

Evidence is in `_local/final_integration/`, including `regression-execution.json`, `provision-runtime.json`,
`browser-results.json`, `internal-browser-results.json`, `preservation.json` and `release-manifest.json`.
The default bundle is `_local/final_integration/release`. No weights, media or credentials are in it.
Preserve the existing Private Space, existing Gradio Secrets, `AVAL_ACCESS_MODE=gradio_auth`, non-root
execution and upload protections. An owner must review and manually publish any candidate revision;
no push/merge/deployment is performed here. Runtime Secrets and non-root behavior follow the official
[Hugging Face Docker Spaces guidance](https://huggingface.co/docs/hub/spaces-sdks-docker).

## J/K. Remaining work and verdicts

| Blocker | Owner role | Next action |
| --- | --- | --- |
| Historical audio provenance and exact numeric reconstruction | Audio producer | Supply independently attested trace/source/checkpoint bindings, original float32 exports and original runtime/batch details; identify unrecoverable evidence explicitly |
| Historical visual batching/layout and frame provenance | Video producer | Recover original export controls and float32 evidence; reconcile frame indices and achieved PTS without relabeling them |
| No accepted numerical/media scope | Independent ML reviewer | Review prespecified TRAIN calibration/confirmation and frozen-threshold sensitivity; approve only a justified bounded scope |
| No general compressed-media inference decoder approval | Backend/media security owner | Implement/validate a bounded general decoder preserving the required preprocessing, clocks and resource isolation; the current OpenCV TRAIN extractor explicitly is not one |
| Live input differs from historical path | Live/frontend + encoder owners | Validate browser PCM/resampling/clipping and canvas/JPEG/RGB transformations on the declared device/browser scope before implementing an approved E3 forward path |
| No independent release decision | Independent release reviewer | Publish a reviewed decision binding exact artifact, preprocessing, numerical, decoder, runtime and input-scope identities; do not replace it with an environment flag |
| Private artifact remote source and hosted performance unverified | Space/artifact owner | Provide a private model repository, full commit and read-only Secret; stage retrieval and benchmark CPU Basic/startup/disk/memory privately |
| Changes not reviewed/merged/published | Repository/Space owner | Review the uncommitted diff and new sources, then manually control PR/merge/publication of the appropriate gated bundle |

| Requested verdict | Status |
| --- | --- |
| UPLOAD ADAPTER | **FIX REQUIRED for general media.** Authenticated, exact-byte TRAIN upload path is implemented and tested |
| LIVE ADAPTER | **FIX REQUIRED.** Capture/contract/closed-readiness integration works; approved live model forwarding is unavailable |
| PUBLIC PREDICTION | **VALIDATION BLOCKED**; neither artifacts nor diagnostics enable it |
| HUGGING FACE RELEASE | **FIX REQUIRED for inference release.** The gated 87-file UI/provisioning bundle is built and ready for owner review; not deployed |

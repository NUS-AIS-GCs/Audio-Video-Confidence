# Stage 2D — Runtime performance and backend integration preparation

2026-10-28, Asia/Singapore. **FIX REQUIRED. Stage 2D stops here.**

A genuine internal backend now runs validated, SHA-whitelisted TRAIN MP4s through both authentic encoders and frozen E3, with optional A2. Models are loaded once and requests are serialized. Both Docker storage profiles preserve every feature value from the preceding Linux five-clip study. Normal user-facing predictions remain disabled; no historical parity contract or frozen threshold changed.

## A. Files changed

Seven new reviewable files; all pre-existing files remain unchanged:

- `deployment/internal_backend.py`: isolated TRAIN request orchestration, reuse, per-stage timing, full integrity checks, cleanup and validated experimental response envelope.
- `deployment/stage2d_profile.py`: genuine cold/warm/inference-mode comparisons, actual checkpoint-load counts, model identity and prior five-case output checks.
- `deployment/run_stage2d_docker.py`: isolated, resource-limited profiling with read-only Windows/native-volume model mounts and automatic task-owned cleanup.
- `deployment/seed_artifact_volume.py`: hash-verified temporary volume seeding; exact byte copy of the seed helper executed in this audit.
- `tests/test_phase8_internal_backend.py`: source/filename/size/clock guards, reuse dispatch, concurrency, cleanup, error sanitization, readiness and genuine response contract checks.
- `docs/phase8_stage2d_internal_contract.md`: internal/public boundary and stable response/lifecycle handoff.
- This report: `docs/phase8_stage2d_runtime_backend.md`.

Audit-only scripts, profile JSON, arrays, logs, image-layer verification and test evidence are retained under ignored `_local/phase8_stage2d/`. No checkpoints or private clips are added to Git.

## B. Runtime cold/warm latency

Both profiles executed in Docker on the same immutable Linux runtime image, with **2 CPU quota, 4 GiB RAM, four PyTorch intra-op threads, one inter-op thread, CPU float32, autocast disabled, eval mode and no gradients**. Source clips and weights are identical. New process/model initialization is measured; hardware/disk caches were not flushed. Samples are five permitted TRAIN clips, with one repeated warm pass; these are observed timings, not a capacity guarantee.

| Measurement | Native read-only volume | Windows read-only bind |
| --- | ---: | ---: |
| Model/backend initialization, including manifest and TRAIN-reference authentication | 5.579 s | 78.735 s |
| First request after loading (Church bell) | 3.348 s | 20.405 s |
| Cold backend initialization + first request | 8.927 s | 99.140 s |
| Mean first pass over all five clips, after loading | 3.606 s | 18.092 s |
| Mean repeated warm request | 3.653 s | 22.128 s |

Native snapshot preparation separately copied/authenticated 40 files, 488165925 bytes, in **21.999 seconds**. Including that one-time seeding cost, native initialization plus first request was **30.926 seconds**, excluding Docker/Python import startup. The source and snapshot hashes match. Snapshots were read-only during inference and the temporary volume was removed afterwards.

The final profiler times the complete `predict_internal` function call from in-memory bytes through its returned response, including source authentication, admission, integrity checks, decoding, inference, alignment, fusion, cleanup, diagnostic/audit-array copying and lock release. Caller file reading, network upload, model initialization, worker imports and JSON encoding are excluded. Mean response validation plus JSON encoding is separately 0.000190 s (native) / 0.000204 s (bind). It is not HTTP round-trip time. Different clip resolutions/codecs affect decode time.

| TRAIN sample | Native warm request | Windows-bind warm request |
| --- | ---: | ---: |
| `c---zaDCTaE__08` | 2.970 s | 19.820 s |
| `22DapbNwnUY__12` | 4.101 s | 19.651 s |
| `0h8HxZaX9wQ__26` | 5.276 s | 21.806 s |
| `6VRcINAY7KE__07` | 3.120 s | 23.995 s |
| `NGeniwxDzJ0__21` | 2.798 s | 25.370 s |

The preserved initial Stage 2D run measured 1.983 s native / 4.738 s bind for warm prediction-core time, with 3.387 s / 16.819 s initialization. The final run above adds precise outer-call and separate JSON timing, but the much larger run-to-run difference cannot be explained by that tiny measurement overhead. Results show substantial host/storage/scheduling variability; no single run is a production latency guarantee. Both runs preserve all feature outputs exactly. Both sets of evidence remain available.

The previous **379.7-second Stage 2C probe is not a single-request latency**. It included initialization, five raw replays, five canonical replays, repeated integrity/export checks, both fusion comparisons and extensive numerical/resource reporting. No speedup ratio is computed against that bulk-probe number.

## C. Per-stage latency

Initialization breakdown:

| Stage | Native volume | Windows bind |
| --- | ---: | ---: |
| manifest_authentication | 0.4325 s | 11.1703 s |
| audio_model_loading | 4.3924 s | 53.9120 s |
| video_model_loading | 0.2362 s | 5.7034 s |
| fusion_and_train_reference_loading | 0.5114 s | 7.9462 s |

Mean stage wall time over the five warm requests:

| Request stage | Native volume | Windows bind |
| --- | ---: | ---: |
| media_validation | 0.107773 s | 0.107486 s |
| integrity_pre | 0.413398 s | 9.499659 s |
| audio_decode | 0.127265 s | 0.117038 s |
| audio_preprocess | 0.005041 s | 0.003537 s |
| audio_encoder | 1.121670 s | 1.069266 s |
| video_decode | 1.072811 s | 1.077670 s |
| video_preprocess | 0.009124 s | 0.007129 s |
| video_encoder | 0.353321 s | 0.339730 s |
| alignment | 0.000240 s | 0.000222 s |
| E3 | 0.000832 s | 0.000581 s |
| A2 | 0.011521 s | 0.001445 s |
| integrity_post | 0.425441 s | 9.899823 s |
| total | 3.653170 s | 22.128194 s |

Audio preprocessing is the actual normalized BEATs filterbank, timed at the existing patch-embedding boundary; audio encoder time includes the remaining authentic BEATs/LoRA/head forward. Video decode time includes BGR→RGB and INTER_LINEAR resizing; video preprocessing is float32 `/255` and ImageNet normalization. Fusion entries are genuine model forwards; integrity work is separately visible. Total also includes small dispatch, hashing/copy and cleanup costs, so it need not equal the displayed stage sum.

The same request verifies checkpoint files and live model fingerprints before and after inference. Native warm integrity time is 0.839 s versus 19.399 s on Windows binds. The measured 6.06× warm-request ratio compares these two new profiles, not the old bulk probe. It is a single controlled audit on this host; no universal latency claim is made.

## D. Peak memory

| Scope | Native volume | Windows bind |
| --- | ---: | ---: |
| Python process peak RSS | 1.188 GiB | 1.189 GiB |
| Container cgroup peak | 1.774 GiB | 1.071 GiB |

Both containers exited 0 with `OOMKilled: false` under the 4 GiB limit. RSS and cgroup memory use different accounting; neither should be substituted for the other. Memory includes model initialization and profiling, not just one neural forward. Full cgroup CPU/throttling/memory counters, timestamped Docker stats and per-stage process CPU seconds are retained. The native profile covers 16 real requests; the bind profile covers 11, each including an E3-only optional-A2 check.

## E. Optimizations implemented and measured

1. **One model registry per worker.** Actual `torch.load` instrumentation records seven restricted loads during initialization in each fresh process and zero further loads through all requests. These seven calls include the existing strict audio configuration/state loader plumbing. Each encoder and fusion model object remains the same across requests. The previous diagnostic also reused models; the new service makes that lifecycle explicit and testable rather than claiming that all prior latency was repeated loading.
2. **Authenticated native artifact snapshot.** Full SHA-256 and live model checks remain enabled. A new temporary volume is populated from read-only originals, every source/copy hash is checked, files are non-writable and the inference mount is read-only. This reduces storage overhead without stat-only trust, removed verification, new weights or image-layer checkpoints. It is an internal runner/storage option, not public activation.
3. **Request-only orchestration.** Each request probes once, decodes audio once and samples video once. TRAIN exports are authenticated/loaded at startup for this audit and are not rescanned for every actual E3 forward. Raw features never receive historical export provenance. The service performs no canonical-reference second replay; those are separate profiling passes.
4. **Original numeric operations retained.** Four/one CPU threads, float32, normalization, producer frame selection, audio crop and original tensor layout/copy operations remain unchanged. No quantization, mixed precision, checkpoint replacement, sampling approximation, retraining or threshold change is introduced.
5. **Eval and inference-mode investigation.** All models stay frozen/eval. `torch.inference_mode()` was compared on all five clips and preserved embeddings and E3/A2 scores exactly. Its mean request time was 3.518560 s versus 3.653170 s for warm `no_grad`, an unconvincing difference at this sample size. `no_grad` remains the default; inference mode is an explicit internal profiling option. PyTorch documents its extra tracking restrictions and evaluation semantics in [the 2.11 API](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.inference_mode.html).
6. **Bounded lifecycle.** A nonblocking request lock, bounded subprocesses/output, private temporary directories, OpenCV capture release, hook removal, result isolation and task-owned container/volume cleanup are implemented. Models intentionally remain resident until worker exit.

All audio/video feature max error, MAE and RMSE against the prior Stage 2C Linux raw outputs are **zero** for all studied requests in both storage profiles. Warm repeats and inference-mode feature/score changes are also zero. The no-A2 request produces exactly the same E3 result. Removing operations that could change floating-point layout/order was deliberately avoided after Stage 2C demonstrated their sensitivity.

## F. Internal E3/A2 execution

The isolated path is approved TRAIN bytes → existing Phase 8 media admission plus zero-origin clock/dimension/FPS checks → producer-compatible decode → authentic encoders → existing EncoderOutput geometry/provenance checks → frozen E3 and optional A2 → experimental response. It is not connected to the normal request handler.

E3 threshold remains `0.009579099009671126`; A2 remains `0.0026640388978268056`. Checkpoint/model hashes, original operating points and complete states remain authenticated. Scores follow the existing Phase 6 float64 stable-sigmoid convention and are explicitly **uncalibrated correspondence scores for original pairing provenance**. Binary decisions remain `score >= frozen threshold`.

The reference comparison below uses authenticated export features at the same single-pair batch size as actual requests:

| Model | Max raw-vs-frozen-export score difference | Mean abs score difference | Max logit difference | Stable decisions |
| --- | ---: | ---: | ---: | ---: |
| E3 | 8.5946856298591e-09 | 1.86972171078992e-09 | 0.000293731689453125 | 5/5 |
| A2 | 2.652042774276e-07 | 6.19024015824365e-08 | 0.000810623168945312 | 5/5 |

A separate controlled Docker replay verifies that recorded request logits match same-batch re-execution exactly. E3 logits also match the prior five-pair batch exactly; A2 changes by at most `9.5367431640625e-7` between batch size one and five. The same difference occurs when replaying both batch sizes on the same loaded frozen model. This is a measured batching effect, not a threshold/weight change or a claim of universal bitwise batch equivalence. Across the native/bind storage profiles, warm repeats and the inference-mode comparison, actual E3/A2 request scores and logits are unchanged. Full rows are in `fusion-check/fusion-request-check.json`.

These are genuine observed internal computations; no placeholder, annotation-derived or synthetic prediction is used. All original positive pairs remain positive and near sigmoid saturation. Stability does not establish encoder parity, accuracy on negatives, calibration or physical-source causality. Existing strict export comparisons still show **1,591 audio / 12,800** and **59 video / 25,600** float16 differences for the Linux raw decode profile, unchanged from Stage 2C. No tolerance is active.

## G. Frontend/backend contract readiness

The [new internal contract](phase8_stage2d_internal_contract.md) defines versioned status, sanitized errors, file/byte admission, frozen E3 score/decision, optional A2, timing and readiness. `response_for_review` removes audit tensors and validates the body; all **27 completed genuine response bodies**, including E3-only responses, pass the final contract validator.

Successful offline execution is labeled `EXPERIMENTAL_INTERNAL_TRAIN_ONLY`, with `production_predictions_enabled: false`, public `MODEL_NOT_READY`, and null event/temporal outputs. The validator rejects changed hashes/thresholds/decision semantics, missing E3, nonfinite values, unvalidated event output and readiness promotion. Private audit feature arrays are not JSON/UI fields. Ordinary frontend requests continue to use the existing admission envelope with null predictions.

Branch boundaries: the current local worktree is `feature/phase8-backend-stage1`; local refs and cached remote refs, including `feature/phase8-final-system`, were inspected read-only. There is one local worktree. Teammates’ active workspace state is not available from these refs; no branch was checked out or modified. `deployment/gradio_app.py`, `deployment/runtime.py`, app/UI source and all prior frontend-facing contracts are preserved byte-for-byte. This stage prepares an interface; it does not integrate a public prediction callback.

## H. Security verification

- **Authentication/public readiness:** existing Gradio startup requires valid configured auth; launch uses auth, private callback API, no share/monitoring, and hides errors. Existing readiness remains code-owned and closed. Public admission/regression tests pass; the new internal class is not imported or invoked by Gradio.
- **Media:** 50 MiB byte limit; normal format/codec/stream/duration rules preserved. Internal MP4 requests additionally require the approved TRAIN identity/content hash, zero starts, bounded video geometry/FPS and successful original-bin reads. Missing/ambiguous streams, malformed metadata, source mismatch and invalid output are rejected. Arbitrary unwhitelisted bytes never reach OpenCV.
- **Bounds/cleanup:** inherited eight-second process deadline and 256 KiB diagnostic-output cap; 64 MiB decoder allocation limit; bounded WAV output (322,048 bytes), exact 160,000-sample finite waveform, one internal concurrent request. Per-request private files and hooks are removed on success/error/timeout; tests exercise cleanup after decoder and model failures. Existing frontend concurrency 1 and queue size 8 are unchanged.
- **Container:** same immutable image `sha256:0ee95003d0f32ab19176ffe86e8cdc015644a0d48c03571e81c94b07858dbaab`; no new build, ports or server. Network none, read-only root/source/artifact mounts, dropped capabilities, no-new-privileges, 2 CPUs, 4 GiB RAM, 128 PIDs and 512 MiB tmpfs. Only audit results/tmpfs are writable during inference. The seed-only container writes solely to its new temporary volume; inference uses the image’s non-root user.
- **Image layers:** a streaming audit checked all **12/12 saved image-manifest layers**, 39,897 regular entries, including empty layers. No known private payload SHA-256 matches and no project checkpoint/media entries were found. No image archive was materialized. `.dockerignore` still excludes `_local`, checkpoints, arrays and private media. This verifies the reused image as well as the source build exclusions.
- **Numerical/timing gates:** source streams share observed zero origins and one container, but fractional-FPS achieved video PTS are retained separately from intended bin labels. `phase6_shared_clock_accepted` remains false; no fabricated `shared_clock_pts_v1` or export provenance is attached. Strict parity, inactive tolerance policy and public gates are untouched.

## I. Full regression

**Focused: 217 passed, 9 passing subtests, zero skips/failures/errors. Full regression: 841 passed, 1,346 passing subtests, 2 unavailable-CUDA skips, zero failures/errors.** Both child process exits are 0. One existing PyTorch weight-norm deprecation warning. Final XML, complete logs and independent exit-code records are retained. The volume-seeding source promoted into `deployment` is byte-identical to the executed helper; the final coordinator CLI was checked. `git diff --check` passes.

## J. Historical preservation

Repository: `NUS-AIS-GCs/Audio-Video-Confidence`; branch `feature/phase8-backend-stage1`; HEAD `eff4e2fe6ed33650c07fb8f719fae9a665a7faf1`. All **1,104 baseline files** match preflight SHA-256 hashes, including Stage 1/2B/2C implementations, reports, authentic checkpoints/exports and previous measurement evidence. Separately, all **28 protected archived reports** match saved anchors. All proposed changes remain new uncommitted files. No merge, commit, push, deployment, TEST evaluation, model/threshold tuning, or GC1 access occurred.

## K. Remaining external dependencies

1. Producer float32 intermediate outputs, original waveform/pixel-cache hashes, exact source/runtime/batch/layout evidence and decoder reconciliation remain outstanding. Stage 2C measured 85 one-PCM-step waveform differences between decode stacks; this stage preserves the existing Linux profile rather than hiding them.
2. Exact historical encoder parity continues to fail. Independently reviewed numerical budgets and achieved-PTS/decoder policy are absent. The separately named candidate policy remains inactive and cannot substitute for exact parity.
3. Only five permitted TRAIN clips are available. The required full 6,678 primary TRAIN-pair release study and independent acceptance review remain incomplete.
4. A secure general compressed-media decoder satisfying the existing strict shared-clock contract and production admission evidence is still required. The SHA-whitelisted local study is not a general upload decoder.
5. Frontend/backend owners must review the eventual post-gate public contract and rollout separately. Current backend preparation and finite scores cannot activate it. Fresh image building or Hugging Face deployment was not performed or requested for this stage.

## L. Decision and reproduction

**FIX REQUIRED.** Runtime/backend preparation is concrete and reviewable, genuine warm inference is measured, all prior Linux outputs are preserved, and security/regression checks pass. Final inference readiness still depends on external parity/timing evidence and independent review. Normal predictions remain `MODEL_NOT_READY`. Stop after Stage 2D.

From this repository in host PowerShell, reuse the preserved Stage 2C authenticated staging set and select a new audit run name:

```powershell
$python = (Resolve-Path "_local/phase8_stage1/venv/Scripts/python.exe").Path
$docker = Join-Path $env:LOCALAPPDATA "Programs/DockerDesktop/resources/bin/docker.exe"
& $docker version
& $python deployment/run_stage2d_docker.py --run-name profile-recheck
if ($LASTEXITCODE -ne 0) { throw "Stage 2D profiling failed; inspect saved execution logs" }
```

The coordinator rejects an existing run directory, authenticates an immutable image and every model snapshot, profiles native/bind storage, records actual model loads, verifies prior five-case tensors, and removes only its own containers/volume. No server or production prediction route starts. All inputs remain read-only; no checkpoints enter image layers.

Primary evidence: `_local/phase8_stage2d/profile-final/{native-volume,windows-bind}/run/profile.json`, per-case arrays, `execution.json`, `stats.jsonl`, `seed.json`, `response-contract-validation.json`, `image-layer-audit.json`, final focused/regression XML/logs and preservation records. The first profile is preserved separately under `profile-01`. Exact request CPU/wall times and frozen model/encoder stamps are included for independent review.

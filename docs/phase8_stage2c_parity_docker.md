# Phase 8 Stage 2C — Parity reconciliation and Docker encoder verification

2026-10-28, Asia/Singapore. **FIX REQUIRED. Stop after Stage 2C.**

Genuine Docker execution is now verified. Numerical parity remains unresolved: encoder runtime changes and a separate audio decoder-stack difference are measured. The strict historical gate, frozen thresholds and public prediction block are unchanged. No retraining, held-out TEST scoring, calibration, deployment, merge, commit or push occurred.

## Preflight and preservation

Repository `NUS-AIS-GCs/Audio-Video-Confidence`; branch `feature/phase8-backend-stage1`; HEAD `eff4e2fe6ed33650c07fb8f719fae9a665a7faf1`. All pre-existing Stage 1/2B files remain uncommitted and unchanged. The Stage 2C baseline covers **965 files**, including protected source, original artifacts/exports, previous reports, measurements, logs and audit scripts (excluding disposable test/cache directories). Before/after hashes show zero changes. All 28 protected archived Phase 7 reports match their saved hashes. Teammate frontend branches and GC1 were not accessed.

The isolated audit staging set contains **40 authenticated model/source files**, plus SHA-bound canonical inputs. Only TRAIN exports and the five permitted TRAIN MP4s were staged; no held-out arrays were mounted. Checkpoints, media, staging and all measurements remain under Git-ignored `_local/phase8_stage2c/`. Source artifacts remain untouched. The externally supplied audit manifest SHA-256 is `65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e`.

| Model | Frozen checkpoint SHA-256 | Unchanged threshold |
| --- | --- | ---: |
| E3 | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` | 0.009579099009671126 |
| A2 | `0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12` | 0.0026640388978268056 |

Both candidates use the original strict checkpoint/metadata/integrity/operating-point loaders and live model fingerprints. Encoder loading remains restricted (`weights_only=True`), with complete trained-state coverage. Audio base SHA-256 is `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34`; adapter/head is `d4a0f78403caafde2c3c39d961c9819fbae8caebaaef4b824034c678da860692`; video is `6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec`. Normalization values and class ordering are authenticated and unchanged.

## A. Audio drift diagnosis

The context combines two historical CPU profiles. Stage 1 recorded 572 audio float16 mismatches. Stage 2B’s explicitly four-thread paired probe recorded 568, as its original report states. The new four-thread host replay exactly reproduces the Stage 2B audio and video **float32 arrays**, not just their mismatch totals.

A fresh process reran a copy of the preserved original Stage 1 script, changing only its output directory and ROOT locator. It reproduced the **aggregate 572** audio mismatch count, but not the original float32 values or mismatch set: the maximum live-to-live change is `7.510185241699219e-06`, with 549 common mismatches, 23 only in the old run, and 23 only in the replay. Aggregate-count agreement must not be described as exact numerical reproduction.

| Sample (video-bundle order) | Original Stage 1 mismatches | Fresh original-script mismatches | Host four-thread mismatches | Docker same-canonical-input mismatches | Docker raw MP4 mismatches |
| --- | ---: | ---: | ---: | ---: | ---: |
| `c---zaDCTaE__08` | 132 | 134 | 132 | 135 | 254 |
| `22DapbNwnUY__12` | 96 | 98 | 99 | 99 | 138 |
| `0h8HxZaX9wQ__26` | 99 | 98 | 97 | 97 | 358 |
| `6VRcINAY7KE__07` | 166 | 163 | 164 | 163 | 183 |
| `NGeniwxDzJ0__21` | 79 | 79 | 76 | 77 | 658 |

Controlled host experiments keep source media, waveform bytes, weights, normalization and feature order fixed. They test execution profiles, not model variants; no profile is selected because it gives fewer mismatches.

| Controlled profile | Audio mismatches / 12,800 | Video mismatches / 25,600 |
| --- | ---: | ---: |
| Four-thread baseline | 568 | 66 |
| `threads20` | 570 | 66 |
| `threads1` | 635 | 66 |
| `mkldnn_off` | 569 | 57 |
| `batch2_audio_batch5_video` | 571 | 38 |

Operation-level trace, one fixed Church bell TRAIN clip, same loaded authentic model:

| Boundary | Max abs change, 20 vs 4 CPU threads |
| --- | ---: |
| normalized_fbank | 1.49011611938477e-07 |
| transformer_tokens | 5.7220458984375e-06 |
| pooled_and_standardized | 4.1574239730835e-06 |
| embeddings | 4.29153442382812e-06 |
| probabilities | 1.37370079755783e-08 |

Within each tested thread setting, immediate repeats were exactly equal through every traced stage. Divergence first appears in the normalized filterbank and propagates through BEATs and the projection head. This demonstrates one concrete CPU arithmetic source; it does not isolate the original GPU/export discrepancy. All dropout modules are in eval mode. Kaldi fbank’s actual installed dither default is 0.0. No dropout, augmentation, waveform repair, normalization epsilon or class/coordinate permutation was introduced.

NumPy float16 conversion and PyTorch `.half()` agree for every observed feature value. The saved mean/std buffers equal the original JSON values exactly. Audio pooling retains time-major tokens, eight frequency patches, original time-centre bin assignment and the trained GELU projection. A universal ULP threshold would be unreliable near zero; exact mismatch locations and rounding-cell excess are recorded separately.

Producer audio environment: Python 3.13.16, Torch/TorchAudio 2.11.0+cu130, NumPy 2.1.3, Pandas 2.2.3, FFmpeg 6.1.1 and Tesla T4. Host: Python 3.11.16, Torch/TorchAudio 2.11.0+cpu, NumPy 2.2.6, Pandas 2.3.3 and Windows FFmpeg 7.1. Original source defaults export batch 32; this is evidence about source defaults, not an independently locked actual export invocation.

### Separate decoder-stack difference established in Docker

All five MP4 source hashes remain identical across modalities and mounts. The Debian FFmpeg 5.1.9 path and Windows FFmpeg 7.1 path use the same mono/16 kHz/10-second PCM-s16 WAV command, yet differ in exactly **85 of 800,000 decoded samples**. Every difference is exactly **one signed PCM-s16 step**, `1/32768 = 0.000030517578125`. The canonical-input experiment is explicitly separate from raw decoding; canonical inputs were never substituted into an admitted production request.

| TRAIN sample | Different waveform samples / 160,000 | Waveform MAE | Waveform RMSE |
| --- | ---: | ---: | ---: |
| `c---zaDCTaE__08` | 37 | 7.05718994141e-09 | 4.64077951835e-07 |
| `22DapbNwnUY__12` | 4 | 7.62939453125e-10 | 1.52587890625e-07 |
| `0h8HxZaX9wQ__26` | 15 | 2.86102294922e-09 | 2.95485179612e-07 |
| `6VRcINAY7KE__07` | 28 | 5.34057617187e-09 | 4.03709611674e-07 |
| `NGeniwxDzJ0__21` | 1 | 1.90734863281e-10 | 7.62939453125e-08 |

Under the same Linux encoder/runtime, raw-decoded vs canonical audio embeddings differ by max abs `0.0008680820465087891`, MAE `2.0099561038564583e-05`, RMSE `5.5256506455878775e-05`. Thus the raw Docker discrepancy is not solely an encoder arithmetic issue. The experiment separates decoder-stack input drift from runtime drift; it does not identify whether a particular AAC, resampling or quantization implementation causes the one-step differences. Producer FFmpeg 6.1.1 and its original decoded waveforms were not reproduced.

## B. Video drift diagnosis

All five host cases reproduce Stage 2B exactly: 66 embedding float16 mismatches, with all 50/50 per-second labels equal. Linux preserves every selected RGB/resized pixel byte but produces 59 embedding mismatches. Host Torch and NumPy normalization tensors are byte-identical; dtype boundaries remain uint8 → float32/255 → ImageNet normalization → float32 network → float16 reference comparison.

With identical canonical pixels, Linux vs Windows video embeddings differ by max abs `2.6226043701171875e-06`, MAE `2.600271510289076e-07`, RMSE `3.798034881716932e-07`. Host contiguous-layout and backend/batch experiments also change outputs by a few float32 micro-units despite identical input values. Contiguous layout for the first clip changes embeddings by max abs `3.4570693969726562e-06`. These measurements establish execution-dependent arithmetic variation; they do not prove exact equivalence to original exports.

Historical extraction remains `int(t * unrounded_fps)`. At 30000/1001 FPS the selected indices are `[0,29,59,89,119,149,179,209,239,269]`; PTS precede intended seconds by up to 32.3667 ms. At 25/30 FPS the selected frames are at integer seconds. Host seeking, sequential decoding and a separate FFmpeg decode agreed on every selected pixel in Stage 2B; Docker now agrees as well. No invalid, padded, clamped, repeated or missing selected frames are accepted. Actual PTS and stream clocks are retained separately from intended bin labels.

Docker ffprobe confirms H.264/AAC streams with zero starts. Video durations are 10.04 seconds for Church bell, 10.01 for the three fractional-FPS clips, and 10.0 for Ukulele; Church bell audio is 10.04644 seconds, the other audio streams 10.0. All decoded study waveforms have exactly 160,000 samples after the documented `-t 10` crop. The Church bell frame at PTS 10.0 is outside requested seconds 0–9. These observations do not certify the existing Phase 6 exact shared-clock admission contract.

The supplied producer video environment contains ranges/alternatives, not an exact lock: Python `3.11 / 3.14`, Torch `2.14.1+xpu / 2.x`, TorchVision `0.20+ / 0.x`, OpenCV `4.10+ / 5.0.0`, NumPy `2.2+ / 2.5.4`. Current export source defaults to batch 16 and a DataLoader-stacked contiguous tensor; live inference uses batch 1 and a different memory layout. The exact historical batch/device/source revision and original frame-cache hashes are not supplied. Therefore selection matches the documented historical algorithm, but cannot be cryptographically compared to the absent original pixel cache.

### Aggregate embedding errors against authenticated frozen TRAIN float16 rows

| Execution/input profile | Modality | Max abs error | MAE | RMSE | Float16 mismatches |
| --- | --- | ---: | ---: | ---: | ---: |
| Windows raw | audio | 0.00381565093994141 | 0.000159402651867152 | 0.000344128923612881 | 568 |
| Windows raw | video | 0.00193881988525391 | 0.0001631338761905 | 0.000262401859745843 | 66 |
| Linux canonical | audio | 0.00381755828857422 | 0.000159401993453955 | 0.000344132163182637 | 571 |
| Linux canonical | video | 0.0019383430480957 | 0.000163134221684516 | 0.000262405923139214 | 59 |
| Linux raw | audio | 0.00382709503173828 | 0.000165060548624699 | 0.000348687688883317 | 1591 |
| Linux raw | video | 0.0019383430480957 | 0.000163134221684516 | 0.000262405923139214 | 59 |

All supplied parity rows match both candidates’ authenticated original TRAIN exports. Every studied host/Linux raw/canonical modality keeps 50/50 label agreement. Full per-clip probability metrics, dtypes, exact `[clip, second, feature]` mismatch coordinates/values, quantiles and per-second/feature distributions are in the measurement JSON and `reconciliation.json`. All five embedding cases continue to fail strict exact parity.

## C. Candidate numerical parity policy

The separately named **`TRAIN_FLOAT16_CELL_PLUS_RUNTIME_V1`** is specified in [the independent-review design](phase8_stage2c_parity_policy.md). It is inactive and has no serving activation path. It checks provenance, shapes, finite outputs, temporal mapping, exact class order, per-second labels and downstream E3/A2 decisions, while reporting max/MAE/RMSE and exact float16 discrepancies.

Its scientifically justified storage interval is the asymmetric midpoint cell between neighboring binary16 values. Quantization alone permits **zero excess outside that cell**. Additional max/MAE/RMSE runtime budgets are **unset**: neither five clips nor stable saturated scores justify universal numeric limits. A complete independently reviewed controlled TRAIN study or a valid propagated forward-error derivation is required. Runtime-budget calibration must hold canonical decoded inputs fixed; decoder-stack differences require separate reconciliation. Missing budgets or provenance mean `BLOCKED_UNREVIEWED_RUNTIME_BUDGET`, never approximate acceptance. The historical exact gate is byte-for-byte unchanged.

PyTorch documents that arithmetic order, batching and platform/device differences can change floating-point results; this supports the investigation, not any particular tolerance. [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html), [reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html). Neighboring representable values are obtained with [NumPy nextafter](https://numpy.org/doc/stable/reference/generated/numpy.nextafter.html).

## D. Docker execution status

**VERIFIED**, not merely prepared. Docker Desktop is installed per-user at `%LOCALAPPDATA%/Programs/DockerDesktop/resources/bin/docker.exe`. The agent’s PATH did not contain it, the earlier check covered only the usual Program Files installation, and the sandbox returned Access Denied for the per-user binary. Approved direct access revealed a working CLI but an absent Linux-engine pipe. `docker desktop start` started the existing engine; Docker Desktop 4.97.0 / Engine 29.9.2 then responded successfully. The prior “unavailable” observation was not evidence that Docker was uninstalled.

The existing `avalidate:stage2a` image was reused by immutable ID `sha256:0ee95003d0f32ab19176ffe86e8cdc015644a0d48c03571e81c94b07858dbaab`. No new image build was necessary or claimed. Current source directories were mounted read-only over the runtime image’s code. Its Python 3.11.17, Torch/TorchAudio 2.11.0+cpu, TorchVision 0.26.0+cpu, NumPy 2.2.6, Pandas 2.3.3, SoundFile 0.13.1, OpenCV headless 4.12.0.88 and remaining deployment pins match. FFmpeg 5.1.9 and ffprobe executed. Both authentic encoders, original/canonical comparisons, and frozen E3/A2 ran successfully; final container exit 0, `OOMKilled: false`.

The root filesystem, source, input and artifact mounts were read-only; only the new result mount and bounded tmpfs were writable. Network `none`, no published ports, capabilities dropped, no-new-privileges, 128 PIDs, 2 CPU quota, 4 GiB RAM, 512 MiB tmpfs. No checkpoint was copied into an image layer or added to Git. The uniquely named test container was removed after evidence collection; no Gradio server was started.

Execution-policy and harness failures are distinguished from success: Windows blocked direct `.ps1` execution, so the coordinator is Python callable from PowerShell; system policy was not changed. The first staged set omitted the original fusion integrity sidecars. Host preflight failed, and its associated first Docker setup was manually stopped (exit 137, no OOM). Those records are retained as failed attempts. Corrected staging includes both SHA-bound sidecars and now rejects incomplete role sets before model construction. Only the later `docker-final` exit-0 run is claimed as verified.

### Measured resource use and latency

| Measurement | Verified result |
| --- | ---: |
| Full probe wall time (after imports; both raw/canonical paths and diagnostics) | 379.719 s |
| Python process CPU time during probe | 84.376 CPU-s |
| Process peak RSS | 1.290 GiB |
| Container cgroup peak memory | 1.883 GiB / 4 GiB |
| Peak sampled Docker CPU (one-core basis) | 200.07%; configured quota: 2 CPUs |
| Mean raw decode + paired-predictor latency per clip | 24.196 s |
| Mean canonical audio predictor latency per clip | 19.461 s |
| Mean canonical video predictor latency per clip | 5.639 s |

These are current audited-path timings: predictor calls include repeated artifact hashing and model integrity checks; raw timings also include decoding. They are not isolated neural-forward benchmarks or throughput guarantees. Windows/WSL bind-mount I/O and CPU quota contention contribute to wall time. Full per-clip latency, process CPU, cgroup CPU/throttling/memory counters and 117 timestamped Docker stats samples are retained. The probe ran concurrently with host diagnostics/tests, so timing is an environment observation, not a controlled capacity estimate.

### Reproducible host PowerShell commands

From this repository, reuse the preserved staged inputs. Results must go to a new run directory so earlier evidence cannot be overwritten:

```powershell
$python = (Resolve-Path "_local/phase8_stage1/venv/Scripts/python.exe").Path
$docker = Join-Path $env:LOCALAPPDATA "Programs/DockerDesktop/resources/bin/docker.exe"
& $docker desktop start
& $docker version
& $python deployment/run_stage2c_docker.py --run-name docker-recheck
if ($LASTEXITCODE -ne 0) { throw "Encoder audit failed; inspect the saved Docker log" }
# If a compatible local runtime image is absent, build a test-only image:
# & $python deployment/run_stage2c_docker.py --build-runtime --run-name docker-built-recheck
```

The optional build uses the unchanged Dockerfile and pinned dependencies; that fresh-build branch was prepared but **not executed**. The default helper records the actual image ID, validates runtime versions and acquisition manifest, mounts artifacts read-only, runs both encoders and E3/A2, samples resource usage, records logs/exit status, and removes only its own container. Existing staged-input preparation is preserved in `_local/phase8_stage2c/prepare.py`; it deliberately refuses to overwrite staging. Run the host study with `_local/phase8_stage2c/run_host.py` only after choosing a new output directory, because it also preserves completed evidence.

## E. E3/A2 stability

| Runtime/input | Model | Max score difference vs its frozen TRAIN reference | Mean abs score difference | Max logit difference | Decisions stable |
| --- | --- | ---: | ---: | ---: | ---: |
| Windows raw | E3 | 2.42659359273034e-09 | 6.28186236362183e-10 | 0.000288009643554688 | 5/5 |
| Windows raw | A2 | 2.65828034362592e-07 | 6.02303206198584e-08 | 0.000812530517578125 | 5/5 |
| Linux canonical | E3 | 2.42659359273034e-09 | 6.27256491192441e-10 | 0.000286102294921875 | 5/5 |
| Linux canonical | A2 | 2.66139912108443e-07 | 6.02674907312917e-08 | 0.000813484191894531 | 5/5 |
| Linux raw | E3 | 8.5946855188368e-09 | 1.8697217774033e-09 | 0.000293731689453125 | 5/5 |
| Linux raw | A2 | 2.652042774276e-07 | 6.18509404359813e-08 | 0.000810623168945312 | 5/5 |

Every controlled host backend/thread/batch variant also retained all five E3/A2 decisions. Thresholds are unchanged. All are original positive pairs with large margins (minimum Linux reference margin E3 0.990368, A2 0.997009) and sigmoid saturation. This is an internal feature-sensitivity experiment, not an accuracy/negative-pair study, calibrated confidence claim or release acceptance. Newly generated tensors were never assigned copied export provenance to bypass `TrainParityFusion`.

## F. Files changed

Only these **new** reviewable files were added for Stage 2C:

- `deployment/parity_metrics.py`: float64 descriptive metrics, exact float16 locations and rounding-cell analysis; no tolerance or admission switch.
- `deployment/stage2c_probe.py`: externally hash-bound TRAIN-only raw/canonical study, runtime/profile variants, both authentic encoders, E3/A2 and latency/resource evidence.
- `deployment/run_stage2c_docker.py`: reproducible isolated-container coordinator and optional test-runtime build.
- `tests/test_phase8_stage2c.py`: rounding/tie/finite/shape tests and missing/tampered provenance, path escape and held-out-role rejection.
- `docs/phase8_stage2c_parity_policy.md` and this report.

All prior source, Stage 1/2B files, original downloads, checkpoint metadata and thresholds are unchanged. Audit-only staging, scripts, arrays, mismatch coordinates, logs, JUnit reports and resource measurements remain ignored local evidence.

## G. Test results and preservation

**Focused: 179 passed, 9 passing subtests, no failures/errors/skips. Full regression: 803 passed, 1,346 passing subtests, 2 unavailable-CUDA skips, no failures/errors.** Both subprocess exit codes are 0. One existing PyTorch weight-norm deprecation warning. Full logs/JUnit/independent execution records are in `_local/phase8_stage2c/`. Genuine host probe, Docker encoder probe, decoder-only comparison and original-script audio replay all completed with exit 0; setup failures are retained separately as described above.

`git diff --check` passes. All 965 baseline files and all 28 archived report files are preserved. The unchanged Phase 6/8 readiness and upload regression checks pass; public requests remain `MODEL_NOT_READY` with null predictions. The offline diagnostics have no server/production activation path. No authentication, normal media validation or upload/resource limits were weakened.

## H. Remaining blockers and evidence-backed next steps

1. Strict historical parity still fails for both encoders. Preserve the exact gate; do not promote approximate agreement to historical parity.
2. Obtain producer float32 features, original pixel/waveform cache hashes and an exact source/runtime/batch/layout record. Current evidence proves some local arithmetic mechanisms and a decoder-stack discrepancy, but not the complete original export cause.
3. Reconcile and pin the audio decoder stack against producer canonical waveforms. Linux raw decoding differs in 85 PCM samples; encoder tolerances must be calibrated with identical canonical inputs, then decoding validated separately.
4. Run the independently specified full TRAIN study covering 6,678 primary pair identities, with scope/coverage/confirmation evidence and reviewed numerical budgets. The five diagnosis clips cannot justify universal tolerances.
5. Independently review the candidate numerical and achieved-PTS timing contracts. Fractional-FPS selection precedes some nominal bin boundaries; a secure shared-clock production decoder and its admission behavior remain unverified.
6. A fresh image build is optional reproducibility work still unexecuted; genuine model execution inside the existing immutable Linux image is verified. Do not confuse this remaining build branch with lack of Docker encoder execution.

## I. Decision

**FIX REQUIRED.** Stage 2C establishes genuine Linux/Docker encoder and frozen-fusion execution, reproduces the four-thread Stage 2B arrays, identifies CPU arithmetic and decoder-stack contributions, and documents an inactive independently auditable policy. It does not establish exact parity or final readiness. Public predictions remain disabled. Stop after Stage 2C.

Primary local evidence: `_local/phase8_stage2c/host-final/measurements.json`, `docker-final/run/measurements.json`, `reconciliation.json`, `audio_stage_trace.json`, `audio-reproduce20/`, `decoder-comparison/decoder_comparison.json`, `container-execution.json`, `docker-stats.jsonl`, `inputs/manifest.json`, preservation records and focused/regression XML/logs. Full numerical arrays are retained so mismatch sets can be recomputed independently.

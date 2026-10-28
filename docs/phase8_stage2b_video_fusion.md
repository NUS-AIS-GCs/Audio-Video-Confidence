# Phase 8 Stage 2B — Video parity and genuine fusion integration

Audit: 2026-10-28, Asia/Singapore. **FIX REQUIRED. Stage 2B stops here.**

Authentic video acquisition, five-clip genuine audio/video CPU execution and internal frozen E3/A2 sensitivity testing completed. Both modalities fail the unchanged exact float16 TRAIN parity gate. Normal requests retain `MODEL_NOT_READY`, `predictions: null`, and `inference_enabled: false`. No public inference was enabled.

## A. Video bundle authentication

Approved [video bundle](https://drive.google.com/drive/folders/1KJ4tpDsyDfs2CwBPGnIKqAsQ3jgoLggN) and [parity samples](https://drive.google.com/drive/folders/1h7vg_Tzc30uTZlMEJNE-stEw5nvM44G5). All 14 files were acquired unchanged through authenticated connector download references into ignored `_local/video_handoff/stage2b_20261028/`. Original source IDs, URLs, MIME types, byte counts and hashes are in `_local/phase8_stage2b/acquisition.json`. `SHA256SUMS.txt` covers every downloaded file. The connector transfer initially hit a network restriction; approved host networking completed the downloads.

| File | SHA-256 |
| --- | --- |
| `README.md` | `0776ff383516140c90e123a8a7bf5cb4f25d7f02a7d496df77a56d12af6cc021` |
| `environment.json` | `323b5e3f9477db336f7ba2f5568a94089ae5fea1d8172001d926dcbc39bd790a` |
| `model_card.json` | `12f069feab94d3324f80cf9de08390a5f1a6fdaa7acdb293ada1690262925a1b` |
| `class_names.json` | `05abcc9bba8940f745c2a6842fc268a0f25a5e689f4e1f70193aaa18d00908ac` |
| `weights.pt` | `6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec` |
| `parity_samples/README.md` | `3a0a76f78382076fd19ce9a44ef945aae1f6874556ce23330bf9f7e7dd9af5a3` |
| `parity_samples/probs.npy` | `6a964b38edb9df4d23e0d06b3315328941e37011a578655aa9d1e78b4e355b17` |
| `parity_samples/emb.npy` | `d97b5a5cfc8225ae582f51ae8fb86cf8d975bb70405f0f6685c8f6f12ccae9b6` |
| `parity_samples/samples.csv` | `8c4d2ab4ced1a809260ac0068498d1da8f08d172ee7b3495e560d55bfbc1e93e` |
| `parity_samples/NGeniwxDzJ0.mp4` | `1c31affd018480ddadf6871a2f1fe8394401e01162706c88d97b8417a4a32d49` |
| `parity_samples/6VRcINAY7KE.mp4` | `b2d6311395e665a92e0f0bd5ca4241ab48a40fa6df4d5deb4c4e8986e1224eec` |
| `parity_samples/0h8HxZaX9wQ.mp4` | `ef859eee52fce905fa5347d799bc9ec19e9abad9c2de271991e752803c96db2c` |
| `parity_samples/22DapbNwnUY.mp4` | `ce046d31c90fc63193aee4fb607e2d62caffc1940e85ce6a84179b2df3ce50ba` |
| `parity_samples/c---zaDCTaE.mp4` | `08b2961cf71add0697844a1a68e39e4d34be1c171eaec380d34441c417a2498d` |

The required weights digest matches exactly. The acquisition-bound model-card digest and its two payload hashes are verified before construction. Trust is the user-approved source plus these byte bindings; the bundle is not digitally signed. Environment/parity files have acquisition hashes rather than producer-signed checksums.

Restricted `torch.load(..., weights_only=True, map_location="cpu")` sees keys `model`, `config`, `epoch`, `val`; config is exactly `{backbone: resnet18, emb_dim: 512}`. All 122 state tensors are strictly covered, shape/dtype/finite checked; trained classifier weight/bias are `(29,512)` / `(29,)`. The backbone is ResNet-18, its FC and 512D projection are Identity, and the classifier is Dropout(0.3) → Linear(512,29). No substitute/pretrained download occurs. Weights are frozen, eval mode, CPU float32, autocast disabled.

The original card has no `config` object, unlike the older `VisionPredictor` constructor expectation. New `deployment/video_bundle.py:VerifiedVideoPredictor` adapts the authenticated schema through the existing Phase 6 strict loader and inherits the producer’s `predict_frames` numerical path. Historical producer code is preserved.

Exact ordered class vocabulary, identical across both cards and both authenticated TRAIN exports (zero-based):

```json
["Accordion", "Acoustic guitar", "Baby cry, infant cry", "Banjo", "Bark", "Bus", "Cat", "Chainsaw", "Church bell", "Clock", "Female speech, woman speaking", "Fixed-wing aircraft, airplane", "Flute", "Frying (food)", "Goat", "Helicopter", "Horse", "Male speech, man speaking", "Mandolin", "Motorcycle", "Race car, auto racing", "Rodents, rats, mice", "Shofar", "Toilet flush", "Train horn", "Truck", "Ukulele", "Violin, fiddle", "background"]
```

Preprocessing exactly matches the supplied card and original extraction code: 1 FPS over intended seconds 0–9, `int(second * source_fps)` with unrounded FPS, RGB uint8, 224×224 using `cv2.INTER_LINEAR`, float32 division by 255, ImageNet mean `[.485,.456,.406]` and std `[.229,.224,.225]`. No augmentation or new feature normalization was introduced.

## B. Video raw-to-export parity measurements

All five sample IDs, clip IDs, categories and CSV TRAIN row indices match authenticated frozen TRAIN metadata. Supplied embedding and probability rows match the E3- and A2-bound original TRAIN exports byte-for-byte. Consequently the errors below are identical against both supplied arrays and frozen TRAIN rows.

Genuine outputs: embeddings **(5,10,512) float32**, probabilities **(5,10,29) float32**. Both original references are float16. Error subtraction uses float64. Float16 mismatch counts test the unchanged historical roundtrip condition.

| Sample ID | TRAIN row | Emb max abs | Emb MAE | Emb RMSE | Emb float16 mismatches / 5,120 | Prob max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c---zaDCTaE__08` | 0 | 0.00185632705688 | 0.000174708694656 | 0.000289071126865 | 17 | 0.000238478183746 |
| `22DapbNwnUY__12` | 1167 | 0.000973701477051 | 0.000161436162364 | 0.000245622622648 | 11 | 0.000224232673645 |
| `0h8HxZaX9wQ__26` | 1305 | 0.000976800918579 | 0.000167506276102 | 0.000274022136794 | 12 | 0.000222265720367 |
| `6VRcINAY7KE__07` | 2628 | 0.000976085662842 | 0.000149623617676 | 0.000232807780371 | 12 | 0.000208139419556 |
| `NGeniwxDzJ0__21` | 3000 | 0.00193881988525 | 0.000162394630155 | 0.000266633637774 | 14 | 0.00022679567337 |

| Sample ID | Probability MAE | Probability RMSE | Probability float16 mismatches / 290 |
| --- | ---: | ---: | ---: |
| `c---zaDCTaE__08` | 5.52236037612e-06 | 3.13340999162e-05 | 17 |
| `22DapbNwnUY__12` | 5.03314078324e-06 | 2.67035127311e-05 | 48 |
| `0h8HxZaX9wQ__26` | 5.53953455172e-06 | 3.04168798772e-05 | 11 |
| `6VRcINAY7KE__07` | 5.16760725304e-06 | 2.74005002358e-05 | 53 |
| `NGeniwxDzJ0__21` | 4.94702137872e-06 | 2.57080476822e-05 | 35 |

| Aggregate | Max abs | MAE | RMSE | Float16 differing values |
| --- | ---: | ---: | ---: | ---: |
| Video embeddings | 0.00193881988525391 | 0.0001631338761905 | 0.000262401859745843 | 66 / 25,600 |
| Video probabilities | 0.000238478183746338 | 5.2419328685696e-06 | 2.83964121677569e-05 | 164 / 1,450 |

All five embedding rows fail exact float16 parity. All **50/50 per-second argmax labels agree**. Complete per-clip probability MAE/RMSE/dtypes and both per-second label sequences are in `video_measurements.json`; label agreement is not accuracy or parity acceptance.

Decoder investigation (`media_scan.json`, `decoder_analysis.json`, per-clip FFmpeg logs and `opencv_build.txt`):

- All five containers have valid H.264 video and AAC audio; scans decoded every metadata-declared frame. Video PTS strictly increase; original decoded audio PTS are contiguous. Both streams first present at time zero in each clip. These checks concern media presentation timing, not physical audio/visual causality.
- Source FPS/frame counts: Ukulele 30/300; Flute, Chainsaw and Rodents 30000/1001 FPS with 300 frames; Church bell 25/251. Count/FPS durations are respectively 10.0, 10.01 and 10.04 seconds. The producer explicitly crops audio to 10 seconds; all five decoded mono 16 kHz waveforms contain exactly 160,000 samples without padding or length repair. The Church bell frame at PTS 10.0 is outside requested seconds 0–9 and is not selected.
- At 29.97 FPS, selected indices are `[0,29,59,89,119,149,179,209,239,269]`. Achieved timestamps are `[0,.967633,1.968633,2.969633,3.970633,4.971633,5.972633,6.973633,7.974633,8.975633]` seconds. They fall up to 32.3667 ms before intended bin starts. Rounding FPS to 30 would select different frames for nine bins. At 25 and 30 FPS selected timestamps are exactly integer seconds.
- No failed frame reads, missing metadata-declared frames, duplicate selected indices, repeated images or consecutive identical decoded images were found. No black frame, last-frame repetition or clamp was used. The new offline sampler rejects cases requiring those historical fallbacks rather than silently treating them as valid.
- Sequential OpenCV and random seek extraction produce byte-identical selected RGB/resized pixels on all five clips. Separately decoded FFmpeg 7.1 RGB frames, resized with the same OpenCV operation, also match every selected pixel. OpenCV 4.12.0 uses its bundled FFmpeg libraries (avcodec 58.134.100, avformat 58.76.100, avutil 56.70.100, swscale 5.9.100). No difference between these two local decoder paths explains the feature drift.

The exact original export pixel cache and float32 pre-quantization features are not supplied. The producer environment is only ranges/alternatives (`python 3.11 / 3.14`, `torch 2.14.1+xpu / 2.x`, etc.), not an exact executable lock. CPU/device/kernel/runtime differences are plausible, but the root cause cannot be isolated from these files. Neither decoder drift nor device drift is asserted as a proven explanation.

Proposed separate policy for future review: **`TRAIN_FLOAT16_CELL_PLUS_RUNTIME_V1`**. Preserve identity, class order, preprocessing, validity, source hashes and achieved-PTS checks; compare float32 output to the float16 reference’s asymmetric round-to-nearest interval, plus a separately established modality/runtime error budget. This separates storage rounding from compute drift. On these TRAIN clips, video float16 outputs differ by at most one representable step; the largest excess outside the reference rounding interval is `1.3113021850585938e-6` (audio: `1.3113021850585938e-5`). These are descriptive observations, **not adopted error budgets**. An independent controlled TRAIN study with producer float32 outputs/pixels and fixed runtimes must establish the budget and acceptance scope before implementation. Do not select it from TEST, score saturation or threshold stability. This policy is documented only, not implemented or enabled; historical exact equality is unchanged.

## C. Audio/video temporal alignment

All five MP4 SHA-256 hashes match byte-for-byte between the new video bundle and the previously authenticated [audio bundle](https://drive.google.com/drive/folders/185BtXY93QUt32gd1YVGGQ3fCeEGLXKAI). This is content verification beyond matching filenames. The regenerated canonical waveforms also match the Stage 1 waveform hashes. Both CSVs join by full `sample_id`, not positional order: their supplied row orders differ.

All five original audio/visual pairs are verified in the frozen TRAIN primary pair manifest, SHA-256 `52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f`. Audio and visual row indices are respectively 0,1167,1305,2628,3000 in video-bundle order. Category and annotation start/end agree; annotations are used for identity checks only. There is no independently supplied historical raw-MP4 checksum manifest, so the assertion that acquired media are the original dataset bytes relies on the approved producer handoff and identity evidence, not a newly invented historical source hash.

Audio retains time-major BEATs patches, eight frequency patches averaged, time-centre assignment to seconds, the exact saved 768D mean/std and GELU 256D head. Visual features preserve the trained ResNet 512D coordinate order. Neither modality is sorted, projected anew or L2-normalized for fusion. Class order matches exactly.

Same-container zero-origin clocks and intended bin identities are supported by the five-clip scan. Actual video PTS are retained separately from intended seconds: at fractional FPS the selected image can precede the audio bin boundary. We did not fabricate a `shared_clock_pts_v1` admission manifest or claim that this offline decoder satisfies Phase 6’s exact synchronized canonical boundary. Production clock mapping and duration admission remain unverified.

The current four-thread CPU paired replay re-runs genuine audio as well as video. Audio vs frozen TRAIN:
- embeddings: max abs `0.00381565093994141`, MAE `0.000159402651867152`, RMSE `0.000344128923612881`; float16 mismatches `568`.
- probabilities: max abs `0.000243842601776123`, MAE `4.88073307080967e-06`, RMSE `2.69601986132988e-05`; float16 mismatches `2`.

All audio per-second labels also agree (50/50). Small numerical changes from Stage 1’s CPU replay do not rewrite its historical results or authorize a tolerance. Current per-clip audio metrics and rounding-interval evidence are in `numerical_analysis.json`.

## D. Frozen E3 smoke test

New `deployment/paired_diagnostics.py:InternalTrainPairStudy` authenticates both original TRAIN exports against the frozen candidates, whitelists explicitly permitted TRAIN media with source digests, runs both genuine encoders, and compares frozen-model outputs on raw-generated vs original-export inputs in the same CPU runtime. No copied historical export provenance is attached to raw features; experimental direct model calls are confined to this internal study.

E3 consumes aligned audio `(B,10,256)` and visual `(B,10,512)` float32 tensors with ten explicitly valid bins. Original float16 references are cast to float32. The model performs its saved linear projections/position encoding/attention/LayerNorm/masked mean; no external fusion normalization is applied. Scores are float64 sigmoids of float32 correspondence logits, with decision `score >= frozen threshold`; they are uncalibrated pairing provenance scores.

| Model | Frozen threshold | Max score difference | Mean abs score difference | Max logit difference | Stable decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| E3 | 0.009579099009671126 | 2.42659359273034e-09 | 6.28186236362183e-10 | 0.000288009643554688 | 5/5 |
| A2 | 0.0026640388978268056 | 2.65828034362592e-07 | 6.02303206198584e-08 | 0.000812530517578125 | 5/5 |

| TRAIN sample | E3 export score | E3 raw score | A2 export score | A2 raw score |
| --- | ---: | ---: | ---: | ---: |
| `c---zaDCTaE__08` | 0.9999999996978539 | 0.9999999996978666 | 0.9999812510063598 | 0.9999812444253289 |
| `22DapbNwnUY__12` | 0.9999975636447745 | 0.9999975629429814 | 0.9999635794415429 | 0.9999635824631092 |
| `0h8HxZaX9wQ__26` | 0.9999997658684865 | 0.9999997658805435 | 0.9998716232636379 | 0.9998716459082083 |
| `6VRcINAY7KE__07` | 0.9999999949200251 | 0.9999999949204998 | 0.9996725992868536 | 0.999672865114888 |
| `NGeniwxDzJ0__21` | 0.999946988656679 | 0.9999469862300854 | 0.9999868851171665 | 0.9999868881935675 |

All five original pairs remain decision 1 for both models. These are near-saturated positive-pair scores, not an evaluation set, evidence about negatives, a calibrated confidence measure or parity acceptance. Full-precision logits/scores and source evidence are in `paired_measurements.json`. The real subprocess completion is independently recorded in `paired_probe.execution.json`; exit code is 0. No training, calibration or threshold change occurred.

## E. A2 comparison

Executed as shown above using its own unchanged weights, schema and threshold. Both original-export and genuine paired features were evaluated. This is the same restricted TRAIN diagnostic scope as E3.

## F. Backend integration status

Added four isolated modules/files: authenticated `video_bundle.py`, explicit offline `train_video_diagnostics.py`, internal `paired_diagnostics.py`, and `stage2b.env.example`; added rejection/real-bundle tests in `test_phase8_video_bundle.py`. The raw MP4 → source/identity checks → producer audio/video decoding → authentic BEATs+LoRA/ResNet → original-order tensors → frozen E3/A2 experimental score path has executed for all five approved clips.

The sampler is explicitly an offline approved-artifact diagnostic, not a secure arbitrary-upload decoder. The existing Phase 6/8 public path, authentication, media validation, process caps and upload limits are preserved. `startup_readiness` on actual artifact bindings reports `MODEL_NOT_READY`, no predictions and inference disabled. Its remaining issues are PATH FFmpeg/ffprobe absence, unimplemented verified raw decoder, and unverified raw-to-TRAIN parity. A pinned imageio FFmpeg binary used offline does not establish availability of the standard serving tools. No normal request routes to these diagnostics; no readiness flag or bypass was introduced.

Strict TRAIN admission has not passed: both raw embedding modalities differ after float16 roundtrip. No `TRAIN_EXPORT_PARITY_MATCHED` result is claimed. The release gate still requires the full 6,678 primary TRAIN pair identities, both modalities, valid ten-bin inputs and independently authenticated provenance.

## G. Docker compatibility

Host candidate runtime executed: Python 3.11.16, Torch/TorchAudio 2.11.0+cpu, TorchVision 0.26.0+cpu, NumPy 2.2.6, OpenCV headless distribution 4.12.0.88, SoundFile 0.13.1. All existing deployment direct pins pass readiness checks. The Dockerfile already copies `deployment`, `audio`, `vision` and `fusion`; no artifact COPY is added. `.dockerignore` excludes `_local`, checkpoints, arrays and compressed media. Downloaded files remain Git-ignored.

Docker is absent from PATH and the usual Windows installation path. **Image build, Linux imports, dependency installation and container execution are UNVERIFIED.** The image pins Python 3.11.17 and Debian FFmpeg 5.1.9, which differ from the executed Windows study. No Docker success is inferred from CPU execution.

Run these commands from the repository on a Docker-capable host. They build a local image and run only readiness/encoder-load diagnostics, with read-only artifact mounts and no network in the container:

```powershell
$audioBundle = (Resolve-Path "_local/audio_handoff/ghanshyam_20261027").Path
$videoBundle = (Resolve-Path "_local/video_handoff/stage2b_20261028").Path
$normalization = (Resolve-Path "_local/phase8_stage2b/derived").Path
$e3 = (Resolve-Path "_local/fusion/phase4_improvement/study_20261024_seed42_01/E3").Path
$a2 = (Resolve-Path "_local/fusion/phase5_ablations/study_20261025_seed42_01/A2").Path
docker version
docker build --tag avalidate:stage2b --file Dockerfile .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed; execution unverified" }
docker run --rm --network none --read-only --cap-drop ALL `
  --security-opt no-new-privileges --memory 4g --cpus 2 --pids-limit 128 `
  --tmpfs /tmp:rw,nosuid,size=512m --env-file deployment/stage2b.env.example `
  --mount "type=bind,source=$audioBundle,target=/models/audio,readonly" `
  --mount "type=bind,source=$videoBundle,target=/models/vision,readonly" `
  --mount "type=bind,source=$normalization,target=/models/integration/audio,readonly" `
  --mount "type=bind,source=$e3,target=/models/fusion/E3,readonly" `
  --mount "type=bind,source=$a2,target=/models/fusion/A2,readonly" `
  avalidate:stage2b python -m deployment.encoder_diagnostics --load
$diagnosticExit = $LASTEXITCODE
Write-Host "Diagnostic exit code: $diagnosticExit (2 means intentionally not ready)"
```

Exit code 2 from this diagnostic is intentional and is not a passed readiness gate. Inspect JSON for both strict loads and remaining issues. Container raw parity and secure shared-clock decoding need their own execution evidence before final readiness.

## H. Source and model preservation

Branch `feature/phase8-backend-stage1` remains at `eff4e2fe6ed33650c07fb8f719fae9a665a7faf1`. Read-only `git ls-remote origin refs/heads/main` confirmed that exact latest main revision at the start. No branch checkout/fetch mutation, merge, commit, push or deployment was performed. No GC1 repository or teammates’ frontend branch was accessed.

The 865-file Stage 2B before/after snapshot covers tracked source, existing uncommitted Stage 1 files, historical artifact/export bindings, downloaded audio/video evidence and prior audio measurements. **Zero files changed.** Separately, all 28 protected Phase 7 archived report files still match saved smoke hashes (approved read-only check). Stage 1 reports remain historical snapshots and are not rewritten.

| Frozen model | Checkpoint SHA-256 | Threshold |
| --- | --- | ---: |
| E3 | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` | 0.009579099009671126 |
| A2 | `0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12` | 0.0026640388978268056 |

E3 metadata/operating-point digests remain `fd6abe3f0e62f09e19c096f82b6516195a2c8dba93cd3d534e7f88fcfec48e58` / `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad`. A2 metadata/operating-point digests remain `aadd9c73e7675249f8192912656ca624aa6d77d50b10cea3f29a1968cce8d460` / `16994e04c6a8421ec8bc22c1fac33691a4a0228ba544f6e57b0730f98c95b384`. Frozen loaders independently verify all bindings and live model fingerprints. Audio base, adapter/head and derived normalization hashes are unchanged from Stage 1 and explicitly listed in `deployment/stage2b.env.example`.

Final focused validation: **154 passed, 9 passing subtests, no skips/failures/errors**. Full regression: **778 passed, 1,346 passing subtests, 2 unavailable-CUDA skips, no failures/errors**. One PyTorch weight-norm deprecation warning. This host run successfully executed the previously blocked Matplotlib tests; no Windows security policy or dependency pins were changed. JUnit XML, complete logs and independent child exit-code records are retained in `_local/phase8_stage2b/`. `git diff --check` passed. The study scores only the five permitted TRAIN pairs; no real TEST data or evaluation outcomes were used for policy selection.

## I. Remaining blockers

1. Exact audio and video float16 raw-to-TRAIN parity fail. The proposed tolerance-aware study has not been accepted or implemented.
2. Producer float32 intermediate outputs/pixel-cache evidence and an exact export runtime/source binding are missing; the numerical cause is unresolved.
3. Only five permitted TRAIN clips are available; the full required 6,678-pair release study is incomplete.
4. Production secure compressed decoding, explicit shared-clock/PTS mapping for fractional FPS, source duration boundary admission, and serving FFmpeg/ffprobe availability are not verified.
5. Docker build and Linux/container parity are unexecuted.

## J. Decision

**FIX REQUIRED.** Authentic paired inference and stable frozen-model smoke results are demonstrated for five approved TRAIN clips. Final readiness is not established. Keep public predictions disabled and stop after Stage 2B.

Reproduction/evidence: `_local/phase8_stage2b/video_probe.py`, `paired_probe.py`, `inspect_media.py`, `decoder_analysis.py`, `numerical_analysis.py`, `run_checked.py`, acquisition and measurement JSON, NumPy raw outputs, timing logs, readiness JSON, preservation records and focused/regression XML/logs. Audit scripts and all acquired/model/data files remain ignored local evidence; proposed backend code, tests, environment example and this report are the only new reviewable source files.

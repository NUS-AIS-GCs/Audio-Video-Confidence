# General upload release dossier ? DRAFT / NOT APPROVED

2026-10-29, Asia/Singapore. No independent release approval exists. This is an engineering candidate for isolated research, not production authorization.

## Decision and scope

- General upload implementation: **READY within the restricted candidate profile below**.
- Numerical and temporal release validation: **BLOCKED**. Descriptive compatibility checks pass on five TRAIN sources; exact historical parity and independently reviewed clock acceptance do not.
- Public inference authorization: **BLOCKED**. `release_gate.py` is unchanged; public uploads return `MODEL_NOT_READY`, null predictions, no raw scores or classifications. Live remains diagnostics-only.
- Ready for live integration: **NO**.
- No benchmark accuracy is claimed. No TEST data, retraining, replacement checkpoints, frozen-threshold changes, Git commit/push/merge or deployment.

## Implementation and boundary

`deployment/general_decode.py` admits bytes through the existing validator, probes and scans decoded timestamps, and decodes private temporary media under process limits. It never takes a client filesystem path. `deployment/general_inference.py` loads the verified authentic BEATs iter3 + trained LoRA/head, trained ResNet-18 and frozen E3 once per worker. The nonblocking worker lock rejects concurrent inference. Six `torch.load` calls occur during construction; there are **zero request-time checkpoint loads**, with one instance of each model. Artifact identities, evaluation/gradient state, class order and E3 fingerprint are checked around inference.

`deployment/general_research_app.py` is a separate optional entry point. `deployment/ui/general_research_section.py` supplies a clearly labeled research result card and sanitized typed JSON. Scores are uncalibrated sigmoid correspondence values and do not prove physical sound causation. The response includes raw score, classification, frozen threshold, checkpoint/model hashes, source/profile hashes, latency and achieved timestamps. Audit arrays stay private and are stripped at the HTTP boundary; neither event nor temporal predictions are invented. General results use their own ABI and never claim a TRAIN source/export signature.

`deployment/gradio_app.py` adds an optional dependency, default `None`. The default Docker command, authenticated TRAIN demonstration, public admission card, live service, release gate and Stage 2H ownership/transport remain intact. Existing login-specific opaque handles, revocation recheck, bounded read, lease-finally deletion, private queue (concurrency one, queue eight), result cancellation/clearing and disabled media downloads apply to the research tab.

`deployment/prepare_release.py` includes the optional dependency closure for precommit bundle review; it does not select or activate the research entry point. The prepared 90-file public gated bundle excludes the research app and offline validator entry points. `.gitattributes` prevents Windows CRLF conversion of the byte-hashed artifact catalog and stabilizes the new research module source hashes. The catalog was restored to its **unchanged committed LF bytes**; its digest and contents were not changed.

## Media and preprocessing profile

Profile ID: `aval.ffmpeg-cfr-bgr-cv-linear.v1.NOT_APPROVED`.

- Existing admission: at most 50 MiB, one audio and one video stream, explicit per-stream durations within 10 ? 0.05 seconds and matching modalities; original extension/container/brand/codec rules. MP4/MOV/MKV require H.264 + AAC; WebM requires VP9 + Opus. Missing stream duration metadata is rejected, not inferred from container duration or a tag.
- Additional candidate scope: constant frame rate 1?120 fps, at most 1920?1080, 8-bit `yuv420p`, square or unspecified pixel aspect ratio, zero format/audio/video origins, no rotation/HDR, mono/stereo 8?96 kHz audio. Dynamic geometry, gaps, VFR, ambiguous/missing streams and incompatible clock resolution are rejected. This is intentionally a narrow compatibility scope, not all files carrying an allowed extension.
- Scan every decoded audio/video frame before producing tensors. Native PTS must agree with CFR frame count or contiguous native audio sample count within **half the source time-base tick**. This is a fixed timestamp quantization check; it is not an encoder numerical tolerance or historical shared-clock acceptance policy. Time bases must be positive and at most 1 ms. No gap repair, padding, duration guessing, timeline relabeling or synchronization certification.
- Audio: FFmpeg mono downmix and 16 kHz resampling, `pcm_s16le`, first 10 seconds; read as float32 PCM /32768, exactly 160,000 finite samples. Reuse the verified producer AudioPredictor/BEATs filterbank, frozen 768D normalization, LoRA and trained head. Audio embeddings `(10,256)`; class probabilities `(10,29)`.
- Video: producer selection `int(second ? rational fps)`, indices reported with actual decoded PTS. FFmpeg BGR24 pixels ? OpenCV BGR-to-RGB ? `INTER_LINEAR` 224?224, then unchanged ImageNet scale/normalization and trained ResNet-18. Visual embeddings `(10,512)`; class probabilities `(10,29)`.
- Ten intended bins are used by frozen E3 with its existing mask. At 30000/1001 fps, indices are 0,29,59,?,269; achieved times are 0,0.9676333333,?,8.9756333333 s. Offsets are explicitly reported (largest observed ?32.3667 ms). This inherited frame-index policy does **not** establish approved sample/frame simultaneity.

## Resource and isolation profile

Research initialization fails closed unless Linux kernel/cgroup evidence shows nonroot UID, no capabilities, no-new-privileges, read-only root, loopback-only networking, memory ?4 GiB, CPU quota ?2 and PID limit ?128. It cannot start in an ordinary networked hosted configuration. The research panel is exercised over loopback **inside** this container; no public port is published. Production HTTP remains the ordinary gated app.

Each probe/scan/decode subprocess uses `prlimit`: 1 GiB address space, 15 CPU seconds, 64 MiB output-file limit and 64 file descriptors; 20-second wall deadline and combined stdout/stderr limit 512 KiB. Input protocols are local-file-only; the demuxer is forced and MOV external/absolute data references are disabled. No server environment credentials are inherited by media children. Raw pixel output is bounded to 62,208,000 bytes; private decoded media are deleted on success/error. The deployed container test adds `/tmp` tmpfs ?512 MiB. A request performs at most four bounded media subprocesses; model work has fixed tensor sizes and one slot under container limits. There is no claim of an end-to-end prediction wall deadline or production latency SLA.

CPU Torch execution: four intraop threads, one interop thread, MKLDNN enabled, CPU float32, no autocast or gradients. The container CPU quota remains two. FFmpeg 5.1.9 Debian Bookworm; pinned packages are recorded in the evidence profile. Final candidate Docker image ID: `sha256:95f13c82a81335775d7e8002013d05eb9bbed42bc3c7d037b36bb38026b5acfe` (OCI config `sha256:18f8da72b5af2c09dd9548425153f52937ec2693f073fd8349603b3d95ab1d07`). Source HEAD and uncommitted inventory below must be reviewed together; HEAD alone does not contain this implementation.

Branch: `feature/general-upload-inference`. HEAD: `dc6973f642a62b3e60b2564a36dba2fbffe8a0fd`.

Runtime profile SHA-256: `2b5a0097a9f437a57f96ab765b238d5d7e963f59ab5a9c02ec418dec66089a48`.

| Bound executable/source | SHA-256 |
|---|---|
| `ffmpeg` | `0dafc1360bb07743f76abeb1e4ae16b0aaa1331e9041a0e8d6a8524111dc0cbb` |
| `ffprobe` | `0e598fe6d247bfe63757f5c839dc145a166b3f876f2875ca88a306a5b8ec5a58` |
| `prlimit` | `663634070079386b7401ccc9fb92522ec3ece10f07f84a83fe96ec3ecb0bc74b` |
| `general_decode.py` | `fb757ca62565d5443da56ffaa112aaac9c89e875d621ea88de2119dd5a416115` |
| `general_inference.py` | `47215d00500bb5a2c9bc7c10a9529bfeca9822d1a397db78b8731a65c78c135c` |

Exact runtime packages:

```json
{
  "torch": "2.11.0+cpu",
  "torchvision": "0.26.0+cpu",
  "torchaudio": "2.11.0+cpu",
  "numpy": "2.2.6",
  "pandas": "2.3.3",
  "soundfile": "0.13.1",
  "tqdm": "4.67.1",
  "opencv-python-headless": "4.12.0.88",
  "pillow": "12.0.0",
  "gradio": "6.30.0"
}
```

## Frozen artifact inventory

The full original byte sizes and SHA-256 values are also bound in [the code-owned catalog](../deployment/artifact_catalog.json), SHA-256 `a836d05cd9071282965578c964d3d555405ac02a2f631a9beb2539381bc95418`. Provision privately using the existing provisioner, then mount read-only. No model bytes, credentials, TRAIN media or derived arrays belong in Git or the image. The optional A2 artifacts are unchanged and are not executed by this new service.

| Runtime artifact | SHA-256 |
|---|---|
| `E3/best.pt` | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` |
| `E3/checkpoint_integrity.json` | `771aa0972cf67082107aabf43c9119ab5a1f2840d85c10d12524f3944319166d` |
| `E3/checkpoint_metadata.json` | `fd6abe3f0e62f09e19c096f82b6516195a2c8dba93cd3d534e7f88fcfec48e58` |
| `E3/frozen_operating_point.json` | `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad` |
| `audio/class_names.json` | `84d4763979cefd73af2248743067e1e48f368d908a701237e8d3616757921100` |
| `audio/encoder.pt` | `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34` |
| `audio/environment.json` | `bb53adcedf93ec9c04d293c7f39b0389561830f5eef8cb9741d239e437e1f2da` |
| `audio/model_card.json` | `3b98967d720e7fad3de3af37f57487ebd49cf48dc357cef8dbde394a86470bfc` |
| `audio/norm_stats.json` | `c0ac268380479149d759f7ba605b881d415c45a83599875d30ef492ef85295e1` |
| `audio/weights.pt` | `d4a0f78403caafde2c3c39d961c9819fbae8caebaaef4b824034c678da860692` |
| `video/class_names.json` | `05abcc9bba8940f745c2a6842fc268a0f25a5e689f4e1f70193aaa18d00908ac` |
| `video/environment.json` | `323b5e3f9477db336f7ba2f5568a94089ae5fea1d8172001d926dcbc39bd790a` |
| `video/model_card.json` | `12f069feab94d3324f80cf9de08390a5f1a6fdaa7acdb293ada1690262925a1b` |
| `video/weights.pt` | `6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec` |

E3 threshold **0.009579099009671126**, model fingerprint `d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025`, checkpoint `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925`. No values or bindings were modified.

## Genuine TRAIN evidence

The acquisition manifest is authenticated before reading all 40 model/reference files and five canonical input arrays. Manifest SHA-256: `65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e`. Only the five prespecified TRAIN IDs were used. Swaps pair video i with audio (i+1) mod 5, specified before observing scores. Compressed streams are copied into new MP4 files without re-encoding and passed through the complete new decoder/model path. They are not replaced by a score fixture or an embedding lookup.

| Case (video; swapped audio is next row cyclically) | Original raw E3 score | Swapped raw E3 score |
|---|---:|---:|
| `c---zaDCTaE__08` | 0.9999999996978804 | 3.85276063285511e-06 |
| `22DapbNwnUY__12` | 0.9999975629290364 | 0.0005871553196661409 |
| `0h8HxZaX9wQ__26` | 0.9999997659062186 | 3.868810522365176e-07 |
| `6VRcINAY7KE__07` | 0.9999999949204514 | 0.007293270978074229 |
| `NGeniwxDzJ0__21` | 0.9999469800619933 | 3.698014650016988e-12 |

All originals are above the frozen threshold; all swaps are below it. This is **not** benchmark accuracy. All ten logits and scores exactly match the preserved raw-TRAIN run (`_local/premium_ui/genuine-native-run/validation.json`, SHA-256 `32a9477ce6922bd1500df695c7b690835af43478b299e9978202d52ddcd96546`). Each freshly decoded swap logit also equals the forward pass using independently extracted constituent features.

The newly admitted metadata remux has a source hash outside the five original hashes and exactly reproduces original 1. It proves new-byte admission in the research path, **not unseen-content generalization**. A repeated full decode/forward of original 1 returns identical waveform, pixels, embeddings and logit. A separate new generated-content fixture is described below; it is a transport test, not an accuracy example.

Measured startup 106.594 s, per-request 48.075?96.011 s for this Windows bind-mounted, concurrently exercised environment; offline audit total 1,306.280 s. Peak **process** RSS 1,183,804 KiB (~1,156 MiB); this is not aggregate cgroup peak. The same models stayed loaded throughout. Repeated full integrity reads dominate bind-mount latency; no hash checks were disabled to accelerate the test.

## Numerical and temporal assessment

| Comparison across five TRAIN sources | Measurement | Interpretation |
|---|---|---|
| Candidate vs same-runtime producer audio decode | Exact, all 800,000 float32 samples | Current FFmpeg command compatibility on these bytes |
| Candidate vs canonical acquired waveforms | 85/800,000 samples differ; maximum 0.000030517578125 (one PCM step) | Historical raw decode parity still fails |
| Candidate vs producer and canonical RGB pixels | Exact, all 50 selected frames | Current extraction/resize compatibility on these sources |
| Candidate vs canonical-input audio embeddings | Maximum float32 difference 0.0008680820465087891 | Small waveform changes propagate through genuine BEATs |
| Candidate vs canonical-input video embeddings | Exact float32 | Same pixels and inherited predictor on this runtime |
| Candidate audio vs historical float16 TRAIN exports | 1,591/12,800 values differ after float16 cast; max absolute float32/reference difference 0.0038270950317382812 | Includes runtime/decode/storage effects, not an accepted tolerance |
| Candidate video vs historical float16 TRAIN exports | 59/25,600 values differ after cast; max absolute difference 0.0019383430480957031 | Historical float32/kernel export evidence still missing |
| Canonical-input vs candidate original E3 scores | Max absolute difference 6.1680920371287584e-9; same five decisions | Descriptive downstream sensitivity only |
| Closest observed TRAIN swap to threshold | Score 0.007293270978074229; margin 0.0022858280315968963 | Does not cover arbitrary near-boundary inputs |

No new `allclose`, `atol`, numerical policy, shared-clock stamp or release rule was introduced. Historical producer float32 pre-cast arrays, exact runtime/kernel/export provenance and complete audio preprocessing trace provenance cannot be recovered from float16 arrays alone. Existing [handoff/parity blockers](final_prediction_live_handoff.md) and [inactive numerical policy](phase8_stage2c_parity_policy.md) remain unresolved. Frozen normalization, class order, encoders, E3/A2 artifacts and thresholds remain unchanged.

## Security and regression evidence

- Focused general/prediction-adapter/frontend/live/upload security: **160 passed**.
- Full Python: **1,068 passed**, **1,346 subtests passed**, two CUDA-dependent skips, one upstream deprecation warning. No tests weakened.
- JavaScript: **14 passed**.
- Candidate Docker build: passed; pinned base/dependency layers unchanged; no model artifacts included.
- Genuine research HTTP queue: passed authenticated loading/result display, unauthenticated config/upload/queue rejection, cross-login rejection, media download denial, replay/stale handle rejection and lease cleanup. The public tab stays gated after the actual models load. No audit tensors or TRAIN provenance appear in HTTP results.
- New unit boundary tests: clock/profile rejection matrix, native fractional timestamps, VFR/gaps/dynamic geometry/rotation/HDR, missing kernel isolation evidence, subprocess timeout/diagnostic exhaustion, sanitized errors, absent child credentials, lock contention, failed decode cleanup, login revocation and queue cancellation/clearing.
- Actual corrupt/missing-audio/missing-video/short/shifted-clock/oversize/unsupported-extension files: rejected without predictions. Worker contention returns `RESOURCE_BUSY`. Temporary decoded media were absent after processing. All 40 acquired model/reference and five input hashes reverified unchanged at the end.

## Frozen E3 sensitivity check

Thirty genuine frozen-E3 forwards compare all ten prespecified pairs using raw float32 features, their float16?float32 round trip, and authenticated original float16 TRAIN exports. This supplemental run uses preserved TRAIN float32 feature arrays with recorded digests; it first requires **all ten raw logits and scores to equal the fresh general-media audit exactly**. Historical exports are authenticated through the original acquisition manifest. It is a diagnostic reuse of existing TRAIN evidence, not new independent boundary data. The offline validator now performs the same sensitivity function directly on freshly generated features in future complete audits.

All ten decisions stay the same across the three variants. The closest observed swap changes from **0.007293270978074229** (raw) to **0.00729423424373297** (float16 round trip), or **0.007293226097807169** (historical exports); threshold remains **0.009579099009671126**. Largest observed score change is **9.63265658740324e-7**. These are measurements on the prespecified examples; they do not justify a universal tolerance, establish arbitrary near-threshold stability or activate the inactive numerical policy.

## New generated-content and container checks

The newly generated 10-second, 224?224/25 fps gray-video + mono 16 kHz/1 kHz sine fixture was encoded as H.264/AAC and passed through the actual models. Its unapproved research score was **0.6474167767652332**, logit **0.6077035665512085**, classification **Original-like pairing**, latency **67.6428 s**. This is an artificial transport fixture, not a natural original-pair example, ground-truth success, benchmark measurement or proof of sound causation. It demonstrates actual inference on content outside the five TRAIN fixtures without inventing a result or using TEST data.

Source SHA-256: `fcdfb98a20a853794ae8d42917d3f915d1385fc6a0b7e9dd30a30d6a5b209900`. The runtime profile matches the TRAIN and HTTP audits exactly.

| Additional format fixture | Observed result | Qualified inference claim |
|---|---|---|
| QuickTime MOV, copied H.264/AAC streams | Decodes; waveform and selected pixels exactly equal MP4 | Decoder compatibility on this fixture; no independent release approval |
| Matroska MKV, copied H.264/AAC | Rejected `UPLOAD_VALIDATION_FAILED` due missing explicit stream durations | No positive MKV scope qualification |
| WebM, VP9/Opus | Rejected `UPLOAD_VALIDATION_FAILED` due missing explicit stream durations | No positive WebM scope qualification; Opus origin policy also needs review |

**Positive general-media qualification is currently MP4/MOV under the narrow profile, not generic MKV/WebM support.** All four extensions retain the existing admission rules; missing durations are not bypassed or reconstructed from tags. Broadening these clocks/metadata rules requires separate implementation and scientific review.

The full Stage 2H smoke on the candidate image passed actual TCP authentication, private queue-only execution, forbidden download/proxy/stream routes, FileData injection rejection, global four-file capacity, two bounded stalled upload readers with 30-second timeout, oversized-body cleanup, real **300.106-second** upload expiry, stale-handle rejection, unclean-restart orphan deletion/login invalidation and graceful-shutdown deletion. The image has no model files and `/models` was empty in this default-app test. The production command remains gated.

## Reproduction (local research only)

Build `avalidate:general-upload-candidate` from the reviewed source with the existing Dockerfile. Authenticate/provision the real bundles privately; do not download substitute checkpoints. From PowerShell at the repository root:

```powershell
$docker = 'C:/Users/venka/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
$models = (Resolve-Path '_local/phase8_stage2c/models').Path
$inputs = (Resolve-Path '_local/phase8_stage2c/inputs').Path
$evidence = (Resolve-Path '_local/general_upload').Path
& $docker run --rm --network none --read-only --cap-drop ALL `
  --security-opt no-new-privileges --memory 4g --memory-swap 4g `
  --cpus 2 --pids-limit 128 --tmpfs /tmp:rw,nosuid,size=512m `
  -e PYTHONPATH=/app --mount "type=bind,source=$models,target=/models,readonly" `
  --mount "type=bind,source=$inputs,target=/inputs,readonly" `
  --mount "type=bind,source=$evidence,target=/results" `
  avalidate:general-upload-candidate python -B -m deployment.general_validation `
  --models /models --inputs /inputs `
  --manifest-sha256 65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e `
  --output /results/validation-repeat.json
```

The opt-in scripts `tests/integration_general_research_http.py` and `tests/integration_general_scope.py` run under the same limits with `/models` and `/results` mounts (mount their reviewed script into the container). No host port or network bridge is needed. They generate their credentials/fixtures privately and expose no Secret in results. To exercise the separate app entry point, inject existing authentication Secrets at runtime and select `python -m deployment.general_research_app` inside the same strict sandbox; this does not change the default Docker command or permit production inference.

## Image and source traceability

The full genuine TRAIN audit used `avalidate:final-integration`, image ID `sha256:4e4afd158f0e2afe701790c0079b64ecc92b8f099cd03811147df4f4e1e0fb14`, with the current deployment source bind-mounted read-only. The genuine HTTP/generated-content and Stage 2H lifecycle checks used the first candidate image (`sha256:1c9a4623beb556b4648be14f2b75d5f1016480d58bdd2a683dc4d3daaf234164`). The final rebuild adds the offline sensitivity audit; core decoder, service, UI, auth and runtime source are unchanged. A separate check on the **final** image verifies the source hashes below match the reviewed working files, confirms no models/media in the image, and constructs the default gated app with no research tab. This distinguishes the complete lifecycle run from the final image inventory check.

| Final image source | SHA-256 |
|---|---|
| `deployment/general_decode.py` | `fb757ca62565d5443da56ffaa112aaac9c89e875d621ea88de2119dd5a416115` |
| `deployment/general_inference.py` | `47215d00500bb5a2c9bc7c10a9529bfeca9822d1a397db78b8731a65c78c135c` |
| `deployment/general_validation.py` | `986b96c5336f80e2af3ffe9c429788863235cecd2ef7c1d92ca9f38c42da3152` |
| `deployment/general_research_app.py` | `6cb0dd3743dbdb799380f67e0080b6fd02091e0b89b6010809bf8f84e5396f98` |
| `deployment/ui/general_research_section.py` | `efe8fa06bc31444d3884cef40bd7c02dd05c295a2c28a33ba84852a386abb319` |
| `deployment/gradio_app.py` | `ea4e8801fd5280263c6d79505ab6f58dbf2cc12603c8662b0f244bf4e3254d93` |
| `deployment/prepare_release.py` | `bd2742e73c8f2fc28cba766731cacc089edb0ddbfb24628fabfb9fd51a6b1e98` |

The final image command remains `python -m deployment.gradio_app`, user `1000:1000`. No deployment or hosted configuration was changed.

## Independent review required

| Review field | Current value |
|---|---|
| Reviewer identity / independence | **null ? not supplied** |
| Signed decision / document link / digest | **null ? not supplied** |
| Exact approved artifact catalog and E3 identities | **not approved** |
| Exact source/image/runtime/decoder/preprocessing profile | **not approved** |
| Supported media, clock behavior, duration and resource scope | **not approved** |
| Justified numerical policy and independent boundary evaluation | **inactive / unresolved** |
| Upload prediction authorization | **BLOCKED** |
| Live prediction authorization | **BLOCKED** |

A release requires independently authenticated evidence binding **all** fields, including resolution of historical parity or an explicitly justified reviewed numerical policy, achieved timestamps/shared-clock policy, independent media/decision-boundary evaluation without TEST tuning, resource behavior, security review and precise upload/live scopes. A matching hash, successful TRAIN diagnostic, environment flag or this draft is not authorization. Any future authorization must be reviewed into the code-owned gate; this revision contains no active approved scope.

## Next actions for live webcam/microphone

1. Preserve diagnostics-only capture until a separately reviewed live input profile exists. Browser PCM/resampling, clipping, channel mixing, timestamps, frame selection, canvas resizing/JPEG and dropped-frame behavior are different producers from the compressed-file decoder.
2. Collect authenticated, consented live reference captures and compare actual PCM/RGB at every stage against the frozen encoder representations; report clocks, drift, loss and achieved timestamps. Never assume shapes or nominal one-second bins establish parity.
3. Specify bounded synchronized windows, missing-data/rejection rules and backpressure; test cross-user ownership, revocation, stale-result cancellation, cleanup and resource exhaustion with real browser capture.
4. Resolve/independently review numerical and temporal acceptance, then bind exact live preprocessing/runtime/artifacts and operating point in a distinct approval. Only after that review should live E3 wiring and a code-owned release gate change be implemented. Never promote this research profile or its source stamps automatically.

## Git review inventory and excluded local material

Combine the decoder, worker, optional UI, tests and this dossier as one gated engineering PR: they share one experimental ABI and keep the public interface closed. Preserve the previous merged implementation. Do not stage by directory or with `git add .`.

Reviewed implementation inventory:

- `.gitattributes`
- `deployment/general_decode.py`
- `deployment/general_inference.py`
- `deployment/general_research_app.py`
- `deployment/general_validation.py`
- `deployment/ui/general_research_section.py`
- `deployment/gradio_app.py`
- `deployment/prepare_release.py`
- `tests/test_general_media.py`
- `tests/integration_general_research_http.py`
- `tests/integration_general_scope.py`
- `tests/integration_general_sensitivity.py`
- `docs/general_upload_release_draft.md`
- `README.md` (research scope/documentation link only)

Exclude `_local/**` (models, media, checkpoints, decoded arrays, credentials if operator-supplied, all local evidence and generated bundles), environments/caches, plus the pre-existing untracked `docs/phase8_final_audio_trace_verification.md`, `docs/phase8_final_video_frame_parity_verification.md`, `docs/phase8_stage2g_deployment.md` and `requirements-fusion-training.txt`. These unrelated files remain preserved. The catalog has no Git content diff: only its local line endings were restored to match the immutable committed identity. No staging, commit, push, merge or deployment occurred.

Suggested PR title: **Add isolated genuine general-upload inference with bounded decoding**.

Suggested description: Add a bounded CFR/zero-origin media decoder and an optional authenticated research worker that runs the original BEATs+LoRA, ResNet-18 and frozen E3 models. Report genuine scores, frozen identities and achieved timestamps through a separate research ABI; preserve the default MODEL_NOT_READY and diagnostics-only live gates. Include TRAIN/swapped/remux/generated-input evidence, ownership/resource tests and this NOT APPROVED release dossier. Historical parity and independent numerical/temporal authorization remain blocked.

## Local evidence identities

These ignored local artifacts contain measurements, not release authority. Preserve them locally; do not add models/media/arrays to Git. They are reproducible with the reviewed audit entry point and opt-in HTTP/scope scripts.

| Local evidence under `_local/general_upload/` | SHA-256 |
|---|---|
| `validation.json` | `13c3cf6d68704e7a37ec13ec36b801b84a6852deaf475fc56f1472cd121806e7` |
| `http-smoke.json` | `824521d64b659c4b2bb514568442c8a1436fc98ee7f9ded763816c448d997b92` |
| `generated-scope.json` | `664c6176af0c0467f24d5479ff688e22c62ff21c3722eafd45d9c6941af07901` |
| `decoder-probe.json` | `294c8f664699659f689d40236de979d2e3078262beb7fc5044ba4ce2bb02008e` |
| `supplemental-comparison.json` | `034f49e68417791a9d84750ceb46d8bfc21a95a4a201c342fee47bd7cbeef9fb` |
| `python-regression.log` | `c3925952737e1a58cf8cfa271094f7ff6cd5bebe7a1528f59daaf5dc2e4a4c33` |
| `focused-tests.log` | `79b7e006c7f86babba50a969bf8b32e3aa0c6ee6758fa5d99ef48a530bebd484` |
| `javascript-tests.log` | `eae2bcfefcdef3f0e7525bb37edc10c38503e04828a2e0e3e2ddf8418058576f` |
| `default-smoke.log` | `d999ed94fecb1ef5d421ace1d3cef4941fc67b85ed4c8929307d284442a13704` |
| `gated-bundle-manifest.json` | `1ca32d4702a8df6959bd62faee2a8208044bf8744882b94229c1e4fa260f3917` |

Working-tree preservation audit found no missing baseline files. The four unrelated untracked files above are byte-identical. Protected runtime, encoder, original TRAIN backend, release-gate, live/transport and scientific policy files are unchanged. No credentials, binary artifacts or media were found in the explicit source/test/documentation inventory; the Git index remains unchanged. The artifact catalog matches the committed blob byte for byte.

Additional local evidence:

| Local evidence under `_local/general_upload/` | SHA-256 |
|---|---|
| `sensitivity.json` | `c33bb529638ab05c08906f7f61584bcff918f2138b4778c83fdcb9eaf4b7e399` |
| `final-image.json` | `a88d7774c9698851c2eac821d65b06d2944942c7b28c95ddbd39b7206f2473aa` |
| `final-unit-tests.log` | `4caa21add13a1950c3c1b8a1d0045b2421aefd2ca274777a54e51d9374b2d3b0` |
| `docker-build.log` | `dd79767ebb375346fb27c3df9eed6fa0d4b89487d0087eb1f90b5133f69a46e9` |

The final general-boundary test rerun passed **46 tests**. The unchanged 1,068-test full regression and 160-test focused run above include these 46 cases. Opt-in sensitivity reproduction uses `tests/integration_general_sensitivity.py`, the same isolated runtime, `/models` and `/inputs` mounts, preserved TRAIN float32 features at `/reference`, and the fresh audit at `/results/validation.json`. It emits exact feature/evidence digests and rejects a baseline logit mismatch; it creates no release approval.

# Live-input compatibility dossier ? DRAFT / NOT APPROVED

2026-10-29, Asia/Singapore. Independent reviewer/sign-off: **not supplied**. Engineering research authorization and a signed capture ticket do not constitute scientific or public inference approval.

## Verdicts

| Condition | Verdict |
|---|---|
| Live research adapter | **READY for bounded, authenticated fixture research** |
| Live input compatibility | **BLOCKED** |
| Public live prediction | **BLOCKED** |
| Ready for reviewed PR | **YES ? gated engineering/research scope only** |
| Physical webcam/microphone end-to-end validation | **NOT ACHIEVED; one attempt rejected, no retry** |

Branch `feature/live-inference-integration`, HEAD `745d915d36cab91ca7fefa749a16ef0515ea3e94`; fetched origin/main at the same HEAD. The implementation is uncommitted. No stage, commit, push, merge, deployment, retraining, checkpoint substitution or frozen-threshold change occurred. Work remained in GC2.

## A. Completed architecture and inventory

The optional local capture broker binds **127.0.0.1 only**, requires the existing Gradio authentication and Stage 2H body/ownership guards, and rejects foreign Host/Origin values. The ordinary Upload and Live tabs keep their closed prediction gates. The extra research tab requires explicit capture consent, uses the existing AudioWorklet/camera controller, and returns a random completed-window ticket. No model is loaded or invoked by its HTTP callbacks.

The broker writes a bounded signed package into an operator-private local spool. A separate **no-network** Linux container consumes that spool, reusing `GeneralResearchService` and its verified BEATs iter3 + trained LoRA/head, trained ResNet-18 and frozen E3. `_forward_window` is shared with general uploads; model loading was not duplicated. The offline adapter returns a typed, signed research result, and only the original authenticated login with its cancellation capability can poll it. The JSON/card reports one actual raw uncalibrated E3 correspondence score and classification, full threshold/model identities, source/profile hashes, latency and timing metadata. Event and temporal predictions remain null. Correspondence does not prove physical sound causation.

Reviewed changes (19 files):

| File | Purpose |
|---|---|
| `.gitattributes` | Stable LF bytes for hash-bound research sources |
| `deployment/general_inference.py` | Factor the authentic forward through the already-loaded models |
| `deployment/gradio_app.py` | Optional capture research tab; default `None` |
| `deployment/prepare_release.py` | Reviewed optional import-closure candidates; no publishing |
| `deployment/ui/live_section.py` | Opt-in consent and private seal/poll/cancel callbacks |
| `deployment/web/live_capture.js` | Research metadata, seal/replay polling and stale-result cancellation |
| `deployment/web/live_research_result.js` | Research-only text rendering, absent from ordinary live rendering |
| `deployment/live/research_store.py` | Private HMAC-signed capture/grant/result files and short leases |
| `deployment/live/research_capture.py` | Authenticated bounded capture producer and revocation/expiry |
| `deployment/live/replay.py` | Genuine offline model adapter, cleanup and process watchdogs |
| `deployment/live_research_app.py` | Loopback capture-only entry point |
| `tests/test_live_research.py` | Ownership, timing, malformed input, cleanup and watchdog tests |
| `tests/js/test_live_security.js` | Consent, late seal and Reset cancellation; existing tests preserved |
| `tests/integration_live_browser.cjs` | Opt-in real browser/replay harness; fake devices by default |
| `tests/prepare_live_browser_fixtures.py` | Manifest-authenticated TRAIN references; no TEST data |
| `tests/integration_live_preprocessing.cjs` | Actual Chromium audio/canvas transformations, no devices |
| `tests/integration_live_metrology.py` | Genuine encoder/E3 comparisons on those controlled fixtures |
| `docs/live_input_compatibility_draft.md` | This draft evidence/review dossier |
| `README.md` | Local research scope and dossier link |

Keep `_local/**` excluded, including every capture package, key, model, TRAIN source, waveform/frame array, browser fixture, log and generated bundle. Preserve the unrelated pre-existing untracked `docs/phase8_final_audio_trace_verification.md`, `docs/phase8_final_video_frame_parity_verification.md`, `docs/phase8_stage2g_deployment.md`, and `requirements-fusion-training.txt`; none belongs in this PR. The artifact catalog matches its committed blob byte for byte despite its local status marker; it has no Git content diff.

## B. Captured input and research transport

Profile `aval.browser-pcm16-jpeg.offline-research.v1.NOT_APPROVED`:

- Browser requests one microphone channel, no echo cancellation/noise suppression/automatic gain control, and ideal 640?480/30 fps video. Actual device sample rate/channel settings are recorded. The research AudioWorklet explicitly uses WebAudio speakers-to-mono mixing.
- AudioContext requests 16 kHz, with supported device/context fallback rates 16,000/22,050/24,000/32,000/44,100/48,000 Hz. Its resampler is implementation-defined. Float32 audio is quantized using **JavaScript Math.round(x?32768)**, clipped to signed int16, transported little-endian PCM, then divided by 32768. If needed, the complete native ten-second window is resampled using unchanged torchaudio behavior and clipped. There is no padding/gap repair in the research adapter.
- Browser canvas longest side ?640, JPEG quality 0.9; server checks actual JPEG header dimensions before decode, then uses existing OpenCV JPEG decode, BGR?RGB and INTER_LINEAR 224?224. The authentic trained video predictor retains its original ImageNet scaling/normalization.
- Audio chunks must be exactly contiguous, including small discontinuities tolerated only by the legacy diagnostics buffer. Frames carry actual selected capture time, monotonically increasing presented-frame ID/media time and timestamp kind. Missing bins, reused IDs, non-JPEG/oversized images, unsupported profiles, device loss and reported dropped frames are rejected.
- A completed **valid** ten-second window may follow initial incomplete bins. The final fix preserves both original source seconds/timestamps and local window coordinates; it does not relabel achieved times as exact bins. A regression covers missing initial frame 0 followed by a complete window at source seconds 1?10. Capture length remains bounded at 13 seconds of audio, 64 anchors, four frames/eight chunks/two seconds of audio per push. Only a complete selected ten-second window reaches inference.
- Repeated AudioContext/performance anchors must progress monotonically over at least nine seconds and stay within the existing **100 ms structural buffer limit**. Selected camera frames also stay within that existing limit of their audio bin. This reuses a capture-quality screen, not an independently approved scientific synchronization tolerance. Sensor delay, hardware clocks and absolute A/V simultaneity are not certified.
- The broker records hashes of its actual resampled waveform and decoded RGB frames. Replay independently rebuilds tensors from native PCM/JPEG and reports exact-match booleans and hashes, rather than assuming matching shapes prove equality.

Each package/grant/result is authenticated with a random local 32-byte key. Capture bytes are bounded at 8 MiB; result files at 128 KiB. Paths are generated from validated 64-hex tickets; clients never supply filesystem paths. Four research sessions/exports may be outstanding. Private files publish atomically. A ticket expires after at most 300 seconds, and a signed **five-second authentication lease**, renewed by the broker every 0.5 seconds, prevents broker death from leaving a five-minute authorization. Revoked/expired leases cannot be resurrected. Authentication/ownership, grant and runtime integrity are checked before and after model work. The trusted local signing key authorizes research transport only; it is not an independent model-release signature or evidence of sensor attestation.

Stop/Reset/capture replacement/page hide invalidate the relevant client generation, release devices, clear displays and request revocation. A late seal is revoked and a late poll cannot render after cancellation. Logout removes the pinned Gradio tokens; the broker clears active capture state and revokes grants. Shutdown/restart invalidates outstanding grants. Captured files are deleted after replay, error, cancellation, expiry or watchdog timeout; the CLI drops private audit-array references after publication. A signed result file prevents reprocessing from overwriting a prior result. Cancelled grants are removed as well as their media/results, so repeated captures cannot accumulate inactive ticket files. One exclusive process lock and the shared worker inference lock bound model concurrency.

## C. Authentic browser fixture result (not a personal capture)

The final browser harness used **Chromium fake webcam/microphone devices** with real getUserMedia, AudioContext/AudioWorklet, canvas/JPEG, authenticated Gradio callbacks, private signed spool, isolated genuine encoders and frozen E3. This demonstrates actual live-input model execution; the generated camera/beep content is not a natural pairing accuracy example or sound-source causation test.

- Browser: `156.0.8078.48`.
- Actual fixture device/context: `{"auto_gain_control": false, "browser_resampler": "AudioContext-implementation-defined", "consent": true, "context_rate": 16000, "device_channels": 2, "device_rate": 44100, "echo_cancellation": false, "mono_mix": "explicit-speakers-mono", "noise_suppression": false}`.
- Raw E3 score **0.011994182992678471**, logit **-4.411266803741455**, **Original-like pairing**.
- Inference/revalidation latency **9.310039 s**.
- Live runtime profile SHA-256 `3de195d7ae91b5f7b0025c247475e5748da6650d722f6b282068ccd171705344`.
- Input source SHA-256 `183c7609171b1f79df6c2136e9441a8c1764c69aa0c0d9c235e3fe312acfae5b`.
- Worst selected-frame offset **61.380000 ms**; maximum anchor-clock drift **0.121000 ms**.
- Broker/replay waveform exact: **True**; RGB frames exact: **True**. This equality applies to this package/runtime, not historical TRAIN parity.

| Bin | Achieved local seconds | Achieved source seconds | Presented-frame ID | Timestamp kind |
|---:|---:|---:|---:|---|
| 0 | 0.061380000095367426 | 0.061380000095367426 | 9 | captureTime |
| 1 | 1.0101800000953676 | 1.0101800000953676 | 28 | captureTime |
| 2 | 1.9974800000953674 | 1.9974800000953674 | 48 | captureTime |
| 3 | 2.9987800000953673 | 2.9987800000953673 | 68 | captureTime |
| 4 | 3.9990800000953675 | 3.9990800000953675 | 88 | captureTime |
| 5 | 4.999680000095367 | 4.999680000095367 | 108 | captureTime |
| 6 | 6.0009800000953675 | 6.0009800000953675 | 128 | captureTime |
| 7 | 7.005680000095366 | 7.005680000095366 | 148 | captureTime |
| 8 | 8.009580000095369 | 8.009580000095369 | 168 | captureTime |
| 9 | 8.999980000095368 | 8.999980000095368 | 188 | captureTime |

The final automated test also passed unauthenticated RPC rejection, known-ticket polling/cancellation rejection from a second login of the same username, foreign Host/Origin rejection, direct file/proxy/stream denial, actual capture deletion, Reset revocation and stale-display clearing. No browser page errors occurred. Ordinary public/live readiness remained closed throughout.

## D. One consented physical attempt ? not a successful inference

The user explicitly consented to one local ten-second webcam/microphone research window. The persistent `physical-attempt.json` marker was created before that sole device request. The fake-device flags were removed and all nonlocal browser HTTP requests were blocked. No screenshot, physical JPEG, PCM or personal capture package was copied into the evidence archive.

The attempt failed to yield a complete valid window and stopped with **LIVE_RESEARCH_CAPTURE_TOO_LONG**. The harness subsequently timed out waiting for a result; **no physical E3 score/classification was produced**. Available logs do not distinguish which underlying readiness condition prevented completion. Device absence was not established, and a missing/muted modality or timing failure must not be guessed. The physical spool was checked and contains only the empty worker lock, with no media package. The local broker/listener and research worker stopped.

The later-complete-window fix was subsequently validated using fixtures and the exact rebuilt image. **It was not retested on physical devices**, and is not claimed to explain or resolve the physical attempt's failure. No additional physical recordings were made. Another device capture requires fresh explicit consent; do not remove the attempt marker to bypass this constraint.

## E. Controlled preprocessing and numerical compatibility

The separate comparison uses manifest-authenticated original TRAIN source `c---zaDCTaE__08`, plus a high-resolution source-2 image probe. Source PCM and all ten original source frames were reconstructed using pinned FFmpeg and checked against producer OpenCV frames. Both audio branches consume the same 48 kHz stereo float32 PCM produced by FFmpeg from the original AAC source; FFmpeg then makes the reference 16 kHz mono PCM16. Actual Chromium OfflineAudioContext performed 48 kHz stereo?16 kHz mono processing; actual canvas/JPEG operations produced the replay images. No TEST data or threshold fitting was used. This controlled file intervention differs from natural hardware capture and is not an accuracy benchmark.

| Comparison | Actual measurement |
|---|---|
| Browser PCM vs FFmpeg 16 kHz mono | 157,908/160,000 samples differ; max absolute 0.028839111328125; MAE 0.0013922426223754883; RMSE 0.002335365658048198 |
| Native 48 kHz PCM then server torchaudio vs FFmpeg | Max absolute 0.02266600728034973; MAE 0.0003890931255164859; RMSE 0.0006362066387797242 |
| Ten browser-JPEG/server RGB224 frames vs producer RGB | 802,361/1,505,280 values differ; max 21; MAE 0.8816964285714286; RMSE 1.469180534669712 |
| Frozen ImageNet-normalized pixels | Max difference 0.3660128116607666; MAE 0.015268218600707224 |
| BEATs+LoRA embeddings, browser vs reference | Max float32 difference 1.1010283827781677; MAE 0.1116346191381131; RMSE 0.19954637417934265 |
| ResNet-18 embeddings, browser vs reference | Max float32 difference 0.11936450004577637; MAE 0.01679406590315331; RMSE 0.023994364575662914 |

Canvas pixels at the actual canvas size matched the equivalent OpenCV resize exactly for the two probed images. JPEG plus actual server resize nevertheless differs: max RGB224 difference 16 for source 0 and 44 for the high-resolution probe. A diagnostic second browser-canvas resize to 224 differs by at most one RGB unit from OpenCV resizing without JPEG; **this second canvas resize is not used by the live model path**, which retains server INTER_LINEAR.

The measured rounding probe confirms that the common PCM scale does not mean identical conversion. For inputs `[1.2,-1.2,1,-1,0.5/32768,-0.5/32768,1.5/32768,-1.5/32768]`, Chromium produced `[32767,-32768,32767,-32768,1,0,2,-1]`; pinned FFmpeg produced `[32767,-32768,32767,-32768,0,0,2,-2]`. Saturation rails match in this probe; half-step ties do not. No conversion behavior or numerical acceptance rule was changed to hide these differences.

Actual frozen E3 controlled comparison:

| Input path | Logit | Raw score |
|---|---:|---:|
| FFmpeg/OpenCV reference | 21.920507431030273 | 0.9999999996979738 |
| Actual Chromium intervention | 21.597803115844727 | 0.999999999582945 |

The decision stayed Original-like for this one pair, but its logit changed by **-0.3227043151855469**. Score saturation hides that change; it is not evidence of numerical equivalence, arbitrary near-threshold stability or generalization. Request-time checkpoint loads during both controlled forwards: **0**.

Historical export provenance gaps remain: original pre-cast float32 values cannot be recovered from float16 TRAIN exports; exact historical device/kernel/runtime and complete audio preprocessing provenance remain unresolved. Existing [general-upload dossier](general_upload_release_draft.md) and [scientific handoff](final_prediction_live_handoff.md) remain applicable. No new allclose/atol, numerical policy, approved shared-clock stamp or readiness exception was introduced.

## F. Frozen models and exact runtime

E3 threshold **0.009579099009671126**; model fingerprint `d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025`; checkpoint `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925`. The original audio/video normalization and class orders remain unchanged. A2 was not used and remains intact. The acquisition audit reverified **all 40 model/reference and five canonical-input hashes** unchanged.

Final tested image: `sha256:b654cbece0ad53dc4399239ab97b6483677ed05d86a781ad298ada2b8fdd18d8`. Default command remains `python -m deployment.gradio_app`, user `1000:1000`. The full final-image checks distinguish the optional research path from the ordinary gated deployment. The image has no weights, media or private signing key.

Model inference requires Linux cgroup evidence: UID nonzero, zero capabilities, no-new-privileges, read-only root, loopback-only/no-network routes, ?4 GiB memory, ?2 CPU quota and ?128 PIDs. Tests use `--network none`, `--read-only`, `--cap-drop ALL`, `--security-opt no-new-privileges`, 4 GiB RAM/no extra swap, two CPUs, 128 PIDs and a 512 MiB `/tmp` tmpfs. The private spool is the only research exchange and is writable for cleanup; the key/model mounts are read-only. A networked worker was explicitly tested and refused with `GENERAL_RESEARCH_SANDBOX_REQUIRED`.

CPU float32 inference uses four Torch intraop threads, one interop thread, MKLDNN enabled, gradients and autocast disabled. Startup and each active job have 180-second process watchdogs; a job timeout deletes capture media and produces no score, then terminates the worker. Exact package versions, decoder executable hashes, model catalog and source bindings are included in the genuine result profile. No latency SLA is established by the fixture timing.

| Final image source | SHA-256 |
|---|---|
| `deployment/general_inference.py` | `b725e7699d6f80755a172887545a4cc56221af64b42c24c234809f84e4f7dc61` |
| `deployment/gradio_app.py` | `40531d9d7dbc0457bd2a42fe4ad4fe6e54f2b1ef6d89deeb3848ba9e7f969e2e` |
| `deployment/live/replay.py` | `d3669370819747fa074d6d6246bbafa9b339c66a7e44d49ddf7b7ac023f02ca1` |
| `deployment/live/research_capture.py` | `265e0c7e9d66de73573577856ae1d70a505d8392d35d779f5f002ce09db9a3a0` |
| `deployment/live/research_store.py` | `b4e3cbc48178d4221cc78d926714ed26363d168897111ce15332c4cc9883daa2` |
| `deployment/live_research_app.py` | `10025165fa365109f4e3e6b571ae51d48114a4b0ab7c0bd2ae846337158f81ca` |
| `deployment/ui/live_section.py` | `84c2b8f07c67ab2d42fe58c1161e88f555c357cefa8a9bee2429a3379717d7ce` |
| `deployment/web/live_capture.js` | `d7262185ffb7af0b2902168396f1c782e981450e23d92c3ef76f3c39f97bc779` |
| `deployment/web/live_research_result.js` | `4ed2b7bf7401a883b283eddfcb6995678d5c915d3f5ed97f3ff10b02ff6f0652` |

The code-owned [artifact catalog](../deployment/artifact_catalog.json), SHA-256 `a836d05cd9071282965578c964d3d555405ac02a2f631a9beb2539381bc95418`, binds the following exact runtime files and byte sizes:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `E3/best.pt` | 6706435 | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` |
| `E3/checkpoint_integrity.json` | 281 | `771aa0972cf67082107aabf43c9119ab5a1f2840d85c10d12524f3944319166d` |
| `E3/checkpoint_metadata.json` | 7369 | `fd6abe3f0e62f09e19c096f82b6516195a2c8dba93cd3d534e7f88fcfec48e58` |
| `E3/frozen_operating_point.json` | 457 | `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad` |
| `audio/class_names.json` | 472 | `84d4763979cefd73af2248743067e1e48f368d908a701237e8d3616757921100` |
| `audio/encoder.pt` | 361499833 | `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34` |
| `audio/environment.json` | 255 | `bb53adcedf93ec9c04d293c7f39b0389561830f5eef8cb9741d239e437e1f2da` |
| `audio/model_card.json` | 1198 | `3b98967d720e7fad3de3af37f57487ebd49cf48dc357cef8dbde394a86470bfc` |
| `audio/norm_stats.json` | 32759 | `c0ac268380479149d759f7ba605b881d415c45a83599875d30ef492ef85295e1` |
| `audio/weights.pt` | 2013703 | `d4a0f78403caafde2c3c39d961c9819fbae8caebaaef4b824034c678da860692` |
| `video/class_names.json` | 531 | `05abcc9bba8940f745c2a6842fc268a0f25a5e689f4e1f70193aaa18d00908ac` |
| `video/environment.json` | 183 | `323b5e3f9477db336f7ba2f5568a94089ae5fea1d8172001d926dcbc39bd790a` |
| `video/model_card.json` | 809 | `12f069feab94d3324f80cf9de08390a5f1a6fdaa7acdb293ada1690262925a1b` |
| `video/weights.pt` | 44839691 | `6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec` |

## G. Executed tests and failure accounting

- Final focused live/general/prediction/frontend/upload tests: **182 passed**. Focused live research alone: **22 passed**.
- Final Python regression: **1,090 passed**, **1,346 subtests passed**, **two CUDA-dependent skips**, one upstream deprecation warning. Existing tests were not weakened.
- Final JavaScript regression: **17 passed**, zero failures/skips, including consent-before-device-use, late seal after Stop and Reset cancellation.
- Exact final image: genuine authenticated browser/replay succeeded; cross-login, loopback Host/Origin, artifact-route controls, resource sandbox and ordinary closed gates passed. Broker/replay tensors match exactly on the captured fixture.
- Final-image Stage 2H lifecycle/security smoke: **PASSED**, including actual **301.074 s** expiry, four-file capacity, two stalled TCP readers/30-second timeout, injection and direct-route rejection, cross-login isolation, unclean restart/orphan cleanup and graceful deletion.
- Genuine preprocessing audit: completed actual Chromium resampling/mixing/quantization/JPEG, unchanged authentic encoders and E3; all acquired hashes preserved. It reports differences, not a parity pass.
- Watchdog, metadata/path/tamper/expiry/single-use guards, exact audio continuity, missing/repeated frames, clock drift, active-session logout cleanup and four-outstanding-window budget passed focused tests. The later-window fixture preserves true source coordinates and exact broker/replay tensors.
- **Physical attempt failed** as described in D; no fabricated personal score is reported.

Earlier development failures were corrected and rerun: a Windows text-encoding-corrupted tab selector; test mock code that did not await asynchronous device setup; an image-audit false positive that searched controller source rather than the default HTML template. Review also corrected pre-decode bound ordering, broker-crash revocation/cleanup, result publication and the first-window-only assumption. Earlier image results are not substituted for the final image's browser/regression/security checks. The physical attempt preceded the last window-selection fix and remains a separate failed validation.

## H. Remaining blockers and independent review

No failing fixture/regression test is being hidden. The remaining unresolved physical readiness failure and absent independent input compatibility mean there is **no claim of production live readiness**. Client-reported frame IDs/clocks are quality diagnostics, not independent hardware attestations. Real camera/microphone startup, delays, device resampling/channel behavior, dropped-frame reporting and absolute sensor alignment still need consented device-specific evaluation. The controlled TRAIN intervention is far from the threshold and does not cover independent near-boundary behavior.

| Independent evidence required | Current status |
|---|---|
| Reviewer identity / independence | **null** |
| Authenticated sign-off document/link/digest | **null** |
| Exact artifact, source/image/runtime and browser/device profile | **not approved** |
| Justified numerical acceptance and historical provenance resolution | **inactive / unresolved** |
| Physical A/V offset, latency, drift, loss and near-boundary scope | **not established** |
| Explicit public upload/live authorization | **absent; both gates closed** |

A successful HMAC transport check, compatible tensor shape, one unchanged E3 decision, supplied artifact hash or this draft cannot activate the code-owned release gate. A future approval must independently bind the exact media/preprocessing/model/runtime/resource scope and numerical/clock policy, and be reviewed into code. The current revision contains no approved public live scope.

## I. Reproduction and exact next steps

From the GC2 repository root, use the existing pinned Python test environment and Playwright/Chrome installation:

```powershell
& _local/phase8_stage1/venv/Scripts/python.exe -B -m pytest tests -q --basetemp=_local/live_integration/review-tests
node --test tests/js/*.js
node tests/integration_live_browser.cjs  # fake devices only; authenticated local app + offline worker
```

The browser harness expects the reviewed image tag `avalidate:live-research-candidate`, authenticated original bundles at `_local/phase8_stage2c/models`, and Playwright at `_local/premium_ui/browser/node_modules`. It generates private login Secrets at runtime and never prints them. The controlled preparation script runs inside the pinned image with `/models` and `/inputs` read-only and `/results` pointing to ignored `_local/live_integration`; run `node tests/integration_live_preprocessing.cjs`, then `tests/integration_live_metrology.py` inside the same isolated image with `/fixtures`, `/models`, `/inputs` read-only and `/results` writable.

For manual research, supply the existing `AVAL_ACCESS_MODE=gradio_auth` and private Gradio login Secrets, then run:

```powershell
& _local/phase8_stage1/venv/Scripts/python.exe -B -m deployment.live_research_app --private-dir _local/live_integration/manual-private --port 7861
```

In a separate terminal, after the broker creates its private key/spool:

```powershell
$docker = 'C:/Users/venka/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
$models = (Resolve-Path '_local/phase8_stage2c/models').Path
$private = (Resolve-Path '_local/live_integration/manual-private').Path
& $docker run --rm --network none --read-only --cap-drop ALL `
  --security-opt no-new-privileges --memory 4g --memory-swap 4g --cpus 2 --pids-limit 128 `
  --tmpfs /tmp:rw,nosuid,size=512m -e PYTHONPATH=/app `
  --mount "type=bind,source=$models,target=/models,readonly" `
  --mount "type=bind,source=$private/research.key,target=/research.key,readonly" `
  --mount "type=bind,source=$private/spool,target=/spool" `
  avalidate:live-research-candidate python -B -m deployment.live.replay `
  --models /models --spool /spool --key-file /research.key --watch-seconds 240
```

Do not publish a worker port, move the worker into an ordinary networked Hugging Face Space, copy keys/weights into an image, or expose an unaudited inference endpoint. This architecture needs a local operator-controlled capture broker and private file exchange; it is not a standard hosted live-prediction deployment.

Next actions:

1. Review the explicitly listed source/tests/documentation together as one gated engineering PR; retain all unrelated untracked work. Suggested title: **Add authenticated offline live-input research replay**. No Git mutation is performed by this handoff.
2. Investigate physical readiness using sanitized diagnostics and independently checked test fixtures. Obtain fresh explicit consent before any new physical recording; the prior one-attempt consent is consumed and its marker must remain.
3. Collect a reviewed device/browser corpus and calibrated A/V reference signals. Compare PCM, RGB, clocks, encoder numerics and frozen-E3 boundary behavior independently; do not tune on TEST data or infer universal tolerances from these TRAIN fixtures.
4. Resolve or explicitly review historical numerical provenance/acceptance and define a distinct live-input scope. Keep public Upload and Live predictions disabled until independent authorization is bound into the release gate.

## Local evidence

Evidence stays under ignored `_local/live_integration/`. Raw models, keys, media and fixtures must not enter Git. Exact evidence hashes and the final Stage 2H result are appended after final verification.

Final evidence checks on image `sha256:b654cbece0ad53dc4399239ab97b6483677ed05d86a781ad298ada2b8fdd18d8` completed successfully. The full regression was rerun after the cancellation-metadata cleanup: **1,090 passed + 1,346 subtests**, two CUDA skips. The final focused run passed **182 tests** and JavaScript **17 tests**. The physical attempt remains the explicitly failed validation in D. All 261 baseline files remain present; only eight intended tracked source/documentation files changed, and eleven reviewed files were added. The four unrelated untracked files and protected model/policy/backend/transport sources are preserved. No files are staged.

| Local evidence under `_local/live_integration/` | SHA-256 |
|---|---|
| `browser-fixture-result.json` | `b15eea19a44465d5f41662518bcc6168bdc2a73a3fbbd25111b42ac464f0cd16` |
| `metrology.json` | `528c905ddbc07ec6c12ae8aa435a38c8367654bbc5702973a7a46f4b43a71fc0` |
| `fixture-identities.json` | `6e17e99e8945e924441fd664c2db5a90a043dbed6d189a99898ba5f97ebcafb8` |
| `browser-comparison-profile.json` | `3d8273fff5e604e3855b704da148b4ed56adfa42ed314e825126120f3f47e644` |
| `python-handoff-regression.log` | `d5d5592038bf8c71508fe30acf297debaa47df7a08d0548e2cfb3dac247ac60e` |
| `focused-handoff.log` | `66e514b7fdd7c3b548aded97402c7a4fdd15cdd5c1d2b87648224a1aa88db49a` |
| `javascript-handoff.log` | `f13b1dc89c3816fd5d7016e73af98ce768d2cff3162ee631e01621904331261f` |
| `default-smoke.log` | `5e164663b8c32317494bd8454b9f9305f73b70d0741b4ad84dc671bace360fd8` |
| `final-image.json` | `3bc87bb75f3d9edda01d0f762361fadb470b6f3008d4b96bb0f45232212d3344` |
| `networked-refusal.log` | `f4adc7374a02813358443b85eb5ab670ea7321334817c0111625d29c49c79a9f` |
| `docker-build-final.log` | `8092d4e79eb3882d7d43f56dd7b6c9a57ccd747f39b1eda862e68ec743a8af6c` |
| `physical-attempt.json` | `cf4fed7504e9b88bc04f53c181a7408c90951273f45c29cb831712a69f63399a` |
| `physical-browser-error.txt` | `b02ecceb9edde5fcf2b0e69940ab29d512933ef927c309d656e306506bf40331` |

The exact changed-file sizes/hashes and excluded-local inventory are recorded in `_local/live_integration/handoff-inventory.json`; it remains local review evidence. Neither that inventory nor these hashes confer release approval.

Follow-up: the [capture-duration diagnosis](live_capture_duration_diagnosis.md) records the subsequent packet-finalization fix, preserved limits and newer test/image evidence. The results and image bindings in this dossier describe the earlier handoff; no additional physical recording was made.

# AValidate prediction and UI release review

2026-10-29, Asia/Singapore. Branch `feature/prediction-adapter-premium-ui`, based on freshly fetched
`origin/main` at `53ede8f5b824c19355acf1f9f06ff33ca6601315`. Changes remain uncommitted.

Public uploads still return the original admission contract. Genuine inference is available only in
the separate authenticated, experimental TRAIN demonstration. No independently reviewed public
release authorization was found. No model, threshold, preprocessing policy or parity gate changed.

## Changed-file inventory

| File | Change |
| --- | --- |
| `README.md` | Current interface, internal entry point and this handoff |
| `deployment/gradio_app.py` | Four-tab shell, native login branding, optional internal tab |
| `deployment/ui/theme.py` | Local dark theme, responsive shell, login styling, architecture and model information |
| `deployment/ui/upload_section.py` | Two-column upload/results, empty score state, scalar-handle reset fix |
| `deployment/ui/live_section.py` | Timer, progress, Reset, microphone label, focus and reduced-motion styling |
| `deployment/web/live_capture.js` | Sample-clock timer, Reset, clearing completed cleanup messages |
| `deployment/internal_app.py` (new) | Separate authenticated TRAIN-only entry point |
| `deployment/prediction_adapter.py` (new) | Fixed TRAIN cases, authentic backend adapter and strict typed E3 response |
| `deployment/ui/internal_section.py` (new) | Private internal callback and validated genuine result card |
| `deployment/validate_prediction_demo.py` (new) | Offline authentic cases, repeats, hashes, loading and resource diagnostics |
| `deployment/prepare_release.py` (new) | Import-closure allowlist bundle and SHA-256 manifest; never publishes |
| `tests/test_prediction_adapter.py` (new) | Authentication, scope, identity, numeric, decision-boundary and renderer tests |
| `tests/test_phase8_upload_ui.py` | Assert reset comes from scalar handle |
| `tests/test_phase8_deployment.py` | Extend component fixture and actual pinned launch signature for layout/CSS |
| `tests/js/test_live_security.js` | Timer/Reset/cleanup test and corresponding DOM fixture |
| `docs/prediction_ui_release.md` (new) | This review and manual release procedure |

Existing untracked audio/video trace reports, Stage 2G report and fusion-training requirements were
left untouched. No tests were removed or weakened. All new test fixtures are clearly separated from
authentic inference evidence.

## Interface and authentication

Deep navy surfaces, cyan/teal accents, restrained gradients, local system fonts, rounded panels and
responsive navigation replace the prior stacked layout. Upload and result panels sit side by side
on desktop and stack on mobile. The public score track has no fill or numeric value. Admission
success explicitly says **Model verification unavailable**. Architecture copy describes pairing
correspondence and corresponding time segments; it makes no physical sound-causation claim.

Gradio 6.30's native username/password form, password input, error behavior, POST `/login` and token
registry remain authoritative. Its unauthenticated view does not load Blocks custom CSS, so trusted
server-owned CSS is included through its documented `auth_message` HTML slot. The native button
label **Login** is retained. No replacement login route, client credentials or browser credential
storage is introduced. Anonymous config/upload/queue requests remain unauthorized.

Browser testing found that a File-change reset was rejected by the Stage 2H FileData transport guard.
The reset now follows the opaque scalar handle's change event. The guard still rejects FileData
before Gradio cache traversal, and the private verification callback still resolves a login-owned lease.

Screenshots are local review artifacts, deliberately outside the release bundle:

- [Branded login](../_local/premium_ui/screens/login-desktop.png) and [mobile login](../_local/premium_ui/screens/login-mobile.png).
- [Upload dashboard](../_local/premium_ui/screens/upload-desktop.png), [admitted media with closed gate](../_local/premium_ui/screens/upload-admitted-desktop.png), [390px layout](../_local/premium_ui/screens/upload-390.png).
- [Architecture](../_local/premium_ui/screens/architecture-desktop.png), [capture in progress](../_local/premium_ui/screens/live-recording-desktop.png), [capture diagnostics](../_local/premium_ui/screens/live-desktop.png).
- [Genuine authenticated internal E3 result](../_local/premium_ui/screens/internal-genuine-result.png).

## Inference architecture and contracts

The default Docker command remains `python -m deployment.gradio_app`. It does not construct the
internal adapter, load its checkpoints, or expose its tab. Its existing response remains:

```json
{"status":"MODEL_NOT_READY","predictions":null,"error_code":"RAW_INFERENCE_GATE_CLOSED"}
```

Existing sanitized readiness `issues` are preserved. Rejection codes remain unchanged. Unexpected
internal results cannot become public predictions through the public renderer or JSON sanitizer.

The separate `python -m deployment.internal_app` entry point requires existing Gradio authentication,
read-only authorized model/input mounts and `AVAL_INTERNAL_MANIFEST_SHA256`. Use the existing acquired
manifest digest `65b9c6573ba6fef65ead16da1aeb1e7803fe5ee57c6328a3b89f485f2306187e`.
`AVAL_MODEL_ROOT` defaults to `/models`; `AVAL_INTERNAL_INPUT_ROOT` defaults to `/inputs`.
Missing/mismatched artifacts refuse internal startup with a sanitized error.

The existing `InternalTrainBackend` authenticates model cards, trained artifacts, reference exports,
decoder binaries and exact permitted TRAIN source bytes. It performs bounded audio decoding and
the existing TRAIN frame extraction, validates `(1,10,256)` audio and `(1,10,512)` video features,
and records achieved PTS separately from intended seconds. The adapter selects five original cases
and five prespecified cyclic audio swaps. It never accepts a client filesystem path or arbitrary
compressed media. Models load once per worker; a nonblocking lock serializes case execution.
Request files/features are temporary and the validated UI response contains no audit tensors.

`PairingPrediction` validates schema `aval.train-pairing.v1`, fixed case/source identities, finite
sigmoid/logit agreement, exact E3 artifact bindings and frozen threshold. It emits the actual raw score,
classification, threshold, model/checkpoint hashes, version, processing time and explicit
`EXPERIMENTAL_INTERNAL_TRAIN_ONLY` / `production_predictions_enabled: false` fields. Event and
temporal predictions are null. The decision is **score >= frozen threshold**. Raw scores retain full
display precision; a near-one value is not rounded to a displayed 1. The score is uncalibrated.
The original A2 comparison remains available in the unchanged internal backend.

## Authentic validation and limits

Only the five previously permitted TRAIN sources were used. Positive cases keep their original
pairing; negative cases use audio from the next permitted source with the selected source's video.
These cases were specified in code before scoring; they are not a performance benchmark.

| Case | Genuine E3 score | Frozen decision |
| --- | ---: | --- |
| original-1 | 0.9999999996978804 | Original-like |
| original-2 | 0.9999975629290364 | Original-like |
| original-3 | 0.9999997659062186 | Original-like |
| original-4 | 0.9999999949204514 | Original-like |
| original-5 | 0.9999469800619933 | Original-like |
| swapped-1 | 0.00000385276063285511 | Audio-substituted-like |
| swapped-2 | 0.0005871553196661409 | Audio-substituted-like |
| swapped-3 | 0.0000003868810522365176 | Audio-substituted-like |
| swapped-4 | 0.007293270978074229 | Audio-substituted-like |
| swapped-5 | 0.000000000003698014650016988 | Audio-substituted-like |

The unchanged E3 threshold is **0.009579099009671126**. All ten cases repeated identically on each
storage profile, and scores/logits were identical across Windows-bind and native-volume runs.
Seven restricted `torch.load` calls occurred during initialization; none occurred during requests.
Audio, video, E3 and A2 each loaded once. All 40 model-bundle files retained their SHA-256 hashes.
The authenticated internal HTTP callback also executed real inference and rendered its validated result.

With 2 CPU quota, four intra-op threads, one inter-op thread, CPU float32, 4 GiB RAM and read-only
models, native-volume initialization was 8.160 s; mean case time was 7.557 s first pass / 7.529 s repeat.
Cases include both one-source originals and two-source swaps. Native peak process RSS was 1558.6 MiB.
Windows-bind initialization was 115.469 s and mean first-case time 55.833 s. Concurrent host activity
and storage affect these observations; these are not hosted capacity guarantees. Verified temporary
volume seeding is a separate cost (38.442 s in the native validation run).

The smallest observed score-to-threshold distance was 0.0022858280315968963. An adjacent-value unit
test covers the exact >= decision boundary; the authentic study does **not** establish independent
near-threshold coverage or numerical acceptance. Historical exact parity remains unresolved,
runtime tolerance acceptance remains inactive, and general decoder/shared-clock approval remains
pending. The new diagnostics read permitted TRAIN sources only; no TEST-driven tuning was performed.
No per-second fused presence head exists.

## Security, regression and release

Final full regression: **1,002 passed, 1,346 passing subtests, two unavailable-CUDA skips**, exit 0.
The earlier focused run passed 117 tests; the final full suite also includes the scalar-reset guard
and exact-boundary test. All seven Node-reported JavaScript tests passed. One pre-existing PyTorch
weight-norm deprecation warning remains.

Full regression and focused tests cover authentication, session/upload isolation, malformed inputs,
resource limits, cleanup, internal/public contract separation and existing model contracts. Browser
checks use synthetic media/devices: native login error/success, local video preview, valid admission,
390/768/1440px layouts, ten-second capture contract, Reset, reduced motion and no page JavaScript
errors. Physical camera/microphone hardware and other browser engines were not tested.

The full Docker HTTP smoke exercises the actual non-root CMD on port 7860, missing-auth refusal,
unauthorized requests, cross-login theft attempts, direct REST and FileData rejection, valid media
admission with null predictions, malformed codecs/duration, upload limits, stalled TCP bodies,
four-file capacity, the unchanged five-minute expiry, restart invalidation and shutdown cleanup.
Layer scanning checks private artifact hashes, media/weight paths, credential patterns and image
environment secrets. Evidence and exact exit codes are in `_local/premium_ui/`.

The final exact-bundle Docker build and full HTTP smoke both exited 0. Production upload expiry
was observed at 301.116 seconds; isolation, restart and graceful cleanup checks passed. All 13 image
layers (40,921 regular files) were scanned with no findings. The final image ID is
`sha256:d7daaf8f5d21812ee81fff9f9763924c2f00a77e9b872956e008be73afe17f6f`.

Preservation checks retain all 224 untouched tracked files and all 40 authentic model files,
including byte-identical runtime, internal backend, encoder construction and fusion/parity gates.
The Git index is empty. Existing research checkpoints and archives are not release contents.

`deployment.prepare_release` produces an 83-file import-closure bundle and SHA-256 manifest under
`_local`. The final reviewed directory is `_local/premium_ui/release-reviewed-final`, with its
sibling manifest. It includes exact Docker ignore allowlists to exclude unknown stale remote files.
It contains source only, with no weights, datasets, user media, credentials or internal demo entry
point. It starts the gated public application; the experimental demo runs from the source checkout.

Manual publishing procedure:

1. Review this branch diff, source hashes, screenshots and the final test evidence; obtain owner review.
2. Check the current head of the existing `vvenkata/avalidate-audio-video-verification` Space before
   publishing, and compare its files/settings with the reviewed bundle. Do not create a replacement Space.
3. Keep visibility Private, Docker port 7860, UID 1000, `AVAL_ACCESS_MODE=gradio_auth`, existing
   Secrets and the default gated command. Publish only the reviewed manifest files using the existing
   Space publishing workflow and a checked target revision. No publishing is performed by these tools.
4. After an owner-authorized publication, repeat sign-in, valid upload/null-result, isolation, capture,
   expiry and shutdown checks on the hosted Space. Roll back to its recorded prior revision on failure.

The hosting configuration is consistent with the official
[Hugging Face Docker Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-docker).
CPU Basic suitability for full inference has not been established. Authorized artifact retrieval,
startup storage, memory, worker concurrency and hardware selection require separate review. Never
put private checkpoints into the image or the Space source repository.

## Release verdict

| Scope | Verdict |
| --- | --- |
| UI | READY |
| Internal prediction | READY for the authenticated, fixed TRAIN demonstration only |
| Public prediction | VALIDATION BLOCKED |
| Hosted release | READY for owner review of the gated bundle; no deployment performed |

Public prediction blockers are exact historical audio/video compatibility, an independently accepted
numerical policy, approved general compressed-media decoding/shared-clock behavior, a reviewed tested
media scope with prespecified TRAIN negatives/near-threshold coverage, and a separately documented
independent release decision binding all preprocessing/model artifacts. The five-clip study cannot
authorize arbitrary tolerances or remove any of those gates. No commit, push, merge, retraining,
checkpoint substitution, threshold edit or Hugging Face deployment was performed.

# Phase 8: final system and deployment readiness

Historical audit scope: Tasks 8.1–8.2, deployment readiness and evidence-supported integration
planning. The capability statements in the original audit below describe that audit snapshot; they
are not a current inventory of Phase 8 files. This document is retained as historical evidence.

## Current implementation status (Tasks 8.3–8.4)

The Docker/Gradio readiness foundation is implemented in `Dockerfile`, `deployment/`, and the root
`README.md`. The image has not been built or cold-started, and no Hugging Face Space has been
created or published. Raw-media inference remains disabled: trained raw encoder artifacts,
verified media decoding, and raw-to-TRAIN export parity are still missing. Public deployment has
not been approved. Upload handling requires configured Gradio authentication and server-side
container/codec admission; these controls still require targeted security review and a verified
container build.

No encoder was trained or run on raw media. No held-out TEST or challenge cohort was scored, and
no model weights were downloaded.

## Repository and phase dependency

At the time of the original 8.1–8.2 audit, the work was on `feature/phase8-final-system`, based on `origin/main` at
`3587a854bb8fdf9638bf72c8d67d2368b517b6f5`. The current `origin/main` includes the Phase 6
and Phase 7 merge commits. At that historical point the branch had no Phase 8 implementation yet;
Tasks 8.3–8.4 have since added the Docker/Gradio foundation described above. The pre-existing
untracked `requirements-fusion-training.txt` remains outside the deployment deliverables.

The current full-suite baseline for this branch is 685 tests: 683 passed and two CUDA-dependent
tests were skipped. An earlier audit recorded 553 tests on a previous snapshot; that historical
count is superseded and is not the baseline for this branch.

## Readiness definitions

**READY** means the specified capability is present and verified for its stated boundary.
**MISSING** means a required file, dependency, or model head is absent. **UNVERIFIED** means
source or exported evidence exists, but the claimed execution, provenance, or parity has not
been established. Export availability alone does not make an encoder executable.

## Audio encoder readiness

| Dependency or contract | Status | Evidence and boundary |
|---|---|---|
| BEATs iter3 architecture | UNVERIFIED | Vendored model code exists. The raw encoder has not been reconstructed and executed in this environment. |
| Exact BEATs iter3 base checkpoint | MISSING | `_local/drive/pretrained/BEATs_iter3*.pt` is absent. `iter3_as2m` is a different model and is not a valid replacement. |
| Trained `beats_lora_r8_v1` LoRA and embedding head | MISSING | Expected `_local/drive/artifacts/audio_baseline/beats_lora_r8_v1/best.pt` is absent. |
| Frozen 768-dimensional normalization statistics and cache layout | MISSING | Expected `_local/drive/cache/beats_iter3/norm_stats.json` and the associated `train_X.npy` layout evidence are absent. |
| Audio inference dependency stack | MISSING | This environment has CPU PyTorch 2.14.1 and NumPy 2.4.6, but no `torchaudio`, `pandas`, `soundfile`, or `tqdm`. A deployment-compatible locked set has not been established. |
| Historical audio preprocessing | UNVERIFIED | Source specifies mono 16 kHz, 160,000 samples, BEATs 128-bin fbank (25 ms window / 10 ms hop), scale 32768, fixed fbank normalization, q/v LoRA rank 8 / alpha 16, per-second pooling, frozen 768D normalization, and a 768→256 GELU embedding head. Raw parity and checkpoint binding cannot be checked without the artifacts. |
| Expected audio embedding | UNVERIFIED | `(B,10,256)` is the declared contract. TRAIN exports exist, but they do not prove raw encoder execution or checkpoint origin. |
| Audio event classifier | MISSING for deployment | A trained 29-class branch head is described by the training code and historical evidence, but the required base, trained adapter/head and normalization are unavailable for execution. |

The only file in `pre_trained_encoders/` is `audio_feature.h5` (42,426,464 bytes). It is a feature
artifact, not an encoder checkpoint. The legacy BEATs loading path and normalization code also rely
on assumptions that need to be bound to the exact saved artifact before deployment.

## Video encoder readiness

| Dependency or contract | Status | Evidence and boundary |
|---|---|---|
| ResNet-18 architecture | UNVERIFIED | `vision/model.py` defines the architecture and a 512-dimensional representation, but raw inference has not been reconstructed and executed here. |
| Trained `resnet18_v1` checkpoint | MISSING | Expected `_local/drive/artifacts/vision_baseline/resnet18_v1/best.pt` is absent. An external path in historical provenance is not a local artifact. ImageNet initialization alone is not a substitute. |
| Video inference dependency stack | MISSING | This environment has no `torchvision`, OpenCV, Pillow, or `ffmpeg` executable. No compatible deployment lock or decoder image exists. |
| Historical frame preprocessing | UNVERIFIED | Source samples ten nominal one-fps frames, converts BGR to RGB, resizes directly to 224×224 with OpenCV `INTER_LINEAR`, divides by 255, then applies ImageNet channel mean/std. It can substitute a last or black frame on decode failure and uses an FPS fallback; these behaviors do not establish valid observed frames. |
| Expected video embedding | UNVERIFIED | `(B,10,512)` is the declared contract. TRAIN exports exist, but they do not prove raw execution or checkpoint origin. |
| Visual event classifier | MISSING for deployment | A trained 29-class per-second branch head is described in the source and historical evidence, but its trained checkpoint is unavailable. |

## Phase 6–7 integration status

| Component | Status | Current boundary |
|---|---|---|
| Frozen E3 and A2 feature adapters | READY for validated feature batches | Selected checkpoints, model fingerprints, operating-point provenance, and model-specific frozen thresholds are checked. CPU inference accepts the existing `(B,10,256)` audio and `(B,10,512)` visual features. This is not raw-media inference. |
| Frozen thresholds | READY and immutable in this scope | E3: `0.009579099009671126`; A2: `0.0026640388978268056`. The inclusive decision is sigmoid score `>=` its bound threshold. No threshold was tuned here. |
| Phase 3 JSD reference | READY as an authenticated TRAIN score source | Phase 7 admission/reporting verifies the archived TRAIN evidence and provenance. JSD is not an executable raw encoder or learned fused model. |
| Encoder artifact interfaces | READY as a fail-closed loader contract | `fusion/encoder_interfaces.py` validates declared artifact roles, hashes, configurations and dependency availability. Current readiness remains blocked because required files/dependencies are missing. It does not download weights. |
| Canonical raw-media boundary | READY as validation only | `RawMediaInput` describes finite mono float32 ten-second audio and ten RGB frames, with explicit timing. `encode_raw_media` rejects execution because encoder weights, normalization, and parity are unavailable. |
| Decoded-media orchestration | READY as an integration skeleton | `fusion/end_to_end.py` checks readiness, timing, validity and frozen fusion bindings. Existing strict decoder supports canonical local tensors, not arbitrary compressed uploads. |
| TRAIN export parity | UNVERIFIED for raw encoders | Existing E3/A2 integration is authenticated against feature exports. It does not prove either encoder reproduces those exports from raw audio/video. Full encoder replay/parity must precede raw inference. |
| Phase 7 evaluation and reporting | READY for authenticated TRAIN-only archived scores | It authenticates the frozen JSD/E3/A2 TRAIN evidence. It does not authorize or perform validation, TEST, or challenge scoring in this task. |
| Existing web application | NOT A MODEL INFERENCE PATH | `app.py` labels its output as placeholder/annotation lookup behavior and reads AVE ground-truth annotations. It must not be deployed as a trained prediction interface. |

The integration chain intended for a later verified deployment is:

```text
uploaded pair
  → audited demux/decode and timestamps
  → strict ten-second audio/video canonical tensors
  → verified BEATs+LoRA and trained ResNet-18 encoders
  → TRAIN-export parity gate and explicit validity masks
  → frozen E3/A2 feature adapters
  → typed outputs with model/checkpoint/threshold provenance
```

Each stage must fail closed on missing artifacts, invalid inputs, decode failures, timestamp
mismatch, nonfinite tensors, hash mismatch, unsupported media, or parity mismatch. No random,
ImageNet-only, alternate-BEATs, annotation-lookup, or synthetic fallback may be promoted to a
real prediction.

## Supported input and output scope

The current strict proposed raw input contract is a synchronized ten-second pair: mono float32
audio at 16 kHz with exactly 160,000 samples, plus ten RGB uint8 frames of shape
`(10,224,224,3)` at timestamps 0–9 seconds. Origins and durations must be explicit and equal.
This is a tensor boundary, not a user-facing media decoder. Arbitrary duration, resampling,
downmixing, padding/trimming, frame substitution, variable-frame-rate selection, and independent
stream clock alignment are not supported by verified deployment code.

| User-facing output | Genuine trained head available? | Can it run on uploaded raw media now? | Safe interpretation |
|---|---|---|---|
| Binary pairing correspondence | Yes: frozen E3 and A2 binary fusion classifiers | No. Feature-batch inference is available; raw encoders/decoder are blocked. | Original/substituted pairing-provenance prediction under the frozen model. This does not establish physical sound-source causality or identity. |
| Correspondence score | Yes: E3/A2 sigmoid outputs | No, for the same raw-media blocker | Uncalibrated sigmoid score for the binary correspondence target; do not present it as a calibrated probability. |
| Audio event category | A trained 29-class unimodal head is documented | No; executable checkpoint chain is incomplete | Unimodal audio label only, if exact artifacts and parity are later verified. |
| Visual event category | A trained 29-class unimodal head is documented | No; trained checkpoint is missing | Unimodal visual label only, if exact checkpoint and parity are later verified. |
| Fused event classification | No | No | No genuine trained fused 29-class head is present. E3/A2 do not supply one. |
| Per-second audiovisual presence | No fused temporal presence head | No | Do not derive per-second decisions from E3/A2 pair scores or attention weights. |

## Proposal-required heads and evaluation still needed

AVE has per-second labels for 28 foreground event classes and background. The branch event heads
are distinct unimodal outputs; their existence does not create a fused audiovisual head. If the
proposal requires a fused event class or temporal audiovisual-presence output, the project needs
an explicitly trained and checkpointed head with a documented target mapping, audio/video label
alignment, valid-time and missing-label policy, architecture, loss, and train-only fitting procedure.
The semantics must state whether a positive bin means an event is audible, visible, or both; these
targets cannot be inferred from the binary correspondence label.

The required sequence is: establish permitted TRAIN labels and timing; train the specific head on
TRAIN only; select architecture/epoch and any calibration on the existing development validation
protocol; freeze the model and full provenance; add synthetic and data-contract tests; perform
TRAIN replay and label/prediction identity checks; then obtain explicit authorization for final
held-out TEST scoring. Keep same-category challenge scoring separate and deferred. No event or
temporal metrics are reported by this readiness audit.

## Hugging Face deployment recommendation

**Recommendation once artifacts and parity gates pass: a Docker Space running a small Gradio UI.**
Docker provides a place to pin the operating-system decoder libraries and compatible Python,
PyTorch, TorchAudio, TorchVision, OpenCV, and FFmpeg stack. Gradio can provide the upload and
result interface inside that controlled image. The Dockerfile and Gradio readiness application were
added in Tasks 8.3–8.4. This remains a foundation, not a verified deployment: the image has not been
built, the app has not been cold-started, and there is still no verified raw-media predictor.

Hugging Face offers Gradio and Docker Space SDKs and documents Space hardware, storage, and
runtime configuration. Hardware availability, account plan requirements, prices, sleep behavior,
and limits can change; verify the current official documentation before choosing or budgeting a
Space: [Spaces overview](https://huggingface.co/docs/hub/spaces-overview),
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[GPU Spaces](https://huggingface.co/docs/hub/main/en/spaces-gpus), and
[Spaces storage](https://huggingface.co/docs/hub/spaces-storage).

### Artifact and runtime design

- Store the exact encoder and fusion files in a versioned, access-controlled model repository;
  pin a revision and expected SHA-256 values in reviewed deployment configuration. Fetch only
  those pinned files. Verify hashes before safe loading and fail readiness on missing or mismatched
  content. Never put private weights in the application Git repository or public Space files.
- Record architecture/configuration, selected checkpoint digest, normalization digest, producer
  and decoder versions, Python and package lock, preprocessing profile, and the frozen E3/A2
  operating-point digests. Do not change the Phase 6–7 provenance contracts or thresholds.
- Build a locked container with a deliberately selected CPU or CUDA PyTorch stack and pinned
  FFmpeg/OpenCV/system libraries. Current source requirement files are research-oriented and
  unpinned or broad; they do not constitute a deployment lock. Gradio and a production serving
  framework are not installed in the current environment.
- Use a read-only mounted model repository where supported, or fetch the exact pinned model
  revision into a controlled cache. Avoid persistent storage for user media. Model caching may
  survive only according to the chosen Space storage/runtime configuration; test cold startup and
  restart behavior.
- Keep credentials in Space Secrets, never in source or logs. A public interface should contain no
  model-repository token; use private access only when required. Do not create or request credentials
  as part of this phase.

### Input safety, privacy, and service limits

For a later prototype, accept only one validated audio-video pair with a ten-second maximum until
longer clips have their own training and parity contract. Set a measured upload byte limit, batch
size one, and queue concurrency one initially. These are proposed conservative defaults, not
benchmarked capacity claims. Validate actual container duration, both stream durations, origin,
timestamps, channels, sample rate, frame availability, and decodability. Reject clips with unsupported
codecs, absent modalities, stream mismatch, or decode substitution. Do not silently pad, trim, or
stretch streams to fit.

Write uploads into request-scoped temporary storage with unpredictable names and delete them in
a `finally` path on success and failure. Do not persist uploads or log content, filenames, or
annotation-derived labels. Keep logs to request ID, stage, duration, output status, and sanitized
error code. Add tests for cleanup under exceptions, size/duration rejection, concurrent requests,
and absence of request content in logs before any public launch. Use private/restricted access until
the application output and privacy posture have been reviewed.

### CPU and GPU feasibility

CPU E3/A2 feature inference has been exercised. It is not evidence of end-to-end raw encoder
latency, peak memory, or concurrency. CPU raw feasibility is **UNVERIFIED** because neither raw
encoder is currently runnable. GPU feasibility is also **UNVERIFIED**: this environment is CPU-only,
there is no CUDA benchmark, and no target Space hardware is selected. Do not promise a free GPU,
startup time, throughput, or a maximum upload size. After artifact recovery and parity, benchmark
cold start, one warm ten-second request, peak host/device memory, decoder latency, and restart/cache
behavior on the intended Space tier. Select CPU or GPU from those measurements and the current
official hardware catalog.

## Minimum implementation sequence

1. Recover exact BEATs iter3 base, trained `beats_lora_r8_v1` checkpoint/config, frozen 768D
   normalization assets, and trained `resnet18_v1` checkpoint through an approved source. Verify
   independent hashes and bind each artifact to the existing `EncoderSpec`; do not substitute weights.
2. Recover a small permitted TRAIN-only raw audio/video fixture and its original manifest identities.
   Establish decoder/tool versions, audio downmix/resample rules, stream clock origin, frame selection,
   and explicit per-bin validity without using labels to fabricate validity.
3. Install and lock a compatible environment. Reconstruct both encoders with safe loading and exact
   training configuration. Verify `(B,10,256)` and `(B,10,512)` shapes, finiteness, eval mode, frozen
   parameters, and deterministic repeated CPU output.
4. Replay the raw TRAIN fixture through each encoder and require exact or prospectively approved
   tolerance parity with its authenticated TRAIN export, including source rows, sample IDs, temporal
   bins, labels and validity. Preserve the existing feature exports and thresholds. Any mismatch blocks
   raw inference pending investigation.
5. Connect the verified decoder and encoders to `MediaFusionPipeline`, then verify that E3/A2 frozen
   feature outputs match their existing behavior. Add rejection and mutation tests for all public input
   and artifact boundaries. Do not promote internal helper behavior into a public API without validation.
6. Implement the Docker/Gradio surface with bounded uploads, temporary-file cleanup, safe model loading,
   controlled logs, pinned dependencies, health/readiness states, and private access. Benchmark the
   selected target hardware and cold-start behavior before publication.
7. If proposal outputs require fused event or temporal predictions, train and freeze the separately
   specified heads under the documented split/label rules, then repeat the parity and development
   validation audit. Obtain explicit authorization before any final held-out TEST or challenge evaluation.

## Evaluation, leakage, and release gates

Phase 7 provides authenticated archived TRAIN comparisons for JSD, E3 and A2. Its metrics and
reporting contracts mark sigmoid outputs uncalibrated; Brier score and reliability bins are
diagnostics, not calibration proof. Those archived results are historical evidence, not new scores
from this task. No validation, held-out TEST, or same-category challenge predictions were created,
loaded for scoring, or used for threshold selection here.

The official annotations have known cross-split source overlap: 10 unique source videos are shared
between TRAIN and validation, 8 between TRAIN and TEST, and 0 between validation and TEST. This
does not alter the official split; it limits source-level independence/generalization claims and
requires deduplication-aware interpretation. Pair-row splits must not be described as source-video
independent.

Before final release, require all of the following:

- Exact raw encoder and normalization artifacts with independent hash/provenance binding.
- Compatible locked dependencies and pinned decoder/system libraries.
- Passing synthetic contract and full regression suites.
- TRAIN raw-to-export parity, identity/label/source-row alignment, and explicit validity proof.
- End-to-end CPU/GPU measurements on intended hardware, with documented upload and concurrency limits.
- Complete security/privacy review of media handling, secrets, logs, temporary cleanup, and access control.
- Separate, genuinely trained event/temporal heads if those outputs remain proposal requirements.
- Frozen model/checkpoint/threshold manifests and an approved, authorized held-out evaluation plan.
- Independent review before Space creation or public release.

**Readiness conclusion:** Phase 8 Tasks 8.1–8.2 are ready as a documented foundation. Genuine raw-media
prediction and deployment remain blocked by missing verified encoder artifacts, decoder/runtime
dependencies, and TRAIN raw-to-export parity. Do not publish or describe the existing annotation-driven
application as a trained inference system.

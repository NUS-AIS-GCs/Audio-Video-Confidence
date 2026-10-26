# Phase 8 Tasks 8.3–8.4: deployment foundation

This change adds a Docker/Gradio readiness surface only. It is **not deployed**, does not load
example predictions, and keeps inference disabled. With required Gradio authentication configured,
the application starts with `MODEL_NOT_READY` until verified assets, exact runtime dependencies, a
supported decoder path, and raw-to-TRAIN export parity are available. Even then, the current Phase 6
parity bridge only admits authenticated TRAIN identities; support for arbitrary public uploads
requires a separately reviewed inference contract.

## Architecture

```text
Hugging Face Docker Space (future)
  → pinned Python/CPU PyTorch container + FFmpeg/ffprobe
  → deployment.gradio_app (separate from app.py)
  → Gradio authentication → private managed upload path
  → upload size/extension/container/codec/stream/duration checks
  → bounded ffprobe output and time
  → MODEL_NOT_READY response; no prediction path is enabled

Future after independent gates:
  upload → audited decoder → canonical tensors + shared-clock evidence
  → fusion.encoder_interfaces.load_encoder (hash-verified CPU weights)
  → Phase 6 MediaFusionPipeline + genuine TRAIN-export parity
  → frozen E3/A2 Phase 4/5 adapters with unchanged checkpoint/threshold bindings
```

`deployment/runtime.py` checks pinned package versions, `ffmpeg`/`ffprobe`, configurable encoder
artifact paths and expected SHA-256 values, and the fixed E3/A2 checkpoint/metadata/operating-point
hashes from the existing Phase 6 adapters. It never downloads model artifacts. If encoder loading is
requested, it verifies bytes before and after safe `weights_only=True` reads, obtains architecture
configuration from those verified payloads, then delegates construction to `load_encoder`. That
function enforces exact producer configuration and full trained-state coverage.

The public Gradio callback uses a bytes input, applies a 50 MiB default upload cap, validates
exactly one audio and one video stream with duration within 50 ms of ten seconds, writes a
request-private temporary copy, invokes `ffprobe` without a shell, and removes the copy on every
exit path. This is admission validation only. It returns `predictions: null` and
`MODEL_NOT_READY` for an otherwise valid upload because no trusted compressed-media decoder or
raw parity gate exists. It does not convert media to canonical tensors.

The Phase 6 integration seam accepts only a concrete, non-synthetic `MediaFusionPipeline` and
`DecodedTensorSource`; it requires the pipeline's `TRAIN_EXPORT_PARITY_MATCHED` result with both
frozen E3 and A2 outputs. The upload handler does not call this seam. The current Phase 6
`TrainParityFusion` is restricted to authenticated TRAIN identities and cannot certify an unseen
user upload.

## Runtime specification

- Base: official `python:3.11.17-slim-bookworm`, Linux amd64, pinned by the current image digest in
  `Dockerfile`. The container runs as UID 1000, matching Docker Spaces' documented runtime user.
- CPU-only PyTorch family: `torch==2.11.0+cpu`, `torchvision==0.26.0+cpu`,
  `torchaudio==2.11.0+cpu`, using the official CPU wheel index.
- Direct Python dependencies are exact-pinned in `requirements-deployment.txt`, including NumPy,
  pandas, SoundFile, tqdm, headless OpenCV, Pillow, and Gradio. They are candidate deployment pins,
  and the startup check verifies each direct pin including Pillow. They are not a validated encoder
  runtime; parity and supported-build verification are outstanding.
- FFmpeg/ffprobe are installed from Debian Bookworm. The `ffmpeg` package version is pinned in
  `Dockerfile`. The OS repository still supplies transitive system packages; record the built image
  digest and freeze a complete transitive package lock before any public deployment.
- No CUDA/GPU is selected or required. CPU feature inference does not establish raw encoder
  latency or memory feasibility. CPU/GPU performance remains unbenchmarked.
- Hugging Face Space metadata is in the new root `README.md` (`sdk: docker`, port 7860). A Docker
  Space uses the repository's `Dockerfile` and one externally exposed app port. See the official
  [Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker) and
  [Gradio launch documentation](https://www.gradio.app/docs/gradio/blocks).

## Artifact configuration and trust

Mount model files read-only under `/models`; no model files are copied into this repository image.
The runtime supports path overrides rooted under `AVAL_MODEL_ROOT` and requires operator-supplied
hashes for the four encoder artifacts. The hash values must come from an independently reviewed
artifact manifest and be entered through protected deployment configuration. They are not read
from the artifact itself. The selected E3/A2 hashes remain fixed in `fusion/inference.py`; only
their model directory paths are configurable.

| Environment variable | Default / purpose |
|---|---|
| `AVAL_MODEL_ROOT` | `/models`; artifact root. Paths outside this directory reject. |
| `AVAL_AUDIO_BASE_PATH` / `AVAL_AUDIO_BASE_SHA256` | BEATs iter3 base checkpoint path and expected digest. |
| `AVAL_AUDIO_TRAINED_PATH` / `AVAL_AUDIO_TRAINED_SHA256` | `beats_lora_r8_v1` checkpoint containing the trained LoRA/head state. |
| `AVAL_AUDIO_NORMALIZATION_PATH` / `AVAL_AUDIO_NORMALIZATION_SHA256` | Frozen BEATs 768D means/std and schema. |
| `AVAL_VISION_TRAINED_PATH` / `AVAL_VISION_TRAINED_SHA256` | Trained `resnet18_v1` checkpoint. |
| `AVAL_E3_DIR`, `AVAL_A2_DIR` | Directories containing the fixed selected fusion checkpoints and evidence. |
| `AVAL_MAX_UPLOAD_BYTES` | Positive byte limit, default 52,428,800; values above 50 MiB reject. |
| `AVAL_ACCESS_MODE` | Required exact value `gradio_auth`; other values disable startup. |
| `AVAL_GRADIO_USERNAME` / `AVAL_GRADIO_PASSWORD` | Required non-empty Gradio credentials; supply as protected deployment secrets. Never put values in source or logs. |
| `GRADIO_TEMP_DIR` | `/tmp/gradio` in Docker; Gradio-managed uploads must resolve beneath this directory. |
| `PORT` | Gradio listen port, default 7860. |

The app requires supported Gradio username/password authentication before it starts. Missing or
unsupported access configuration stops startup, so there is no unauthenticated upload callback.
Gradio launch authentication protects the UI and direct HTTP event/API requests. The app also sets
`api_visibility="private"`, but that is only API discovery/client visibility and is not an
authentication control. Set the Hugging Face Space itself to **Private** during initial testing as
defense in depth; Space visibility is configured outside this repository and is not verifiable by
the app. Do not publish a public Space until access control and upload handling pass separate review.

The application contains no credentials in source, HF token, model-download client, annotation lookup, or
model-weight fallback. A private model repository token is not needed by this foundation. If a
future deployment needs one, provide it as a runtime Space Secret and use an explicitly pinned
repository revision; never add it to build arguments, source, output, or logs.

## Server-side upload policy

The Gradio file component uses its managed filepath mode so the callback receives the server-side
temporary filename. The callback resolves the path and requires it to remain under `GRADIO_TEMP_DIR`,
checks the actual byte count before reading at most `max_upload_bytes + 1`, preserves only its
basename for suffix admission, and deletes the managed upload in a `finally` path. Missing or
unpreserved filename metadata fails closed. The application then copies the bounded bytes to its own
request-scoped temporary file, probes actual container and stream metadata, and removes that copy.
Direct event/API requests pass through the same callback and authentication gate.

Accepted suffix/container/codec combinations are deliberately narrow:

| Suffix | ffprobe container name must include | Video / audio codecs |
|---|---|---|
| `.mp4` | `mp4` plus MP4 major brand (`isom`, `iso2`, `mp41`, or `mp42`) | H.264 / AAC |
| `.mov` | `mov` plus QuickTime major brand (`qt  `) | H.264 / AAC |
| `.mkv` | `matroska` | H.264 / AAC |
| `.webm` | `webm` | VP9 / Opus |

The suffix is not trusted as content proof; both suffix and ffprobe metadata must agree with the
profile. Exactly two streams are accepted: one audio and one video stream. Missing/ambiguous codecs,
extra streams, unknown containers, absent per-stream durations, nonfinite durations, or clips
outside ten seconds ±50 ms reject. The target encoder decoder contract is still unverified; this
allowlist only bounds upload admission and does not enable decoding or predictions.

## Readiness and errors

Startup reports `MODEL_NOT_READY` with sanitized categories for missing or mismatched package
versions, missing `ffmpeg`/`ffprobe`, absent artifact hashes/files, incorrect hashes, missing
E3/A2 evidence, `NOT_IMPLEMENTED` raw decoder, and `NOT_VERIFIED` TRAIN parity. It never prints
artifact paths, digests, uploaded filenames, content, or exception text. It starts the UI so the
blockers are visible; the upload button is explicitly an admission check and cannot run inference.

Upload errors have stable codes for missing upload, oversize upload, unsupported extension,
container or codec, malformed metadata, missing/extra streams, invalid or mismatched duration,
probe timeout/output overflow, and unavailable decoder dependencies.
All responses keep `predictions` null unless a future separately audited implementation replaces
the current disabled inference gate. Gradio API visibility is private, the launch requires
application credentials and does not create a share URL, analytics and monitoring are disabled,
concurrency is one, queue capacity is eight, and Gradio's upload cap is configured at launch.

## Privacy and temporary storage

The UI receives a Gradio-managed path, verifies that it is inside `GRADIO_TEMP_DIR`, reads at most
50 MiB plus one byte, and removes the managed file after handling. The callback separately writes
one request-scoped generated-name copy and removes it in a context manager on success and failure.
Container execution is non-root. `ffprobe` has an eight-second deadline, a 64 MiB FFmpeg allocation
ceiling, and a 256 KiB combined stdout/stderr cap; overflow kills the child and returns a sanitized
rejection. Logs contain only status and allow-listed error codes. No uploaded media is intentionally
persisted. Gradio upload queue capacity is eight with one callback at a time.

These application limits do not impose portable per-process CPU/memory limits. For local Linux
Docker smoke tests, apply cgroup controls such as `--memory`, `--cpus`, and `--pids-limit` to `docker
run`; select values from measured requirements, since none have been benchmarked. Hugging Face
Space CPU/RAM limits are controlled by the selected Space hardware and must be verified there.
`GRADIO_TEMP_DIR` and model storage are separate. Docker build, cold start, cleanup behavior, and
Space ephemeral-storage behavior remain unverified.

## Build and local run commands

From the repository root:

```powershell
docker build --platform linux/amd64 -t avalidate-phase8 .
# Populate these variables from a protected local secret store before running.
$env:AVAL_ACCESS_MODE = 'gradio_auth'
# AVAL_GRADIO_USERNAME and AVAL_GRADIO_PASSWORD must also be populated securely.
docker run --rm -p 7860:7860 `
  -v "${PWD}/path/to/verified-models:/models:ro" `
  -e AVAL_ACCESS_MODE `
  -e AVAL_GRADIO_USERNAME `
  -e AVAL_GRADIO_PASSWORD `
  -e AVAL_AUDIO_BASE_SHA256=REPLACE_WITH_REVIEWED_SHA256 `
  -e AVAL_AUDIO_TRAINED_SHA256=REPLACE_WITH_REVIEWED_SHA256 `
  -e AVAL_AUDIO_NORMALIZATION_SHA256=REPLACE_WITH_REVIEWED_SHA256 `
  -e AVAL_VISION_TRAINED_SHA256=REPLACE_WITH_REVIEWED_SHA256 `
  avalidate-phase8
```

These commands only build and run the local readiness UI; with the current assets it must report
`MODEL_NOT_READY`. Do not push the repository to a Space as part of this task.

## Unresolved assets and gates

Missing local artifacts include:

- Exact self-supervised BEATs iter3 base checkpoint with its original `cfg` and state.
- Trained `beats_lora_r8_v1` state/config containing LoRA and the embedding/event head tensors.
- Frozen `beats_iter3` 768-dimensional normalization JSON and cache-layout provenance.
- Trained `resnet18_v1` checkpoint and exact training configuration.
- Existing E3/A2 checkpoint evidence in the deployment model mount.
- Authorized raw TRAIN audio/video fixtures and source manifest identities for parity replay.

Also unresolved: exact supported encoder/runtime combination; full transitive Python and OS package
lock; reproducible FFmpeg/OpenCV decoder signatures; compressed upload decoding with PTS and
validity evidence; raw encoder replay against authenticated TRAIN exports; unseen-media inference
contract; and CPU/GPU cold-start, latency, memory, and concurrency measurements. Do not enable
genuine predictions until those gates pass and an independent review accepts the resulting
provenance. Do not infer temporal predictions from attention weights. Any future correspondence
score remains uncalibrated unless a separate calibration protocol is approved and evaluated.

## Verification scope

Synthetic tests exercise readiness aggregation, artifact hash rejection, path boundaries, upload
size/duration/modality rejection, sanitized logging, temp-file cleanup, no annotation fallback, and
the nonsynthetic Phase 6 result gate. They do not load encoders, calculate TEST/challenge metrics,
or claim real-media predictions. The existing full regression suite remains the compatibility gate.

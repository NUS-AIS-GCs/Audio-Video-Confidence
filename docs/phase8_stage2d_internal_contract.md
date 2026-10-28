# Stage 2D internal backend contract

`aval.internal-train-inference.v1` is a **local, experimental TRAIN-only interface**.
It is not connected to Gradio, a network endpoint, or production readiness.
`deployment/runtime.py`, `deployment/gradio_app.py` and the existing Phase 6
strict parity pipeline remain unchanged.

## Current public boundary

The existing Gradio entry point requires configured credentials, private upload
admission, one concurrent callback and a queue of at most eight. Accepted media
still returns `MODEL_NOT_READY`, `predictions: null`, and
`RAW_INFERENCE_GATE_CLOSED`. Rejected uploads retain sanitized error codes.
The authentication and admission contract in `phase8_backend_integration_contract.md`
continues to apply. No user-controlled model, checkpoint, threshold, TRAIN ID or
provenance claim is accepted by a public prediction endpoint.

## Internal request

`InternalTrainBackend` is constructed once per local worker from the authenticated
model/input manifest and exact decoder binaries. The frozen E3 and A2 models,
audio/video encoders and validated TRAIN references are loaded once. Every
request retains complete file SHA-256 and live model-integrity checks.

`predict_internal(sample_id, content, filename, inference_mode=False, include_a2=True)`
accepts only one of the five explicitly permitted TRAIN IDs and its exact
SHA-bound MP4 bytes. The ID is an offline audit capability, not a future public
multipart field. A single nonblocking lock rejects overlapping requests with
`RESOURCE_BUSY`; there is no unbounded internal queue. This worker must have the
audited CPU/thread configuration set before construction/use (the measured
profile is four intra-op and one inter-op thread).

Admission reuses Phase 8 filename, byte limit, container, codec, stream count and
duration checks. The internal subset accepts MP4 only. It additionally verifies
zero stream origins, bounded dimensions/frame rate and the approved source hash.
The byte limit remains 50 MiB; timing tolerance remains the existing 50 ms.

Audio is decoded once to mono 16 kHz PCM-s16 WAV, using the producer's explicit
10-second crop; decoded output is bounded and must contain exactly 160,000 finite
float32 samples in [-1,1]. No padding/length repair is allowed. Video is sampled
once using the documented producer convention with RGB/INTER_LINEAR resizing.
Missing frames, failed seeks, duplicates caused by clamping, or FPS disagreement
are rejected. Actual PTS are retained alongside intended seconds.

These clips share a source container and observed zero origins. At fractional
FPS, historical `int(t*fps)` selects images before some intended bin starts.
This internal convention is **not** promoted to a verified Phase 6 shared-clock
manifest. `phase6_shared_clock_accepted` is always false. The new path uses the
existing `EncoderOutput` geometry/stamps and frozen model schemas, but cannot
issue `TRAIN_EXPORT_PARITY_MATCHED` or copy historical export provenance onto
new features. A separately reviewed decoder/timing policy remains required.

## Response

Call `response_for_review` before serializing a result. It removes private audit
tensors and validates the internal response contract. The schema requires:

| Field | Meaning |
| --- | --- |
| `schema` | Exactly `aval.internal-train-inference.v1` |
| `request_id` | Opaque generated identifier on completed experiments |
| `status` | `EXPERIMENTAL_INTERNAL_TRAIN_ONLY` on successful local execution; otherwise `REJECTED` or `MODEL_NOT_READY` |
| `predictions` | Mandatory E3 and optional A2 for completed experiments; null on failure |
| `production_predictions_enabled` | Always false |
| `public_readiness` | `MODEL_NOT_READY`, inference disabled, null public predictions |
| `event_predictions`, `temporal_predictions` | Always null |
| `timings` | Wall and process CPU seconds by stage; total request wall time |
| `diagnostics` | Approved source hash/identity, immutable encoder stamps, clocks/achieved PTS, and execution profile |
| `error_code` | Sanitized uppercase code on failure; no traceback, path, bytes or exception details |

Each E3/A2 result contains the actual float32 `logit`, float64 sigmoid `score`,
`predicted_target`, `decision`, `score_semantics`, frozen `threshold`, and original
checkpoint/model hashes. `decision` is `original` iff `score >= threshold`, else
`substituted`; target is respectively 1 or 0. Thresholds remain:

- E3: `0.009579099009671126`.
- A2: `0.0026640388978268056`.

`score_semantics` is exactly **uncalibrated sigmoid score for original pairing
provenance**. The score is neither a calibrated probability nor evidence of
physical sound-source causality. No event-class or per-second presence output is
exposed. The classifier operations retained for numerical fidelity do not grant
validation of such UI outputs.

The response validator rejects changed model/threshold bindings, missing E3,
nonfinite scores, changed score semantics, inconsistent binary decisions,
readiness promotion, and unvalidated event/temporal output. Ordinary frontend
requests continue to use the existing public admission response, not this internal
success status.

The response's `timings.total` covers the prediction core through cleanup. The
profiler additionally measures the complete Python call boundary (including
final diagnostics/audit-copy construction and lock release) and separately times
response validation plus JSON encoding. Neither includes client network upload.

## Timing and lifecycle

Initialization records manifest authentication, authentic audio/video loading,
fusion/TRAIN-reference loading and total startup. Request stages are admission,
integrity before/after execution, audio decode, normalized filterbank, BEATs/LoRA,
video decode/RGB/resize, video normalization, ResNet, feature alignment, E3 and
optional A2. Decoder/probe deadlines and output caps are preserved. The audio
decode file cap is 322,048 bytes; ffprobe/FFmpeg combined diagnostic output uses
the existing 256 KiB cap and eight-second deadline. Docker supplies process/RAM
quotas; arbitrary unwhitelisted input never reaches OpenCV.

Every request uses private temporary files; success, timeout and model-error paths
release them. OpenCV capture, timing hooks, subprocess pipes and the request lock
are released. Models stay resident intentionally until worker exit. The profiler
checks actual `torch.load` calls and model identities through repeated requests.

The optional native Docker volume is seeded from read-only source artifacts,
with every original and copied SHA-256 checked. It is mounted read-only during
inference and removed after the audit. This preserves all integrity checks while
avoiding Windows bind-mount I/O; checkpoint bytes never enter image layers.
`inference_mode()` is measured as an explicit experiment. `no_grad` remains the
default because the observed speed difference was negligible.

Future production activation still requires independent raw-to-export parity,
full TRAIN coverage, reviewed numerical/timing policies, secure shared-clock
decoding and release review. A finite internal score cannot satisfy those gates.

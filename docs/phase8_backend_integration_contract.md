# Phase 8 Backend Integration Contract

## Scope and current state

The agreed integration branch is `main`. At this review, `main` is at merge commit
`2866995` and includes the Phase 8 foundation merge (`c62967f`, PR #20), plus the
Phase 6 and Phase 7 dependencies. The existing Gradio surface is an authenticated
upload-admission/readiness UI. It is not a raw-media prediction endpoint.

This document defines the backend boundary to share with the frontend team. The
future prediction response below is a contract proposal; it is not implemented or
enabled. Until exact encoder artifacts, the decoder, and TRAIN raw-to-export parity
are verified, an accepted upload must still return `MODEL_NOT_READY` with no
predictions. No source changes here load weights or open validation, TEST, or
challenge samples.

## Phase 6 encoder contracts

`fusion.encoder_interfaces.EncoderSpec` requires explicit producer identity,
canonical actual checkpoint configuration, exact artifact roles and SHA-256
digests, a fixed preprocessing signature, and pinned dependency versions. Unknown
hashes are rejected. Artifacts are hashed before safe loading and rechecked after
loading; tensor names, shapes, dtypes, finiteness, and strict state coverage must
match the reconstructed architecture. Loading is CPU-only, `weights_only=True`,
eval mode, no gradients, and no fallback to randomly initialized or alternate
weights. A loaded encoder is fingerprinted and checked for later mutation.

### Audio: `beats_lora_r8_v1 / beats_iter3`

The required files (using the Phase 8 `/models` defaults) are:

| Role | Default path | Required schema |
|---|---|---|
| Self-supervised base | `/models/audio/BEATs_iter3.pt` | Dictionary with original `cfg` and complete `model` state. Configuration must reconstruct non-finetuned BEATs iter3: 12 transformer layers, 768-dimensional embeddings, patch size 16. |
| Trained adapter/head | `/models/audio/beats_lora_r8_v1/best.pt` | Dictionary with `config` exactly `{kind: beats_lora, encoder: iter3, r: 8, alpha: 16.0, emb_dim: 256, features: beats_iter3}` and `model` containing every trained LoRA q/v and embedding-head parameter, with exact names/shapes/dtypes. The frozen base state is loaded from the separate base file. |
| Frozen normalization | `/models/audio/beats_iter3/norm_stats.json` | JSON with `dim: 768`, `encoder: "BEATs iter3"`, finite 768-element `mean` and finite, strictly positive 768-element `std`. |

The encoder output is finite CPU float32 `(B,10,256)`. The old 29-class branch
classifier is not the output used by the E3/A2 correspondence models.

### Video: `resnet18_v1`

`/models/vision/resnet18_v1/best.pt` is a dictionary with saved `config` and full
`model` state. The saved configuration must identify `backbone: resnet18` and
`emb_dim: 512`; the config is bound to the actual artifact, not supplied as a
default. Reconstruction uses `pretrained=False` and strict state loading. The
checkpoint includes the trained branch classifier, while the E3/A2 feature path
uses the 512-dimensional ResNet-18 representation with the 29-class head excluded.
The encoder output is finite CPU float32 `(B,10,512)`.

The production environment must match the deployment pins in
`deployment/runtime.py` and `requirements-deployment.txt`: Torch 2.11.0+cpu,
TorchVision 0.26.0+cpu, TorchAudio 2.11.0+cpu, NumPy 2.2.6, Pandas 2.3.3,
SoundFile 0.13.1, tqdm 4.67.1, OpenCV headless 4.12.0.88, Pillow 12.0.0, and
Gradio 6.30.0. The encoder-specific dependency lists are independently pinned by
their `EncoderSpec`. A Docker build and actual artifact load are still required to
establish runtime compatibility.

## Canonical decode and temporal alignment

The backend boundary into the existing encoders is `RawMediaInput`, not a file
path. For one clip it requires:

- Mono finite float32 audio with exactly 160,000 samples, values in `[-1,1]`, at
  16,000 Hz, exactly 10 seconds, and local start time 0. No padding or trimming.
- Ten RGB uint8 frames shaped `(10,224,224,3)`, exactly 10 seconds, with frame
  timestamps `(0,1,...,9)` seconds and local start time 0.
- Explicit audio and video validity masks, each all true. Missing audio or frames
  cannot be repaired by reusing another frame, inserting silence, or masking bins.
- A decoder manifest bound to the exact waveform and frame bytes, decoder
  signature, media durations/origins, timestamp list, sample IDs, and both masks.

The decoder must preserve one shared stream clock and verify packet/frame PTS and
audio timing before producing these tensors. It must explicitly downmix/resample
to the required audio tensor and select one actual video frame per second at the
declared timestamps, convert BGR to RGB, and resize to 224x224 with OpenCV
`INTER_LINEAR`. Stream order, presentation timestamps, duration, and source hash
must be checked. Fail on a missing frame, decode error, gap, unsupported variable
frame timing, or stream-clock mismatch. Do not copy the historical extractor's
last-frame/black-frame fallback or its `int(second * fps)` approximation into the
strict deployment path.

BEATs then applies its 128-bin filterbank (25 ms window, 10 ms hop, scale 32768,
fixed normalization `(fbank - 15.41663) / (2 * 6.55582)`), the trained q/v LoRA
(rank 8, alpha 16), averages its eight frequency patches and pools time centres
into ten one-second bins, applies the frozen 768D TRAIN mean/std, and maps through
the trained 768-to-256 GELU embedding head. Keep this execution in float32 with
autocast disabled. ResNet-18 receives NCHW float32 frames divided by 255, followed
by ImageNet channel mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`;
export-time augmentation is disabled. Both outputs retain the original ten bins.

`DecodedTensorDecoder` currently accepts only already-decoded `.npy` waveform and
frame arrays plus a hashed `gc2-decoded-ten-bin-v1` manifest. It does not decode
compressed uploads. The deployment FFprobe admission check checks container,
codecs, stream count, and duration; it does not establish PTS parity or decode
frames. Connecting that validator to a real, bounded decoder is a remaining gate.

There are producer/replay details to resolve before declaring preprocessing
identical. `fusion.encoder_interfaces.AUDIO_PREPROCESSING` currently says
`no_pad_trim`, while the historical audio producer calls `audio.features.load_wav`,
which pads short decoded WAVs with silence and trims longer ones to 160,000
samples (after requiring 16 kHz). `vision.extract_frames` samples with
`int(second * fps)` and repeats the last valid frame or a black frame after decode
failure. The strict raw contract disallows those repairs and requires actual
validity. Therefore, first prove the authenticated TRAIN source clips used no
padding/trimming/frame fallback, or reconcile the producer signature and replay
policy through an independently reviewed change. Do not weaken production
rejection behavior just to force parity. The exact export comparison must expose
any mismatch.

## Authenticated TRAIN parity evidence

Parity must use the authenticated Phase 7 primary TRAIN pair manifest and the
checkpoint-bound producer exports, never labels or records supplied by an upload
caller. The Phase 7 audit records **6,678 unique primary pair identities**: 3,339
original-pair targets (`1`) and 3,339 substituted-pair targets (`0`). The
parity harness must consume every primary TRAIN identity and retain, per pair,
`audio_sample_id`, `visual_sample_id`, `audio_row`, `visual_row`, source clip IDs,
and `original_pair_target` from the authenticated manifest. The original pair
target is correspondence provenance, not physical sound-source causality.

For each modality, resolve source IDs and row indices against the authenticated
producer `train_ids.csv` and TRAIN export archive. Encode each distinct modality
source from its permitted raw TRAIN media, then map results back to every pair
row without changing order. Check identity, source row, and target alignment
before comparing values. The feature reference is the authenticated
`train_emb.npy`; its labels/IDs and producer declarations are separately bound by
the existing checkpoint metadata and Phase 7 freeze. Do not use event labels to
create encoder validity or to alter embeddings.

The current `TrainParityFusion` bridge requires each emitted representation,
cast to float16, to be exactly array-equal to the authenticated TRAIN export for
that identity. Its current gate requires both modalities and all ten bins valid.
This is an exact compatibility gate, not an assumed numerical tolerance. If
producer/decoder evidence supports a different deterministic tolerance, propose
and independently review that rule before changing the gate. Parity work compares
encoder features; it does not create new evaluation metrics or authorize
validation, TEST, or challenge scoring. Preserve the existing E3/A2 checkpoints,
threshold files, feature exports, and archived evaluation reports byte-for-byte.

## Frontend-facing request and response

### Current behavior

The current authenticated Gradio callback accepts one media file. It returns a
JSON object with `status`, `predictions`, and, on failure, `error_code`; readiness
issues may also be present. A syntactically supported upload currently returns:

```json
{
  "status": "MODEL_NOT_READY",
  "predictions": null,
  "error_code": "RAW_INFERENCE_GATE_CLOSED",
  "issues": [
    {"category": "raw_media_decoder", "code": "NOT_IMPLEMENTED"},
    {"category": "raw_to_train_export_parity", "code": "NOT_VERIFIED"}
  ]
}
```

The two `issues` shown are mandatory raw-inference gates; the real readiness
response may include additional missing-artifact or dependency issues.

Malformed, unsupported, or oversized media returns `status: "REJECTED"`,
`predictions: null`, and a sanitized `error_code`. Authentication remains at the
Gradio launch boundary. This current callback behavior is the only supported
frontend integration point today; callers must display not-ready/rejected state
and must never manufacture a prediction from an error.

### Proposed post-gate contract (not implemented)

Request: one authenticated multipart request containing a single 10-second
audio-video file in the accepted media profile. Do not accept a caller-supplied
model, threshold, split, label, sample ID, source row, or claimed model
provenance. Generate a server-side opaque request ID. The backend always invokes
both frozen correspondence heads so the frontend cannot select or redefine a
checkpoint/threshold.

Successful response after every gate passes:

```json
{
  "schema": "aval.backend-inference.v1",
  "request_id": "server-generated-opaque-id",
  "status": "PREDICTED",
  "predictions": {
    "E3": {
      "decision": "original",
      "score": 0.91,
      "score_semantics": "uncalibrated sigmoid score for original pairing provenance",
      "threshold": 0.009579099009671126,
      "checkpoint_sha256": "e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925",
      "model_sha256": "d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025"
    },
    "A2": {
      "decision": "substituted",
      "score": 0.001,
      "score_semantics": "uncalibrated sigmoid score for original pairing provenance",
      "threshold": 0.0026640388978268056,
      "checkpoint_sha256": "0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12",
      "model_sha256": "1ef5985f9de3e7b9f2994b076ebd16c1b0cd7d4922c0842d12d301aa202c2b8e"
    }
  },
  "interpretation": "pairing-provenance correspondence; not physical-source causality",
  "temporal_predictions": null,
  "event_predictions": null
}
```

The example numbers are illustrative, not observed predictions. Decision is
`original` iff `score >= threshold`, otherwise `substituted`; positive class is
1 = original pairing provenance. Scores are uncalibrated and must not be labeled
probabilities. The backend, not the frontend, owns frozen threshold/model
bindings. A blocked or invalid request uses the current `MODEL_NOT_READY` or
`REJECTED` envelope with `predictions: null`; the future public wrapper must
sanitize internal exception details and never return paths, stack traces, or
uploaded bytes. Event and temporal fields remain null until separate genuine
trained heads exist.

## Release gates and handoff

1. Recover the exact four encoder inputs above from their approved protected
   source; record independently verified digests and producer/config provenance.
2. Verify dependency/runtime versions and load both encoders with strict,
   `weights_only=True` state validation; confirm CPU eval/frozen behavior,
   deterministic output, and `(B,10,256)` / `(B,10,512)` shapes.
3. Implement and audit compressed-media decode plus manifest generation; prove
   shared-clock timing, frame availability, cleanup, limits, and decoder identity.
4. Run complete authenticated TRAIN encoder replay and exact float16 export parity
   for every primary pair identity and original source row. Any mismatch blocks
   raw inference.
5. Run the existing `MediaFusionPipeline` and confirm both E3/A2 outputs retain
   their fixed checkpoint, model fingerprint, and operating point bindings.
6. Add live Docker/Gradio integration tests and keep the app at `MODEL_NOT_READY`
   until all prior gates pass. No train/validation/TEST/challenge scoring or
   threshold fitting is part of this integration preparation.

The E3 and A2 frozen thresholds remain `0.009579099009671126` and
`0.0026640388978268056`. No model, threshold, historical feature export, manifest,
or evaluation report is to be regenerated or rewritten during this work.

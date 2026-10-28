# Phase 8 Stage 1 — authentic audio acquisition and five-clip TRAIN probe

2026-10-27, Asia/Singapore. Branch: `feature/phase8-backend-stage1`, based on
`eff4e2fe6ed33650c07fb8f719fae9a665a7faf1`. No commit, push, training, deployment,
held-out scoring, threshold tuning or production raw inference occurred.

**Authentic audio access and Python 3.11 CPU execution are READY FOR TRAIN PARITY
VALIDATION. Exact historical audio parity remains BLOCKED:** all five permitted
clips fail the unchanged float16 array-equality condition. No numerical tolerance
has been adopted. The earlier Stage 1 report is a historical pre-acquisition
snapshot; this report supersedes its audio-access and execution status only.

## Acquisition and integrity

Source: [Ghanshyam's audio inference bundle](https://drive.google.com/drive/folders/185BtXY93QUt32gd1YVGGQ3fCeEGLXKAI),
with [five explicitly permitted TRAIN clips](https://drive.google.com/drive/folders/1Pml4ByzllIT2yEsYIM0f8t0Wznn9uP_b).
All 17 original files, 372,388,217 bytes total, are retained unchanged under
Git-ignored `_local/audio_handoff/ghanshyam_20261027/`. The Drive connector's
256 MiB limit prevented downloading the 361 MB base checkpoint through that
connector; an authorized direct Google Drive download of the same file ID was
used. Size and producer checksum matched. No alternate checkpoint was fetched.

| Original file | SHA-256 |
| --- | --- |
| `encoder.pt` | `8d1b234032a9ccff353612dc6c20982346dc2968b205b79d97303eb5e77bfb34` |
| `weights.pt` | `d4a0f78403caafde2c3c39d961c9819fbae8caebaaef4b824034c678da860692` |
| `norm_stats.json` | `c0ac268380479149d759f7ba605b881d415c45a83599875d30ef492ef85295e1` |
| `model_card.json` | `3b98967d720e7fad3de3af37f57487ebd49cf48dc357cef8dbde394a86470bfc` |

Every file's ID, source URL, byte count and SHA-256 is recorded in
`_local/phase8_audio_stage1/acquisition.json` and `SHA256SUMS.txt`. All five
producer-declared payload hashes (base, trained weights, normalization, class
names, environment) match the original model card. Trust comes from the explicitly
approved source plus these byte bindings; the bundle is not digitally signed.
Files without producer-declared hashes were bound to their acquired bytes and
checked against frozen TRAIN evidence where possible.

## Safe loading and actual architecture

The current repository includes PR #15's `audio.infer.AudioPredictor`,
`audio.beats_lora.BeatsLoRA`, and `audio.beats_extract`. The producer constructor
does not enforce the card hashes and the original base loader explicitly uses
`weights_only=False`. Those historical files were not modified or executed as
unsafe loaders.

New `deployment/audio_bundle.py:VerifiedAudioPredictor` verifies the externally
bound model-card hash and all payload hashes, uses restricted checkpoint loads,
and delegates complete model/state verification to the existing Phase 6 loader.
It then executes the original inherited `AudioPredictor.predict_waveform` method
with the strictly loaded authentic model. It replaces loader plumbing only; the
filterbank, LoRA, pooling, normalization, GELU projection and classifier numerical
operations remain the producer's existing implementations. No permissive pickle
fallback, replacement weights or downloads are available in this adapter.

Verified actual configuration/state:

- Self-supervised BEATs iter3: 12 layers, 768 hidden dimensions, patch size 16.
  The entire base state loads strictly before LoRA injection.
- Saved adapter config exactly `{kind: beats_lora, encoder: iter3, r: 8,
  alpha: 16.0, emb_dim: 256, features: beats_iter3}`.
- 24 q/v LoRA modules, rank 8, scale 2; all 48 LoRA tensors and four trained head
  tensors are required and strictly covered (52 trained tensors total).
- Embedding head `Dropout(0.1) → Linear(768,256) → GELU`; classifier
  `Dropout(0.3) → Linear(256,29)`. Weights are frozen, eval mode, no gradients,
  CPU float32 with autocast disabled.
- Original normalization contains only 768-element mean/std arrays. A separate
  derived `normalization.phase6.json` adds identity/geometry and source digests,
  justified by the actual base/trained configurations. Mean/std values are
  preserved verbatim with no recomputation or added epsilon. Original bytes are
  never rewritten. Derived SHA-256 is
  `3b9e260b5c09c4356b4a607c1dd49b9699251a2e425761c1c1fdca394d97a25f`.

## TRAIN identity and reference authentication

Both frozen E3/A2 checkpoints and their metadata/operating-point hashes were
verified. `load_verified_train_exports` authenticated the original audio/vision
TRAIN exports against each checkpoint. All five supplied `emb.npy` and
`probs.npy` rows match the frozen audio TRAIN exports **byte-for-byte**.
The supplied vocabulary also matches the authenticated audio class map.

Each `samples.csv` identity, row index, clip ID, category and annotation interval
matches its original TRAIN metadata. Each matching original audio/video pair is
present in the authenticated primary TRAIN pair manifest, SHA-256
`52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f`.
Annotation fields were used only to check identity, never as encoder inputs,
validity masks or score features. No TEST/validation/challenge arrays were scored.

## Genuine CPU measurements

All five MP4s were decoded using the producer command's `-vn -ac 1 -ar 16000 -t 10`
WAV path. Every decoded waveform already had exactly 160,000 mono float32 samples
at 16 kHz; no waveform padding or subsequent length repair was needed. MP4, decoded
WAV and canonical waveform hashes are retained. This audio-only study does not
establish a shared audio/video media clock or a verified production decoder.

Errors below compare genuine float32 output to the original float16 reference
cast to float64 for measurement. Exact float16 mismatch counts additionally test
the unchanged historical representation-equality condition.

| TRAIN sample ID | Audio row | Embedding max abs | MAE | RMSE | Float16 mismatches / 2,560 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `0h8HxZaX9wQ__26` | 1305 | 0.00273895264 | 0.000129336949 | 0.000284256720 | 99 |
| `NGeniwxDzJ0__21` | 3000 | 0.00171995163 | 0.000138722204 | 0.000273996494 | 79 |
| `22DapbNwnUY__12` | 1167 | 0.00192832947 | 0.000165712969 | 0.000332505889 | 96 |
| `6VRcINAY7KE__07` | 2628 | 0.00372982025 | 0.000149551811 | 0.000333144630 | 166 |
| `c---zaDCTaE__08` | 0 | 0.00381946564 | 0.000213741951 | 0.000463414713 | 132 |

| Aggregate over all five clips | Maximum absolute error | MAE | RMSE | Float16 mismatches |
| --- | ---: | ---: | ---: | ---: |
| Embeddings | 0.0038194656372070312 | 0.00015941317694585067 | 0.0003441437691373159 | 572 / 12,800 |
| Probabilities | 0.00024384260177612305 | 0.000004879756329137833 | 0.00002695813003067708 | 2 / 1,450 |

Per-second argmax labels match on all 50 bins. This is agreement with the frozen
outputs, not an accuracy benchmark or evidence that the parity gate passes.
All five embeddings fail exact equality after the float16 roundtrip. The producer
CSV also reports nonzero live/export differences; its approximate-rounding prose
does not authorize a tolerance. Differences cannot be attributed solely to a
specific library/device/decoder change from this one experiment.

## Frozen fusion sensitivity, separate from raw admission

For each authenticated original TRAIN pair, frozen visual features were held at
their independently verified export values. The unchanged, hash-verified frozen
models were evaluated with reference audio and reconstructed audio in the same
CPU runtime. This is an explicitly offline feature-sensitivity experiment, not a
`TrainParityFusion` raw admission or production prediction. No new audio was given
copied historical export provenance. No raw-video parity is claimed.

| Model | Maximum absolute sigmoid-score difference | Mean absolute score difference | Maximum absolute logit difference |
| --- | ---: | ---: | ---: |
| E3 | 2.830885525995086e-9 | 7.3978976278255e-10 | 0.000347137451171875 |
| A2 | 2.4773860574089923e-7 | 5.695844222941559e-8 | 0.0007572174072265625 |

These five original-pair scores are near sigmoid saturation; small score changes
do not justify feature compatibility or a tolerance. E3/A2 thresholds remain
`0.009579099009671126` / `0.0026640388978268056`. Full-precision per-pair scores,
logit differences, identities and source hashes are in `measurements.json`.

## Python 3.11 CPU and Docker boundary

Executed candidate: Python `3.11.16`, Torch/TorchAudio `2.11.0+cpu`, NumPy `2.2.6`,
Pandas `2.3.3`, SoundFile `0.13.1`; the existing direct deployment pins were retained.
The local `imageio-ffmpeg==0.6.0` executable reports FFmpeg 7.1 and is hash-recorded.
Producer-reported environment: Python `3.13.16`, Torch/TorchAudio `2.11.0+cu130`,
NumPy `2.1.3`, Pandas `2.2.3`, FFmpeg 6.1.1, CUDA/Tesla T4. These are different
execution environments; the original producer freeze/environment files are retained.

Docker is unavailable on this host, including its usual Windows installation path.
No image build or Linux/container execution was performed. The Dockerfile pins
Python 3.11.17 and FFmpeg 5.1.9; compatibility with that exact Linux environment
therefore remains unverified. The existing Dockerfile already copies the new
deployment module. `deployment/audio-bundle-stage1.env.example` maps original
bundle artifacts and the separately derived normalization into `/models` with
their actual hashes. Mount sources read-only; do not bake artifacts into Git/image.
The two hash-bound encoders both loaded through the existing Phase 6 CPU adapter;
the availability diagnostic still returns `MODEL_NOT_READY` with null predictions.
It reports `NOT_VERIFIED` for the code-owned admission gate; the five-clip study
separately establishes that its exact audio equality condition fails. The diagnostic
also reports `ffmpeg`/`ffprobe` absent from the host PATH. The standalone probe
used the explicit imageio-ffmpeg executable; that does not supply `ffprobe` or
establish readiness of the application's standard decoder dependencies.

The genuine predictor does not import `audio.evaluate` or its test-only
scikit-learn dependency. `predict_video` is deliberately unavailable on the new
adapter; audited offline waveform diagnostics do not open a compressed-media
serving endpoint. The existing decoder and TRAIN-parity readiness gates are unchanged.

## Evidence, validation and remaining work

Local audit directory: `_local/phase8_audio_stage1/`.

- `acquisition.json`, `drive_inventory.json`, `SHA256SUMS.txt`: original source/byte evidence.
- `run_probe.py`, `probe.log`, `measurements.json`: reproducible five-clip measurement path/results.
- `live_train_embeddings.npy`, `live_train_probabilities.npy`, `decoded/`: genuine outputs and waveforms.
- `derived/normalization.phase6.json`: explicitly derived schema bridge with source digests.
- `focused-final.xml` / `focused-final.log`, `regression.xml` / `regression.log`: test evidence.
- `readiness.json`: actual per-modality loads with overall readiness closed.

The earlier 623-file historical preservation baseline is unchanged, including
frozen E3/A2 state/thresholds, exports, preprocessing contracts and archived evidence.
Existing predictor/frontend source was not edited. Added files for this continuation:
`deployment/audio_bundle.py`, `deployment/audio-bundle-stage1.env.example`,
`tests/test_phase8_audio_bundle.py`, and this report. All downloads/results are ignored.

Focused tests: **39 passed, zero skips/failures/errors**, including real strict
audio/video loads. Complete regression: **740 passed, 2 failed, 11 setup errors,
2 unavailable-CUDA skips, and 1,346 passing subtests**. All failures/errors are
in existing Phase 7 plotting tests because Windows Application Control blocks
Matplotlib's `_c_internal_utils` DLL. The security policy was not changed. The
only additional warning is PyTorch's weight-norm deprecation warning.

The focused tests cover original bundle hashes, restricted loading, normalization
identity, complete base/LoRA/head architecture, malformed artifacts, and canonical
waveform rejection. Their synthetic input shape checks are separate from the five
genuine clip results.

Remaining blockers: exact audio float16 parity fails; only five of the required
TRAIN sources are supplied; full authenticated TRAIN replay, raw-video parity,
shared-clock decoder verification and actual Docker execution remain outstanding.
An independently reviewed compatibility study would be required before any policy
change. No such policy change or tolerance selection occurred here.

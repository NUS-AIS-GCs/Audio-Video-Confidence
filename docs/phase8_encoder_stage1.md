# Phase 8 real encoder integration — Stage 1

Audit date: 2026-10-27, Asia/Singapore. **FIX REQUIRED** before real paired-media
Stage 2 execution. Backend preparation and authentic video loading are implemented;
genuine audio execution and both raw-to-TRAIN parity studies remain blocked.
Raw upload predictions remain disabled.

## Repository and preservation

Repository: `NUS-AIS-GCs/Audio-Video-Confidence`. Fetched `origin/main` at
`eff4e2fe6ed33650c07fb8f719fae9a665a7faf1`, then created
`feature/phase8-backend-stage1` from that exact revision. This includes the newly
merged `audio/infer.py:AudioPredictor` and `vision/infer.py:VisionPredictor`.
No commit, push, deployment, training, threshold tuning or frontend change was made.

Pre-existing untracked `docs/video_export_handoff.md` and
`requirements-fusion-training.txt` were preserved. A before/after byte-hash snapshot
covers 623 pre-existing files (354,035,762 bytes): tracked source/contracts,
historical artifacts, exports, dataset evidence, downloaded video artifacts and
handoff evidence. The final preservation result is recorded under
`_local/phase8_stage1/preservation_after.json`. No existing file is intentionally
modified by this stage; all proposed backend work is in new files.

Separately, all 28 files across the four historical Phase 7 smoke/audit report
directories match the saved Phase 7 smoke SHA-256 anchors. Those directories
required an approved read-only check outside the sandbox. No archive was rewritten.

The frozen E3/A2 loader and `TrainParityFusion` are unchanged. In particular:

- E3 checkpoint SHA-256 remains
  `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925`,
  threshold `0.009579099009671126`.
- A2 checkpoint SHA-256 remains
  `0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12`,
  threshold `0.0026640388978268056`.
- Exact float16 array equality, independent modality provenance, all-valid ten-bin
  inputs and authenticated TRAIN identities remain required. The complete release
  study must cover the contract's 6,678 primary TRAIN pair identities. A few clips
  or a shape smoke test cannot satisfy that release gate.

## Audio execution and TRAIN measurements

**BLOCKED — zero genuine clips executed.** The owner-reported location is
`Google Drive → AValidate/inference/audio/`; no direct accessible link or verified
local path is available. Connected Drive discovery found `audio_final`, which
contains precomputed arrays/IDs/notes, not the executable encoder bundle. Local
artifact discovery likewise found no authentic BEATs base, LoRA/head, normalization
bundle or permitted TRAIN media. The user confirmed that Stage 1 should continue
without substituting these assets.

Required authentic producer bundle files are `encoder.pt`, `weights.pt`,
`norm_stats.json`, `model_card.json`, `class_names.json`, `environment.json`, and
the supplied `parity_samples/` clips with `samples.csv`, `emb.npy`, `probs.npy`.
The owner source names these files; their availability and hashes are not assumed.
Producer-controlled hashes/source identity and actual saved configuration must be
verified on receipt. A filename or a locally computed digest alone does not prove
producer authenticity or checkpoint-to-export lineage.

The Phase 6 audio boundary expects roles `base`, `trained`, `normalization`, the
exact saved iter3/r8/alpha16/256D configuration, and normalization JSON declaring
768D BEATs iter3 statistics. The new producer bundle exporter currently writes
only mean/std in its normalization JSON. Any mapping of that bundle into the
strict deployment layout must preserve original bytes and establish the missing
metadata from authenticated producer evidence; no schema or signature was relaxed.

| Measurement | Result |
| --- | --- |
| Genuine AudioPredictor execution | BLOCKED: bundle inaccessible |
| Authenticated raw TRAIN clips measured | 0 |
| Audio maximum absolute embedding error | Not measured (`null`), not zero |
| Audio MAE / RMSE | Not measured (`null`) |
| Exact float16 TRAIN equality | NOT VERIFIED |
| Frozen E3 score differences | Not measured: no trustworthy reconstructed audio/aligned pair |
| Frozen A2 score differences | Not measured: same blocker |

`deployment.encoder_diagnostics.embedding_differences` prepares descriptive maximum
absolute error, MAE, RMSE and exact float16-difference counts. It validates shape,
finiteness and export representability but does not authenticate split/identity,
select a tolerance, invoke fusion, or replace the historical parity gate. Its unit
fixtures are numerical tests, not observed audio results. No validation, TEST or
challenge features were scored or used to choose a tolerance.

## Authentic video compatibility

Exact source: [Swaraj's shared checkpoint](https://drive.google.com/file/d/122b4V9DkDgd3KqmJjbS8OCXzz8vujjdc/view).
Local Git-ignored path:
`_local/drive/artifacts/vision_baseline/resnet18_v1/best.pt`.

- Bytes: 44,839,691.
- Recorded and reverified SHA-256:
  `6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec`.
- Restricted `torch.load(weights_only=True, map_location='cpu')`; actual config
  `{"emb_dim":512,"backbone":"resnet18"}`, epoch 9.
- Full strict state loading succeeded, including all 122 state tensors and the
  trained 29-class classifier. No ImageNet download or alternate weights were used.
- Reconstructed ResNet-18 has `fc=Identity`, Identity 512D projection, and
  `Dropout(0.3) → Linear(512,29)`. Dropout is an architecture setting, not a tensor
  stored in the checkpoint. The classifier is excluded from fusion embeddings.
- Actual candidate CPU execution produced finite float32 `(1,10,512)` embeddings
  and `(1,10,29)` logits. Repeated embeddings were identical within the same run;
  eval/frozen/no-grad behavior and caller autocast restoration were checked.
- Inputs for that test were explicitly synthetic diagnostic pixels. They establish
  architecture/output compatibility only, not raw decoding or TRAIN export parity.

All 12 earlier downloaded champion export files match the frozen local copies by
SHA-256. That identity does not establish which executable checkpoint produced
them. Permitted raw TRAIN clips, export runtime/decoder records and complete
checkpoint-to-export provenance remain unavailable. **Video TRAIN parity BLOCKED.**

## Backend and Docker preparation

`deployment/encoders.py` independently constructs each modality as the existing
Phase 6 `FrozenEncoder`. Audio absence does not prevent video inspection. It checks
trusted configured hashes and root-contained paths, safely reads actual checkpoint
configurations, enforces the existing dependency pins and strict state loader, and
checks real dependency imports so missing DLLs/binary incompatibility fail closed.
Failures are sanitized codes, with no fallback model.

`deployment/encoder_diagnostics.py` adds an operator-only local CLI:

```text
python -m deployment.encoder_diagnostics --load --output /tmp/encoder-stage1.json
```

It checks each modality independently, reports actual deployment readiness and
always returns `MODEL_NOT_READY`, `predictions: null`, `inference_enabled: false`.
Exit status is 2. A successful checkpoint load never promotes prediction readiness.
The frontend-facing callback, authentication, decoder and parity gates are unchanged.

Use `deployment/encoder-stage1.env.example` for the existing `/models` layout.
Only the acquired vision hash is supplied; unknown audio hashes remain empty.
The current Dockerfile already copies the whole `deployment/` directory, so it
includes these new modules without changing its application entrypoint. Artifacts
stay outside the image and should be mounted read-only. After a separately verified
image build, an operator can run diagnostics without starting the UI:

```text
docker run --rm --network none --env-file deployment/encoder-stage1.env.example --mount type=bind,src=<absolute-approved-model-root>,dst=/models,readonly --entrypoint python <locally-built-image> -m deployment.encoder_diagnostics --load
```

The isolated Windows CPU test environment matches every existing direct deployment
pin: Torch `2.11.0+cpu`, TorchVision `0.26.0+cpu`, TorchAudio `2.11.0+cpu`, NumPy
`2.2.6`, Pandas `2.3.3`, SoundFile `0.13.1`, tqdm `4.67.1`, OpenCV headless
`4.12.0.88`, Pillow `12.0.0`, Gradio `6.30.0`. These are candidate execution
versions, not inferred historical export versions. Dependencies were installed in
`_local/phase8_stage1/venv`; the pre-existing `.venv` was not modified. PyTorch
packages came from its official CPU index, other packages from PyPI. The observed
test environment also includes pytest and scikit-learn (needed by the existing
AudioPredictor tests, which import `audio.evaluate`). The strict Phase 6 audio
adapter does not import that evaluation module.

Actual local diagnostic result: video `LOADED_PARITY_UNVERIFIED`, audio
`MISSING_ARTIFACT`, overall `MODEL_NOT_READY`. `ffmpeg`, `ffprobe` and Docker are
not installed/on PATH on this host. No image build or Linux/container execution
was possible; Windows success is not Docker execution evidence. The compressed
decoder and both raw TRAIN parity gates remain unresolved.

## Validation and files

Focused result: **23 passed, zero skips/failures/errors**, including all 18 new
Stage 1 tests and five existing audio/video producer tests, in 15.54 seconds.
The authentic video tests passed rather than being skipped.

Complete `pytest tests` run: **724 passed, 2 failed, 11 setup errors, 2 skipped**,
plus **1,346 passing subtests**, in 115.87 seconds. Both skips are CUDA-only tests
on this CPU host. All failures/errors are in existing Phase 7 plotting tests:
Windows Application Control rejected Matplotlib's `_c_internal_utils` DLL.
The suite was run in full; it is **not green**. No test was removed or mocked to
hide that host execution failure, and no security policy was changed. Earlier
runs exposed missing test-only dependencies, which were installed locally before
this final complete run. `regression-final.log` retains the actual failure evidence.

The tests cover authentic checkpoint loading and shape execution (explicitly
skipped with a reason when that local artifact/runtime is unavailable), invalid
architecture/head/state, missing or changed hashes, malformed checkpoints,
dependency/version/import failures, path escapes, and readiness staying closed
even when both loader calls succeed. Existing synthetic producer tests do not
constitute authentic BEATs inference.

New proposed repository files:

- `deployment/encoders.py`
- `deployment/encoder_diagnostics.py`
- `deployment/encoder-stage1.env.example`
- `tests/test_phase8_encoder_stage1.py`
- `docs/phase8_encoder_stage1.md`

Execution logs, environment inventory, preservation snapshots and readiness JSON
are Git-ignored under `_local/phase8_stage1/`. Existing untracked files are not
part of the proposed Stage 1 change. No artifacts were staged or committed.

Reproduce the final complete command in the isolated environment, with `TMP`,
`TEMP` and `MPLCONFIGDIR` set to writable workspace-local directories:

```text
_local/phase8_stage1/venv/Scripts/python.exe -B -m pytest tests -q -ra -p no:cacheprovider --basetemp <new-workspace-temp-directory>
```

The observed package inventory is `requirements-observed.txt` in the local audit
directory. Besides the candidate deployment pins it includes pytest `9.1.1`,
scikit-learn `1.9.1`, Matplotlib `3.11.2`, W&B `0.31.0`, and h5py `3.16.0`.
W&B interactions in the regression tests are mocks; no live tracking run started.

## Stage 2 prerequisites

1. Accessible authentic audio bundle, trusted producer hashes/lineage, actual saved
   configuration and normalization evidence; then genuine AudioPredictor and
   strict-adapter execution on permitted TRAIN clips.
2. Permitted original TRAIN media with pinned identities/source rows/hashes and
   verified decode timing/validity for both modalities. Existing embedding-only
   reference rows cannot substitute for raw media.
3. Independent TRAIN error/parity measurements under the unchanged policy, plus
   E3/A2 score-difference diagnostics only for trustworthy aligned features.
4. A Docker-capable host and a real build/cold-start test with approved model mounts.
5. A permitted environment where the complete plotting regression suite can load
   Matplotlib's compiled dependency. The current host's Application Control block
   remains an external validation blocker.

Stop after this Stage 1 preparation. Do not enable raw predictions while any gate
is unresolved.

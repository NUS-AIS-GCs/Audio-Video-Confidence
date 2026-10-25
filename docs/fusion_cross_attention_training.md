# Phase 5 Tasks 5.4–5.6: initial real cross-attention experiment

Status: **COMPLETE — READY FOR TRAINING AUDIT.** Executed 2026-10-24
(Asia/Singapore). Exactly one real experiment completed. Cross-attention
underperformed both historical references; the protocol was not changed in
response. No ablations or challenge/test evaluation were performed.

## Implementation and LOW fix

`fusion/model_interface.py` now accepts exact numerical dropout equivalence,
including integer `0` versus float `0.0` in either direction. Booleans, NaN/Inf,
unsupported numeric types and differences as small as `1e-15` still fail. Model
identity, dimensions, heads, module structure and position buffers retain strict
validation. Two new regressions and an independent direct probe passed before
training integration proceeded.

The new `fusion/train_cross_attention.py` entry point shares the Phase 4 guarded
experiment lifecycle through `fusion/train_mlp.py::_run_experiment`. The public
Phase 4 wrapper remains MLP-only with its original defaults and policies. Loaders,
distinct-index coverage, BCE, optimizer, finite-gradient checks, early stopping,
threshold selection, safe bound checkpoints and verified Forge finalization are
reused. There is no copied epoch/tracking system.

The opt-in `EmbeddingDataset.from_scoped_json` loads TRAIN/validation bundles
only. Every pair, target, source row, selected-split provenance, quota, order and
local report field is deterministically regenerated through Phase 2. Only three
all-inventory leakage-context fields stay opaque; their original bytes remain
protected by the original manifest hash. Dataset provenance retains that original
hash. The legacy all-split reader is unchanged.

Pinned official-reference evidence is rebound through fresh byte hashes. Only
TRAIN/validation annotation files are parsed. Final-test producer labels, arrays,
manifest contents, HDF5 orders and predictions are never parsed. Held-out files
participate only as opaque integrity hashes. Challenge pair records are checked
structurally, without constructing a challenge dataset or gathering/scoring its
feature items.

Synthetic reload testing exposed a CPU kernel dispatch difference of `1.19e-7`
when parameter gradient flags changed at freeze time. Every Phase 5 evaluation
now uses frozen parameter flags, restoring training flags afterward. Selected
checkpoint evaluation uses the identical mode. Phase 4 evaluation is unchanged,
and exact reload equality was retained without substituting a tolerance.

## Frozen configuration and data

| Setting | Actual value |
|---|---|
| Model | Bidirectional attention; audio 256→256, visual 512→256 |
| Heads / positions | 4 / fixed sinusoidal, original ten one-second bins |
| Attention/residual/feed-forward dropout | 0.1 / 0.1 / 0.1 |
| Feed-forward / activation / normalization | 512 per branch / GELU / residual LayerNorm |
| Pooling / parameters | Explicit masked mean / 1,251,841 |
| Model/training seed; pair seed | 42; 42 |
| Objective | Unweighted mean BCEWithLogitsLoss |
| Optimizer | AdamW, LR 0.0003, weight decay 0.01 |
| Betas / epsilon / clipping | `(0.9,0.999)` / `1e-8` / maximum gradient norm 5 |
| Batch / maximum epochs / patience | 64 / 30 / 5 |
| Device / threads / scheduler | CPU / 4 / none; CUDA unavailable |
| Epoch selection | Strictly greater validation-primary ROC-AUC; earliest exact tie |
| Epoch F1 | Diagnostic threshold 0.5; no epoch threshold fitting |
| Final threshold | Once after checkpoint freeze; maximum validation-primary binary F1, largest-threshold tie |
| Decision | Float64 `exp(-logaddexp(0,-logit)) >= threshold` |

These are initial settings, not validated optima. `config.json`,
`preregistration.json`, source/data hashes, versions and a deterministic code ZIP
were persisted before tracking startup or optimization. Guards protect the live
and persisted configuration before optimization, each epoch, selection freeze
and successful finalization. No configuration search, adaptive changes or resume
was used.

TRAIN has **6,678 primary pairs** (3,339 original, 3,339 substituted); validation
has **804** (402/402). The separate **402-pair validation challenge was not
evaluated**. Targets encode pairing provenance. Predictive inputs contain only
embeddings and explicit producer-mask intersection, without metadata or
event-label-derived validity.

Original TRAIN manifest SHA-256:
`52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f`.
Validation manifest SHA-256:
`0ef344fa96e64d4ee7f44bee4082a2b1ad5c881504040c306cf6d00172322178`.
Source inventory fingerprint:
`c8ec1448ab79e7c5fcc044678e9375bece168c7f2adfbbe5fa7a57c0ca957949`.
Every primary source ID, row index and target matches historical E3 and JSD.
Ten TRAIN/validation source videos overlap under official AVE assignments;
annotation identities remain split-isolated.

## Actual training and frozen validation

The run completed **14 epochs and 1,470 optimizer steps**. Every completed epoch
visited all 6,678 distinct indices, including the last 22-pair minibatch: 105
steps per epoch. Epoch **9** was selected, with 945 checkpointed steps. Five
subsequent non-improving epochs exhausted patience. All finite-value safeguards
passed. Training plus primary validation took **79.105 seconds**, excluding
integrity, checkpoint and network work.

| Epoch | TRAIN BCE | Validation BCE | Validation AUC | Validation AP | F1 at 0.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.692051 | 0.595019 | 0.747723 | 0.706653 | 0.650323 |
| 2 | 0.374607 | 0.373826 | 0.917224 | 0.900057 | 0.832695 |
| 3 | 0.212271 | 0.336528 | 0.931926 | 0.914462 | 0.874845 |
| 4 | 0.163732 | 0.312882 | 0.948417 | 0.936088 | 0.874222 |
| 5 | 0.113102 | 0.308909 | 0.952204 | 0.946552 | 0.876095 |
| 6 | 0.095829 | 0.381449 | 0.944593 | 0.932009 | 0.866324 |
| 7 | 0.070580 | 0.399866 | 0.946202 | 0.944508 | 0.858942 |
| 8 | 0.057587 | 0.399296 | 0.947514 | 0.943413 | 0.884956 |
| **9** | **0.043109** | **0.441370** | **0.956474** | **0.951058** | **0.855228** |
| 10 | 0.028298 | 0.405469 | 0.956369 | 0.950457 | 0.878672 |
| 11 | 0.034494 | 0.472228 | 0.955057 | 0.953802 | 0.855615 |
| 12 | 0.028122 | 0.466752 | 0.955719 | 0.952322 | 0.872123 |
| 13 | 0.020941 | 0.497304 | 0.954884 | 0.952510 | 0.868661 |
| 14 | 0.022761 | 0.556156 | 0.952278 | 0.948739 | 0.865633 |

Selected validation ROC-AUC is **0.9564738496571867**, AP
**0.9510582382362514**, and final binary F1 **0.9027611044417767** at frozen
threshold **0.0406411536485028**. Accuracy is 0.8992537313432836; BCE is
0.44137032233660495. Confusion: **TP 376, TN 347, FP 55, FN 26**.
Selected TRAIN evaluation AUC/AP/F1 are
0.9998847422095825 / 0.9998835178326826 / 0.9848060185868122.

A fresh safe checkpoint load reproduced all 804 validation logits exactly before
threshold fitting. An independent post-run process repeated that check and
verified row references, pairwise AUC, grouped AP, confusion counts, stable BCE
and exact-rational F1 threshold optimization. AP differed only by `1e-16`
arithmetic roundoff; BCE differed by `2.95e-9` from the sample-weighted float32
batch calculation. Thirty parameter tensors changed from seed-42 initialization.
Source ZIP contents and the complete output manifest were verified independently.

Checkpoint SHA-256:
`50fa1d9675c54485b207abe16be47ce7acc5f0a7fcce52df3b21934aa40ea088`.
Selected state fingerprint:
`0999caff3ae997d2a95bd7d1c7bf58c69c09b4d3eb43cf1257da7a0bc51c8bcb`.

## Forge and reproducibility evidence

Destination: `https://api.forge.coreweave.com`, entity `GC2-AValidate`, project
`avalidate`, group `phase5_bidirectional_cross_attention`, name
`phase5_cross_attention_initial_seed42`.

* [Completed run hiyb44br](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/hiyb44br)
* [Selected checkpoint v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase5-cross-attention-selected-checkpoint/v0)
* [Reproducibility reports v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase5-cross-attention-reproducibility-reports/v0)

Authenticated readiness upload/read-back preceded optimization. Final uploads,
remote summaries, all 14 epoch rows and local input/output checks preceded SDK
success and local completion. A separate API check confirmed state `finished`,
repeated summary/history checks, downloaded both artifact versions and matched
all **22 downloaded file hashes** to local bytes. No additional run was created.
Credentials were not printed or stored in the repository.

Actual command (the completed output must never be overwritten):

```powershell
.\.venv\Scripts\python.exe -B -m fusion.train_cross_attention `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --reference-root _local/datasets/AVE_reference `
  --reference-evidence _local/fusion/ave_reference_verified/verification.json `
  --integration-root _local/fusion/task28_real `
  --baseline-root _local/fusion/phase3_real_experiments/run_20261022_01 `
  --mlp-reference-root _local/fusion/phase4_improvement/study_20261024_seed42_01/E3 `
  --output-dir _local/fusion/phase5_training/run_20261024_seed42_01 --device cpu
```

Python 3.11.16, PyTorch 2.14.1+cpu, NumPy 2.4.6, W&B 0.31.0. The required suite
passed: **467 tests run, 465 passed, two CUDA skips**. Eight new offline lifecycle
tests and two dropout regressions cover the new contracts; W&B unit tests use
mocks and are separate from this genuine experiment.

Four historical MLP checkpoints retain identical states, metadata and synthetic
predictions against the original committed loader. All 280 protected files are
unchanged, including producers, manifests, historical experiments and the
pre-existing untracked requirements file. Phase 1–3/teammate code and GC1 were
not accessed for changes. TRAIN/validation-only production inputs were verified
before allocating optimization resources.

Changes in this batch: `fusion/model_interface.py`, `fusion/embedding_data.py`,
`fusion/train_mlp.py`, new `fusion/train_cross_attention.py`,
`tests/test_fusion_checkpoint_binding.py`, new
`tests/test_fusion_cross_attention_training.py`, and these Phase 5 documents.
Ignored evidence is under `_local/fusion/phase5_training/`, including the run,
independent local verification and downloaded remote verification. No weights,
raw embeddings or credentials are staged. Branch remains
`feature/fusion-phase5-cross-attention`, HEAD `38d4171`; included Phase 4 commit
`76add86` remains outside main. No commit/push/merge/PR occurred. CUDA, exact resume
and calibrated confidence remain unverified/unsupported. See the comparison report.

# Phase 4 Tasks 4.4–4.6: learned fusion training and validation

Status: **COMPLETE — READY FOR FOCUSED TRAINING AUDIT.**

Execution date: 2026-10-23 (Asia/Singapore). This report covers one prespecified
initial experiment on GC2 AValidate. GC1 was not accessed. No commit, push, merge
or PR creation is part of this batch.

## Reproducible workflow

The new modules separate optimization/numeric evaluation (`fusion/mlp_training.py`),
online-only tracking (`fusion/mlp_tracking.py`) and verified experiment orchestration
(`fusion/train_mlp.py`). Existing Phase 1/2 and Phase 3 production code is reused
read-only. The audited Phase 3 test-portability fix and Phase 4 float32 range guard
are preserved. Training dependencies are in `requirements-fusion-training.txt`.
The six files added in this batch are those three modules, that requirements
file, `tests/test_fusion_mlp_training.py` and this report. No audited foundation
or Phase 1–3 production module was edited.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-fusion-training.txt
$env:WANDB_BASE_URL = 'https://api.forge.coreweave.com'
$env:WANDB_ENTITY = 'GC2-AValidate'
$env:WANDB_PROJECT = 'avalidate'
# This workspace already has the optional reference adapter isolated locally:
$env:PYTHONPATH = (Resolve-Path _local/tools/ave_reference).Path
.\.venv\Scripts\python.exe -B -m fusion.train_mlp `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --reference-root _local/datasets/AVE_reference `
  --reference-evidence _local/fusion/ave_reference_verified/verification.json `
  --integration-root _local/fusion/task28_real `
  --baseline-root _local/fusion/phase3_real_experiments/run_20261022_01 `
  --output-dir _local/fusion/phase4_training/run_20261023_seed42_01 `
  --device cpu --seed 42 --pair-seed 42 --batch-size 64 --epochs 30 `
  --patience 5 --cpu-threads 4
```

Every invocation requires a fresh Git-ignored output inside GC2, separate from
inputs and prior baseline evidence. Reproductions must select another fresh
directory; the completed run is never overwritten. There is no resume mode or
claim of exact resume. CPU is explicit; a requested unavailable CUDA device fails.
No automatic architecture, pair-policy or hyperparameter search is implemented.

## Prespecified configuration and data boundary

The 558,081-parameter model concatenates per-bin audio `(B,10,256)` and vision
`(B,10,512)` embeddings into 768 features. The shared MLP uses hidden widths
512, 256 and 128, ReLU and dropout 0.2 after each hidden layer. Explicit validity
masks determine the temporal mean, followed by one linear correspondence head.
Missing masks use all ten exported bins as the documented operating assumption;
AVE event intervals do not provide validity or independent presence targets.

Settings: deterministic initialization and training seed 42; AdamW learning rate
0.001, weight decay 0.0001, betas (0.9, 0.999), epsilon 1e-8; unweighted mean
BCEWithLogitsLoss; batch size 64, maximum 30 epochs, patience 5; maximum gradient
norm 5; four CPU threads; no scheduler and no pretrained fused weights.

Expected inventories are 6,678 TRAIN primary pairs (3,339 originals and 3,339
authorized easy negatives), 804 validation primary pairs (402/402), and a separate
402-negative same-category validation challenge. Original modality row references,
all original annotation identities and the empty-annotation sample are preserved.
No metadata, targets, categories or pair types enter feature tensors.

Before optimization, the runner reuses the audited source/export/manifest hash
verifier and freshly verifies the pinned authors' official HDF5 references. It
validates complete typed TRAIN and validation manifests through Phase 2, and
checks Phase 3 completion/output hashes and identical source/manifest inventories.
Only TRAIN/validation manifests are deserialized. All-split arrays/metadata and
opaque final-test hashes remain structural context; no final-test prediction or
performance report is read or computed.

## Selection, threshold and failure lifecycle

Each epoch trains only TRAIN primary data with seeded shuffling and retains the
last partial minibatch. Validation uses `.eval()` and inference mode without
parameter updates. Losses are sample-weighted means. Gradient statistics are
pre-clipping norms; clipping and parameter-finiteness checks precede continued
training. Epoch duration covers optimization and primary validation, excluding
checkpoint/network overhead.

A checkpoint improves only on strictly greater validation-primary ROC-AUC.
Exact AUC ties retain the earliest epoch; AP and challenge metrics do not break
ties. Five consecutive non-improving epochs stop optimization. Epoch accuracy/F1
use a fixed diagnostic threshold 0.5; no threshold optimization occurs per epoch.

After selection, a fresh model safely loads the run-owned, hash-verified checkpoint
with `torch.load(..., weights_only=True)`. All selected validation logits must
reproduce exactly in evaluation mode. The final threshold maximizes binary F1
on those validation-primary probabilities only, with exact-rational F1 comparison
and largest-threshold tie resolution. Decisions use probability >= threshold.
The model is frozen before TRAIN reporting and separate validation-challenge
diagnostics. Float64 sigmoid conversion avoids needless float32 saturation.

Validation selects both the epoch and threshold and reports their resulting
performance. These estimates are not unbiased held-out results. Sigmoid outputs
are classifier probabilities without independent calibration evidence. The
official split overlaps 10 source videos between TRAIN/validation and 8 between
TRAIN/test, with no validation/test source overlap. No annotations move splits.
Same-category substitution rejection does not establish physical causality.

`state.json` records verification, online-write preflight, optimization, validation,
reload/freeze and upload stages. Exceptions and interrupts preserve FAILED or
INTERRUPTED state and never publish `completion.json`. SDK errors are reduced to
nonsecret stage/type descriptions. Saved best checkpoints include optimizer state
and step count, but no RNG/loader-position restoration is implemented.

## Forge and reproducibility evidence

The required destination is `https://api.forge.coreweave.com`, entity `GC2-AValidate`,
project `avalidate`, group `phase4_learned_fusion_mlp`. Authentication/project access
and an uploaded/read-back readiness artifact must succeed before any real
optimizer step. Offline mode, alternate destinations and silent fallback fail.

Actual epoch-step curves log train/validation loss, validation ROC-AUC and AP,
fixed-0.5 F1/accuracy, learning rate, gradient norms and duration. Final summaries
include selected epoch, frozen threshold, confusion counts, validation metrics
and separate challenge acceptance/rejection. Selected weights/metadata and
reproducibility reports are versioned artifacts. No raw AVE media, embeddings,
credentials or unnecessary input binaries are uploaded.

Ignored local evidence includes configuration and versions, input/code hashes,
a deterministic source-code ZIP for the uncommitted implementation, complete
epoch history, selected checkpoint and metadata, frozen policy, TRAIN/validation
score CSVs, separate challenge scores, baseline comparison, reload verification,
before/after identity checks, output manifest and tracking identity. Completion
requires uploaded artifact read-back plus matching remote final summaries and
every logged epoch curve, followed by unchanged local input/code/report checks.

## Execution results

The single initial run completed successfully. Early stopping fired at epoch 20
after five non-improving epochs; **epoch 15** was selected by the prespecified AUC
rule. There were 2,100 actual optimizer steps (105 per epoch), with 1,575 steps in
the selected checkpoint. Each epoch retained the final 22-item TRAIN minibatch.
No architecture, sampling policy or hyperparameter was changed to improve results.

Execution used Python 3.11.16, NumPy 2.4.6, PyTorch 2.14.1+cpu, W&B 0.31.0 and the
isolated h5py 3.16.0 reference reader. CUDA was unavailable. Optimization and
primary validation took approximately 1.3–1.4 seconds per epoch with four CPU
threads; complete integrity checks, checkpoint/network overhead are additional.

The final frozen cutoff is **0.05357356021051435**. All 804 selected validation
logits reproduced exactly in a fresh model on the same CPU/backend. Independent
arithmetic over saved prediction records reproduced AUC/AP, confusion counts,
F1, the exact-rational threshold optimum, and BCE to floating-point tolerance.
Pair identities/row references matched the original TRAIN/validation manifests.
The source-code ZIP matches every recorded current code hash. Original input,
manifest, official reference, baseline-evidence and code hashes remain unchanged.

| Selected model metric | TRAIN primary | Validation primary |
|---|---:|---:|
| Count | 6,678 | 804 |
| BCE loss (evaluation mode) | 0.002028043 | 0.519720569 |
| ROC-AUC | 1.000000 | 0.961746 |
| Noninterpolated AP | 1.000000 | 0.957185 |
| Accuracy at frozen cutoff | 0.995058 | 0.904229 |
| Binary F1 at frozen cutoff | 0.995083 | 0.906440 |
| Precision | 0.990214 | 0.885986 |
| Recall | 1.000000 | 0.927861 |
| Descriptive Brier score | 0.000432 | 0.104397 |

Validation confusion matrix (rows = actual, columns = predicted):

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | TN 354 | FP 48 |
| Actual positive | FN 29 | TP 373 |

The separate validation same-category challenge contained 402 provenance
negatives: **61 rejected (15.1741%)**, **341 falsely accepted (84.8259%)** at the
frozen cutoff. These examples affected neither optimization, checkpoint selection
nor threshold fitting. One-class challenge ROC-AUC/AP are undefined, and are not
combined with primary metrics.

| Validation metric | Phase 3 Jensen–Shannon | MLP | MLP minus baseline |
|---|---:|---:|---:|
| ROC-AUC | 0.969642 | 0.961746 | -0.007896 |
| AP | 0.972805 | 0.957185 | -0.015620 |
| F1 | 0.917293 | 0.906440 | -0.010853 |
| Same-category rejection | 15.6716% | 15.1741% | -0.4975 percentage points |

**This initial MLP underperforms the audited baseline on all four comparison
metrics.** The comparison verified identical source export hashes, complete
pair manifests, artifact context and original pairing policy. TRAIN-minus-
validation gaps are larger for the MLP: AUC 0.038254 versus 0.030307, AP 0.042815
versus 0.027143, and F1 0.088643 versus 0.074096. Perfect TRAIN ranking and much
higher validation BCE show a substantial generalization gap. No additional run
was performed to improve the baseline comparison. No calibration-quality or
final-test generalization claim follows from these validation results.

### Actual epoch history

Training loss uses training-mode dropout; final TRAIN loss above uses the frozen
selected model in evaluation mode. Epoch F1 uses the fixed 0.5 diagnostic cutoff,
not the final fitted cutoff. The table preserves all executed epochs, including
the five after the selected checkpoint.

| Epoch | Train loss | Val loss | Val AUC | Val AP | Val F1 at 0.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.474171 | 0.335264 | 0.933974 | 0.920481 | 0.857868 |
| 2 | 0.190337 | 0.290525 | 0.951239 | 0.942707 | 0.890566 |
| 3 | 0.122942 | 0.605951 | 0.940973 | 0.931628 | 0.747059 |
| 4 | 0.094053 | 0.363883 | 0.952266 | 0.946944 | 0.870229 |
| 5 | 0.069209 | 0.345653 | 0.954667 | 0.951071 | 0.871212 |
| 6 | 0.064022 | 0.540952 | 0.931697 | 0.930455 | 0.821622 |
| 7 | 0.045075 | 0.458604 | 0.948782 | 0.945212 | 0.843085 |
| 8 | 0.037078 | 0.423300 | 0.957161 | 0.951234 | 0.858653 |
| 9 | 0.039528 | 0.413973 | 0.954364 | 0.948676 | 0.877680 |
| 10 | 0.022703 | 0.551121 | 0.956505 | 0.952098 | 0.864652 |
| 11 | 0.023037 | 0.776170 | 0.957464 | 0.957105 | 0.798271 |
| 12 | 0.039596 | 0.423542 | 0.953621 | 0.952519 | 0.849462 |
| 13 | 0.016317 | 0.671723 | 0.956548 | 0.957984 | 0.865870 |
| 14 | 0.035179 | 0.405265 | 0.958528 | 0.957780 | 0.881013 |
| **15** | **0.009205** | **0.519721** | **0.961746** | **0.957185** | **0.857895** |
| 16 | 0.017302 | 0.426926 | 0.958052 | 0.954412 | 0.881101 |
| 17 | 0.019623 | 0.704049 | 0.958961 | 0.958297 | 0.839065 |
| 18 | 0.014075 | 0.601574 | 0.954952 | 0.950785 | 0.880309 |
| 19 | 0.009191 | 0.574365 | 0.961344 | 0.957317 | 0.879690 |
| 20 | 0.016854 | 0.611666 | 0.958361 | 0.955961 | 0.830556 |

### Run, checkpoints and fingerprints

- [CoreWeave Forge run de5u89au](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/de5u89au).
- [Selected checkpoint artifact v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase4-mlp-selected-checkpoint/v0),
  digest `d6df1fee820e4f5eae10fc5fb693fef3`.
- [Reproducibility/evaluation artifact v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase4-mlp-reproducibility-reports/v0),
  digest `e0fd7ee8d091b5ee2367978dc02105bd`.
- Ignored output: `_local/fusion/phase4_training/run_20261023_seed42_01/`.
- Selected checkpoint: `best.pt`, 6,706,435 bytes including optimizer state;
  SHA-256 `39fccb99c7e06c1d6be3db4bb321a8818932a5c91b23954843836713a9f0ac34`.
- Model tensor fingerprint:
  `061675a9acdc2de861c39b80db6384191feb94a164cf68e70be844255342aedb`.
- Code fingerprint:
  `baa703efca6410175ab7ad4829742cb48ecaa48fb43c867aef80a9e091fca4a9`.
- Configuration fingerprint:
  `68634b80b99b7d476f09a6eac38e9778b91347184bd9bd6f4f1ae63d5ba6a3e8`.
- Verified artifact context:
  `c8ec1448ab79e7c5fcc044678e9375bece168c7f2adfbbe5fa7a57c0ca957949`.

Forge completion verified the final summary, both artifact digests and every
metric in all 20 epoch-history records. Independent local evidence is saved in
`_local/fusion/phase4_training/independent_verification_run01.json`; verification
scripts and any downloaded audit copies remain ignored and create no new runs.
An independent authenticated read-back confirmed the run is `finished` and all
20 epoch records match. Both final artifacts were downloaded to
`_local/fusion/phase4_training/remote_verification_run01/`: all three model-artifact
files and all 19 report-artifact files matched local SHA-256 values. This evidence
is in that directory's `verification.json`. No checkpoint was loaded from an
untrusted download; the run-owned local checkpoint was verified and loaded safely.

HEAD remains `00838cc3fdec0c4073c410940701b785a11f986c` on
`feature/fusion-phase4-mlp`. The exact uncommitted implementation is recorded in
the source ZIP and per-file hashes. A fresh remote check on 2026-10-23 confirmed
main at `1846d9076eea354ddf463469921314c9dfb05a05` and the Phase 3 branch at
`00838cc3fdec0c4073c410940701b785a11f986c`, still unmerged into main. Phase 4 depends
on that Phase 3 implementation; no upstream integration or PR action was taken.

## Synthetic verification

Verification: 365 tests, 364 passes and one unavailable-CUDA skip;
zero errors/failures. The 17 new offline tests cover optimization, no-grad
validation, modes, exact selection ties, threshold fitting, challenge/test guards,
checkpoint integrity/reload, deterministic repeat runs, failed/interrupted runs,
online-only destination checks and mocked Forge artifact/history verification.
They create no real W&B runs and never optimize the real AVE dataset.

The complete suite was run again after real training with the same successful
365-test result. Syntax parsing passed for 33 fusion/test Python files; scoped
whitespace checks and credential-pattern inspection passed; `pip check` reported
no broken requirements. Checkpoints, score records, download mirrors and audit
evidence remain ignored under `_local/`, with no new large tracked artifacts.
The Phase 3 test-portability modification remains intact. No remaining pipeline
correctness defect or tracking blocker was confirmed. Limitations are the genuine
baseline underperformance/generalization gap, validation reuse, source-video
overlap, unverified CUDA execution, encoder provenance based on producer
declarations plus byte continuity, and no exact-resume implementation.

## Post-audit reliability remediation (2026-10-24)

The subsequent independent audit found two guard defects affecting future runs:
successful Forge finalization preceded evidence read-back, and optimization
coverage checked sample count without checking distinct original dataset indices.
The recorded first experiment remains valid; its outputs and finished run
`de5u89au` are unchanged. No additional real training or Forge run was performed.

The tracker now uploads and waits for artifacts, verifies final summary values
(including the manifest hash), all epoch curves and artifact digests while the
run remains active, checks local input/source/output integrity, and only then
calls `finish(exit_code=0)`. Active-run visibility is retried at most three times,
one second apart; public API run caches are cleared and history caching is
disabled. Failed uploads/read-back/local checks request `finish(exit_code=1)`;
cleanup is idempotent and SDK exception text remains suppressed.

W&B finalization is not a distributed transaction. If successful finalization
itself raises after the SDK has marked the run finished, a later failure request
may be ignored and remote state cannot be inferred or reversed safely. Such a
case remains locally `INCOMPLETE`, explicitly records unknown finalization state,
and requires read-only reconciliation. Local completion-file writes can also
fail after remote finalization; they cannot be made atomic with the server.
There is no claim that this offline remediation verified active-run visibility
on Forge. Delayed evidence beyond the bounded retries fails closed.

Training now attaches original dataset indices through a private, worker-safe
dataset adapter. The original dataset and public loader batches remain unchanged;
only the internal optimization loader carries the third, non-predictive index
field. It reuses the actual batch sampler and seeded generator. Each consumed
batch rejects duplicate indices before its update, and epoch completion requires
every original index exactly once. Missing coverage fails at epoch end; earlier
valid updates are not rolled back, so the runner must fail the experiment.
Targets and indices never enter model features. Unbatched loaders and custom
collators are explicitly unsupported, since they cannot preserve this audited
index-to-item association. Partial final minibatches and worker prefetch remain
supported. Validation inference does not use this training-only adapter.

Offline regression tests cover ordering, upload/read-back/finalization failures,
unknown terminal states, repeated cleanup, distinct-index coverage, identical
feature values, shuffle determinism, worker prefetch, feature schemas, validation
separation and checkpoint/selection compatibility. The original archived source
fingerprint above identifies the historical experiment; current source hashes
intentionally change with these fixes and do not relabel that experiment.

Remediation verification: the unchanged baseline suite had 365 tests (364 passed,
one CUDA skip). The final suite has 384 tests (383 passed, one CUDA skip), including
19 new regression tests. Controlled no-update doubles reproduced both original
failures before editing: 16 visits to one index were accepted, and a summary
mismatch left the SDK finalized with exit code zero. The same probes now reject
the duplicate sampler and request exit code one with local `INCOMPLETE` status.
All 73 inventoried original experiment files, producer exports, pair manifests
and top-level Phase 3 baseline reports retained their SHA-256 values. Syntax
parsing, scoped whitespace/secret checks, `git diff --check` and `pip check` passed.
The existing Phase 3 test-portability correction remains unchanged. Read-only
remote inspection still shows Phase 3 at `00838cc` and main at `1846d90`; no
Phase 4 remote branch or upstream is configured, and no Git publication occurred.

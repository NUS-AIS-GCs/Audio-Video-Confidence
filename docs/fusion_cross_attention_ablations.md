# Phase 5 Tasks 5.7–5.8: controlled architectural ablations

Status: **COMPLETE — READY FOR INDEPENDENT ABLATION AUDIT.** Exactly three
additional real experiments completed on 2026-10-25 (Asia/Singapore), each with
one attempt. No experiment was restarted. The frozen attention ranking selected
**A2, video-to-audio only**. Its primary-validation ROC-AUC was
**0.9608549293334323**, above the original attention run by
**0.0043810796762455695**, below historical E3 MLP by **0.00916437711937823**
and below JSD by **0.008786911215068915**. These are descriptive single-seed
development-validation results, without significance or generalization claims.

## Frozen inventory and implementation

| Variant | Retained attention | Positions | Pooled classifier input | Parameters |
|---|---|---|---|---:|
| Initial reference | Audio Q → video K/V and video Q → audio K/V | Sinusoidal | Concatenated 256-dimensional updated streams, 512 total | 1,251,841 |
| A1 | Audio Q → video K/V | Sinusoidal | 256-dimensional updated audio stream | 724,481 |
| A2 | Video Q → audio K/V | Sinusoidal | 256-dimensional updated video stream | 724,481 |
| A3 | Both original directions | Disabled explicitly | Original 512-dimensional concatenation | 1,251,841 |

Each one-way stream includes the query residual and cross-modal attended values,
followed by the original normalization/feed-forward update and masked temporal
mean. Both learned modality projections remain. The removed branch has no
parameters, modules or replacement tensors; its optional diagnostic output is
`None`. Removing one 527,104-parameter branch and 256 classifier weights saves
527,360 parameters. These comparisons are **not parameter matched**. A3 retains
the original ten bins and their source-row order; its registered positional
buffer is reconstructed as zeros by the validated `position_encoding="none"`
configuration. No temporal reordering or shuffling was used for training.

The original bidirectional configuration keeps its exact serialized fields,
checkpoint schema and model fingerprint behavior. A read-only
`attention_direction` property declares its existing topology. One-way variants
use a distinct strict configuration/model type with an explicit required
direction, dispatched by the model registry. Architecture-bound checkpoint
validation checks the actual module topology, parameter/buffer registration and
serialized state. Historical MLP metadata need no attention-specific fields.

Seed controls are unchanged: seed 42 for initialization and training, CPU
initialization within an RNG fork, deterministic algorithms and the original
loader seeding. Removing modules changes the subsequent initialization/dropout
draw placement; retained modules are not individually coupled across all
topologies. The seed is controlled, but this is not a matched-initialization or
multi-seed causal comparison.

| Fixed training setting | Value for A1, A2 and A3 |
|---|---|
| Shared width / heads | 256 / 4 |
| Attention / residual / feed-forward dropout | 0.1 / 0.1 / 0.1 |
| Feed-forward / activation | 512 per retained branch / GELU |
| Normalization / pooling | Original residual LayerNorm / explicit masked mean |
| Loss / class weighting | Mean BCEWithLogitsLoss / unweighted |
| Optimizer | AdamW, LR 0.0003, weight decay 0.01 |
| Betas / epsilon / gradient clipping | (0.9, 0.999) / 1e-8 / max norm 5 |
| Batch / maximum epochs / patience | 64 / 30 / 5 |
| Model/training seed / pair seed | 42 / 42 |
| Device / CPU threads / workers | CPU / 4 / 0; CUDA unavailable |
| Scheduler / resume / adaptive changes | None / none / none |
| Epoch selection | Strictly greater primary-validation AUC; earliest exact tie |
| Threshold selection | Once after best-checkpoint freeze; primary-validation max binary F1, largest-threshold tie |
| Decision scores | Float64 `exp(-logaddexp(0,-logit)) >= threshold` |
| Attention-variant ranking | Full-precision validation AUC, then AP, then F1 descending; final fixed order initial/A1/A2/A3 |

All three configurations, topology counts, original/E3/JSD references, selection
rules, source snapshot and input hashes were frozen **before** any ablation
optimization in `protocol.json`. Its SHA-256 is
`33cfdb7e5dd8853e847ab1c2b6208c58923f181cd3e8caded4c6e92ad8e599f6`.
The identical protocol and digest are embedded in each uploaded per-run
preregistration. Guards check source, plan and every variant configuration
before each epoch and finalization. An attempt ledger records exactly A1/A2/A3
with `attempt_number=1`; failure stops the study without automatic retries.

## Data scope and observed training

Each epoch used all **6,678 primary TRAIN pairs**, including 3,339 original and
3,339 substituted pairs. There were 104 full batches plus the final 22-example
batch: **105 optimizer steps per epoch**. Distinct-index coverage was checked
each epoch. Each validation pass used **804 primary pairs**, balanced 402/402.
No labels, source identities, split indicators or pairing metadata enter model
forward inputs; only the original aligned audio/visual features and validity
mask do. Validation and final checkpoint evaluation use the same frozen
parameter flags and deterministic CPU execution, and validation does not update
the optimizer.

The separate 402-pair same-category challenge was never constructed as a
feature dataset or scored. Final-test arrays, labels, manifest contents and
predictions were not parsed. Existing held-out evidence participated only in
opaque byte hashes under the established integrity contract. Structural
regeneration of challenge pair records during manifest verification did not
gather features or perform inference.

| Model | Best epoch | Completed epochs | Actual total optimizer steps | Steps in selected checkpoint |
|---|---:|---:|---:|---:|
| Initial reference | 9 | 14 | 1,470 | 945 |
| A1 | 9 | 14 | 1,470 | 945 |
| A2 | 18 | 23 | 2,415 | 1,890 |
| A3 | 11 | 16 | 1,680 | 1,155 |
| Historical E3 MLP | 17 | 22 | 2,310 | 1,785 |

All three new runs stopped after five epochs without a strictly better AUC.
Epoch histories contain train/validation losses, AUC/AP, diagnostic F1 at 0.5,
gradient norms, learning rate, durations, selection flags and cumulative steps.
The epoch F1 diagnostics did not fit operating thresholds during training.

Source inventory fingerprint:
`c8ec1448ab79e7c5fcc044678e9375bece168c7f2adfbbe5fa7a57c0ca957949`.
Original TRAIN manifest:
`52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f`.
Original validation manifest:
`0ef344fa96e64d4ee7f44bee4082a2b1ad5c881504040c306cf6d00172322178`.
All five learned models have the same ordered 804 validation identities,
producer source rows, targets and source references. The corresponding
reference-row fingerprint is
`95c2af7c5dae243fb8f500f6d38b76a3e5d0e9d2253b44d290c8097b7172b724`.
The JSD reference uses the previously verified identical inputs and pairing.

## Complete primary-validation comparison

Losses below are deterministic evaluation BCE at the selected checkpoint;
training loss is evaluated without training dropout. They are distinct from
the stochastic per-epoch training averages in `history.json`.

| Learned model | Train loss | Val loss | Val ROC-AUC | Val AP | Val F1 |
|---|---:|---:|---:|---:|---:|
| Initial bidirectional | 0.019389009 | 0.441370322 | 0.956473850 | 0.951058238 | 0.902761104 |
| A1 audio-to-video | 0.028419505 | 0.381364227 | 0.954357565 | 0.948288901 | 0.907063197 |
| A2 video-to-audio | 0.000896839 | 0.557757082 | 0.960854929 | 0.958387720 | 0.905263158 |
| A3 without position | 0.013134949 | 0.457523619 | 0.959976238 | 0.955512095 | 0.898263027 |
| Historical E3 MLP | 0.002825481 | 0.482346303 | 0.970019306 | 0.968596681 | 0.913669065 |

| Learned model | Precision | Recall | TP | TN | FP | FN | Frozen threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0.872389791 | 0.935323383 | 376 | 347 | 55 | 26 | 0.0406411536485028 |
| A1 | 0.903703704 | 0.910447761 | 366 | 363 | 39 | 36 | 0.25910291122020657 |
| A2 | 0.854304636 | 0.962686567 | 387 | 336 | 66 | 15 | 0.0026640388978268056 |
| A3 | 0.896039604 | 0.900497512 | 362 | 360 | 42 | 40 | 0.10440009964709925 |
| E3 | 0.881944444 | 0.947761194 | 381 | 351 | 51 | 21 | 0.009579099009671126 |

**Separate non-trainable reference: JSD.** Zero trainable fusion parameters;
epochs and training BCE are inapplicable. Validation AUC **0.9696418405485012**,
AP **0.9728053582663733**, F1 **0.9172932330827067**, precision
**0.9242424242424242**, recall **0.9104477611940298**; TP 366, TN 372, FP 30,
FN 36; frozen threshold **0.32543530977887525**. JSD scores are not logits from a
trained BCE model, so a comparable validation BCE is not reported.

Differences below are model minus reference in raw metric units, rounded for
display. JSON evidence retains full precision for all AUC/AP/F1 differences.

| Attention model | ΔAUC initial | ΔAP initial | ΔF1 initial | ΔAUC E3 | ΔAP E3 | ΔF1 E3 | ΔAUC JSD | ΔAP JSD | ΔF1 JSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial | 0 | 0 | 0 | -0.013545457 | -0.017538443 | -0.010907960 | -0.013167991 | -0.021747120 | -0.014532129 |
| A1 | -0.002116284 | -0.002769337 | +0.004302093 | -0.015661741 | -0.020307780 | -0.006605868 | -0.015284275 | -0.024516457 | -0.010230036 |
| A2 | +0.004381080 | +0.007329481 | +0.002502053 | -0.009164377 | -0.010208962 | -0.008405907 | -0.008786911 | -0.014417639 | -0.012030075 |
| A3 | +0.003502389 | +0.004453857 | -0.004498077 | -0.010043068 | -0.013084586 | -0.015406037 | -0.009665602 | -0.017293263 | -0.019030206 |

Frozen ranking: **A2 → A3 → initial → A1**. No metric ties required secondary
rules. E3 and JSD are reference comparators, outside attention-variant selection.
A2 leads A3 in AUC by only **0.0008786911215068471**. A1 has the highest
thresholded F1 of the attention models; the prespecified AUC-first policy still
selects A2. E3 and JSD exceed every attention variant in AUC, AP and F1.

These observations provide no evidence that the reverse branch or sinusoidal
positions improve this particular seed's development AUC. They do not establish
that either component is generally harmful or unnecessary. Unequal capacity,
initialization draw placement, stochastic optimization and different selected
epochs limit architectural attribution. A2's very small train loss, higher
validation loss and low operating threshold also show why AUC alone should not
be interpreted as calibrated confidence or good probability estimates.

## Reproducibility and preservation

Before real optimization, the full suite passed: **478 tests run, 476 passed,
two CUDA skips**. Eleven new tests cover topology/QKV direction, retained
residual streams, masks, modality dependence, finite gradients, counts,
configurations, checkpoint round-trips, mismatches, ranking and three-run
lifecycle behavior. Synthetic lifecycle tests use mocked tracking and are
separate from the real study. Failure/tamper tests prove fail-closed behavior
without starting later variants or restarting an attempt.

Independent pre-training probes additionally compared the original source ZIP
against the extended implementation: default initialized tensors, synthetic
outputs and the historical selected checkpoint's outputs were bitwise equal.
Two-dimensional identity-QKV algebra reproduced both directional branches to
floating error below `4e-16`. The four Phase 4 checkpoints were loaded through
the original committed loader and the current loader: states, metadata and
synthetic predictions were identical.

A separate post-training verifier safely loaded each actual selected checkpoint
with `weights_only=True`, checked configuration binding and reconstructed
buffers, and reproduced **all 804 saved logits exactly**. It confirmed trained
weights differ from seed-42 initialization, real AdamW selected-checkpoint step
states, full epoch index coverage and earliest maximum-AUC selection. Independent
pairwise ROC-AUC, grouped AP, confusion/F1 and exact rational threshold searches
matched recorded metrics/policies. Independently evaluated float64 BCE agreed
with float32 evaluation within `1.1e-8`. All five models' ordered validation
references reconciled exactly. The final full comparison equals the comparison
uploaded in A3's reports artifact.

Historical preservation checks re-hashed **280 files / 126,941,543 bytes**, with
no changes to producer exports, original manifests, Phase 3/4 evidence or the
existing requirements file. A separate pre-study inventory protects all **23
non-W&B files** of the initial Phase 5 run, including its selected checkpoint,
configuration, manifests and reports; all remain unchanged. The new source ZIP
and source fingerprints remain equal to the frozen protocol. No dependency was
added; `pip check` and whitespace checks pass. No raw embedding or checkpoint
is staged. No GC1 path was accessed.

The untracked `requirements-fusion-training.txt` remains byte-for-byte identical:
SHA-256 `a01116f4399965d31d989d92ec8d14a07cd518b55208d42a9081efd91f2848ab`.

## Forge evidence and local paths

Destination is CoreWeave Forge, entity `GC2-AValidate`, project `avalidate`, group
`phase5_cross_attention_ablations`. Each readiness upload/read-back completed
before optimization. Uploads, remote histories/summaries, input/output checks
and study source guards passed before successful SDK finalization and local
completion. All per-run histories and artifacts are immutable recorded versions.
A separate authenticated read-only API pass confirmed all three runs are
`finished`, matched their complete configurations and final summaries, verified
all **14/23/16 epoch rows**, and streamed all **66 artifact files** (22 per run)
to SHA-256/MD5 hashes matching local bytes and remote manifest entries. It
created no runs and wrote no historical artifacts. The four historical Phase 4
runs and initial Phase 5 run also retain their terminal state, selected AUCs and
pinned artifact digests.

| Run | Selected checkpoint | Reproducibility reports |
|---|---|---|
| [A1: 2wbi8en8](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/2wbi8en8) | [A1 model v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase5-ablation-a1-selected-checkpoint/v0) | [A1 reports v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase5-ablation-a1-reproducibility-reports/v0) |
| [A2: 6mo5xu4k](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/6mo5xu4k) | [A2 model v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase5-ablation-a2-selected-checkpoint/v0) | [A2 reports v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase5-ablation-a2-reproducibility-reports/v0) |
| [A3: usp3ura0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/usp3ura0) | [A3 model v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase5-ablation-a3-selected-checkpoint/v0) | [A3 reports v0, including full comparison](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase5-ablation-a3-reproducibility-reports/v0) |

References: [initial attention](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/hiyb44br),
[historical E3 MLP](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/c7mvo4sd).
No historical run was resumed or written by this task.

Ignored local study evidence is at
`_local/fusion/phase5_ablations/study_20261025_seed42_01/`: `protocol.json`,
`source_code.zip`, attempt/completion ledgers, full-precision
`primary_comparison.json`, `primary_comparison.csv`, and A1/A2/A3 run directories.
Each run contains the configuration, preregistration, input/output hashes,
history, selection, safe checkpoint/metadata, frozen threshold, selected TRAIN
and primary-validation scores/reports, reload evidence, comparison and tracking
record. Independent verifier scripts/reports are siblings of this study root,
outside the completed per-run output manifests.

Actual one-time invocation (completed outputs must never be overwritten):

```powershell
.\.venv\Scripts\python.exe -B -m fusion.cross_attention_ablations `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --reference-root _local/datasets/AVE_reference `
  --reference-evidence _local/fusion/ave_reference_verified/verification.json `
  --integration-root _local/fusion/task28_real `
  --baseline-root _local/fusion/phase3_real_experiments/run_20261022_01 `
  --mlp-reference-root _local/fusion/phase4_improvement/study_20261024_seed42_01/E3 `
  --initial-root _local/fusion/phase5_training/run_20261024_seed42_01 `
  --study-root _local/fusion/phase5_ablations/study_20261025_seed42_01 --device cpu
```

Python 3.11.16, PyTorch 2.14.1+cpu, NumPy 2.4.6, W&B 0.31.0.

## Scope, upstream state and limitations

Changes made for this task: `fusion/cross_attention_model.py`,
`fusion/model_interface.py`, `fusion/training_contracts.py`,
`fusion/train_cross_attention.py`, `fusion/mlp_study.py`, new
`fusion/cross_attention_ablations.py`, new
`tests/test_fusion_cross_attention_ablations.py`, and this document. Earlier
uncommitted Phase 5 work remains in place; the full working-tree diff includes
that prior work. The public Phase 4 runner behavior and defaults remain intact.

Branch remains `feature/fusion-phase5-cross-attention`, HEAD
`38d4171d1b9c6aefab476a87fa8fc144999191d9`. Phase 4 commit
`76add86568daed6f7dbb31659f26c020b264962c` is included in this branch. A live
read-only remote-tip check found Phase 4 still at that commit and main at
`467c7568def85d3f9270756ce1ce98d861958c45`; Phase 4 remains an upstream merge
dependency. The index is empty. No commit, push, merge or PR was performed.

No implementation defect was confirmed in these checks. Scientific limitations
remain: one seed; development data reused for epoch, threshold and architecture
selection; added selection optimism; capacity and initialization differences;
ten shared source IDs across official TRAIN/validation annotation splits;
targets represent pairing provenance rather than general audiovisual truth;
uncalibrated scores; and ten coarse aligned temporal bins. Historical E3 used
its own prespecified LR/dropout, so it is a reference rather than an isolated
architectural control. CUDA behavior is untested in this CPU environment.
Exact training resume is unsupported. No same-category challenge or final-test
evaluation was conducted; no additional experiments are part of this task.

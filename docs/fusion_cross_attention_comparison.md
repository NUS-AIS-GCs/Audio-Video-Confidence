# Phase 5 initial attention versus historical primary references

Status: **COMPLETE — the initial attention model underperforms both references
on all three reported validation metrics.** This is one frozen seed-42 result,
not evidence that every attention architecture is inferior.

| Model | Validation ROC-AUC | Validation AP | Validation binary F1 |
|---|---:|---:|---:|
| Phase 5 attention, epoch 9 | 0.956473850 | 0.951058238 | 0.902761104 |
| Phase 4 selected E3 MLP | 0.970019306 | 0.968596681 | 0.913669065 |
| Phase 3 Jensen–Shannon | 0.969641841 | 0.972805358 | 0.917293233 |

| Attention minus reference | ROC-AUC delta | AP delta | F1 delta |
|---|---:|---:|---:|
| Versus E3 | -0.013545457 | -0.017538443 | -0.010907960 |
| Versus JSD | -0.013167991 | -0.021747120 | -0.014532129 |

These are absolute differences. Producer hashes, primary manifest identities,
source IDs, row indices and targets match both references exactly. Higher scores
mean original pairing. Shared ROC-AUC and noninterpolated AP definitions apply.
Binary F1 uses each method's validation-fitted frozen cutoff and `>=`; these are
not all F1-at-0.5 values. The attention cutoff is 0.0406411536485028.

| Actual pairing | Predicted original | Predicted substituted |
|---|---:|---:|
| Original (402) | TP 376 | FN 26 |
| Substituted (402) | FP 55 | TN 347 |

Only primary-validation ROC-AUC selected the epoch. Epoch 9 won; five subsequent
non-improving epochs stopped the run at epoch 14. Lower BCE at another epoch did
not change that decision. Threshold fitting occurred once after checkpoint freeze.

No same-category challenge measurement was computed or used in this comparison.
Previously viewed historical challenge results are omitted from comparison
inputs. The 402-pair attention challenge remains reserved for a later authorized
frozen-model diagnostic. No final-test labels/predictions were parsed or scored;
held-out files were used only as opaque integrity hashes.

Validation selects epochs and the threshold and supplies the reported metrics.
Historical E3 additionally came from its documented configuration-selection study.
These are reused development measurements, not independent held-out generalization
estimates. One seed provides no statistical superiority/inferiority claim.

The attention TRAIN evaluation AUC (0.999884742) versus validation AUC
(0.956473850) shows a development gap without identifying its cause. No additional
experiment or responsive hyperparameter/architecture change was made.

Official AVE source-video overlap remains, including ten TRAIN/validation sources.
Pair labels encode provenance, not physical source causality. Ten one-second
embedding bins limit precise temporal/source attribution. Attention maps are not
causal evidence. Validity comes from producers, not AVE event intervals. Sigmoid
outputs and descriptive Brier scores do not establish calibrated confidence.

The [training report](fusion_cross_attention_training.md) contains the frozen
configuration, epoch history and integrity evidence. The genuine experiment is
[Forge run hiyb44br](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/hiyb44br).
The result is ready for training audit. No ablations or further challenge/test
experiments are part of this batch.

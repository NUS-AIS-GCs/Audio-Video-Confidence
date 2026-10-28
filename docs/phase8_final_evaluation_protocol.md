# Frozen final-evaluation preparation — Stage 2E

Status: **documented, execution blocked**. This protocol grants no permission to
read or score held-out TEST tensors. Stage 2E uses only the existing five approved
TRAIN clips. Independent parity approval, review of this protocol, a reviewed
TEST-capable reader and explicit final-evaluation authorization are prerequisites.
Do not relax the existing Phase 7 TRAIN/validation admission checks to run it.

## Cohort and task

The future primary benchmark is the official AVE **TEST** split and the already
frozen Phase 2 cross-category pairing policy: 402 original pairs and 402
audio-swapped pairs, 804 total. Authenticate its manifest, exports and identity
mapping against the original source evidence before execution. These counts are
the existing split/pair specification, not newly measured TEST results.

An original pair is `(audio_i, visual_i)`, target **1**, with the same authenticated
sample identity. A substituted pair is `(audio_j, visual_i)`, target **0**, from
the same official split, with different sample and source-clip identities and a
different event category under the frozen primary policy. Preserve audio/visual
row indices, both source IDs and pair order. Category labels establish the pairing
manifest; they must never become encoder features or validity masks.

Use every authenticated primary pair exactly once. Missing source media,
invalid decode, mismatched provenance, nonfinite scores or incomplete coverage
block a complete primary result. Record all failure counts and identities; do not
silently drop failures and report the remaining cohort as the benchmark. The
separate same-category challenge is all-negative and must remain a separate
analysis, subject to its own authorization. Do not pool it into primary AUC/AP.

## Leakage and interpretation

Existing metadata audits describe 3,339/402/402 TRAIN/validation/TEST annotation
rows, yielding 6,678/804/804 primary pairs. Across all 4,143 annotation IDs there
are 4,097 distinct source videos. Historical source-ID overlaps are 10 between
TRAIN/validation, 8 between TRAIN/TEST and 0 between validation/TEST. References:
[Phase 7 reporting](phase7_evaluation_reporting.md) and
[real-data audit](fusion_real_data_validation.md). Do not rerun TEST scoring to
verify those historical metadata counts during Stage 2E.

The official primary split is therefore not source-disjoint. State encoder
training exposure, repeated-source dependence and this limitation prominently.
Validation was reused for architecture, epoch and threshold selection; E3 epoch
17 and A2 epoch 18 results on that split are development results with selection
optimism. The five current positive TRAIN cases establish neither negative-pair
performance nor generalization.

Before scores are opened, a separately labeled source-disjoint sensitivity
cohort may be registered using metadata alone: exclude a pair if **either** its
audio or visual source occurs in TRAIN or development validation. Freeze and
review the resulting identity list and exclusions before inference. Preserve
the official primary cohort and historical reports. This supplementary analysis
cannot undo encoder exposure or turn the original official split into a
source-disjoint experiment. Do not add or revise cohorts after seeing results.

## Frozen models and decision rule

E3 is primary; A2 is the fixed comparison. Do not select a new winner on TEST.
Bindings must match `fusion.inference.CANDIDATES` and original metadata,
operating-point and integrity files, rechecked before and after inference.

| Binding | E3 | A2 |
| --- | --- | --- |
| Threshold | `0.009579099009671126` | `0.0026640388978268056` |
| Checkpoint SHA-256 | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` | `0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12` |
| Model SHA-256 | `d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025` | `1ef5985f9de3e7b9f2994b076ebd16c1b0cd7d4922c0842d12d301aa202c2b8e` |
| Metadata SHA-256 | `fd6abe3f0e62f09e19c096f82b6516195a2c8dba93cd3d534e7f88fcfec48e58` | `aadd9c73e7675249f8192912656ca624aa6d77d50b10cea3f29a1968cce8d460` |
| Operating-point SHA-256 | `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad` | `16994e04c6a8421ec8bc22c1fac33691a4a0228ba544f6e57b0730f98c95b384` |

Execute CPU eval/frozen float32 models, with autocast disabled, and persist raw
logits plus the stable float64 sigmoid used by the frozen adapter. Predict 1
(`original`) iff score **>=** that model's threshold. Fix batch size, thread
counts, library builds, decoder, tensor ordering and source-code hashes before
execution; record them. Do not round displayed scores before classification or
ranking. Batch-size numerical effects remain reportable, not tunable.

## Feature benchmark versus raw-media benchmark

The primary frozen-model comparison uses the authenticated producer feature
exports and original pair mapping, retaining their historical timing assumption.
A separately labeled raw end-to-end comparison may use the same prespecified
cohort only after complete TRAIN coverage and approval of encoder numerical and
decoder/timing policy. Report its exact and any independently approved numerical
compatibility evidence separately; approximate compatibility is not exact parity.

The current `InternalTrainBackend` accepts only five original SHA-bound TRAIN
MP4s. It cannot implement this benchmark, arbitrary audio substitutions or TEST
admission. A future audited adapter must bind **both** source streams to the
frozen pair identities without introducing remux/re-encoding artifacts or
invented timestamps. Neither an internal score nor a feature-only benchmark
establishes raw production readiness. A TEST-capable reporting adapter must also
replace the current development-only report wording under separate review.

## Metrics and reporting

Report support, coverage, exclusions/failures, TP/FP/TN/FN, accuracy, positive-class
precision, recall and F1 at the frozen threshold for both models. Compute ROC-AUC
and non-interpolated average precision (AP) from continuous full-precision scores
where both classes exist. AP is not trapezoidal PR area. Reuse the tie handling
and numeric conventions of `fusion.evaluation_metrics`/`baseline_metrics` after
the TEST reader/reporting boundary has been reviewed.

Preserve the existing explicit undefined policy: zero-denominator classification
values are 0 **with their reason**, while one-class ROC-AUC/AP are null with a
reason. Publish undefined and failed results; do not describe them as passing.
Optional macro-F1 and descriptive curves must retain fixed labels and the same
cohort. Repeated sources/pairs are dependent; no IID confidence-interval claim
is authorized. Any source-aware uncertainty analysis needs a separately frozen
method before scores are inspected.

Scores measure **uncalibrated original-pairing provenance**, not physical
sound-source causality. Do not label them calibrated confidence/probability,
infer event accuracy from the retained encoder heads, or invent temporal
predictions. Reliability bins, if later reported, are descriptive and do not
establish calibration. No TEST calibration fitting is permitted.

## Execution freeze and acceptance

Before opening TEST predictions, archive independently trusted hashes of this
reviewed protocol, cohort/source manifests, exports, model/threshold files,
approved parity/timing policy, code and runtime profile. Register all planned
analyses and the accountable reviewer. Preserve immutable per-pair full-precision
outputs, failures and final metric derivations for independent recomputation.

No retraining, checkpoint reselection, threshold fitting, calibration fitting,
tolerance selection, decoder selection, cohort revision or other evaluation-driven
tuning is allowed. A failed run is investigated transparently and versioned; it
does not authorize tuning against TEST. Stage 2E stops at this document.

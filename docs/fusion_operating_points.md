# Phase 3 Tasks 3.4–3.5: operating points and score reliability

This batch adds evaluation machinery and offline synthetic tests. It performs no
real-data threshold fitting, experiments, held-out test scoring, neural training,
W&B calls, commit, push or merge. The independently audited Tasks 3.1–3.3 files
and their APIs are unchanged, as are Phase 1/2 and teammate audio/vision code.

Preflight on 2026-10-22 (Asia/Singapore) found the correct GC2 remote and
`feature/fusion-phase3-baselines` branch, exactly the five previously audited
untracked files, and **245 passing tests** (8.221 seconds). Exports and production
pair manifests remain Git-ignored. Phase 2 PR #13 remains a dependency until
merged; the preceding independent audit confirmed it open and unmerged. This
implementation does not assume that local Phase 2 availability means it is merged.

## Public interfaces

| Module | API | Purpose |
| --- | --- | --- |
| `fusion.baseline_protocol` | `validate_score_dataset(records, manifest, audio, vision, *, split, partition, source_reference, artifact_inventory_fingerprint=None)` | Bind records to exactly one complete, validated Phase 2 manifest partition |
| | `ScoreDataset`, `ScoreProvenance` | Immutable score/identity/provenance snapshots and deterministic source fingerprint |
| `fusion.baseline_thresholds` | `threshold_candidates(dataset, *, method)` | Exhaustive validation-primary operating-point sweep |
| | `select_threshold(dataset, *, method)`, `select_thresholds(dataset)` | Fit one/all four methods on validation primary pairs |
| | `freeze_threshold(policy)` | Explicit immutable freeze step with policy fingerprint |
| | `classification_metrics(targets, scores, *, threshold)` | Numeric confusion and classification utility |
| | `ConfusionCounts`, `ClassificationMetrics`, `OperatingPoint`, `ThresholdPolicy`, `FrozenThresholdPolicy` | Frozen typed outputs/evidence |
| `fusion.baseline_evaluation` | `score_reliability(dataset, *, method)` | Fixed-bin descriptive score/target summaries |
| | `threshold_sensitivity(policy)` | Full validation-fitting sweep and equally optimal candidate thresholds |
| | `evaluate_primary(dataset, frozen)` | Train/validation primary report using an already frozen policy |
| | `evaluate_challenge(dataset, frozen)` | Separate same-category false-acceptance report |
| | `evaluation_report_to_json(report)` | Deterministic report JSON, sorted keys and final newline |
| | `frozen_policy_to_json(frozen)` | Deterministic frozen-policy JSON with full validation selection evidence |

All code uses only NumPy and the standard library. No additional packages,
models, calibrators, network calls or secrets are needed.

## Record validation and source binding

Create records through the unchanged probability adapter and `make_score_records`.
Then call `validate_score_dataset` with the same Phase 2 manifest and original
bundles. This calls the existing Phase 2 manifest validator and audited Phase 3
record evaluator. It checks exact pair partition coverage, original row/identity
references, generation provenance, canonical order, duplicate identities, split,
pair type, derived binary target, every method's finite score in `[0,1]`, and
aggregation identifier. Truncated, mixed, missing-score, malformed and test
records fail. Ordinary input lists are copied to tuples; custom container
subclasses are unsupported.

`ScoreDataset` stores only one split/partition. Its SHA-256 fingerprint binds the
protocol version, full score records, pair metadata, generation provenance,
source reference, available source-video overlap context and missing-metadata
diagnostics. Changing scores, identities or context changes the fingerprint.
Cross-modal probabilities remain the only predictive features; metadata and true
targets enter only validation, fitting and diagnostics. Empty annotations retain
the audited UNKNOWN semantic status and eligibility masks; fitting the separate
original-pair provenance target invents no semantic/event supervision.

The ingestion factory validates against the current bundles. Typed dataset and
policy constructors/revalidation check structural consistency, not external
authenticity: they are not signed validation certificates. Fingerprints cannot
prove that a producer attached correct IDs, that supplied predictions came from
the claimed checkpoint, or that a supplied source reference is truthful. The
optional artifact inventory fingerprint is an explicit declaration from the
Phase 2 integration context, not a claim that these APIs freshly hashed files.
Real experiment callers must perform the existing artifact/reference verification
before ingestion and preserve its completion report and hashes. There is no
untrusted JSON-to-policy loader in this batch; output JSON is archival evidence.

## Validation fitting and exhaustive thresholds

Fitting accepts only a validated, nonempty **validation primary** dataset with
both binary target classes. Train, test, challenge, mixed and one-class sources
fail explicitly with `ContractError`; no fallback deployment threshold is invented.
One-class operating-point metrics and nonempty candidate sweeps remain available
as diagnostics even though one-class threshold fitting is unsupported.

Each of the four existing methods is fitted separately. The fixed objective is
`maximum_validation_binary_f1`. Classification is **original iff score >=
threshold**. F1 is compared as the exact rational `2TP / (2TP + FP + FN)`.
Among exactly equal objective values, choose the **largest threshold**; this
deterministically favors fewer acceptances at the same F1.

For U distinct observed validation scores `s1 < ... < sU`, candidates are:

```text
s1, s2, ..., sU, nextafter(sU, +infinity)
```

These represent all U+1 possible prediction partitions under `>=`. The first
candidate accepts all examples. The final finite float64 sentinel rejects all,
including when the observed maximum is 1. Tied scores cannot be separated.
Thresholds in gaps or below the minimum yield no additional prediction partition,
so they are deliberately omitted. All-negative candidates are supported even
though, with both classes present, maximum binary F1 always favors some positive
acceptances. The sweep uses sorting and cumulative counts, O(N log N).

A constant score yields exactly two candidates. With both classes present,
maximum F1 selects the observed constant (all-positive); the all-negative
candidate is still reported. A threshold may exceed 1 by one float64 ULP for
the sentinel. Score bounds remain `[0,1]`; threshold values are operating cutoffs,
not probability estimates.

`ThresholdPolicy` stores method, cutoff, objective, tie rule, validation class
counts, selected confusion/metric result, candidate count, source fingerprint,
full immutable validation evidence and protocol version. Public policy consumers
re-run class-owned validation, recompute the optimum from that evidence and
check fingerprints. Reconstructed altered policies, missing fields, misleading
instance validation hooks and unsupported subclasses cannot waive these checks.
This prevents inconsistent policy reuse; it is not a defense against arbitrary
code execution or an external producer fabricating an entirely new dataset.

## Classification metrics and undefined cases

For `positive = original recording pair`, the confusion counts are TP/TN/FP/FN.
With P actual positives, N actual negatives and T=P+N:

| Metric | Definition | Zero/undefined convention |
| --- | --- | --- |
| Accuracy | `(TP+TN)/T` | `None` if T=0 |
| Precision | `TP/(TP+FP)` | 0 if no predicted positives |
| Recall | `TP/P` | 0 if P=0 |
| Binary F1 | `2TP/(2TP+FP+FN)` | 0 if denominator is zero |
| Negative-class F1 | `2TN/(2TN+FP+FN)` | 0 if denominator is zero; used in macro-F1 |
| Macro-F1 | Mean of positive and negative F1 | Fixed labels `[0,1]`, including absent classes; empty input gives 0 |
| Specificity | `TN/N` | 0 if N=0 |
| Balanced accuracy | `(recall+specificity)/2` | `None` if either actual class is absent |
| Positive prediction rate | `(TP+FP)/T` | `None` if T=0 |

Thus a perfectly rejected negative-only set has binary F1 0 and macro-F1 0.5,
not 1. Its balanced accuracy is undefined. This fixed-label convention is
deliberate and must accompany reports. Class counts remain explicit.

`evaluate_primary` also retains the earlier threshold-independent metrics and
all/positive/negative score distributions. Average precision remains the earlier
noninterpolated stepwise PR integral, **not trapezoidal PR-AUC**. One-class and
empty ranking metrics retain explicit undefined states under that contract.

## Calibration limitations and descriptive reliability

Top-1, dot-product, cosine and Jensen–Shannon similarities are semantic agreement
scores, not estimates of `P(original pair)`. High agreement may occur for shared
background confidence or a same-category substituted soundtrack. Scores cannot
establish physical causality or waveform synchronization.

`score_reliability` uses ten **fixed** bins with edges `i/10` for i=0..10.
Each is left-inclusive/right-exclusive, except `[0.9,1.0]` includes 1. Exactly
0.1 goes into the second bin. Every bin reports bounds, closure, count, positive
and negative composition, mean score and observed original-pair frequency.
Empty bins retain zero counts and `None` mean/frequency. The report explicitly
labels its score meaning `semantic_similarity_not_a_probability`.

Observed bin frequencies are descriptive and depend on sampling ratios, source
reuse and split composition; they are not fitted probabilities, a calibration
curve for a probability predictor, or an independent generalization estimate.
This batch intentionally implements **no ECE, Brier score, isotonic regression,
Platt scaling or probability calibrator**. A future probabilistic extension needs
an explicit prediction-probability contract and separate fitting/evaluation or
cross-fitting protocol before those metrics would be justified.

`threshold_sensitivity` reports every validation operating point and the exact
set of equally optimal candidate thresholds. This describes changes in counts
and metrics across cutoffs on the fitting data. It does not claim bootstrap
stability, uncertainty bounds, a held-out sensitivity estimate, or calibrator
generalization. Challenge scores never select, revise or tie-break thresholds.

## Separate challenge evaluation

Pass only the same-category different-source partition to `evaluate_challenge`.
It requires a frozen policy fitted exclusively on validation primary pairs.
All challenge provenance targets must be 0; incompatible record types, positive
targets, mixed partitions and wrong split declarations are rejected.

For C challenge pairs, A scores at or above the cutoff are classified original:

```text
false acceptance rate = A / C
rejection rate        = (C - A) / C
```

The report carries C, zero positive count, C negative count, accepted/rejected
counts, both rates, score distribution, fixed-bin composition, method, frozen
threshold, split, fingerprints, complete provenance and source-overlap warnings.
For C=0, both rates are `None`. No standalone challenge ROC-AUC, AP or F1 is
produced, and challenges are never pooled into primary headline metrics.

## Lifecycle and reproducible reports

The supported lifecycle is:

1. Use the established Phase 2 artifact/reference verification and unchanged
   Phase 3 scoring APIs on an authorized train/validation partition.
2. Bind complete score records to the validated manifest with
   `validate_score_dataset`.
3. Fit validation-primary thresholds, one per method, with the fixed objective.
4. Explicitly call `freeze_threshold`; save its canonical JSON and validation
   score/artifact evidence before any future final-test work.
5. Produce separate train/validation primary and challenge diagnostic reports
   using that frozen policy. No frozen policy means no report.

Reports carry fitting/evaluation source fingerprints and provenance, objective,
tie rule, fit class counts and reused fitting pair/source-video counts. Validation
reports explicitly disclose that validation was used for threshold selection and
make no independent performance/generalization claim; training reports are
labeled training diagnostics. Merely changing a source reference does not hide
pair/source reuse counts.

`evaluation_report_to_json` emits sorted-key UTF-8-safe JSON text with deterministic
float representations, explicit nulls and no clocks, random IDs or inferred
environment fields. `frozen_policy_to_json` additionally preserves the complete
selection evidence. These are output serializers, not validation credentials for
arbitrary loaded reports.

No API in these new modules accepts a test `ScoreDataset`, even with a frozen
policy or reconstructed objects. No final-test entry point is provided. A future
final-test operation must be a separately authorized evaluation step with an
already frozen, verified policy and previously frozen model/pair configuration;
it must never refit from test scores or labels. The pure numeric helper, like
the audited earlier array-only helpers, cannot infer provenance from arrays and
must not be used to bypass dataset guards.

Phase 2's available source-video overlaps and missing-metadata context propagate
into datasets and reports, rather than being silently treated as source-disjoint.
Historical official split overlap counts are 10 train/val, 8 train/test and 0
val/test; new report warnings come from the supplied validated metadata, not
hardcoded constants. Supplementary source-disjoint analyses cannot undo prior
unimodal encoder exposure or silently replace the official partition protocol.

## Usage for future Tasks 3.6–3.7

The following illustrates APIs only; it is not a real experiment executed here:

```python
from fusion.baseline_data import extract_pair_probabilities
from fusion.baseline_metrics import make_score_records
from fusion.baseline_protocol import validate_score_dataset
from fusion.baseline_thresholds import select_thresholds, freeze_threshold
from fusion.baseline_evaluation import (
    evaluate_primary, evaluate_challenge, evaluation_report_to_json,
    frozen_policy_to_json, threshold_sensitivity,
)

# Existing validated bundles and Phase 2 validation manifest, with verified files.
data = extract_pair_probabilities(val_manifest, audio, vision, split="val")
primary = validate_score_dataset(
    make_score_records(data.primary), val_manifest, audio, vision,
    split="val", partition="primary", source_reference="verified-val-score-artifact",
    artifact_inventory_fingerprint=verified_inventory_fingerprint,
)
challenge = validate_score_dataset(
    make_score_records(data.challenge), val_manifest, audio, vision,
    split="val", partition="challenge", source_reference="verified-val-score-artifact",
    artifact_inventory_fingerprint=verified_inventory_fingerprint,
)
for selected in select_thresholds(primary):
    frozen = freeze_threshold(selected)
    frozen_json = frozen_policy_to_json(frozen)
    primary_json = evaluation_report_to_json(evaluate_primary(primary, frozen))
    challenge_json = evaluation_report_to_json(evaluate_challenge(challenge, frozen))
    sensitivity = threshold_sensitivity(selected)
    # Later experiment code persists evidence/reports and logs the approved run.
```

Tasks 3.6–3.7 must establish the actual experiment configuration, artifact hashes,
negative budgets, source reuse and validation allocation/reuse before executing
real evaluations. Freeze any model/method selection and operating policies before
a separately authorized final test. Do not use test-derived model selection,
negative mining, architecture choice, cutoff choice or calibration.

All future actual GC2 training/evaluation runs must use:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

Log the existing mandatory experiment metadata and distinguish noninterpolated
AP from any separately defined PR-AUC. Preserve frozen-policy/score/report
fingerprints, validation-reuse statements and separate challenge results. Synthetic
unit tests and structural checks create no W&B run. No API keys are read, stored,
printed or exposed by these modules or this implementation batch.

## Verification

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
git diff --check
```

New offline synthetic tests exercise candidate exhaustiveness against an
independent prediction-partition oracle across all 81 four-score assignments,
exact F1 tie resolution, all four methods, all-positive/all-negative and constant
scores, 256 confusion-matrix cases against a mathematical oracle, zero division,
one-class/empty behavior, immutable policies and malformed reconstruction attempts,
split/partition/duplicate/coverage rejection, source binding, score ranges,
fixed-bin boundaries, no raw-score probability metrics, challenge acceptance,
validation reuse, threshold sensitivity and deterministic JSON. Existing tests
retain Phase 1/2 and earlier Phase 3 compatibility coverage.

Final verification on 2026-10-22:

- **279 tests passed** in 9.941 seconds: the 245 existing tests plus 34 new
  operating-point tests. The complete required unittest command ran in the
  existing `.venv` with bytecode generation disabled.
- An additional independent randomized synthetic probe matched **120 threshold
  fits and 813 operating-point confusion counts** against simple mathematical
  oracles. No real data were used in this probe or the new tests.
- SHA-256 comparisons confirmed that all five audited Tasks 3.1–3.3 files are
  byte-for-byte unchanged. Existing tracked files have no diff.
- New files pass syntax parsing, imports, trailing-whitespace and final-newline
  checks. `git diff --check` passes. Runtime dependencies remain only NumPy and
  the Python standard library; dependency files are unchanged.
- Git scope is exactly the five original untracked files plus three new runtime
  modules, this document and one new test file: ten untracked files total, no
  staged changes. Local exports and production manifests remain ignored and
  unchanged; this batch did not open their prediction arrays or fit real cutoffs.
- No W&B run, API-key access, full real-data experiment, final-test scoring,
  neural training, commit, push or merge occurred. Tasks 3.6–3.7 are ready to use
  these APIs in a later authorized batch under the documented experiment protocol.

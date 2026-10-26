# Phase 7 Tasks 7.3–7.4 — authenticated TRAIN reporting

Implemented on **2026-10-26, Asia/Singapore**, in
`NUS-AIS-GCs/Audio-Video-Confidence`, branch `feature/phase7-evaluation`.
Scope is deterministic reporting and authenticated **TRAIN-primary** integration.
No validation, TEST or same-category challenge performance is evaluated here.

## Dependency and preservation preflight

HEAD remains `298097d21cbf0397c0ad43e92ca80aef98771c0e`, the Phase 6 feature tip.
The main merge base remains `abbcb97fb4c3cb34bdcca913e149f994eb154e4b`.
Phase 6 remains an intact unmerged dependency; its source, tests and documents
are unchanged. The audited Phase 7 Tasks 7.1–7.2 admission/metric implementation
and tests are also unchanged by this batch.

The 590-file historical preservation baseline was checked before work. All
protected files matched, including checkpoints, embeddings/probability exports,
pair manifests and score archives. The existing untracked
`requirements-fusion-training.txt` remains 141 bytes with SHA-256
`0b6230b82b66ec1d2537681d7c5cdc7e0a91e660bef01ba154fe8a68bdd23e08`.
The Git index is unchanged and there are no staged changes.

Only four source/document files are added:

- `fusion/evaluation_reporting.py`
- `fusion/evaluation_plots.py`
- `tests/test_fusion_evaluation_reporting.py`
- This document.

No historical implementation, frozen threshold, checkpoint, manifest or metric
definition is edited. Matplotlib was already listed in `requirements-video.txt`;
version **3.10.7** was installed in the local virtual environment. Requirements
files were not changed. Runtime/dependency caches remain ignored.

## Architecture and authenticated admission

The [audited foundation](phase7_evaluation_foundation.md) remains the authority
for metric mathematics, target meanings and archive authentication.

| Component | Responsibility |
|---|---|
| Existing `evaluation_admission` | Authenticate the committed freeze and transitive archive/checkpoint/source bindings; admit complete TRAIN-primary predictions only |
| Existing `evaluation_contracts` | Enforce exact frozen model/threshold association, trust levels and identity/label/source-row alignment |
| Existing `evaluation_metrics` | Compute the independently verified metrics, curves and diagnostic bins |
| New `evaluation_reporting` | Require all three authenticated admissions, build a portable comparison, publish complete artifacts without overwriting existing reports |
| New `evaluation_plots` | Draw already-computed coordinates, counts and bins using noninteractive Agg; no metric computation or threshold fitting |

Public interfaces:

```python
build_train_comparison(admissions, *, split="train", partition="primary")
write_train_comparison(admissions, *, repository_root, output_directory,
                       split="train", partition="primary")
generate_train_report(*, repository_root, output_directory,
                      split="train", partition="primary")
```

`generate_train_report` loads exactly JSD, E3 and A2 through the restricted
admission reader. The other two entry points accept exactly three authenticated
admissions, not raw scores, exploratory sets, report dictionaries or caller trust
flags. Unknown/duplicate identities, incorrect checkpoints, mixed trust,
incomplete/mismatched samples and non-TRAIN/non-primary declarations reject.

All models pass fresh authentication and full alignment **before any metrics**.
Each existing evaluator repeats authentication. Alignment is checked again after
computation and after rendering, before publication. Every report fingerprint
must still match the authenticated evidence. Changed files or admission content
cannot reuse an earlier success to publish an authoritative report.

The fixed presentation order is **JSD, E3, A2**. No metric-dependent sorting,
winner selection, threshold optimization or model substitution is implemented.
The exact frozen thresholds remain:

| Model | Frozen threshold | Score interpretation |
|---|---:|---|
| JSD | `0.32543530977887525` | Mean base-2 JSD semantic similarity; not a probability |
| E3 | `0.009579099009671126` | Uncalibrated float64 sigmoid estimate |
| A2 | `0.0026640388978268056` | Uncalibrated float64 sigmoid estimate |

Decisions retain `score >= threshold`. Threshold provenance identifies the
validation-primary fitting cohort, frozen objective/tie rule, evidence digest
and historical prediction fingerprint. No validation scores are evaluated in
this batch. The historical JSD policy contains development fitting evidence;
reading that pinned policy to verify its threshold is not a new validation
evaluation. No TEST or challenge score archive is opened for metric computation.

## Portable report schema and publication

The JSON schema is `gc2-phase7-authenticated-train-report-v1`.

| Field | Meaning |
|---|---|
| `evidence_authenticity` | Always `authenticated_archived_train` after successful admission |
| `split`, `partition`, `cohort` | Explicit TRAIN-primary diagnostic scope |
| `model_order`, `model_selection_performed` | Fixed order and explicit `false` selection status |
| `count`, `support`, target meanings | Complete aligned cohort size and binary provenance class support |
| `alignment` | Canonical identity digest, identity count and authenticated pair-manifest digest |
| `producer_versions` | Frozen audio and visual producer identities |
| `models[].artifact` | Selected checkpoint or baseline output-manifest role and SHA-256 |
| `models[].threshold_provenance` | Exact threshold, decision rule and frozen fitting evidence |
| `models[].prediction_fingerprint`, `provenance` | Authenticated predictions, score archive and overlap-evidence digests |
| `models[].metrics`, `curves`, `confusion_matrix`, `reliability` | Existing validated outputs, without redefining metric mathematics |
| `undefined_policy`, `scientific_limitations` | Explicit undefined conventions and scientific interpretation limits |
| `rendering` | Matplotlib version, Agg backend, font and DPI |

Portable reports contain no absolute/private machine paths, output locations,
timestamps, run IDs or tracking credentials. Source files are identified by
semantic role and digests; even machine-local relative evidence references are
omitted. JSON preserves full numerical precision and rejects NaN/Infinity.
Per-sample metadata is not duplicated into the report; canonical identity and
prediction fingerprints retain the binding to the authenticated inputs.

Output must be a **fresh child directory under `_local/phase7_reporting/`**,
which is covered by the repository's existing `_local/` Git ignore rule.
Absolute paths, traversal, the reporting root itself, other evidence directories
and existing destinations reject. Neither historical evidence nor an earlier
report can be overwritten. Files are rendered in a new owned staging directory
and published together only after success. Failed rendering/authentication
removes only that staging directory and leaves no destination report.

Each report directory contains exactly:

| Artifact | Content |
|---|---|
| `report.json` | Full portable comparison data |
| `manifest.json` | Sorted artifact names, SHA-256 and byte sizes for the JSON and five PNGs |
| `roc_curve.png` | Observed ROC threshold coordinates for all three models |
| `precision_recall_curve.png` | Observed PR threshold coordinates and noninterpolated AP labels |
| `confusion_matrices.png` | Three raw-count matrices, exact thresholds and fixed label order |
| `reliability_diagnostics.png` | Occupied score bins with sample counts and explicit interpretation |
| `metrics_table.png` | Classification/ranking metrics, Brier diagnostics, class support and thresholds |

PNG output uses `FigureCanvasAgg`, bundled DejaVu Sans, 120 DPI, explicit style
defaults and fixed metadata. It does not use interactive pyplot windows or GUI
backends. Byte-level determinism was verified in the same installed runtime;
cross-version raster identity is not claimed. The PNG metadata has no creation
timestamp. Dependency-load and rendering failures are explicit errors.

## Metric-to-plot definitions

- **ROC:** x is false positive rate, y is true positive rate/recall. Only the
  existing tied-score threshold coordinates are drawn, with no interpolation,
  smoothing or resampling. Legends identify each model and its ROC-AUC.
- **PR:** x is positive-class recall, y is positive-class precision. The legend
  reports **average precision**, not trapezoidal PR-AUC. Only stored coordinates
  are drawn.
- **Confusion matrices:** rows are true labels and columns predicted labels;
  order `[0,1]` means `[[TN,FP],[FN,TP]]`. Cells show integer counts, with common
  scaling, original/substituted labels and each exact model threshold.
- **Reliability:** x is the mean score and y the observed original-pair frequency
  in each occupied fixed bin. Empty bins are not replaced with zeros. Bin counts
  are labeled. E3/A2 are explicitly uncalibrated diagnostics; `y=x` is a reference
  line, not a fitted calibration curve. JSD has descriptive similarity bins and
  no probability-reference diagonal.
- **Table:** accuracy, precision, recall, positive F1, fixed two-class Macro-F1,
  ROC-AUC, AP, Brier diagnostic, class support and exact frozen thresholds. Display
  metrics use six decimals; JSON retains full precision. Zero-filled undefined
  classification values are marked, while unavailable metrics show their reason.

One-class ROC/AP/curves follow the audited foundation's explicit undefined policy.
Missing curves are annotated and get no invented points. JSD Brier is unavailable
with `score_not_a_probability`. Every figure displays TRAIN scope, authenticity,
support, pair-manifest provenance and the generalization/causality limitations.
No confidence intervals, significance tests or attention-derived temporal outputs
are introduced.

## Internal Helper Interfaces and Trust Boundaries

The report builder returns an ordinary mutable Python dictionary for inspection.
Callers can change its contents, including textual authenticity fields. That
dictionary is not an authenticated publication artifact and must not be used as
independent proof of model provenance. The supported report writer does not
accept arbitrary caller-supplied report dictionaries: it reconstructs the report
from authenticated E3/A2/JSD admissions and rechecks evidence before publication.

The private curve-drawing helper assumes its coordinates came from validated
metrics. Direct calls with externally injected NaN or other nonfinite coordinates
are not independently rejected by that helper. It is internal, not a supported
public plotting or publication API. The supported reporting pipeline generates
coordinates from validated metric contracts and does not expose that direct
injection route. These guarantees do not mean that all helper calls are safe for
arbitrary caller input.

## Reproduction

Use the existing Python environment and archived evidence. Matplotlib 3.10.7 is
the tested rendering version; install it into this repository's environment if
needed, without modifying the preserved requirements file:

```powershell
uv pip install --python .venv/Scripts/python.exe "matplotlib==3.10.7"
$env:MPLCONFIGDIR = Join-Path $PWD '_local/phase7_reporting/.matplotlib-cache'
.\.venv\Scripts\python.exe -B -m fusion.evaluation_reporting `
  --repository-root . `
  --output-directory _local/phase7_reporting/train_report_new
```

Choose another fresh output name for each invocation. There is no CLI split,
threshold, score-file, checkpoint or model-selection override. Matplotlib's cache
is kept below the ignored reporting root by the command above.

The Codex sandbox denied reading the newly installed Matplotlib package during
the initial check. The rendering tests, smoke and full suite subsequently passed
outside that sandbox. No ACL, operating-system policy or historical source file
was changed to bypass the restriction. Missing/unreadable plotting packages fail
with an explicit dependency/runtime-access error; they do not publish a report.

## Verified TRAIN smoke evidence

Two genuine reports were generated in ignored directories:

- `_local/phase7_reporting/train_smoke_20261026_01/`
- `_local/phase7_reporting/train_smoke_repeat_20261026_01/`

Both authenticated all **6,678** primary identities, comprising **3,339 original**
and **3,339 substituted** pairs. All seven corresponding output files are
**byte-identical**. All five figure types were visually inspected. Outputs are
retained locally for review, are temporary smoke evidence, and are not staged.

The only score archives opened for evaluation were:

- Phase 3 `scores/train_primary.csv`
- Selected E3 `train_primary_scores.csv`
- Selected A2 `train_primary_scores.csv`

The smoke checked model labels, artifact/archive digests, frozen thresholds,
alignment, portability and absence of encoder execution, threshold fitting and
W&B initialization. Verification is recorded in the ignored
`_local/phase7_reporting/smoke_verification_20261026.json`.

The following are **TRAIN diagnostics**, not validation or held-out results:

| Model | Accuracy | Macro-F1 | ROC-AUC | AP |
|---|---:|---:|---:|---:|
| JSD | 0.991314765 | 0.991314110 | 0.999948426 | 0.999947968 |
| E3 | 0.994010183 | 0.994009968 | 0.999996412 | 0.999996412 |
| A2 | 0.988918838 | 0.988917477 | 1.000000000 | 1.000000000 |

Their rounded display does not alter JSON precision or thresholds. The high
training ranking metrics do not establish generalization or justify a new winner
selection. No conclusions about final TEST or same-category challenge performance
follow from these results.

## Source overlap and final TEST readiness

Official assignments remain unchanged. The archived metadata audit counts unique
underlying source videos: **10 TRAIN/validation**, **8 TRAIN/TEST**, and
**0 validation/TEST** overlaps. These are different annotation rows referring to
the same source videos, not merely pair-row duplicates or naming collisions.

The eight repeated TRAIN/TEST source VideoIDs are:

```text
-9_5_2i4igQ
-PljzzluWNw
-WOcO4B-Jis
0NmA9tKpBdw
0f_iTO5t3hU
0qvQIuQoODo
1cmo1SGxJnc
7iIBkME4bSM
```

These identities are recorded in the pre-existing Phase 3 `dataset_summary.json`
and [reference/data audit](fusion_real_data_validation.md). Multiple annotations
of a shared source do not increase the unique-video overlap count. Annotation
identities are disjoint across official splits, but source content is not.

Before any final TEST or challenge evaluation:

1. Obtain explicit authorization and review a separate cohort-specific admission
   path; this implementation supplies no held-out admission/reporting override.
2. Resolve the Phase 6 dependency through normal review. Freeze exact models,
   thresholds, producer/preprocessing bindings, report schema and metric policies.
3. Authenticate the complete intended cohort and original source rows before
   computing metrics. Keep challenge analysis separate from primary results.
4. Disclose source overlap, development-selection reuse, shared-source dependence
   and single-seed evidence. Preregister any supplemental source-disjoint analysis;
   later filtering cannot undo previous encoder exposure.
5. Require independent calibration evidence for calibrated-confidence claims.
   Require genuine heads/labels for event or temporal tasks and verified raw
   encoder parity for end-to-end raw-media claims. Authenticated pairing archives
   do not establish physical sound-source causality.

## Regression and preservation

```powershell
$env:MPLCONFIGDIR = Join-Path $PWD '_local/phase7_reporting/.matplotlib-cache'
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Final result: **685 tests: 683 passes, two unavailable-CUDA skips, zero
failures/errors, Python exit code 0**, in 95.918 seconds. This includes all 656
existing discovery entries and **29 new reporting tests**. The tests cover exact
plot mappings/orientation, reliability bins, undefined displays, strict portable
JSON, repeated artifact bytes, dependency/render failures, invalid attribution,
checkpoint/archive changes, missing evidence, sample mismatches, mixed trust,
forbidden cohorts, frozen thresholds, no selection and preservation of source
files. Changes during rendering prevent publication.

All **590 protected files / 264,975,100 bytes** remain unchanged. The pre-existing
six audited Phase 7 files, requirements bytes and Git index also match their
preflight hashes. No historical checkpoint, score, manifest or export was
rewritten. Generated figures/reports and logs remain ignored; no files were
staged. The full test log is
`_local/phase7_reporting/full_suite_20261026.log`.

**Tasks 7.3–7.4: COMPLETE. READY FOR PHASE 7 REPORTING AUDIT.**
No training, validation/TEST/challenge evaluation, threshold tuning, W&B run,
commit, push, merge or PR was performed. Work stops after these two tasks.

# Phase 7 evaluation foundation — Tasks 7.1–7.2

Audit date: **2026-10-26, Asia/Singapore**. Repository:
`NUS-AIS-GCs/Audio-Video-Confidence`; branch: `feature/phase7-evaluation`.
Scope: readiness audit, reusable correspondence contracts and synthetic tests.
No held-out TEST or same-category challenge scoring is authorized by this work.

## Preflight and dependency status

HEAD is `298097d21cbf0397c0ad43e92ca80aef98771c0e`, equal to the Phase 6
remote-tracking tip. After `git fetch --no-tags origin`, `origin/main` remains
`abbcb97fb4c3cb34bdcca913e149f994eb154e4b`, the Phase 5 merge. It is the merge
base of this branch and main. Main is an ancestor of HEAD; HEAD is ahead by
`cac2cfe` (Phase 6 foundation) and `298097d` (main synchronization).
Phase 6 is therefore an **unmerged dependency**, consistent with its pending
approval. No merge or approval is inferred from its presence on this branch.
Any eventual Phase 7 PR against current main would also include Phase 6 until
that dependency is merged. This task creates no PR and changes no history.

The initial tracked worktree/index was clean. The pre-existing untracked
`requirements-fusion-training.txt` is preserved, including its 141-byte CRLF
content and current SHA-256
`0b6230b82b66ec1d2537681d7c5cdc7e0a91e660bef01ba154fe8a68bdd23e08`.
This is the measured preflight/final value; the older Phase 6 documentation's
different digest is not the current file baseline. This task did not change it.
Only new Phase 7 files are introduced; no Phase 1–6 file is edited.

## Readiness matrix

The [original proposal](../Documentations/project_proposal_report_V3.md), sections
1.1, 2 and 4, calls for event class, approximately per-second temporal presence,
pairing correspondence, negative difficulty, and confidence reliability.
These are separate scientific targets.

| Proposed task | Available predictions / trained heads | Available labels | Readiness and interpretation |
|---|---|---|---|
| Primary binary correspondence | JSD similarity; trained E3 MLP and A2 cross-attention each produce one pooled binary logit/score per pair | Original-pair target 1; cross-category substituted target 0, from Phase 2 provenance | **READY for contract audit**; future final evaluation requires explicit authorization and frozen inputs. This is pairing provenance, not source causality. |
| Same-category substitutions | Frozen binary heads can score these inputs; historical JSD challenge archive exists | Provenance target 0, semantic category compatibility retained; no physical-source identity truth | **DEFERRED**. Kept separate from primary distribution; future false acceptance/rejection analysis needs authorization. No new challenge scoring here. |
| Hard cross-category negatives | Policy and pair types exist | Production run generated zero hard negatives | **PARTIAL**. Requires approved difficulty policy, fixed cohorts and adequate labels before evaluation. |
| Temporally shifted mismatch | No trained synchrony/shift head or supported shift-pair protocol | Temporal status is `NOT_ASSESSED`; no shift ground truth | **BLOCKED** for timing claims. Original bin alignment is not a synchronization measurement. |
| Unimodal event recognition | Trained audio and visual 29-way per-second heads; archived `(N,10,29)` probability exports | 28 event categories plus background, derived from AVE joint annotations | **PARTIAL**: exported predictions support a future, separately scoped AVE category benchmark. Actual selected encoder checkpoints/configuration and raw parity remain unavailable. Does not validate independent audible/visible labels. |
| Fused clip event classification | E3/A2 have only a binary correspondence head | Original AVE clip category exists; swaps do not define a unique fused event target | **BLOCKED**: needs a genuine trained class head, original-pair eligibility/aggregation policy and validated predictions. Binary scores cannot supply class predictions. |
| Fused per-second joint event presence | No E3/A2 temporal presence head; masked pooling discards per-bin outputs | Original joint AVE event/background interval labels exist; swapped pairs lack fused presence targets | **BLOCKED**: train/validate a temporal head with explicit eligible labels. Attention weights, masks and ten input bins are not temporal predictions. |
| Separate audio audibility / visual visibility | Unimodal heads predict AVE categories, not independently annotated modality presence | AVE intervals are shared joint annotations, not separate modality truth | **BLOCKED** for these claims without suitable independent labels and validated heads/protocols. |
| Correspondence reliability | E3/A2 sigmoid scores; no independent calibrated model | Binary pairing labels available | **PARTIAL**: Brier and reliability diagnostics supported with explicit uncalibrated label. Calibrated-vs-raw comparison/ECE needs a preregistered independent calibration protocol and evidence. JSD is not a probability. |
| Raw-media end-to-end evaluation | Phase 6 feature bridge is implemented; raw encoder execution is blocked | Original annotations/pairing metadata exist | **BLOCKED** by missing producer encoder checkpoints, BEATs base weights and audio normalization/configuration; actual raw-to-export parity remains unverified. |
| Demo event/timeline outputs | `src/pipeline.py` and `src/models/*` use annotation lookups, random inputs and heuristics | Outputs can directly reflect the same annotations used as supposed truth | **NOT VALID evaluation predictions**. Never score annotation lookups as trained predictions. |

`audio/model.py`, `audio/beats_lora.py` and `vision/model.py` implement genuine
per-second class heads. Their prediction tensors are different from the binary
fusion outputs in `fusion/mlp_model.py` and `fusion/cross_attention_model.py`.
Existing `audio/evaluate.py` and `vision/evaluate.py` implement category metrics,
but their command-line paths default to TEST and can initialize tracking; they
were **not executed or imported** for this audit. The demo also uses inclusive
interval ends in places; it is not the authority for AVE interval semantics.

## Evidence inventory and label semantics

| Evidence | Local location and meaning |
|---|---|
| Verified reference metadata | `_local/datasets/AVE_reference/`, `_local/fusion/ave_reference_verified/`; author reference commit `93f843fed72fb17394df76330df958b0c2dbde76` is recorded in the existing audit |
| Producer exports | `_local/embeddings/audio_final/`, `_local/embeddings/vision_final/resnet18_v1/`; embeddings `(N,10,256)` / `(N,10,512)`, probability exports `(N,10,29)`, ordered ID CSVs, class maps and producer declarations |
| Frozen pair manifests | `_local/fusion/task28_real/{train,val,test}_pairs.json`, `validation.json`; original source rows, sampling policy, pair type and split retained |
| Phase 3 JSD evidence | `_local/fusion/phase3_real_experiments/run_20261022_01/`; `scores/{train,val}_primary.csv`, `policies/jensen_shannon.json`, reports/comparison, input/output manifests and tracking receipts |
| Historical challenge evidence | Phase 3 separate challenge score/report files exist; their metrics were not recomputed or used for Phase 7 selection |
| Selected Phase 4 E3 | `_local/fusion/phase4_improvement/study_20261024_seed42_01/E3/`; `best.pt`, metadata, epoch selection, frozen operating point, TRAIN/validation score archives, manifests and checkpoint verification |
| Selected Phase 5 A2 | `_local/fusion/phase5_ablations/study_20261025_seed42_01/A2/`; same evidence structure; selected attention variant, not best learned validation-AUC model |
| Consolidated historical freeze | [Phase 5 freeze JSON](fusion_cross_attention_phase5_freeze.json) and [summary](fusion_cross_attention_phase5_summary.md); selection protocol, validation results, hashes and existing Forge artifact versions |
| Phase 6 scope/limitations | [Feature inference foundation](fusion_end_to_end_inference_foundation.md) and [integration audit](fusion_end_to_end_integration.md); feature inference is distinct from actual raw encoder readiness |

The positive class is **1 = original audio-video pairing provenance**. Negative
**0 = substituted pairing provenance**. Neither label establishes whether a
visible object physically caused the sound. Semantic compatibility and temporal
synchrony are separate annotations/statuses; they must not replace pairing truth.
Ground truth, category IDs, source IDs and annotation intervals are reporting
metadata, never predictive features.

AVE annotations use integer half-open intervals: second `t` has the annotated
event when `start <= t < end`, otherwise background. Categories have indices
0–27; background is index 28. These describe an annotated joint audio-visual
event in the original clip. They are not independent audio audibility, visual
visibility, or causal sound-source labels. Multiple annotation rows may refer
to one source video. The official empty interval `[0,0)` for TRAIN sample
`VWi2ENBuTbw__08` remains a valid provenance-positive annotation with unknown
semantic status and no invented event supervision. Annotation masks must not
become producer-validity masks. Swapping sources supplies no fused temporal
presence target.

## Splits and source-overlap limitations

The existing [reference/data audit](fusion_real_data_validation.md) and Phase 3
`dataset_summary.json` establish:

| Official split | Annotation rows | Primary positives | Primary easy negatives | Separate challenges |
|---|---:|---:|---:|---:|
| TRAIN | 3,339 | 3,339 | 3,339 | 3,339 |
| Validation | 402 | 402 | 402 | 402 |
| Held-out TEST | 402 | 402 | 402 | 402 |

These are inventory counts, **not Phase 7 predictions or TEST measurements**.
There are 4,143 unique annotation identities across 4,097 source videos; no
annotation identity crosses official partitions. Nevertheless, **10 source
videos overlap TRAIN/validation, 8 overlap TRAIN/TEST, and 0 overlap
validation/TEST**, identically in both modality metadata audits. Pair construction
stays within each split and forbids same-source substitutions, but this does not
remove shared source content across official partitions.

Preserve official manifests and disclose that the primary benchmark is not
source-disjoint. Any supplementary source-disjoint cohort must be prespecified
and versioned separately, account for *both* pair sources, and never silently
rewrite historical assignments. Filtering a later fusion evaluation cannot undo
earlier encoder exposure. Multiple pairs share sources, so pair-level observations
are also not independent; no significance claim follows from a single seed.

## Frozen models and threshold provenance

All scores increase toward original pairing. JSD is the mean over ten bins of
`1 - JSD_base2` between the two 29-class distributions; it has no trained
checkpoint. E3/A2 use the historical float64 sigmoid of one pooled logit,
`exp(-logaddexp(0, -logit))`, without independent calibration evidence.

| Model | Exact frozen threshold | Artifact SHA-256 | Operating-point evidence SHA-256 |
|---|---:|---|---|
| JSD | `0.32543530977887525` | `c0a33292cd5c376f4c2267bbc91a4845c246c006d581abc9f400430a3d7b9110` (output manifest) | `f57a8e78e70fe2e1194a449e9edf4e10de823d8b396e47893491da47b70f0f80` |
| E3 | `0.009579099009671126` | `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925` (`best.pt`) | `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad` |
| A2 | `0.0026640388978268056` | `0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12` (`best.pt`) | `16994e04c6a8421ec8bc22c1fac33691a4a0228ba544f6e57b0730f98c95b384` |

The JSD operating-point evidence is `policies/jensen_shannon.json`; E3/A2 use
their respective `frozen_operating_point.json`. Each threshold maximized
**validation-primary binary F1**, with largest-threshold resolution of exact
ties. E3 selected epoch 17; A2 selected epoch 18. No threshold is transferred
between models, rounded for decisions, optimized or refitted here. The rule is
**`score >= threshold`**, including exact equality.

The new read-only `FROZEN_MODELS` registry includes these pins, evidence paths,
validation prediction fingerprints and score interpretation. For JSD the stored
fingerprint is the legacy full validation `ScoreDataset` fingerprint (all baseline
scores and provenance); E3/A2 use their archived prediction fingerprints. They
are provenance identifiers with different historical schemas, not directly
comparable cross-model sample hashes. Cross-model alignment is checked separately.
These pins were checked against actual local artifact bytes; no model was loaded.

## Historical validation versus future held-out results

The following are **existing development-validation results**, transcribed from
the frozen Phase 5 comparison, not newly scored Phase 7 outputs:

| Frozen candidate | Validation ROC-AUC | Validation AP | Validation binary F1 |
|---|---:|---:|---:|
| JSD | 0.9696418405485012 | 0.9728053582663733 | 0.9172932330827067 |
| E3 MLP | 0.9700193064528105 | 0.968596681483803 | 0.9136690647482014 |
| A2 attention | 0.9608549293334323 | 0.9583877196552544 | 0.9052631578947369 |

The same 804 validation pairs (402/402) were reused for epoch, threshold and/or
architecture selection. These values have selection optimism and source overlap;
they are not independent generalization estimates. E3 is the best *learned*
validation-AUC model; JSD has higher AP/F1. A2 is the selected attention variant.
No held-out correspondence result is available from this task. Existing unimodal
producer TEST results must not be copied into this binary comparison.

The frozen comparison plan retains all three candidates and their own thresholds,
same primary pair identities, labels, source references, ten-bin preprocessing,
and score direction. No winner is reselected on TEST. First verify archived TRAIN
identity alignment; later authorize a dedicated final-evaluation reader that
verifies bytes, exact coverage and immutable provenance without fitting anything.
Run the shared metrics only after that gate. Keep challenge analysis separate;
its all-negative provenance labels require false-acceptance/rejection measures,
not a pooled primary ROC-AUC comparison. No TEST reader is supplied here.

## Reusable metric/report contracts

The numerical/contracts modules are `fusion/evaluation_contracts.py` and
`fusion/evaluation_metrics.py`. The remediation adds the restricted
`fusion/evaluation_admission.py` archive reader. These depend on NumPy and
historical contracts, without importing PyTorch, experiment runners or trackers.
Exploratory metrics perform no file I/O; authenticated attribution explicitly
requires verified archived evidence as described below.

| API | Contract |
|---|---|
| `CorrespondencePrediction` | Exact Phase 2 `PairRecord`, explicit integer target matching pair provenance, finite Python float score in `[0,1]`; booleans and reversed encodings rejected |
| `EvaluationProvenance` | Explicit TRAIN/validation `primary` only; pair-manifest, score-archive and source-overlap evidence references/digests |
| `PredictionSet` | Nonempty immutable tuple, model/artifact/score declarations; caller-created sets are exploratory and cannot claim reserved JSD/E3/A2 attribution; duplicate identity `(split,audio_sample_id,visual_sample_id)` rejected regardless of score/type/seed/row order |
| `FrozenOperatingPoint` | Explicit model artifact, threshold, positive direction, validation-primary fitting provenance, evidence hash/fingerprint, objective and tie rule; historical pins cannot be overridden |
| `admit_archived_train_predictions` | Restricted reader returning a loader-created `AuthenticatedPredictionSet` after checking fixed evidence anchors and full TRAIN primary coverage |
| `align_prediction_sets` | Revalidates archive admission and frozen checkpoint bindings for known models; exact identity/target/source-row/category/policy/split/manifest/overlap agreement required. Exploratory comparisons are explicitly unverified and cannot mix with authenticated admissions |
| `evaluate_predictions` | Accepts exploratory sets or authenticated admissions; includes explicit trust status alongside unchanged metrics, curves, bins and per-sample decisions |
| `canonical_json` | Sorted keys, compact deterministic JSON, native scalar/list conversion, `allow_nan=False`; no run IDs, timestamps or NaN/Infinity sentinels |

Records are ordered by existing `pair_order_key` (visual identity, then audio,
within a split). Reports retain both original sample/source IDs, producer row
indices, pair type and sampling provenance. Fingerprints bind sorted records,
targets, scores, model identity and provenance; input permutation cannot alter
the report. Reconstructed inputs are validated again at public evaluation and
alignment boundaries. Exploratory fingerprints and externally supplied hashes
remain declarations; they cannot grant authenticated attribution. Archive
admission uses independently pinned evidence, not those caller declarations.

`validate_binding` checks trust, model/artifact/score-kind association and frozen
operating-point equality before evaluation. Named JSD/E3/A2 require an authenticated
archive admission. Custom exploratory models supply an explicit policy and always
receive unverified attribution. There is no default threshold, search,
fitting function or final-test override in the new API. A curve lists cutoffs for
plotting; it does not select a new operating point.

## Provenance-integrity remediation

The independent audit reproduced two defects: a caller could relabel an arbitrary
score vector as a frozen model by supplying the expected hash strings, and
alignment did not reject incorrect known-model checkpoint declarations. Both
were reproduced again before remediation, without changing historical evidence.

### Explicit trust levels and entry points

`evaluate_predictions(PredictionSet(...), exploratory_policy)` returns
`trust.level = "unverified_exploratory"` and `authenticated_model = null`.
It remains suitable for synthetic mathematical checks. Plain sets claiming E3,
A2 or JSD, including case variants, are rejected by evaluation and alignment.
Possessing the correct checkpoint hash string does not authorize attribution.

`admit_archived_train_predictions("E3", repository_root=...)` is the only supported
construction path for an `AuthenticatedPredictionSet`. It accepts no replacement
hash, checkpoint, producer, cohort, score array or threshold arguments. The root
directory permits relocation of the same authenticated bytes. The return type
has no public data constructor or ordinary dataclass replacement path that can
grant trust. Its `.dataset` property permits inspection; copying that dataset
produces an ordinary unverified set, not an authenticated admission.

Passing an admission to `evaluate_predictions` yields
`trust.level = "authenticated_archived_train"` and the verified model identity.
`align_prediction_sets` marks authenticated comparisons explicitly. It rejects
mixed exploratory/authenticated comparisons and incorrect known-model bindings.
No public boolean, status field or caller-recomputed fingerprint grants authority.

### Independent evidence anchors

The root SHA-256 is
`4fcbd1537ab3e5574ee6d0d06d7d3a2787431741dd14cc6c161577b0a689f9e5`,
the exact bytes of the already committed
`docs/fusion_cross_attention_phase5_freeze.json`. The working file was compared
with `git show HEAD:docs/fusion_cross_attention_phase5_freeze.json` before editing
Phase 7. The root digest is owned by the admission code and has no caller override.

The existing transitive trust chain is:

| Model/evidence | Independent binding |
|---|---|
| A2 | The committed freeze pins its output manifest, checkpoint metadata and preregistration. The output manifest pins `train_primary_scores.csv` and all model/threshold sidecars. |
| E3 | A2's preregistration is pinned by the freeze (`f3896480f457944a5fec504b5a67c0c5247fe86b5a814c91a04f64f79820ea3e`). Its `phase4_e3/output_manifest.json` input pins E3's manifest (`7d9e7908e667977a45c2a1f4f2f40e8685f21a786da944fda0ac8f38ac466598`), which pins the TRAIN score archive and checkpoint/threshold metadata. No new E3 archive hash was invented. |
| JSD | The committed freeze pins its Phase 3 output manifest (`c0a33292cd5c376f4c2267bbc91a4845c246c006d581abc9f400430a3d7b9110`), which pins `scores/train_primary.csv`, producer configuration and frozen threshold policy. |
| Pair identities/rows | The freeze pins the original TRAIN pair manifest (`52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f`) and exporter TRAIN ID/SOURCE/README/class-map bytes. |

The reader checks actual checkpoint bytes as opaque SHA-256 inputs, verifies
model configuration, selected epoch, producer versions, the exact frozen
threshold and validation prediction fingerprint. It never loads or executes
encoder or fusion checkpoints. Existing producer/checkpoint evidence is reused;
raw encoder availability is not a prerequisite for archived prediction admission.

The pinned TRAIN manifest is rebuilt into the existing Phase 1/2 `Record`,
`PairSource`, `PairRecord`, sampling-policy and provenance contracts. Each source
row must reference the exact original record in the authenticated exporter ID
table. Primary CSV rows must match the complete canonical pair sequence,
source identities, original row indices and provenance targets. Duplicate,
missing, reordered or conflicting records fail. Learned scores are also checked
against the archived logits' historical float64 sigmoid; no model inference or
threshold selection occurs. The JSD score comes directly from its pinned archive.

Source bytes are verified before parsing and rechecked before admission succeeds.
Every subsequent evaluation or alignment repeats full admission from the fixed
root and compares the entire prediction fingerprint and dependency inventory.
An old success cannot authorize later changed scores, metadata or files, even
when a caller reconstructs a handle or recomputes its content fingerprint.
Missing files, incompatible evidence and self-consistent rewritten manifests
fail closed against the unchanged code-owned root digest.

### Scope and remaining trust assumptions

Authenticated admission is intentionally limited to **archived TRAIN primary**.
There is no validation, TEST or challenge admission option. No TEST/challenge
prediction archive or array is read. The original TRAIN pair JSON also contains
challenge metadata; only its `primary_pairs` are rebuilt and used. Historical
hash/metadata inventories may mention other cohorts without opening their
predictions. The separate 590-file preservation check still reads protected
files only as opaque bytes.

The trusted code, fixed root digest, SHA-256 collision resistance and historical
audit of the pinned producer artifacts remain trust assumptions. Admission proves
continuity to those recorded artifacts, not independent proof of the original
training execution, raw encoder parity or the truth of all producer assertions.
Arbitrary replacement of executing Python code or its private trust anchors is
outside this boundary. Concurrent hostile mutation during computation is outside
the supported single-process reporting model. Historical source overlap (10/8/0),
development-selection optimism and single-seed limitations remain unchanged.
Authenticated pairing provenance still does not prove physical sound-source
causality, independent modality presence or fused event/temporal predictions.

### Metric definitions and undefined policy

Confusion matrix rows are **true**, columns **predicted**, labels `[0,1]`:
`[[TN,FP],[FN,TP]]`. Positive support is `TP+FN`; negative support is `TN+FP`.

- Accuracy: `(TP+TN)/N`.
- Precision: `TP/(TP+FP)`; recall: `TP/(TP+FN)`.
- Positive binary F1: `2TP/(2TP+FP+FN)`.
- Macro-F1: mean of positive F1 and `2TN/(2TN+FP+FN)`, with the fixed two-label
  universe `[0,1]` even for single-class input.
- ROC-AUC: probability a positive score outranks a negative, with half credit
  for tied scores.
- Average precision: sum of recall increments times precision at each distinct
  descending score group. This is noninterpolated AP, **not trapezoidal PR-AUC**.
- Brier: mean `(score-target)^2`, for probability estimates only. E3/A2 are
  explicitly **uncalibrated sigmoid estimates**: this is a diagnostic, not proof
  of calibration. JSD gets `null` and reason `score_not_a_probability`.
- Reliability: ten fixed `[i/10,(i+1)/10)` bins, last bin including 1. Each
  records support, positives, mean score and observed original-pair frequency.
  Empty-bin means/frequencies are `null`. JSD bins are descriptive score bins.

`classification_metrics` and `ranking_metrics` are reused from Phase 3, preserving
Phase 4/5 definitions. Classification zero denominators give **0**, accompanied
by an explicit undefined reason (including an undefined zero-filled component
of macro-F1). No-positive predictions make precision undefined, while F1 can
remain defined as zero if false negatives exist. Empty evaluation is rejected.
For any one-class input, both ROC-AUC and AP are `null` with
`only_one_target_class`, preserving the historical contract; both curve datasets
are empty with that reason. This deliberate AP convention also applies to an
all-positive input, even though some libraries return 1.

ROC and PR curves group ties and enumerate descending finite score cutoffs with
inclusive `>=` decisions. ROC starts `(FPR,TPR)=(0,0)`; PR starts
`(recall,precision)=(0,1)`. Their common all-negative origin uses threshold `null`
and `above_maximum=true` rather than non-JSON Infinity. All subsequent thresholds
are observed scores. ROC ends `(1,1)` and PR ends at recall 1, precision equal to
positive prevalence. No calibration is fitted and no ECE is claimed. A custom
`calibrated_probability` policy must include calibration evidence reference and
hash; that is a caller assertion requiring future independent audit. None of
the frozen candidates has that status.

## Original foundation verification and preservation

The baseline full suite passed **586 tests: 584 passes, two CUDA skips, zero
failures/errors**, exit code 0, in 56.142 seconds. The 36 new focused synthetic
TRAIN tests pass and cover perfect/inverted/mixed/imbalanced/single-class cases,
undefined metrics, empty/duplicate/nonfinite/wrong-label rejection, equality,
ties, hand-computed ROC/PR, asymmetric confusion orientation, bins, uncalibrated
semantics, frozen threshold/model binding, JSON determinism, and sample alignment.
Synthetic tests do not use real validation, TEST or challenge predictions.

The archived TRAIN-only metadata check confirms identical 6,678 unique primary
pair identities, labels and original row references across JSD/E3/A2, with
3,339 positives and 3,339 negatives. It does not evaluate the archived scores.
Ignored evidence is `_local/phase7_train_alignment.json`.

Before/after preservation hashes cover 590 existing files / 264,975,100 bytes:
tracked Phase 1–6 files, all local research evidence, manifests, encoder exports,
frozen checkpoints and the untracked requirements file. Preservation reads
held-out/challenge files **only as opaque bytes for SHA-256 continuity**, without
deserializing arrays/pairs or computing scores/metrics. Cache/runtime directories
(`wandb`, `uv-cache`, `python`, `tools`, `__pycache__`) are excluded, as in prior
preservation audits. No historical artifact is rewritten. Local Phase 7 audit
scripts/logs are separate ignored additions.

Final command:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Original foundation result: **622 tests: 620 passes, two unavailable-CUDA skips, zero failures/errors,
Python exit code 0**, in 57.457 seconds. All 586 historical discovery entries
remain present and passing/skipped as before; the 36 additions are the Phase 7
tests. Before/after SHA-256 and sizes match for **all 590 protected files**.
At that checkpoint, the tracked working diff and Git index were empty. The four
original Phase 7 files and untouched pre-existing untracked requirements file
were the only nonignored status entries. Whitespace checks passed. Ignored evidence includes
`_local/phase7_baseline_tests.log`, `_local/phase7_full_tests.log`,
`_local/phase7_preservation_before.json` and `_local/phase7_preservation.py`.

That original implementation subsequently failed the independent provenance
audit. The two confirmed defects are corrected by the remediation above; the
latest verification follows. The matrix's head/data/provenance gaps still block
the corresponding final evaluation claims.

## Remediation verification

The two defects were reproduced before editing: all three reserved model names
accepted synthetic scores, and alignment accepted an E3 declaration with an
incorrect checkpoint hash. They now fail at the public evaluation/alignment
boundary. The original 36 evaluation tests remain passing; the test that previously
permitted reserved-model synthetic attribution now requires rejection.

There are **34 new focused archive-admission tests**, for **70 Phase 7 tests** in
total. They use small synthetic byte-backed archives and test-only patches of
private code-owned anchors, never a production hash-override API. They cover
all three spoofed model names, recomputed caller hashes, source row 900, wrong or
missing identities, reordered scores/rows, wrong checkpoints and thresholds,
producer/split changes, duplicate/missing records, failed/missing admission,
trust-level separation and invalidation after source or admitted-content changes.
They also reconstruct handles and recompute their fingerprints to ensure those
operations cannot bypass fresh independent archive authentication.

The genuine local E3, A2 and JSD TRAIN archives were admitted without any patched
anchor. Authenticated alignment matched all **6,678 primary identities** across
the three models. No evaluation metrics were computed in that real-evidence
check; it only verified admission, identity/row alignment and trust status. The
only opened score archives were their existing TRAIN primary CSVs. No encoders
or fusion models executed. Ignored evidence is
`_local/phase7_remediation_admission.json`.

AST comparison confirms `_curves` and `_reliability` are unchanged. After
normalizing only the added trust dispatch/report fields, the entire
`evaluate_predictions` function matches its pre-remediation AST. Metric formulas,
undefined policies, tie handling, sigmoid interpretation, frozen thresholds and
inclusive decisions therefore retain the independently verified mathematics.

The complete required suite passes **656 tests: 654 passes, two unavailable-CUDA
skips, zero failures/errors, exit code 0**. All 590 protected files /
264,975,100 bytes match their original SHA-256 and size baseline. Phase 1–6 source,
checkpoints, exports, manifests and experiment archives are unchanged. The Git
index has the same byte hash and remains unstaged. The untracked requirements
file retains its current documented hash. No credentials, checkpoints or exports
were added to source control.

The remediation changes only the two Phase 7 modules, their original test file
and this document; it adds `fusion/evaluation_admission.py` and
`tests/test_fusion_evaluation_admission.py`. The full-suite log is the separate
ignored `_local/phase7_remediation_full_tests.log`.

**Two provenance corrections: FIXED. READY FOR TARGETED RE-AUDIT.**
No Phase 7 Tasks 7.3–7.4, held-out/challenge evaluation, training, calibration,
W&B runs or Git publication actions were undertaken.

## Gates before final evaluation

1. Audit these contracts and resolve the pending Phase 6 dependency through normal
   review; no Phase 6 merge or implementation change is part of this task.
2. Authorize final TEST and challenge evaluation explicitly. Keep this foundation's
   no-TEST/no-challenge guard; add any future ingestion in a separately reviewed step.
3. Freeze candidates, exact thresholds, hashes, input producer/preprocessing
   contracts, pair cohorts, metric/undefined policies and interpretation before
   looking at outcomes. Verify actual archive/model bytes and full sample coverage.
4. Retain official partitions and disclose source overlap, validation selection
   reuse, shared-source dependence, single-seed evidence and encoder exposure.
   Preregister any supplementary source-disjoint analysis separately.
5. For event/temporal claims, obtain appropriate trained heads, label eligibility
   and genuine prediction provenance. Audit raw encoder assets/parity before any
   end-to-end raw-media claim. Do not use demo annotation lookups as predictions.
6. For calibrated confidence claims, freeze an independent development calibration
   allocation and calibration method/evidence without TEST fitting or threshold
   revision. Reliability plots of raw scores alone do not satisfy this gate.

No training, threshold tuning, checkpoint modification, new W&B run, TEST/challenge
prediction artifact, commit, push, merge or PR was performed. Work stops at
Tasks 7.1–7.2; future held-out work and missing heads are outside this foundation.

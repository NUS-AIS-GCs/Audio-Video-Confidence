# Phase 2 Task 2.1: pair contracts and sampling policy

This task defines metadata, labels, deterministic selection rules, and validation
for future pair manifests. It does not implement the sampling generator, select
real AVE pairs, load checkpoints, train, or establish a production inference path.
The implementation imports only Python's standard library and Phase 1 contracts
(which require NumPy). Import APIs from `fusion.pair_contracts`; Phase 1's public
exports and behavior are unchanged.

## Three separate questions

The **primary binary task** is original-pair versus substituted-pair discrimination.
MATCH (`original_pair_target == 1`) means the exact same annotation sample's
original audio and video. A soundtrack substituted from a different recording
has target 0, including same-category different-source challenges. This is a
recording-provenance label. It does not establish physical sound-source causality,
whether a source is visible, whether an annotated event is active at every second,
or even that the original recording was naturally synchronized or unedited.

**Semantic event compatibility** is a separate, explicitly category-based proxy:
when both source annotations have nonempty intervals, matching foreground
categories are `COMPATIBLE` and different categories are `INCOMPATIBLE`. These
statuses describe annotated category agreement/disagreement, not a perceptual
judgment or logical impossibility of co-occurrence. Same-category swaps are never
automatically semantically incompatible. If either annotation is empty, status is
`UNKNOWN` and semantic supervision is masked. No new physical or perceptual
compatibility annotations are claimed.

**Temporal synchrony** is always `TemporalStatus.NOT_ASSESSED` in this schema.
Equal integer-second indices, original pairing, and matching event labels cannot
prove synchrony. There is no supported shift pair enum, offset field, or positive
synchrony claim. A future waveform-shift extension requires a new version with
re-encoding provenance, offset/sign, padding/cropping, valid overlap, and temporal
evidence. Rolling contextual BEATs embeddings is not waveform re-encoding.

## Repository definitions and deliberate resolution

`Documentations/project_proposal_report_V3.md` describes original synchronized
pairs as positives and unrelated-event swaps as negatives. Section 6.2 in both
`Documentations/AValidate_Complete_Project_Dossier.md` and
`Documentations/AValidate_Vision_Pipeline_Dossier.md` instead defines MATCH as
same-class audio, while their pair tables call time shifts mismatches. These
definitions are not interchangeable. For this task, the user's explicit primary
task is the provenance binary label above; semantic and temporal concepts have
separate fields. Team review must reconcile those older documents before results
are presented under a broader source-matching claim. Teammate documents are not
edited here.

## Pair categories

| PairType | Required sources | Binary target | Semantic proxy if both intervals nonempty |
| --- | --- | --- | --- |
| `ORIGINAL_POSITIVE` | Exact same sample ID and equal complete Record metadata | 1 | COMPATIBLE |
| `CROSS_CLASS_EASY_NEGATIVE` | Different clip IDs and categories; outside approved hard edges | 0 | INCOMPATIBLE |
| `CROSS_CLASS_HARD_NEGATIVE` | Different clip IDs/categories; explicit approved directed hard edge | 0 | INCOMPATIBLE |
| `SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE` | Different clip IDs, same category/index | 0 | COMPATIBLE |

Any empty source interval makes the semantic proxy UNKNOWN regardless of type.
"Easy" names the category-contrast bucket; different labels do not guarantee
perceptual ease. "Hard" means selected by the recorded policy, not demonstrated
perceptual difficulty. Evaluation must keep difficulty buckets distinguishable.

`HardNegativePolicy` has no default taxonomy. Without an applicable policy,
constructing a hard pair fails. The policy contains the exact 29-class vocabulary,
sorted unique directed `(audio_category, visual_category)` edges, a versioned ID,
origin, named approval, evidence reference, and evidence SHA-256. Symmetry is not
assumed: the reverse direction must be explicitly listed if desired.

Supported origins are `MANUAL_APPROVED` (no evidence split) and
`VALIDATION_CONFUSION` (evidence split exactly `val`). Test/train/unknown confusion
sources fail. The team must approve the evidence and freeze its policy before
test evaluation; do not choose edges or tune them after seeing test results.
Allocate validation data or document reuse so policy choice, calibration, threshold
selection, and evaluation are not presented as independent when they are not.
Approval text and evidence hashes are declarations: the code cannot authenticate
an approver or verify the contents of an external evidence file.

## Metadata, masks, and numerical inputs

`PairSource` contains an official split, an unchanged Phase 1 `Record`, and its
original modality-specific row index. Thus sample ID, source clip ID, category,
class index, and original interval remain available. `PairRecord` contains two
sources, enum type, immutable sampling policy, deterministic provenance, and
temporal status. Its target, semantic status, identity, eligibility and candidate
rank are derived properties, so callers cannot assign contradictory labels.

All pair/seed/provenance records are frozen. Hard-policy lists are snapshotted
as tuples. Pair constructors reuse Phase 1's internal record and vocabulary
validators without changing Phase 1. Structural construction alone cannot prove
that a source belongs to a bundle: always call `validate_pair_manifest()` before
using row references.

Public validation accepts exact `PairRecord`, `PairSource`, `SamplingPolicy`,
`SamplingProvenance`, `HardNegativePolicy`, and Phase 1 `Record` types;
subclasses of these input contracts are unsupported. Validation uses class-owned
checks, never a supplied instance's overridden `__post_init__` hook. Reconstructed
objects are checked from their current fields, including nonnegative row indices,
supported enum member identity, and required metadata. Pair ordering also validates
the supplied record before reading its key.

Nested hard-policy fields are revalidated when attaching a policy, computing its
fingerprint, constructing provenance or pairs, and validating manifests. Declared
test-confusion evidence, missing approval text, and malformed evidence hashes are
rejected even if constructor validation was bypassed. This checks structural
approval/evidence metadata; it does not authenticate approval or inspect external
evidence. Supported base-type fields, labels, seed/hash formats, and Phase 1
behavior are unchanged. Supervision flags remain derived from validated pair data.

`EventSupervision` defines eligibility, not actual target tensors or input validity:

| Field | Meaning |
| --- | --- |
| `audio_annotation_eligible` | Audio source has a nonempty original joint AVE annotation |
| `visual_annotation_eligible` | Visual source has a nonempty original joint AVE annotation |
| `joint_event_eligible` | Original positive with both source annotations nonempty |
| `semantic_eligible` | Both source annotations nonempty |

Source-specific eligibility refers only to each source's original AVE annotation;
it is not independent audio audibility or visual visibility ground truth. Swaps
do not supply a fused event class or fused joint-presence target. No per-second
negative correspondence targets are invented from disjoint activity masks.
Downstream losses still need declared target definitions and valid-input policies.

The official `VWi2ENBuTbw__08` sample keeps `[0,0)`, category, row, ID, and train
membership as required by Phase 1.1. Its activity has ten False bins, but neither
modality's validity is inferred. When paired with itself, its provenance binary
target is still 1; this is not a positive annotated event. Its semantic status is
UNKNOWN and all event/semantic eligibility flags are False. If only one side of a
swap is empty, that side's annotation eligibility and both pair-level flags are
False; the other source's original annotation remains available.

Phase 1.1's prohibition on inferring semantic/event correspondence from this row
remains in force. The new binary label uses known recording provenance, a separate
task. Do not relabel unknown semantic compatibility from that binary target.

Model inputs must be numerical embeddings/probabilities selected using validated
row references. Pair metadata objects contain no model input arrays. IDs, class
labels, annotation intervals, provenance, pair type, masks, and targets must not
be concatenated into predictive features. Labels may support sampling, losses,
validation, and stratified reporting only.

## Sampling invariants and deterministic specification

1. Positives use the same exact sample ID with equal metadata, even when modality
   row positions differ. Different annotations sharing one clip stay distinct.
2. Every substituted pair uses different source clip IDs, never another annotation
   of the same video. This applies even to cross-class annotation rows.
3. Audio and video come from the same official split. Membership is checked
   against complete Phase 1 bundles, including exact array-associated row indices.
4. Identity is `(split, audio_sample_id, visual_sample_id)`. Repeated identities
   fail; changing type, seed, or row position does not create a new identity.
   Repetition/augmentation would require a separately reviewed versioned policy.
5. A policy's root seed is an unsigned 64-bit integer; booleans/floats are rejected.
   The version is `original-pair-v1`, method `sha256-rank-v1`.
6. Canonical hash encoding is UTF-8 of Python `json.dumps(value, sort_keys=True,
   separators=(',', ':'), ensure_ascii=True)`. `split_seed(seed, split)` is the
   first 16 SHA-256 hex digits of `[version, root_seed, split]`, interpreted as an
   unsigned integer. Each split has an independent namespace, not shared RNG state.
7. `PairRecord.sampling_priority` is SHA-256 of
   `[method, split_seed, audio_sample_id, visual_sample_id, pair_type.value]` using
   that encoding. Task 2.2 should rank eligible candidates by ascending priority,
   breaking ties by `(visual_sample_id, audio_sample_id, pair_type.value)`.
   Candidate budgets, ratios and anchor policy are not implemented here.
8. Final output is ordered by `pair_order_key`: train, val, test; then visual
   sample ID; then audio sample ID. The manifest validator rejects unordered
   output rather than silently sorting or deduplicating it.
9. Metadata fingerprints bind each modality's split, vocabulary and original row
   sequence. They exclude paths, clocks, modification times and other splits.
   Changing row order changes provenance but not candidate priority. They are not
   binary embedding hashes or proof that a producer attached the right CSV IDs.
10. Global RNG state, Python `hash()`, filesystem ordering, sets, and processing
    one split before another must not determine pair selection. Changing another
    split's contents does not change this split's candidate ranks. A deliberate
    change to the shared approved policy is a new experiment configuration.

The immutable policy fingerprint includes version, root seed, method, and all
hard-policy metadata. `SamplingProvenance` also preserves independent split seed
and audio/visual metadata fingerprints. Future generator quotas/configuration and
artifact binary identities need their own recorded provenance in Task 2.2; this
contract does not certify that a supplied manifest was selected by a particular
quota or without inspecting test results.

## Official split overlap

The readiness audit read the complete audio/vision Drive CSVs and found identical
cross-modal records, no cross-split sample-ID overlap, and shared source counts:

| Source-video overlap | Previously audited count |
| --- | ---: |
| train / val | 10 |
| train / test | 8 |
| val / test | 0 |

Saved output in `notebooks/03_vision_resnet18_local.ipynb` reports the same counts;
`audio/RESULTS.md` also acknowledges cross-split sources. These are historical
manifest findings, not hardcoded validator expectations or a new binary audit.
The manifests had 4,143 annotation rows over 4,097 sources: 44 sources with two
annotations and one with three. Full official partition membership was not
independently verified against downloaded official partition binaries.

`validate_pair_manifest()` invokes Phase 1 leakage auditing on both full available
metadata views, including rows not selected for pairs. Its immutable report lists
source overlaps and missing metadata splits; absence of unprovided split metadata
is not evidence of no overlap. It reports known overlap, keeps official assignments,
and does not allow pair-level cross-split borrowing. A source-disjoint evaluation
claim requires additional controls; fusion filtering cannot undo source exposure
already received by trained unimodal models.

## Public APIs and Task 2.2 handoff

- Enums: `PairType`, `SemanticCompatibility`, `TemporalStatus`, `HardPolicyOrigin`.
- Records: `HardNegativePolicy`, `SamplingPolicy`, `SamplingProvenance`,
  `PairSource`, `PairRecord`, `EventSupervision`, `SourceOverlap`,
  `PairValidationReport`.
- Helpers: `split_seed()`, `pair_order_key()`, `make_sampling_provenance()`,
  `validate_pair_manifest()`.

Task 2.2 must first load/validate real bundles, establish complete sample-ID
alignment, and retain exact source rows. Obtain split provenance with
`make_sampling_provenance(audio, vision, split=..., policy=...)`. Enumerate eligible
candidates within that split under a reviewed budget/anchor policy, rank them by
the specification above, build PairRecords, and sort by `pair_order_key`.
Call `validate_pair_manifest(pairs, audio, vision, policy=policy)` before storing
or consuming a nonempty manifest. Inspect the returned source-overlap and
missing-metadata diagnostics. The helper checks references, fingerprints, policy,
identity and ordering, but does not implement selection or certify selection
frequencies. No Task 2.2 generator or actual AVE pair list is delivered here.

## Decisions requiring team approval before experiments

- Reconcile the older same-class MATCH definition with the explicit primary
  provenance task; approve wording and which claims/metrics belong to each task.
- Approve any hard-neighborhood taxonomy or validation-confusion evidence; hard
  generation stays disabled by default. No category taxonomy is invented here.
- Choose split-specific pair budgets, ratios, anchors, source reuse limits, and
  whether same-class challenges participate in training or only separate evaluation.
- Approve downstream event/presence target definitions and empty/ambiguous-sample
  loss masks; do not treat these eligibility flags as complete training targets.
- Freeze validation use, threshold/calibration boundaries, and overlap reporting;
  decide any supplementary source-disjoint evaluation without changing AVE splits.
- Review a future waveform-level temporal-shift schema and evidence before adding
  temporal labels. That work is explicitly deferred.

## Validation

Use the existing Python 3.11 environment; no new dependencies are required:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Synthetic tests exercise provenance labels versus semantic status, empty masks,
same-source/cross-split exclusions, duplicate identity, metadata/row binding,
approved hard-policy requirements, deterministic rank/order/seed behavior, source
overlap reports, immutable snapshots, and unsupported temporal/enum claims.
Phase 1's full suite remains the regression boundary for artifact loading and
alignment coverage.

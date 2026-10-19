# Phase 2 Tasks 2.6--2.7: integrity accounting and synthetic integration

These tasks add missing leakage accounting and an independent synthetic
integration test. Phase 1, Task 2.1 and Tasks 2.2--2.5 remain unchanged. No real
AVE artifacts are loaded, downloaded or paired; no tensors, models, training,
experiment runs or Task 2.8 implementation are added.

## Reused safeguards

The [generator](fusion_pair_generator.md) and [pair contracts](fusion_pair_policy.md)
already check exact original modality row references, complete alignment,
sample identities, same-split membership, canonical order, independent split
seeds, configuration/policy fingerprints, quotas, exact pair uniqueness,
same-source exclusion, policy evidence declarations, empty supervision masks,
and challenge isolation. `validate_generated_manifest()` remains the authority;
the integrity audit invokes it before producing a report. It does not duplicate
these rules or weaken any contract.

## New public audit boundary

```python
from fusion.pair_integrity import audit_generated_manifest

# manifest, audio and vision are existing validated in-memory objects.
report = audit_generated_manifest(manifest, audio, vision)
```

`audit_generated_manifest(manifest, audio, vision) -> PairIntegrityReport` is
read-only and performs no file or network I/O. It raises `ContractError` on
invalid manifests, sources or policies. It does not return a success report for
a duplicate pair, same-source substitution, cross-split pair, wrong row index,
quota mismatch or invalid evidence declaration. It also inherits the existing
validator's documented boundary: it does not certify an external producer's
ranked selection.

The report and its `MetadataLeakage` snapshots are frozen output records with
immutable tuples. They are diagnostics, not trusted input contracts or persistent
validation credentials. Always re-run the audit against current objects before
consumption. The existing generation manifest remains the pairing authority.

Report fields include:

- `pair_count`, `primary_pair_count`, `challenge_pair_count`, and the existing
  eligible/requested/selected/unselected `pair_type_counts`.
- `source_row_references_checked`: two references per selected pair, including
  legitimate repeated references in different pair identities.
- `same_source_substitutions`, `cross_split_pairs`, `duplicate_pair_identities`,
  and `supplied_split_membership_preserved`. Successful audits have zero violations;
  invalid inputs raise before a report is returned.
- Empty audio/visual source-pair counts and `unknown_semantic_pairs`. These are
  annotation accounting, not modality validity, audibility or visibility.
- The unchanged `generation_provenance`, hard-generation status and full
  `declared_hard_policy`, including origin, approval, directed edges and evidence
  reference, split and SHA-256 declaration.
- Separate audio and vision `metadata_leakage` snapshots for **all available
  metadata**, including rows and splits not selected into this manifest.

Each modality snapshot contains split row counts, within-split duplicate sample
IDs, sample-ID and source-clip-ID overlap by split pair, legitimate multiple
annotations `(split, clip_id, sample_ids)`, missing metadata splits, and a full
metadata fingerprint. Tuple mappings can be read with `dict(snapshot.sample_overlap)`
or `dict(snapshot.source_overlap)`. Split pairs use Phase 1 names such as
`train&val` and `train&test`; IDs are sorted. Output ordering is deterministic.

Within-split repeated sample IDs fail existing Phase 1 validation. Different
sample IDs for one clip are legitimate separate annotations and are listed
separately. The same sample ID declared in two splits is a leakage finding,
distinct from shared clip IDs with different annotations. Such overlaps are
reported without changing or concealing supplied partition assignments. A
successful structural audit does not imply source-disjoint evaluation.

Missing split metadata produces explicit missing-split entries and no invented
zero-overlap result for that split. Historical AVE overlap counts are never
validation constants. Approval/evidence contents are not authenticated; declared
validation-confusion evidence must still be `val`, never test. Manual approvals
and policy freezing require external review before actual experiments.

## Integrity claims and reproducibility

`metadata_integrity` is `VALIDATED_IN_MEMORY`.
`artifact_binary_integrity` is always `NOT_VERIFIED` and
`evidence_contents_verified` is always False in reports returned by this audit.
Phase 1 checks the provided arrays' structure and numeric validity, but neither
that validation nor these metadata fingerprints hashes binary array contents,
remote files, checkpoints or evidence documents.

The additional modality fingerprint hashes the `pair-integrity-v1` namespace,
modality, category vocabulary, requested split set in official order, and all
available split records in their original modality-specific row order. Integer
metadata is canonicalized, including accepted NumPy integers, using Task 2.1's
canonical SHA-256 JSON helper. Dictionary order and process hash randomization
do not affect it. Row order or metadata changes do. It excludes paths, clocks
and binary arrays.

Generation provenance remains split-specific. Changing another split's metadata
must not affect training candidate ranks; it does change this audit's full-context
fingerprint and leakage findings. Finite numerical array changes alone can leave
both metadata fingerprints and the audit report unchanged. This is expected and
is explicitly tested. Nonfinite or structurally invalid arrays still fail.

No manifest serialization/export is needed for the current in-memory boundary;
none is implemented here. Task 2.8 can use existing PairSource fields directly
without reconstructing pair identities. If later persistence is required, define
a versioned format with exact enum/dataclass reconstruction, integer row indices,
ordering, complete provenance and round-trip validation against fresh bundles;
plain JSON dictionaries are not accepted by existing strict contracts.

## Synthetic integration evidence

`tests/test_fusion_pair_integration.py` defines independent in-memory audio and
vision bundles with identifiable numerical rows, shuffled modality row orders,
29-class probabilities and the real empty annotation identity/category
`VWi2ENBuTbw__08`, `Church bell`, `[0,0)`. Other fixture identities and policy
approval/evidence are synthetic. No external files are accessed.

The training fixture has nine annotations from eight source clips. All positive
annotations survive by default, including both `a` annotations and the empty row.
An independent Cartesian eligibility and canonical SHA-256 ranking oracle checks
the exact selected IDs: 9 positives, 5 easy negatives, 3 hard negatives and 12
separate challenges. Eligible counts are 9/50/8/12. It checks targets, semantic
status, supervision masks, ordering and original array references for every pair,
using row markers rather than rebuilding fusion features.

The full synthetic metadata has one shared source across train/val (`a`), one
across train/test (`f`), and none across val/test, with no sample-ID overlap.
Another fixture deliberately repeats `e__02` across train/val and verifies that
sample overlap is reported. These values are test expectations only. Integration
tests also cover reuse modes, infeasible quotas, missing/malformed sources,
duplicates, reconstructed invalid records, policy evidence boundaries, report
immutability, missing split diagnostics, full-context fingerprint changes and
cross-process determinism.

## Task 2.8 requirements (future work, not executed here)

Before consuming actual artifacts, Task 2.8 must:

1. Identify explicit audio/vision export roots, versions and trusted manifest or
   producer references. Preserve official train/val/test assignments; compare IDs
   and counts against official split-specific references rather than assuming
   that a structurally valid bundle proves official origin.
2. Load actual `.npy` arrays with pickles disabled and validate them through
   Phase 1. Require audio embeddings `(N,10,256)`, vision embeddings `(N,10,512)`,
   and **both** probability arrays `(N,10,29)` for each integration split. Phase 1
   permits other positive vision dimensions and optional vision probabilities;
   this integration requirement must therefore be checked explicitly in Task 2.8.
3. Verify exact dtype against the producer's declared export schema: final audio
   embeddings/probabilities are float16; Phase 1 supports float16/32/64 for vision,
   but the actual chosen export's dtype must be recorded and matched. Verify
   numeric finiteness, probability range/normalization and exact row counts.
4. Match each array to its producer's ordered sample-ID CSV and original row
   association. Check unique IDs, source clip IDs, category/index mapping and
   original integer intervals, with only the documented `[0,0)` empty exception.
   Equal array lengths alone cannot prove correct ID order. Reject missing,
   extra or conflicting cross-modal annotations; do not reorder arrays silently.
5. Verify the exact shared 29-class mapping, including background last, with no
   inferred class remapping. Retain all original metadata and empty supervision
   eligibility without inferring input validity from annotations.
6. Stream SHA-256 over the **actual bytes** of every consumed array, ordered ID
   CSV, class mapping and relevant producer provenance/manifest file. Record file
   roles, sizes, hashes, export versions and any checkpoint declarations in a
   separate artifact inventory; compare against trusted producer checksums when
   available. Locally calculating a digest establishes identity, not authenticity.
   Guard against files changing between hashing, loading and consumption; abort
   and revalidate on changes. Do not claim checkpoint or remote-source verification
   merely from a declaration or locally computed array hash.
7. Generate/validate pairs against those exact bundles, audit full available split
   metadata, and reconcile source exposure and hard-policy evidence boundaries.
   Index each modality using its own `PairSource.row_index`, then verify the row's
   Record against `PairSource.record`; retain existing `(split, audio_id, visual_id)`
   identities and separate challenge accounting. Missing held-out metadata limits
   the completeness of the leakage audit.

The eventual concatenated embedding width is 256 + 512 = **768 per temporal bin**.
No concatenation, fusion tensor, MLP, artifact inventory implementation or real
binary verification is performed by Tasks 2.6--2.7. Sampling ratios, challenge
training inclusion, loss masks, evidence approval and evaluation/calibration
boundaries still require explicit experiment decisions.

## Experiment tracking and validation

All future actual GC2 model training/evaluation experiments must use:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

These synthetic checks require no W&B run. Use the normal environment secret
mechanism for future authentication; never hardcode or expose API keys.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

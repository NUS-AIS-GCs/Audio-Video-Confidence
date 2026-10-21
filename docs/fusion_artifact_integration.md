# Phase 2 Task 2.8: local artifact integration

Official reference verification is complete: the supplied mirror metadata matches
the authors' annotation text and official HDF5 split indices for every record.
The [real-data results](fusion_real_data_validation.md) record pair counts,
hashes, source leakage and validation evidence. No models, fusion tensors or
W&B runs are created.

## Local entry point

From the repository root, using the existing Python 3.11 environment:
first run the portable reference verification below. Supply the four mirror
text files, the four pinned author references, and both producer export roots.

```powershell
.\.venv\Scripts\python.exe -B -m fusion.artifact_integration `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --output-dir _local/fusion/task28_real_reproduced `
  --seed 42 `
  --expected-hashes _local/fusion/ave_reference_verified/verified_metadata_hashes.json
```

Exit codes: 0 complete, 2 blocked by missing prerequisites, 1 validation or I/O
failure. The output directory must be fresh and separate from every input root;
use another ignored `_local/fusion/...` directory for subsequent runs. Existing
results are never overwritten. The report is published last and is the completion
marker; ignore a directory without `validation.json` after an interrupted write.

`_local/` is already ignored by the repository's `.gitignore`. This covers both
the `_local/datasets/AVE/` metadata directory and all integration
outputs. Do not add downloaded files, absolute private paths or generated
manifests to Git. No ignore-rule change is necessary.

Public functions in `fusion.artifact_integration`:

- `run_integration(audio_root, vision_root, official_root, *, output_dir=None,
  seed=42, expected_counts=OFFICIAL_COUNTS, expected_hashes=None)`.
- `manifest_to_json(manifest, audio, vision, *, artifact_fingerprint=None)`.
- `manifest_from_json(text, audio, vision, *, expected_artifact_fingerprint=None)`.

The library's `expected_counts` override is for explicitly declared fixture or
subset tests. The CLI always uses official 3339/402/402 counts. A synthetic test
success does not establish real integration.

## Inputs and independently verified official metadata

The inspected exports each contain these twelve files:

```text
class_names.json  README.md  SOURCE.txt
train_emb.npy    train_probs.npy    train_ids.csv
val_emb.npy      val_probs.npy      val_ids.csv
test_emb.npy     test_probs.npy     test_ids.csv
```

Producer notes identify audio run `beats_lora_r8_v1`, features `beats_iter3`, and
vision champion `resnet18_v1`. Those are source declarations, not checkpoint
authentication. The integration never imports or changes teammate exporter code.
Optional Phase 1 validity/timestamp sidecars, when present, are also consumed,
validated and included in the file inventory.

The user supplied `Annotations.txt`, `trainSet.txt`, `valSet.txt` and `testSet.txt`
under `_local/datasets/AVE/`, obtained from the
[Helios1208 mirror](https://huggingface.co/datasets/Helios1208/AVE_Dataset/tree/main).
Filenames and export agreement were not treated as proof of authority.

The authors' repository was pinned to commit
`93f843fed72fb17394df76330df958b0c2dbde76`. Its
[Annotations.txt](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/Annotations.txt)
and [train_order.h5](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/train_order.h5),
[val_order.h5](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/val_order.h5),
[test_order.h5](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/test_order.h5)
were downloaded to ignored `_local/datasets/AVE_reference/`. Their SHA-256,
sizes and pinned URLs are recorded in `sources.json`; the authors' data loader
and README are included as provenance. The data loader indexes arrays directly
with the integer `order` entries, using zero-based Python indexing.

The mirror annotation bytes exactly equal the authors' bytes after removing the
single `Category&VideoID&Quality&StartTime&EndTime` header line. The parser now
accepts that exact header; no annotation records or intervals are changed.
Regression tests check both successful header handling and rejection of malformed
data following it.

For every split, the HDF5 `order` array was checked for integer dtype, rank,
length, uniqueness and bounds. The three arrays partition indices 0..4142
exactly once. Selecting the corresponding authors' annotation rows reproduces
each mirror split file exactly, including quality and row order. Every exporter
sample ID, category, interval, membership and class mapping also matches; both
export row orders happen to equal official order, but ID alignment is still used.
The sole empty row remains `[0,0)` in train.

### Portable upstream verification

The tracked module `fusion/verify_ave_reference.py` requires only explicit local
inputs. It never downloads files, reads an old report, uses exporter metadata as
ground truth, or generates pairs. Supply these files from the pinned author links
above under any chosen `--reference-root`:

```text
Annotations.txt  train_order.h5  val_order.h5  test_order.h5
```

Their SHA-256 values are fixed in the tracked module at commit `93f843f` in full;
an editable local `sources.json` cannot override them. The metadata root must
contain `Annotations.txt`, `trainSet.txt`, `valSet.txt`, and `testSet.txt`.
All eight files are required. Missing references or any hash, category, interval,
identity, membership or order mismatch fails closed, with exit code 1 and no
completed evidence. Only the exact optional first-line mirror header is removed;
the remaining annotation bytes must match exactly, including quality and order.
The CLI always checks 4,143 records and the official 3,339/402/402 counts. HDF5
indices must be integer, unique, in bounds, disjoint, complete and used directly
as zero-based indices, in their original order.

HDF5 reading needs optional `h5py==3.16.0`, separate from lightweight contract
requirements. On a fresh checkout, if an HDF5 reader is not already available,
install it into an ignored target (this is setup, not performed automatically):

```powershell
.\.venv\Scripts\python.exe -m pip install --no-deps --target _local/tools/ave_reference h5py==3.16.0
```

NumPy is already a project prerequisite. To use that isolated reader and verify
the supplied local references, from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path _local/tools/ave_reference).Path
.\.venv\Scripts\python.exe -B -m fusion.verify_ave_reference `
  --metadata-root _local/datasets/AVE `
  --reference-root _local/datasets/AVE_reference `
  --output-dir _local/fusion/ave_reference_verified
```

Omit the `PYTHONPATH` assignment if h5py is already installed in the selected
Python environment. Do not install it again over an existing isolated target.
Input paths can be anywhere; output must be a fresh Git-ignored directory inside
this repository, separate from the inputs. This is enforced before writing.
The library `verify_reference(metadata_root, reference_root, *, output_dir=None)`
also supports verification in memory without writing any files.

It writes deterministic, path-independent `verification.json` (completion marker
last), containing pinned URLs, actual source hashes and sizes, exact comparison
results, mismatch count, split order fingerprints, shared-source counts and
empty-annotation status. `verified_metadata_hashes.json` freezes the four
upstream-verified mirror files for the integration command above. Integration
recomputes the other 24 export hashes itself; these are local integrity evidence,
not trusted producer checksums. No historical export inventory is required.
The previous real run's 28-entry continuity baseline remains historical evidence;
the fresh workflow intentionally supplies four upstream-verified expected roles.
An invalid run emits an error instead of a PASS report. No network access or
optional h5py dependency is needed merely to import the verifier or run its tests.

## Validation and integrity boundaries

The integration performs these steps without rewriting source files:

1. Inventory all required consumed exports and available official reference files.
   Hash actual file bytes with streaming SHA-256; record role, local path and size.
   Check file identity/size/time stamps around hashing. Missing files produce an
   explicit BLOCKED report, never substitute arrays.
2. Reuse Phase 1 loading (`allow_pickle=False`, read-only memory mapping) and numeric
   validation. Additionally require both final modalities' arrays to be float16,
   embeddings `(N,10,256)` audio / `(N,10,512)` vision, and probabilities `(N,10,29)`.
   These stricter integration checks match the inspected producer READMEs; Phase 1
   remains unchanged. Probability sums use its existing float64 sum and 0.001
   absolute tolerance. Reject corrupt arrays, nonfinite values and invalid IDs.
3. Align complete exporter records by sample ID, preserving both original row
   indices. Report missing/extra IDs through existing validation errors. Report
   unequal row positions explicitly; equality of row counts is not alignment.
   Export CSV agreement proves internal metadata consistency, not official origin
   or a producer's actual content-to-CSV row association.
4. When all four official files are present, parse their ampersand-delimited
   annotation format, derive the exact alphabetical 28-event vocabulary plus
   background, and reuse Phase 1 record validation. Mirror the producer's exact-row
   deduplication, but reject duplicate annotation identities. Require the official
   split union to match Annotations and compare every export Record and membership
   against the supplied reference files. Preserve exporter row order even when
   the reference order differs. No predictions or embedding similarities define
   annotation truth, split membership or temporal synchrony.
5. Generate pairs only after all reference checks pass. Run the existing generator
   and integrity audit; export and round-trip every per-split manifest.
6. Re-hash consumed inputs before publishing results and abort on changes. Explicitly
   close successfully opened memory maps on success and error paths. No filesystem
   locking, adversarial race immunity, remote authenticity or checkpoint verification
   is claimed. Consumers must revalidate inputs if they subsequently change.

`validation.json` distinguishes `binary_validation=VALIDATED_LOCAL_FILES`,
local SHA-256 computation, expected-checksum coverage, official reference status,
export alignment and overall COMPLETE/BLOCKED status. A digest identifies local
bytes; it does not prove their authenticity. Optionally provide a local JSON
role-to-digest mapping with `--expected-hashes`, e.g. key `audio/train_emb.npy`.
Unknown roles and mismatches fail. Partial coverage is named in
`expected_hash_roles`; no trusted producer checksums have been supplied for the
earlier blocked run. The completed run checks all 28 expected entries: upstream-
verified metadata hashes and the unchanged earlier local export hashes. Those
export hashes do not authenticate model checkpoints or remote source media.

The nested Tasks 2.6--2.7 metadata audit intentionally keeps
`artifact_binary_integrity=NOT_VERIFIED`: that API does not hash files. Actual file
verification belongs to the outer integration inventory, not a relabeled metadata
fingerprint. Available export metadata is reported for each modality, including
unique source counts and empty annotation row indices/activity masks.

## Initial deterministic sampling and persistence

The initial policy is fixed without consulting test outcomes: seed 42, source
reuse ALLOW, all aligned original positives, one easy negative per positive and
one separate same-category challenge per positive, independently in each official
split. Hard policy is None, hard budget zero and hard enablement False. No quota
is silently reduced. Candidate/work/matching failures abort with no published
report claiming success. The generator's default scan cap is unchanged.

When real references are supplied, the planned requested counts are:

| Split | Positive | Easy | Hard | Separate challenge |
| --- | ---: | ---: | ---: | ---: |
| train | 3339 | 3339 | 0 | 3339 |
| val | 402 | 402 | 0 | 402 |
| test | 402 | 402 | 0 | 402 |

These fixed quotas are checked without relaxation. Actual completed counts and
manifest hashes are recorded in the linked real-data results. Test metadata is
used only to construct that split's evaluation pairs; it never tunes training
configuration.

On complete runs, `train_pairs.json`, `val_pairs.json` and `test_pairs.json` use
`gc2-pair-manifest-v1`. Each contains config, policy, full generation provenance,
count/exclusion report and separate canonically ordered primary/challenge lists.
Pairs contain both complete original sources and integer row indices, type,
identity, binary target, semantic status, temporal status, event eligibility and
both ten-bin annotation masks. Empty annotations retain identity with unknown
semantic status and masked supervision; no modality validity is inferred.

Each file also binds the integration inventory fingerprint; `validation.json`
records every manifest's byte SHA-256 and size. Keep that report with manifests.
`manifest_from_json` requires the expected inventory fingerprint for bound files,
regenerates exact typed records through audited APIs and compares the entire
canonical document. This checks ordering, integer versus float/bool types,
derived masks, unknown fields and full provenance. Duplicate JSON keys and
nonfinite JSON constants fail. Regeneration pays the sampling cost again; this
is deliberate, not a fast untrusted-deserialization shortcut. The reader alone
does not hash input arrays or establish official provenance; use fresh integration
inventory/reference checks before consuming a persisted manifest.

## Verification, scope and experiment tracking

Focused file tests use only temporary synthetic exports and explicitly declared
synthetic reference annotations. They cover the complete happy path, corruption,
truncation, wrong dtypes/shapes, nonfinite values, probabilities, ID errors,
independent official membership/vocabulary checks, missing files, checksum
mismatches, concurrent input changes, deterministic bytes, serialization tampering,
row reconstruction and source immutability. They never overwrite real artifacts.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

No Phase 1, Tasks 2.1--2.7, teammate code or GC1 files are changed. Only small
upstream reference files and an isolated HDF5 verification tool were downloaded,
all under ignored `_local/`. No AVE media, embeddings or checkpoints were
downloaded. No model training, Phase 3 work, commit, push or merge occurs.
Future actual GC2 training/evaluation experiments must use
`WANDB_BASE_URL=https://api.forge.coreweave.com`, `WANDB_ENTITY=GC2-AValidate`,
`WANDB_PROJECT=avalidate`. No W&B run is required here. No API keys are read or
stored by the integration.

The mandatory [evaluation protocol and experiment logging fields](fusion_pair_generator.md#evaluation-protocol)
apply to all future model experiments. This remediation requires no W&B run.

# Task 2.8 real-data validation results

Status: **COMPLETE**, verified on 2026-10-19 (Asia/Singapore). No metadata
mismatches remain. The real pipeline returned exit code 0. No training,
model inference, W&B run, commit, push or merge was performed.

## Metadata authority and verification

The four local metadata files were supplied from the
[Helios1208 mirror](https://huggingface.co/datasets/Helios1208/AVE_Dataset/tree/main).
They were independently checked against the authors' repository, pinned to
commit `93f843fed72fb17394df76330df958b0c2dbde76`:

- [Annotations.txt](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/Annotations.txt).
- [train_order.h5](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/train_order.h5),
  [val_order.h5](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/val_order.h5),
  [test_order.h5](https://github.com/YapengTian/AVE-ECCV18/blob/93f843fed72fb17394df76330df958b0c2dbde76/data/test_order.h5).

The mirror adds one `Category&VideoID&Quality&StartTime&EndTime` header line.
Removing precisely that line yields a **byte-for-byte match**, including row
order, quality, event categories, source IDs and integer intervals, with all
4,143 authors' annotation rows. The source files were not edited. A narrow
Task 2.8 parser change accepts this exact header; malformed data after it still
fails validation. Both cases have regression tests.

The authors' `order` arrays are one-dimensional int64 arrays with 3,339/402/402
entries. Their indices are unique within each split, disjoint across splits,
in bounds, and cover every index 0..4142 exactly once. Direct zero-based indexing
matches the authors' data loader. Selecting annotation rows by these indices
reproduces each mirror split file exactly, including row order and quality.
Every sample ID, category, label index and interval in **both** exporters matches
its corresponding official split. Both export row orders also match official
order; pairing nevertheless uses the audited sample-ID join.

The official empty annotation is author annotation index 26, sample
`VWi2ENBuTbw__08`, `Church bell`, `[0,0)`, in train. Its original export row
index remains 63 in both modalities. Shared-clip annotations remain distinct.
There are 4,143 unique annotation IDs over 4,097 source clips: 44 clips have two
annotations and one has three. No annotation IDs overlap official splits.

The original run's reference evidence is stored under ignored
`_local/datasets/AVE_reference/`:

- `sources.json`: pinned URLs, file sizes and SHA-256 for the six author files.
- `verification.json`: full comparison results, mirror hashes and empty-row evidence.
- `verified_input_hashes.json`: all 28 frozen input hashes enforced by integration.

Fresh verification uses the tracked `fusion/verify_ave_reference.py` module and
the four pinned author files. It does not require those historical JSON files
or the old ignored helper. See the [portable input and optional HDF5 setup](fusion_artifact_integration.md#portable-upstream-verification).

Only small author metadata/provenance files were downloaded. The h5py 3.16.0
reader was installed with `--no-deps` into ignored `_local/tools/ave_reference/`;
project dependencies and the main Python environment were unchanged.

## Real pair generation

Seed 42, source reuse ALLOW, hard policy None, hard enablement False. Each
split requests every original positive, N easy substitutions and N separate
same-category challenges. Settings were fixed before examining test outcomes.
Requested and generated quotas are equal; no eligibility or quota was relaxed.

| Split | Positives | Easy | Hard | Separate challenges | Primary | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 3,339 | 3,339 | 0 | 3,339 | 6,678 | 10,017 |
| val | 402 | 402 | 0 | 402 | 804 | 1,206 |
| test | 402 | 402 | 0 | 402 | 804 | 1,206 |

Total: **12,429 pairs** and **24,858 original modality row references**.
Every split passed the existing pair validator, generation validator, integrity
accounting and typed serialization round trip. The reader regenerated the full
selection and compared every persisted field, including masks and ordering.
A separate pass over saved JSON independently checked all 24,858 source references
against ordered CSV records, unique identities, labels, supervision masks,
source exclusions, partition isolation, counts and every input/output hash.

| Split | Positive pool | Easy pool | Hard pool | Challenge pool | Same-source candidate exclusions |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 3,339 | 10,718,958 | 0 | 426,570 | 3,393 |
| val | 402 | 155,336 | 0 | 5,864 | 404 |
| test | 402 | 155,338 | 0 | 5,864 | 402 |

Actual selected same-source substitutions, cross-split pairs and duplicate pair
identities are all zero. Challenge records remain in their separate partition;
no training distribution is built. Unknown input validity and temporal synchrony
remain unknown/NOT_ASSESSED.

## Binary integrity and output hashes

Both modalities have 3,339/402/402 rows. All twelve arrays are float16: audio
embeddings `(N,10,256)`, vision embeddings `(N,10,512)`, and both probability
arrays `(N,10,29)`. Numeric finiteness, bounds and Phase 1 probability sums passed.
The 28-file inventory includes twelve arrays, six ordered ID CSVs, two class
mappings, four producer notes and four mirror reference files. All expected
hashes matched before loading; input hashes were rechecked before publishing
and independently afterward. The six downloaded author file hashes were also
rechecked. Source bytes were not rewritten.

| Actual array role | Bytes | SHA-256 |
| --- | ---: | --- |
| `audio/test_emb.npy` | 2,058,368 | `13afd5215b736db28ec47123bfbedcb867192ae3a6bbb526f03652ba9d5ffd66` |
| `audio/test_probs.npy` | 233,288 | `d5668d00eec535589f23a41487ec2acbbe68c29cfcd64a5657c024da0a6270a6` |
| `audio/train_emb.npy` | 17,095,808 | `29b59352b293abee1814d6a61900433d5379ae7355c050688ee79b2519a13da9` |
| `audio/train_probs.npy` | 1,936,748 | `07f1ccd454dee5a5227debee5120252c1d6060071dba1ba2e01befadd71353e0` |
| `audio/val_emb.npy` | 2,058,368 | `0707b6bf89569a6dfdc1a87f6de5eeefe338b688b9948c61f266a81bee884dde` |
| `audio/val_probs.npy` | 233,288 | `4da0ebb87986e8abd4b6a751d6293bc9fef73c08c4c664e8bb0033abab4a7d07` |
| `vision/test_emb.npy` | 4,116,608 | `21195172413c3e1326da0102a6435c73b90b1f3c9d6fe513761b5b7be12fe808` |
| `vision/test_probs.npy` | 233,288 | `550953696f698fe3202d40dd6f4dc023e4f648214ba5b5493b9f0db85e1abb85` |
| `vision/train_emb.npy` | 34,191,488 | `bb1f0373c1820ae932e0a9e5732bd47ae51c641ae86271207a929b044ca7575d` |
| `vision/train_probs.npy` | 1,936,748 | `0ef956002f2dab009c7f6036e060b0f587e2d4440add4c85b2a3eef6332074bb` |
| `vision/val_emb.npy` | 4,116,608 | `cfe3a1b750c372eb8c333124e54b04f4aa40a686e5b6fa1102d23b82a9f04208` |
| `vision/val_probs.npy` | 233,288 | `aefbfaa0ba2d54784948c93d272ec315e35a363f868c1003040afe7455413dcc` |

| Persisted manifest | Bytes | SHA-256 |
| --- | ---: | --- |
| `train_pairs.json` | 8,334,066 | `52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f` |
| `val_pairs.json` | 996,145 | `0ef344fa96e64d4ee7f44bee4082a2b1ad5c881504040c306cf6d00172322178` |
| `test_pairs.json` | 1,000,114 | `a268fd573c6e591e4dd81f7c5b9037d2742ebeb3c31b33962ad536a1ccd19b76` |

Artifact inventory fingerprint: `c8ec1448ab79e7c5fcc044678e9375bece168c7f2adfbbe5fa7a57c0ca957949`.

Validation report SHA-256: `ac7ec90c23d3ae3827042eb9cde974884251fa06deefaee881160a8ad873e665`.

Independent upstream verification report SHA-256: `00fd29701e41b7c891f7f4461b8f313600eb4f3ed3f015e794176859f8e82cb5`.

The fingerprint binds all inventory roles, local paths, sizes, hashes and declared
array shapes/dtypes to each manifest. Private paths occur only inside ignored
local reports, not in this document. Expected export hashes establish continuity
with the earlier local inventory; they are not producer-signed checksums. The
metadata comparison anchors annotations and splits to the checked author commit.
Neither operation authenticates model checkpoints, decoded AV content or physical
sound-source causality. The nested metadata-only audit correctly continues to say
`artifact_binary_integrity=NOT_VERIFIED`; the outer integration report is the
actual file-byte validation evidence.

## Leakage and empty-annotation supervision

Both modalities report source overlaps: train/val **10**, train/test **8**,
val/test **0**. These are observed values from the complete verified metadata,
not hardcoded expectations. Annotation-ID overlap is zero for every split pair.
Per-split unique source counts are train 3,312, val 401 and test 402. Official
assignments were preserved; fusion pairing cannot remove prior encoder exposure.
These results do not justify a source-disjoint evaluation claim.

The empty train annotation is retained as an original positive with provenance
target 1; it is not a positive annotated event. It also appears as the audio
source in two selected same-category challenges. All three affected pairs have
UNKNOWN semantic compatibility and masked pair-level semantic/event supervision.
Its own ten-bin annotation activity is all False; the nonempty partner's source
annotation eligibility is retained. No empty row occurs in val/test. Annotation
activity is not an audibility, visibility or input-validity measurement.

## Reproduction and validation

```powershell
$env:PYTHONPATH = (Resolve-Path _local/tools/ave_reference).Path
.\.venv\Scripts\python.exe -B -m fusion.verify_ave_reference `
  --metadata-root _local/datasets/AVE `
  --reference-root _local/datasets/AVE_reference `
  --output-dir _local/fusion/ave_reference_verified
.\.venv\Scripts\python.exe -B -m fusion.artifact_integration `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --output-dir _local/fusion/task28_real_reproduced `
  --seed 42 `
  --expected-hashes _local/fusion/ave_reference_verified/verified_metadata_hashes.json
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Supply the four pinned references and optional h5py reader as documented in the
linked setup instructions. Choose fresh ignored output directories for both
commands; omit `PYTHONPATH` when h5py is already in the selected environment.
The verifier outputs four upstream-verified metadata hashes; integration hashes
all exports afresh. It does not require the original blocked report. The completed
historical run remains untouched at
`_local/fusion/task28_real/`, containing `validation.json`, `summary.md`, three
pair manifests and independent `post_validation.json` evidence. Keep the source
verification evidence with the completed run. No generated manifests, downloaded
files, tool packages or private paths should be committed.

**198 unit tests passed in 7.016 seconds**: 196 prior tests plus the two narrow
mirror-header regression tests. No Phase 1 or Tasks 2.1--2.7 behavior changed.
Runtime configuration, quotas, target semantics and hard-negative gating remain
unchanged. Full metadata verification, real generation and deterministic
serialization checks are complete. Future loss and source-exposure experiment
decisions must follow the mandatory [evaluation protocol](fusion_pair_generator.md#evaluation-protocol):
validation for selection, early stopping and calibration with reuse recorded;
freeze model and threshold before final held-out test; no test-derived mining,
tuning or calibration; report challenges separately; preserve official splits
and acknowledge 10/8/0 source overlap. Supplementary source-disjoint sensitivity
results must be distinguished from official evaluation. Targets express provenance,
not physical-causality proof.

No W&B run was created. Future actual GC2 training/evaluation experiments must
use `WANDB_BASE_URL=https://api.forge.coreweave.com`,
`WANDB_ENTITY=GC2-AValidate`, `WANDB_PROJECT=avalidate`. No credentials were read
or stored, and no training was started.

All future actual training/evaluation runs must also log the mandatory
[configurations, metrics, model versions, checkpoints and experiment records](fusion_pair_generator.md#experiment-tracking).
No W&B run is required for reference verification or this remediation.

## Final audit remediation verification

The tracked verifier passed against the available pinned author files: **4,143
annotations, 3,339/402/402 split records, zero mismatches**, 4,097 source clips,
and 45 sources with multiple annotations. The empty sample remains at author
index 26 in train, with ten False activity bins and UNKNOWN event compatibility.
New evidence is under ignored `_local/fusion/ave_reference_verified/`;
`verification.json` SHA-256 is
`5398cc498eeec2133276da5d2d7fdc19de3d5691979fa8e0782187c638ba724c`.

An isolated checkout smoke check containing only the required verifier source,
base contracts, supplied text files and pinned references produced byte-identical
evidence. It had neither the ignored historical helper nor the blocked report.
All 28 existing integration input hashes still match; the new four-role metadata
hash map agrees with those inputs. Production pair manifests and reports were
not regenerated or rewritten. Pair-generation and Phase 1 source remain unchanged.

The full suite passed **218 tests in 8.378 seconds**: the existing 198 plus 20
offline verifier regressions. The optional HDF5 reader is not required by the unit
suite; the actual reference verification used the already available isolated reader.
The Phase 1 merge dependency remains separate Git work; this remediation does not
commit, push, merge or create a PR.

# Phase 1: fusion artifact contracts

## Scope and repository evidence

`fusion/contracts.py` reads explicit local export paths, validates arrays and
annotation metadata, reports leakage, and returns deterministic join indices.
It imports NumPy and the Python standard library only. It does not download data,
modify exports, load models, generate training pairs, calibrate scores, or train.

Checked-out repository behavior verified before implementation:

- `audio/data_ave.py::load_split()` and `vision/data_ave.py::load_split()` use
  sorted event classes and `sample_id = clip_id + '__' + two-digit class index`.
- Both `second_labels()` functions use the half-open annotation interval.
- `audio/features.py::split_table()` can omit rows without WAVs, whereas
  `vision/features.py::build_split_metadata()` retains rows without frame caches.
  Matching row order or counts must therefore never be assumed.
- `audio/export_embeddings.py::main()` exports embeddings, probabilities, IDs,
  `class_names.json`, and `SOURCE.txt`. The final external handoff additionally
  requires its README. The checked-out exporter does not publish `audio_final/`.
- `vision/export_embeddings.py::main()` currently exports embeddings, IDs, and
  `SOURCE.txt`; it discards probabilities returned by `vision.evaluate.predict()`.
- `vision/extract_frames.py::extract_one()` requests frame index `int(t * fps)`
  for t=0..9. It can repeat/replace failed frames, without exporting validity or
  actual timestamps. The existing demo in `src/` remains a separate prototype.

The external audio source `beats_lora_r8_v1` / feature source `beats_iter3` is the
team-confirmed expectation, not a binary verification or a value inferred from
the directory name. Real artifacts are not included in Phase 1's synthetic tests.

## Files and numeric contracts

Pass a filesystem directory and modality (`audio` or `vision`) explicitly. Split
names are `train`, `val`, `test`; use `val`, not `validation`, in filenames/API calls.
Each NumPy array's row i must describe row i of its own IDs CSV.

| File | Audio final | Vision |
| --- | --- | --- |
| `{split}_emb.npy` | Required `(N,10,256)`, float16 | Required `(N,10,Dv)`, Dv>0, float16/32/64 |
| `{split}_ids.csv` | Required | Required |
| `{split}_probs.npy` | Required `(N,10,29)`, float16 | Optional `(N,10,29)`, float16/32/64 |
| `class_names.json` | Required | Required when probabilities exist; otherwise optional |
| `SOURCE.txt` | Required provenance | Optional provenance |
| `README.md` | Required final handoff description | Optional |
| `{split}_valid.npy` | Optional bool `(N,10)` | Optional bool `(N,10)` |
| `{split}_timestamps.npy` | Optional float16/32/64 `(N,10)` | Optional float16/32/64 `(N,10)` |

Validity and timestamp filenames are the **new contract layer's optional sidecar
convention**, not a claim that current teammates' exporters produce these files.
For vision, validity means the sampled frame represents real input at that bin.
For audio, it means the bin has usable real audio under the producer's policy.
No validity policy is inferred from the waveform/embedding values.

All floating arrays must be finite; embeddings are not required to be normalized.
Integer, complex, object, and unsupported floating dtypes are rejected. Mask
integers are not coerced to booleans. Arrays use read-only NumPy memory mapping
with `allow_pickle=False`; object files and archives disguised as `.npy` fail.
Sources are never rewritten, reordered, renormalized, or cast in place.

Probabilities must lie strictly in [0,1]. Their sums are accumulated in float64
and must differ from one by at most 0.001 (absolute tolerance, zero relative
tolerance), allowing float16 storage rounding. NaN/Inf never pass this check.

Class mappings are explicit JSON lists of 29 unique, nonempty names: 28 events in
Python alphabetical order, followed by exactly `background`. No sorting or
automatic column remapping takes place. Different semantic vocabularies between
modalities fail even if their dimensions agree. The external mapping is the
semantic authority; the module does not copy the demo's hardcoded class list or
invent an independent AVE vocabulary. Therefore a jointly incorrect but internally
consistent external vocabulary cannot be detected without authoritative labels.

Official row counts (3339/402/402) are enforced by default. For deliberate subsets
or synthetic fixtures, explicitly pass `expected_counts=None` or a custom mapping.
This choice does not waive row/array consistency or identity validation. Dynamic
vision dimensions must remain consistent across loaded splits.

## Partial bundles and errors

`load_bundle()` returns a `Bundle`; `validate_bundle()` returns a `ValidationReport`.
`complete` means all required inputs for the **requested** splits are present and
valid. Requesting only `train` is not validation of val/test. Missing required
files appear as full paths in `missing_inputs`. Malformed arrays, IDs CSVs, and
class mappings raise `ContractError` instead of being treated as absent. CSV UTF-8
decoding failures include the path and retain `UnicodeDecodeError` as their chained
cause. Unrelated I/O failures are not silently swallowed. A nonexistent root fails.

`validate_bundle()` independently checks the current requested splits, required
arrays/IDs, record identities and intervals, mappings, and array row counts. It
adds missing-field diagnostics even when a constructed/replaced bundle has an
empty stored `missing_inputs` tuple. The original diagnostics remain conservative
evidence of missing inputs; clearing them cannot waive structural checks. Audio
README/source availability is also checked against the loaded provenance state.
Configured expected counts are retained in `Bundle.expected_counts` and checked
on revalidation, including when bundles are reconstructed.

Vision without probabilities and without a class mapping can be structurally
complete. Its report explicitly says semantic mapping validation is pending;
alignment then checks observed visual category/index identities against the audio
mapping. Missing a mapping while probabilities exist keeps the bundle incomplete;
the report says probability semantics remain pending. It never invents columns.

Missing README/source files keep the strict final audio handoff incomplete. This
lets older audio exports be inspected without calling them the final contract.
Metadata remains available for leakage auditing even when embeddings are missing.

An audio-only bundle can be complete as an audio artifact. Calling alignment
without vision returns a pending result, never a completed cross-modal alignment.

## Identity and deterministic joining

CSV columns must include `sample_id`, `clip_id`, `category`, `start`, `end` with
unique headers. IDs/categories cannot be blank or have outer whitespace. Clip IDs
use letters, digits, underscore, and hyphen; exact YouTube ID length is not
enforced. The sample ID must be its exact clip ID followed by `__00` through
`__27`. Category/index associations must be consistent across available splits
and agree with any supplied mapping. There is no whitespace repair or ID coercion.

`align_on_sample_id(audio, vision, split='train')`:

1. Validates both bundles and their available semantic mappings.
2. Matches exact sample IDs in original audio row order.
3. Checks matched clip ID, category, class index, start, and end for equality.
4. Returns matched IDs plus original `audio_indices` and `visual_indices`.
5. Reports every unmatched audio ID and visual ID; it never silently drops them.

Two annotations of one source video remain separate samples. Never join by
`clip_id`, row position, or filename sorting. A match of duplicate source videos
is not itself evidence of a unique annotation match.

Call `Alignment.require_complete()` before any future consumer indexes arrays.
An incomplete result contains a diagnostic matched subset, **not a training-ready
intersection**. Missing inputs from either loaded bundle also prevent completion.
Complete alignment means every source row on both sides is matched exactly once:
`len(sample_ids) == audio_row_count == visual_row_count`. An empty match set,
including two empty source bundles, cannot pass the guard.

`Alignment` also carries `audio_row_count` and `visual_row_count`, taken from
each validated source embedding array's actual first dimension. They are not
inferred from the number of matches or maximum index. A missing source array has
an unavailable count (`None`) and cannot produce a complete alignment.

The `complete` property and `require_complete()` share the same current-state
structural checks. Both index arrays must be one-dimensional integer NumPy arrays
(booleans excluded), with exactly one index per unique aligned sample ID. Each
modality's indices must be unique, nonnegative, and less than its own explicit
source row count, and their length must equal that count. Together these checks
require each index array to be a permutation of `0..N-1`, while allowing shuffled
visual ordering. Truncating IDs and both index arrays together, or omitting an
interior row, cannot certify a subset as complete. Missing/unmatched records still
prevent completion. Invalid counts, shapes, dtypes, lengths, or duplicate
IDs/indices are never repaired;
`complete` returns false and `require_complete()` raises `ContractError` with the
reason and available partial-input diagnostics.

Index arrays and ID/diagnostic sequences are copied into read-only array/tuple
snapshots on construction. Directly constructed or `dataclasses.replace()`-created
alignments must supply true source counts; older nonempty manual constructions
without counts intentionally no longer pass the guard. Structural validity and
declared bounds cannot prove semantic identity for arbitrary in-bounds permutations
or fabricated source counts. Obtain normal results from `align_on_sample_id()`
and retain their relationship to the same source arrays. This remains a fixed
one-to-one identity join, not a representation for generated negative pairs.

### Immutable record order and public API compatibility

`Bundle.splits` and `Bundle.metadata` are read-only `Mapping` snapshots, with tuple
record sequences and frozen `Record` objects. Construction copies caller-owned
dictionaries and record lists; later mutation of those containers cannot alter
the bundle. Dictionary item assignment is intentionally no longer supported.
Read access such as `bundle.metadata['train']` remains compatible.

For any split with arrays, `SplitArtifacts.records` is the authoritative array
row sequence. The metadata view is retained for compatibility and IDs-only
partial bundles. Validation **requires exact membership and order equality**
between both representations whenever array records are present. Missing,
reversed, truncated, or empty metadata cannot yield a complete alignment with
nonempty arrays. Alignment uses the array-associated sequence after validation.

To construct an intentional alternate bundle, use new dictionaries and
`dataclasses.replace()`, then validate it. Replacing only metadata is an error;
reordering a real export requires reordering every associated array and its
records together. No validator can infer semantic row identity from arbitrary
embedding values if a caller deliberately mislabels both record representations.
`SplitArtifacts.records` and `.embeddings` accept `None` for explicitly missing
inputs in constructed partial bundles. Present empty records are malformed, not
a complete zero-sample split. The external file shapes are unchanged.

The format has no per-row IDs embedded in NumPy arrays. A CSV shuffled without
the corresponding array rows violates the producer contract but cannot be
detected from arbitrary embedding values alone. Hashes establish file identity,
not that the producer attached the correct semantic row IDs. The shuffled-row
test permutes visual CSV and array rows together and verifies preserved indices.

## Temporal semantics and masks

Ten fixed index bins represent [0,1), [1,2), ..., [9,10). The nominal vision
sampling point is the start of each bin (frame-at-second t), preserving the
existing interface. The module does not resample or learn offsets.

`annotation_activity(records)` returns bool `(N,10)` computed using
`start <= t < end`. Start/end must be integer seconds with
`0 <= start < end <= 10`, with the sole empty-interval exception `[0,0)`
described below. Invalid intervals are rejected rather than clipped.
For [6,8), only bins 6 and 7 are active. Booleans and noninteger values,
including floating-point `0.0`, are not accepted as integer seconds.

### Official AVE empty annotation compatibility (Phase 1.1)

The official training annotation is retained by both teammate exporters:

```csv
VWi2ENBuTbw__08,VWi2ENBuTbw,Church bell,0,0
```

The [official AVE annotation file](https://github.com/YapengTian/AVE-ECCV18/blob/master/data/Annotations.txt)
contains `Church bell&VWi2ENBuTbw&good&0&0`. Both
`audio/data_ave.py::second_labels()` and `vision/data_ave.py::second_labels()`
leave all ten labels as background for this empty slice. Their metadata/export
paths retain the row and its original clip category.

The contract accepts exactly integer `[0,0)` as an explicit AVE convention;
this is an interval-format exception, not an ID-based bypass of validation.
It does not permit other empty intervals such as `[5,5)` or `[10,10)`, negative
values, reversed intervals such as `[9,2)`, fractional seconds, or endpoints
outside [0,10]. All identity, class-mapping, array, count, alignment coverage,
and leakage checks still apply.

`annotation_activity()` returns ten False bins for this row: there are no
annotated active joint-event seconds. The sample keeps its original sample ID,
clip ID, `Church bell` category/index, array row, and training membership. It
remains required for complete representation alignment and leakage auditing.
No category, probability, embedding, validity flag, or timestamp is rewritten.
Absent input validity remains unknown; supplied input validity remains independent
of the all-false annotation activity.

For Phase 2, representation retention is separate from eligibility for a loss.
This row must not become a positive annotated event merely because its clip
category is `Church bell`. Its all-false activity can reproduce the producer's
event/background convention, but is not evidence that either modality is absent
or invalid. Any use for presence supervision requires an explicit task policy;
independent audio/visual presence cannot be inferred from it.

Treat its event-compatibility status as **unknown**. Exclude or mask this sample
from semantic event-compatibility/correspondence supervision and its labeled
metrics until a separately justified policy or annotation resolves that status;
do not silently assign MATCH or MISMATCH from category or empty activity.
Original clip pairing is known provenance, not proof of an active compatible
event. Keep the row in the bundles and joins even when a future loss mask excludes
it. Phase 1 supplies no correspondence labels or target-supervision mask.

Keep these concepts separate:

- **Annotation/event presence:** returned activity of the annotated joint AVE
  event. Background does not mean the underlying audio or frame was missing.
- **Input validity:** producer-supplied `{split}_valid.npy`, otherwise `None`.
  Event activity never fills an unavailable validity mask.
- **Target supervision:** a future task-specific mask, not produced in Phase 1.
  Neither correspondence labels nor independent audibility/visibility ground
  truths can be inferred from joint event activity alone.

Real timestamps are `None` when absent. Supplied timestamps must be finite,
nondecreasing, and clip-relative in [0,10). Every position declared valid must
fall in its corresponding bin. If validity is unknown, all supplied timestamps
are checked for bin consistency, without asserting the inputs themselves valid.
Explicitly invalid positions may retain a repeated earlier timestamp, but still
must be finite and inside the clip. There is no implicit NaN-as-unknown convention.

Matching bins and even in-bin timestamps are **not proof of decoded stream
synchronization**. Stream offsets, approximate frame seeking, variable frame rate,
audio padding, receptive fields, or unknown timestamp conventions can matter.
BEATs pooled tokens are contextual and do not isolate exactly one audio second.
The validator does not verify codec presentation timestamps or audiovisual causes.

## Leakage reporting

`audit_leakage(bundle)` distinguishes:

- Duplicate annotation sample IDs within each split (normal loading rejects them).
- Identical sample IDs across splits (`sample_overlap`).
- Shared source-video IDs across splits (`source_overlap`).
- Different annotation sample IDs sharing one clip inside each split
  (`multiple_annotations`).

It reports unavailable metadata splits, and never reassigns/removes samples.
Audit each modality before considering alignment, so unmatched rows are included.
Known historical official-split overlaps must be recomputed from real manifests;
the module does not hardcode previous notebook counts. Different uploads of the
same source under different clip IDs require a separate content-level audit.
Source-disjoint fusion evaluation cannot undo overlap exposure in pretrained or
already fine-tuned unimodal encoders.

## Provenance and read-only behavior

`Bundle.provenance` is JSON-serializable and records resolved artifact paths,
filenames, byte sizes, modification times, original array shapes/dtypes, per-split
sample counts and embedding dimensions, class mapping, and verbatim available
`SOURCE.txt`. Missing provenance stays missing; no model version is guessed.
`hash_files=True` adds SHA-256 checksums in bounded chunks. Hashing is off by
default. Size/mtime without hashing are lightweight identity hints, not cryptographic
verification. Keep the source bundle stable while loading/using memory maps;
Phase 1 does not lock external writers or create immutable snapshot copies.
Validation checks the loaded in-memory structure and provenance, not whether a
file was subsequently deleted or replaced on disk. Reload to inspect changed
external files. Read-only mappings protect record order; they do not make arbitrary
caller-supplied NumPy arrays or provenance dictionaries deeply immutable.

### Bundle serialization and worker processes

`pickle.dumps(bundle)` is unsupported because `MappingProxyType` is not picklable.
Phase 1's loading/validation/alignment interface does not require serialization,
and a future CPU-only pair generator can operate on bundle data in one process
without pickling it. A future multiprocessing implementation is not provided or
assumed by this phase. No custom pickle hooks or unsafe artifact loading are added.

For worker processes, pass explicit artifact paths and load worker-local bundles,
or design a separate serializable dataset representation when that functionality
is needed. Do not pass a `Bundle` directly through a process queue or assume a
dataset containing one can be serialized under Windows multiprocessing spawn.

## Usage and future Phase 2 boundary

```python
from fusion import load_bundle, validate_bundle, align_on_sample_id, audit_leakage

audio = load_bundle('/explicit/path/audio_final', 'audio')
print(validate_bundle(audio))
print(audit_leakage(audio))

vision = load_bundle('/explicit/path/vision_export', 'vision')
aligned = align_on_sample_id(audio, vision, split='train')
aligned.require_complete()
a = audio.splits['train'].embeddings[aligned.audio_indices]
v = vision.splits['train'].embeddings[aligned.visual_indices]
```

The indexed arrays can later feed pair manifests, a probability-agreement baseline
(only when both probabilities/mappings exist), and concatenation-MLP fusion.
IDs and annotation metadata remain joining/validation/supervision information;
never concatenate IDs, categories, intervals, or targets into predictive features.
Unknown validity and task-supervision policies must be resolved before training.
No Phase 2 functionality is implemented by this usage example.

## Lightweight environment and tests

Use an isolated project-local Python 3.11 environment. Only NumPy is required;
tests use standard-library `unittest` and temporary synthetic files.

```powershell
uv venv --python 3.11 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements-fusion.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Downloads require the environment's normal permission/network approval. The actual
validated environment for this change is Python 3.11.16 with NumPy 2.4.6. Python
is installed under ignored `_local/python`, uv cache under `_local/uv-cache`, and
the virtual environment under ignored `.venv`; the system Python is unchanged.
Dependencies are ranged in `requirements-fusion.txt`, not a full environment lock.

Regression fixtures use deterministic SHA-256 byte patterns per sample ID rather
than per-class constants. Same-category clips therefore have distinct synthetic
rows. Tests check that join indices select those actual rows, and cover metadata
reversal/membership mismatches, immutable snapshots, missing required fields in
constructed/replaced bundles, invalid records, and chained CSV encoding errors.
These byte patterns are test markers only, never a proposed predictive feature.
Alignment guard regressions cover both modalities' index lengths, dtypes, rank,
negative/out-of-bounds/duplicate indices, duplicate sample IDs, absent/invalid
source counts, reconstructed containers, immutable snapshots, and partial-input
diagnostics. Existing shuffled-order and same-category tests verify actual row
selection with unchanged valid alignment results. Full-coverage regressions reject
simultaneous truncation, unequal source coverage even with cleared unmatched
diagnostics, and interior-row omissions from either modality. Full permutations
and distinct annotations sharing one source clip remain accepted; empty sources
cannot become a training-ready alignment.

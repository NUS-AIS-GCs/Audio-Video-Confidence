# Phase 3 Tasks 3.1–3.3: probability correspondence baselines

These modules consume Phase 2 pairs and predicted unimodal probabilities. They
require only NumPy and the standard library, with no changes to Phase 1/2,
teammate audio/vision implementations, or production manifests. No neural model,
threshold selection, calibration, final test evaluation, or W&B run is included.

The binary target remains **original recording pair versus substituted pair**.
The four scores measure predicted semantic agreement; they cannot establish
recording provenance, waveform synchrony, visible sources, or physical causality.
Background agreement and same-category substitutions can both score highly.

## Preflight evidence

Checked on 2026-10-22 (Asia/Singapore):

- Remote: `NUS-AIS-GCs/Audio-Video-Confidence`; the initial working tree was clean.
- Switched from `feature/fusion-phase2-pairs` to the required existing
  `feature/fusion-phase3-baselines`. Both pointed to `0e67c34` before this work.
- [Phase 2 PR #13](https://github.com/NUS-AIS-GCs/Audio-Video-Confidence/pull/13)
  was confirmed **open**, with no merge timestamp. This branch depends on its
  implementation; its presence locally does not imply an upstream merge.
- Real exports exist under ignored `_local/embeddings/`; validated Phase 2
  manifests exist under ignored `_local/fusion/task28_real/`.
- Phase 1 contracts, Phase 2 pair policy/generator/integrity/artifact integration,
  real-data validation evidence and the existing evaluation protocol were read.
- Baseline complete suite: **218 tests passed**, 23.149 seconds.

## Public interfaces and separation of responsibilities

| Module | Interface | Purpose |
| --- | --- | --- |
| `fusion.baseline_data` | `extract_pair_probabilities(manifest, audio, vision, *, split, limit_per_partition=None)` | Validate a `GeneratedManifest` and both bundles, then select original probability rows |
| | `ProbabilityFeatures(audio, visual)` | Read-only copies of only `(P,10,29)` prediction arrays |
| | `ProbabilityPartition`, `PairProbabilities` | Keep canonical primary/challenge pairs alongside, outside features |
| `fusion.baselines` | `top1_agreement`, `dot_product_agreement`, `cosine_similarity`, `jensen_shannon_similarity` | Each takes only two prediction arrays and returns `(P,10)` bin scores |
| | `score_probabilities(audio, visual)` | Return `BaselineScores.per_bin` and `.pair_scores` for all four methods |
| `fusion.baseline_metrics` | `make_score_records(partition)` | Score features first, then attach unchanged Phase 2 pair metadata |
| | `ranking_metrics(targets, scores)` | Numeric ROC-AUC/AP utility for synthetic checks or approved train/val evaluation |
| | `summarize_scores(values)` | Deterministic score distribution summary |
| | `evaluate_records(records, *, split, partition)` | Evaluate exactly one train/val partition; reject test and mixed inputs |

`extract_pair_probabilities` calls the existing
`validate_generated_manifest`, which reuses Phase 1 artifact validation and
complete alignment and Phase 2 reference/provenance validation. It rejects
negative/out-of-range indices, mismatched sample/source records, split conflicts,
invalid arrays, inconsistent policy/counts and invalid partition membership before
extracting any features. It requires predicted probabilities in both modalities;
vision embeddings alone are insufficient. Bundles must be passed in audio/vision
order. The adapter does not infer official membership independently: real callers
must retain the Phase 2 official-reference/inventory validation evidence.

Every audio row comes from `pair.audio.row_index`; every visual row comes from
`pair.visual.row_index`. Row orders may differ between modalities. Output order is
the supplied canonical order within each separate partition. An optional
nonnegative integer limit takes the first K pairs **in each partition after full
validation**, for deterministic structural smoke tests. It does not rewrite the
manifest or claim that its prefix satisfies the full generation quotas. Empty
partitions have `(0,10,29)` feature arrays and produce no score records.

Selected rows with any producer-supplied invalid bin are rejected: this batch's
all-ten-bin aggregation has no partial-validity weighting policy. Missing validity
is retained as unknown, not inferred from annotation activity or predictions.
There is no waveform alignment or timestamp resampling: only equal bin indices
0 through 9 are compared.

`ProbabilityFeatures` contains exactly `audio` and `visual`. Ground-truth classes,
targets, IDs, pair types, annotation intervals and masks never enter the scorer.
Sources are not normalized, cast, reordered or overwritten in place. Selected
arrays are copied into read-only snapshots; as elsewhere in these contracts,
read-only NumPy flags are an accidental-mutation safeguard, not a security
boundary against a caller deliberately changing flags or bypassing constructors.

Empty-annotation samples retain their predictions and original row identities.
Their provenance target still follows Phase 2, while semantic compatibility
remains UNKNOWN and semantic/event eligibility remains masked. No active event,
semantic class target or input-validity mask is invented for them.

## Scores and numerical conventions

Inputs to scoring functions have equal floating shapes `(P,10,29)`, use
float16/32/64, and contain finite values in `[0,1]`. Positive-mass vectors must sum
to one within Phase 1's absolute tolerance `0.001`; sums are checked and
renormalized on float64 copies, without changing exports. Logits and arbitrary
unnormalized nonnegative features are rejected. Float64 accumulation limits
float16 storage error.

For normalized vectors p and q in each corresponding bin:

| Method key | Bin score | Convention |
| --- | --- | --- |
| `top1` | `1[argmax(p) == argmax(q)]` | Ties use the lowest vocabulary column |
| `dot_product` | `sum(p * q)` | Includes all classes; uniform agreement is `1/29` |
| `cosine` | `dot(p,q) / (norm(p) * norm(q))` | Identical nonzero vectors score 1 |
| `jensen_shannon` | `1 - JSD_base2(p,q)` | Uses JSD divergence, not its square root |

For Jensen–Shannon, `m=(p+q)/2`,
`JSD=H(m)-(H(p)+H(q))/2`, with base-2 entropy and `0 log2(0)=0`.
The entropy implementation avoids ratios involving midpoints that underflow to
zero. Exact-zero probabilities and subnormal float64 entries remain finite.
Numerical roundoff is clipped into `[0,1]` after valid input checks.

The low-level scorer additionally accepts an exact all-zero vector as an explicit
**abstention**. If either side abstains, every method returns zero for that bin,
including when both sides are zero. Zero is not treated as a normalized
probability distribution or as perfect background/top-1 agreement. Real artifact
loading still rejects zero-mass probability vectors under unchanged Phase 1 rules.

All methods point in the same direction: **higher means more compatible**.
`score_probabilities` averages each method's ten bin scores arithmetically, with
aggregation identifier `mean_all_10_bins`. There is no foreground-only variant,
annotation/activity weighting, missing-bin renormalization, learned weight,
threshold or fitted transform. No alternative aggregation is justified in this
batch. Shared background confidence can dominate all four methods; this is a
documented limitation to assess in future experiments, not corrected with labels.

## Result and metric contracts

`PairScoreRecord` stores the unchanged Phase 2 `pair`, explicit partition,
canonical `(method, score)` tuples and aggregation identifier. Its `identity`,
`split`, `pair_type` and binary `target` are derived from that pair. Original sample
IDs, source clip IDs and row references remain available through `pair.audio` and
`pair.visual`. These fields are evaluation metadata, not predictive inputs.

`evaluate_records` requires unique, canonical records from exactly the named
split/partition. It accepts only `train` or `val` in this batch. It reports total,
positive, negative and per-pair-type counts, plus for each method:

- ROC-AUC: the probability that a positive outranks a negative, counting tied
  positive/negative comparisons as one half. Ascending tied score groups are
  accumulated without dependence on row order within ties.
- Average precision: `sum(delta_recall * precision)` at distinct descending
  score thresholds, including an entire tied group at once. This is the
  noninterpolated stepwise PR integral, **not trapezoidal PR-AUC**. A separate
  trapezoidal PR-AUC statistic is not supplied or mislabeled as AP.
- All/positive/negative score distributions: count, min, quartiles, median, max,
  mean and population standard deviation (`ddof=0`). Quartiles use NumPy's
  explicitly selected linear interpolation. No data-dependent histogram bins.

`MetricValue(value=None, undefined_reason=...)` explicitly marks empty or
one-class input. Both AUC and AP are undefined by this evaluation contract when
either class is absent, including all-positive AP. Empty distributions have
count zero and all numeric fields `None`. Same-category challenges contain only
negative provenance targets, so their standalone ranking metrics are normally
undefined; retain their score distributions without adding primary positives or
mixing the challenge distribution into the main task.

The numeric `ranking_metrics` utility cannot infer split provenance from arrays.
Use `evaluate_records` for provenance-aware reports; do not bypass its test guard
to calculate held-out performance in this implementation phase. Threshold-based
classification metrics, calibration and operating decisions remain Tasks 3.4–3.5.

## Usage with a validated in-memory manifest

```python
from fusion.baseline_data import extract_pair_probabilities
from fusion.baseline_metrics import make_score_records, evaluate_records

# audio, vision and manifest come from the existing Phase 1/2 APIs.
data = extract_pair_probabilities(manifest, audio, vision, split="train")
primary_records = make_score_records(data.primary)
challenge_records = make_score_records(data.challenge)
# Evaluation utility example for future approved train/val use:
# primary_report = evaluate_records(primary_records, split="train", partition="primary")
# challenge_report = evaluate_records(challenge_records, split="train", partition="challenge")
```

For persisted manifests, use the existing `manifest_from_json` with its expected
artifact inventory fingerprint. Keep the Phase 2 `validation.json` completion
marker and manifest digest, and freshly verify source hashes and official
reference membership before use. The reader regenerates and compares the entire
typed document; it does not authenticate external model checkpoints.

## Verification and scope

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
git diff --check
```

`tests/test_fusion_baselines.py` adds **27 synthetic tests**, covering independent
modality row order, original/substituted/hard pairs, empty annotations, canonical
prefixes and empty partitions, feature/target isolation, unchanged source arrays,
invalid references, missing probabilities, modality and split isolation, explicit
invalid input bins, NaN/Inf rejection, all four methods, contradictory/identical
distributions, background dominance, zero norms, float16 normalization,
subnormal entries, symmetry, deterministic ordering and fixed aggregation.
Ranking tests compare AUC/AP against independent pairwise/threshold oracles over
all 81 four-score combinations with three possible score values, as well as tie
permutations, one-class/empty states, score distributions and record validation.
A temporary file-backed test round-trips the Phase 2 serialized contract and
closes its memory maps before Windows temporary-directory cleanup.

Final verification: **245 tests passed** (218 existing + 27 new), 25.626 seconds.
The required command ran through the existing `.venv` with bytecode generation
disabled. Test output is retained locally in ignored
`_local/fusion/phase3_unit_tests.txt`.

The real-data structural smoke test passed on 2026-10-22:

| Scored partition | Split | Pairs | Audio shape | Visual shape | Row/identity mismatches | All four scores finite |
| --- | --- | ---: | --- | --- | ---: | --- |
| Primary | train | 64 | `(64,10,29)` | `(64,10,29)` | 0 | Yes |
| Same-category challenge | train | 64 | `(64,10,29)` | `(64,10,29)` | 0 | Yes |

The canonical first 64 pairs per partition came from the existing production
`train_pairs.json`. The local helper reused Phase 2 `_export_paths`, `_inventory`,
`_check_hashes`, `_load_exports`, `_verify_official` and `manifest_from_json`
rather than implementing another binary validator. All consumed input hashes and
the production train-manifest digest matched the previous Phase 2 completion
report, official metadata matched again, and hashes/manifest bytes remained
unchanged afterwards. These checks establish local byte continuity, not remote
authenticity or checkpoint provenance.

The existing safe reader compares the complete typed document, including its
full-context source-overlap report. Consequently its structural validation opened
the complete bundles read-only, including held-out metadata/arrays. **Only the
train prefixes were extracted and scored**; val/test pairs were not deserialized
or scored, and no real-data metrics, score distributions, threshold decisions or
performance estimates were computed. This is structural smoke evidence, not an
experimental evaluation. No claim of loading only train files is made.

The local smoke helper and structural JSON are retained under ignored
`_local/fusion/phase3_smoke.py` and `phase3_smoke_results.json`. To repeat in this
workspace after checking the existing input roots and Phase 2 completion marker:

```powershell
.\.venv\Scripts\python.exe -B -m _local.fusion.phase3_smoke
```

The helper is local verification evidence, not an additional tracked runtime
dependency. New workspaces should use their own explicit artifact paths and the
existing Phase 2 verification workflow before invoking the adapter.

Git scope consists only of three new Phase 3 modules, this document and the new
test file. Existing tracked files, dependencies, Phase 1/2 contracts, policies,
teammate implementations and production manifests are unchanged. All local data
and verification outputs remain ignored. No commit, push or merge was performed.

All future actual GC2 evaluation/training experiments must use:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

Synthetic tests and structural smoke checks create no W&B run. Full real-data
experiments/logging belong to Tasks 3.6–3.7. No API keys are needed or stored.
This implementation stops at Tasks 3.1–3.3; threshold/calibration work is ready
for a later authorized batch, subject to the existing evaluation protocol and
pending Phase 2 dependency review.

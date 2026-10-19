# Phase 2 Tasks 2.2--2.5: deterministic pair generation

`fusion/pair_generator.py` implements positive, easy-negative, policy-selected
hard-negative, and same-category challenge generation from explicit Phase 1
bundles. It performs no downloads, file writes, model loading, training, or W&B
calls. Development validation uses synthetic fixtures only; no full AVE pair
manifest or real-artifact validation is claimed.

The independently reviewed [Task 2.1 policy](fusion_pair_policy.md) remains the
labeling authority. No Phase 1 or Task 2.1 code or semantics are changed.

## Public interfaces and configuration

Import from `fusion.pair_generator`:

- `GenerationConfig`: one split's pair budgets and selection controls.
- `SourceReuse`: `ALLOW` or `UNIQUE_PER_TYPE`.
- `HardGenerationStatus`: explicit enabled/disabled reason.
- `GenerationProvenance`, `PairTypeCounts`, `GenerationReport`, `GeneratedManifest`:
  frozen, validated output metadata.
- `generate_pairs(audio, vision, *, config, policy) -> GeneratedManifest`.
- `validate_generated_manifest(manifest, audio, vision) -> PairValidationReport`.

The seed and optional approved `HardNegativePolicy` belong to the existing
`SamplingPolicy`. There is no hidden global seed or inferred hard taxonomy.

```python
from fusion import load_bundle
from fusion.pair_contracts import SamplingPolicy
from fusion.pair_generator import GenerationConfig, generate_pairs

# Explicit local paths; this example is not executed as part of development.
audio = load_bundle('/explicit/path/audio_final', 'audio')
vision = load_bundle('/explicit/path/vision_final/resnet18_v1', 'vision')
manifest = generate_pairs(
    audio, vision,
    policy=SamplingPolicy(seed=42),
    config=GenerationConfig(
        split='train', positive_budget=None,
        easy_negative_budget=100, hard_negative_budget=0, challenge_budget=20,
    ),
)
# Separate distributions: only an explicit later decision may mix challenges.
primary_pairs = manifest.primary_pairs
challenge_pairs = manifest.challenge_pairs
```

| Configuration | Default | Meaning |
| --- | --- | --- |
| `split` | required | Exactly train, val, or test |
| `positive_budget` | None | All aligned annotations; integer requests exactly that many positives |
| `easy_negative_budget` | 0 | Exact global easy-negative count for this split |
| `hard_negative_budget` | 0 | Exact global hard-negative count for this split |
| `challenge_budget` | 0 | Exact global challenge count in the separate challenge partition |
| `enable_hard_negatives` | False | Explicit enablement in addition to a valid approved policy |
| `source_reuse` | ALLOW | Reuse recordings across distinct pair identities |
| `max_candidate_evaluations` | 20,000,000 | Limit on the substitution Cartesian-product scan |

These defaults establish no negative ratio or training distribution. Budgets must
be nonnegative integers; booleans, strings and fractional values fail. An explicit
all-zero request is unsupported because Task 2.1 requires a nonempty manifest.
Individual partitions may be empty, including a challenge-only result whose
`primary_pairs` is empty. Unsupported enum members, subclasses and malformed
reconstructed inputs are rejected through class-owned validation.

## Eligibility and pair identities

Generation first reuses Phase 1 validation and complete sample-ID alignment.
All source rows must align, regardless of output pair budgets. Original modality
row indices and unchanged Records supply source identities and metadata.

- Positives pair exactly the same sample ID and annotation metadata. With the
  default positive budget, every source annotation receives a positive pair.
- Easy substitutions have different source clip IDs, sample IDs and categories,
  and are outside every directed hard edge in the supplied policy.
- Hard substitutions have different source IDs/categories and exactly follow a
  supplied directed hard-policy edge. Both explicit enablement and a structurally
  valid approved policy are required.
- Challenges have matching category/index and different source clip/sample IDs.
  They remain individually marked and are stored only in `challenge_pairs`.

Every pair's identity remains `(split, audio_sample_id, visual_sample_id)`.
Source reuse does not permit duplicate identities. Different annotations sharing
one clip stay distinct positives; neither may serve as a different-source negative
for the other. Sample IDs, clip IDs, category labels and annotation intervals are
metadata, never added to predictive feature tensors.

An explicit positive budget smaller than the aligned count selects a positive
subset; it does not truncate the source bundles or negative candidate universe.
Every omitted positive ID is listed in `report.unselected_positive_ids`. The
default retains all rows, including the official empty `[0,0)` sample.

Targets and supervision flags are derived by `PairRecord`, not reimplemented by
the generator. The empty sample keeps its provenance identity and an all-false
annotation activity mask; semantic compatibility is UNKNOWN and event/semantic
supervision remains masked. No input validity is inferred. A same-category swap
has provenance target 0 and category-proxy compatibility when nonempty. No pair
claims verified temporal synchrony, physical causality, or waveform-shift evidence.

## Hard-negative gating and evidence

No approved hard policy is bundled with the generator. Without one, normal
generation reports `DISABLED_NO_POLICY`; an enabled hard request fails explicitly.
A supplied policy with enablement False reports `DISABLED_BY_CONFIG`. A positive
hard budget while enablement is False is an invalid configuration.

Even when generation is disabled, a supplied hard policy's edges remain excluded
from easy negatives, preserving Task 2.1's disjoint bucket definitions. Eligible
hard counts may therefore be nonzero while the requested/selected hard count is
zero. Approval text is not authenticated by code. External review must confirm
the approved evidence and freeze it before evaluation.

The existing contract recursively checks evidence references, SHA-256 formatting,
approval text, vocabulary, directed edges and origin. Validation-confusion policies
must declare `val`, never test; test scores cannot be supplied as a generator
configuration or used to derive neighborhoods here. The generator does not
compute confusion matrices, choose policies, or access held-out outcomes for
training-time candidate ranking.

## Deterministic selection and source reuse

Task 2.1's stable split seed and SHA-256 candidate priority are used unchanged.
The generator reuses the canonical hash helper and priority schema, then constructs
selected PairRecords to derive and revalidate labels. Ties use visual sample ID,
audio sample ID, and pair type. No RNG state, Python process hash, dictionary
insertion order, or filesystem order determines selection.

With `ALLOW`, a bounded heap retains the lowest-ranked requested number of
candidates per type. Many-to-one and one-to-many substitutions are valid when
their complete pair identities differ. Candidate memory is proportional to the
requested budgets rather than the full Cartesian product.

`UNIQUE_PER_TYPE` means each audio source clip is used at most once and each
visual source clip at most once **within each substitution type**. The two roles
are separate; a recording may appear once in each role. Positives are exempt,
so multiple annotations are not collapsed. Easy, hard and challenge pools have
independent reuse constraints, keeping challenge selection isolated from primary
selection. This is not a global no-reuse rule across all pair types or splits.

For this mode, sample edges sharing an audio/visual recording pair are collapsed
to the best-ranked eligible annotation edge. Deterministic breadth-first
augmenting paths find a source matching of the requested size. An early greedy
choice is rerouted when necessary; a feasible quota is not rejected merely
because greedy selection failed. This mode needs memory proportional to its
eligible source-edge graph and may perform more work than heap selection.

All candidates are drawn only from the configured split. Generating other splits
does not consume a shared random stream or alter this split's choices. Reordering
CSV rows changes the preserved indices and source fingerprints, but not eligible
sample identities or their priority ranking under otherwise identical metadata.
Final primary and challenge tuples are each canonically ordered by Task 2.1's
`pair_order_key`.

## Exact budgets, bounded work and reports

Requests exceeding eligible pair counts raise `ContractError`. Unique-source
requests additionally fail if the graph cannot provide the requested matching.
There is no replacement sampling, silent quota reduction, alternate category
fallback, cross-split borrowing, or unbounded rejection loop. Selection produces
all requested quotas or returns no manifest.

Eligible pool sizes and exclusions are counted using category/source histograms.
Substitution selection scans at most `audio_rows * visual_rows` combinations once
when any substitution budget is positive. A request above the configured scan
limit fails before candidate enumeration. Positive-only generation does not scan
that Cartesian product. The scan limit does not bound all matching work or the
memory needed for requested output; large requests still need resource review.

`GenerationReport` includes:

- Eligible, requested, selected and unselected pair counts for all four types.
- Full audio/visual source row counts and explicit omitted positive IDs.
- Candidate evaluations, same-source exclusions and same-sample exclusions.
  Same-sample exclusions are a subset of same-source exclusions, not additive.
- An explicit hard-generation status.
- Task 2.1's source-overlap and missing-metadata-split report, including source
  records not selected into output pairs.

Pool/exclusion counts describe the full configured split, including types with
zero requested budgets. They are not losses or class-balance metrics.

`GenerationProvenance` binds the generator version, every requested configuration
field, seed/hard-policy fingerprint, and Task 2.1's modality-specific metadata
fingerprints. It records an explicit None/all positive budget differently from an
integer budget even when they select the same current rows. It does not hash
embedding/checkpoint bytes or export a manifest file.

Before return, `generate_pairs` revalidates all selected PairRecords with Task 2.1
and `validate_generated_manifest`. The latter independently checks original row
references, unique identities, split/policy consistency, canonical ordering,
challenge isolation, exact quotas, settings/source fingerprints, source reuse,
and histogram-derived counts/exclusions. For externally supplied manifests it
does not recompute the entire ranked selection: it is a structural/quota validator,
not proof that an external producer chose precisely the lowest-ranked candidates.

## Leakage and limitations

Official partition assignments remain unchanged. The prior manifest audit found
10 shared source videos across train/val, 8 across train/test and 0 across val/test.
Fresh overlap reports come from the supplied Phase 1 metadata, not those historical
constants. Pair generation adds no cross-split borrowing. Existing source overlap,
including unimodal encoder exposure, still limits source-disjoint evaluation
claims. Auditing available held-out metadata for leakage is distinct from using
its labels or scores to choose training pairs.

Sampling ratios and supervision/loss policy still require experiment decisions
within the evaluation protocol below. Explicit test-split generation uses that split's
metadata solely for its own evaluation pair construction; it must not feed back
into training policy selection.

## Evaluation protocol

- Use training data only for model fitting. Use validation for model selection,
  early stopping, threshold calibration and any other selected calibration method.
  Record any reuse of validation for multiple purposes, including the selection
  and calibration procedure; do not present reused estimates as independent.
- Select and freeze the model and operating threshold before final held-out test
  evaluation. Never use test-derived hard-negative mining, architecture selection,
  hyperparameter tuning or calibration. Test metadata used to construct its own
  evaluation pairs or audit leakage must not influence training-negative selection.
- Report same-category challenge results separately from primary
  original-versus-easy-negative evaluation. Keep challenges in their separate
  partition; they are not part of the primary training distribution in this protocol.
- Preserve official AVE split assignments and acknowledge source-video overlaps
  of 10/8/0 (train/validation, train/test, validation/test). Any future
  source-disjoint sensitivity evaluation must be clearly labeled supplementary
  and distinguished from official AVE evaluation; it cannot erase prior encoder
  exposure or replace the official results silently.
- Original-versus-substituted targets are recording-provenance labels, not proof
  of physical sound-source causality. Empty annotations remain UNKNOWN for event
  compatibility, with the existing supervision masks; no modality validity is inferred.

Real artifact binary validation remains pending. No actual full AVE pairs, temporal
shift augmentation, serialization/export pipeline, model features, baseline,
training or Tasks 2.6--2.8 are implemented by this work.

## Experiment tracking

These tasks generate pair metadata and create **no W&B training run**. All future
actual training/evaluation experiments are expected to log to:

- Server: `https://api.forge.coreweave.com`
- Entity: `GC2-AValidate`
- Project: `avalidate`

The mandatory environment settings are:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

Every actual model-training and evaluation experiment must log the following,
where applicable (record why a field is inapplicable):

- Model name, architecture and version; Git commit, run name and random seed.
- Dataset split, artifact hashes and pair-generation policy; training
  configuration and hyperparameters.
- Training and validation loss curves; accuracy, precision, recall, macro-F1,
  ROC-AUC and PR-AUC; confusion matrix and threshold-selection method.
- Calibration metrics and per-second metrics when applicable; same-category
  challenge results reported separately from primary evaluation.
- Model checkpoint and final evaluation artifacts, with the W&B run URL in
  experiment records.

Configure authentication through the environment's normal secret mechanism.
Never hardcode credentials or API keys in code, configuration examples, manifests,
or documentation. Do not log personal credentials or raw confidential data.
Metadata verification and this remediation do not train a model and require no run.

## Validation

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests use identifiable synthetic rows and cover all four types, exact quotas,
source exclusions, masks, deterministic priorities and process hash randomization,
row permutations, overlap reporting, insufficient pools, reuse modes, malformed
settings/manifests, candidate-work limits, and unchanged source arrays. Small
matching graphs are checked exhaustively against brute-force feasibility.

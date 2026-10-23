# Phase 4 Tasks 4.1–4.3: learned correspondence foundation

Status: **COMPLETE — foundation audit and test-portability remediation passed.
READY FOR TASKS 4.4–4.6.** The complete suite now runs 348 tests with 347 passes,
no errors/failures and one conditional CUDA skip. Phase 1–3 production code is
unchanged; the Phase 3 runner tests now isolate repository preflight. See the
audit record below for the independently reproduced root cause and one corrected
Phase 4 float-conversion defect. No subsequent task was started.

## Scope and files

New files:

- `fusion/embedding_data.py`: validated embedding dataset and immutable provenance.
- `fusion/mlp_model.py`: configurable shared temporal-bin MLP and correspondence head.
- `fusion/training_contracts.py`: configuration, objective, minibatch/device/seed
  interfaces, split guards, backward safeguards and checkpoint metadata.
- `fusion/smoke_mlp.py`: read-only, untrained TRAIN-only integrity smoke CLI.
- `tests/test_fusion_mlp.py`: offline synthetic unit/integration and smoke-boundary tests.
- `requirements-fusion-mlp.txt`: opt-in PyTorch dependency over existing fusion requirements.
- This document.

This batch contains no training loop, real-data optimization, threshold fitting,
final-test scoring, trained checkpoint, or W&B run. No commit, push, merge or
production manifest rewrite was performed. GC1 was not accessed.

## Data contract and use

`EmbeddingDataset(manifest, audio, vision, split=..., partition=...)` requires
complete Phase 1 bundles and uses the audited Phase 2 manifest serializer/auditor.
`EmbeddingDataset.from_json(...)` first uses the audited safe manifest reader,
including its artifact-inventory fingerprint. It never edits or regenerates
production files; the safe reader regenerates expected metadata in memory for
comparison, as prescribed by Phase 2.

Full manifest construction requires every original annotation, including the
legitimate empty-annotation row. Phase 2 checks exact quotas, original source
rows, identities, ordering, source constraints, partitions and provenance. Full
coverage means coverage of the validated bundle/count context: production bundles
use official counts; synthetic fixtures explicitly provide their own counts.
Deliberately reduced original-positive manifests fail. The optional `limit` is
applied after full validation and marks the resulting dataset incomplete, even
if its numerical limit happens to exceed the partition size. Optimization and
selection loaders reject incomplete datasets.

Primary pairs permit original positives and authorized cross-class easy negatives
only. Same-category challenges use a separate dataset with `partition="challenge"`;
they never enter default loaders or selection. Hard policies are outside this
batch and are also unsupported by the upstream Task 2.8 serialization contract.

Each pair gathers `audio.row_index` and `visual.row_index` independently from the
original producer order. This supports shuffled modality exports. Source arrays
are copied once per modality into immutable byte-backed NumPy storage, preserving
producer dtype. Pair features are gathered lazily, avoiding storage proportional
to duplicated negative-pair embeddings. Finite source values outside float32
range are rejected before conversion can produce infinite minibatch tensors.
Each item returns:

```python
features, target = dataset[index]
# features["audio"]: float32 (10,256)
# features["visual"]: float32 (10,512)
# features["validity"]: bool (10,)
# target: separate float32 (1,)
```

Minibatches are `(B,10,256)`, `(B,10,512)`, boolean `(B,10)` and separate targets
`(B,1)`. For inspection, `dataset.features` materializes immutable `(P,10,256)`
and `(P,10,512)` arrays; `dataset.targets` remains separate. No event label,
sample ID, pair type, provenance, or annotation interval enters embedding tensors.
`dataset.pairs` retains the original frozen Phase 2 pair records. Provenance
records canonical manifest SHA-256, pair-policy fingerprint, optional verified
artifact context, shape/dtype/content source-array hashes, and any file hashes
present in loaded bundle provenance. Use `load_bundle(..., hash_files=True)` when
file-hash inventory is needed in checkpoint evidence. In-memory array hashes and
an externally supplied artifact-context string do not authenticate source files;
the smoke check separately verifies original Phase 2 byte hashes and sizes.

Explicit producer input-validity masks are intersected for the selected source
rows. Missing masks use all ten exported bins as an explicit operating assumption;
this does not verify physical input validity or decoded waveform synchronization.
AVE event intervals never determine these masks. All-masked selected pairs fail.
NaN/Inf values fail even in masked bins. The `[0,0)` annotation retains ten inputs
and its original positive provenance target, with semantic compatibility UNKNOWN.
No new semantic label or independent audio/visual presence target is introduced.

## Model contract

Default `MLPConfig()`:

```text
audio (B,10,256) + visual (B,10,512)
  -> concatenate (B,10,768)
  -> shared Linear(768,512), ReLU, Dropout(0.2)
  -> shared Linear(512,256), ReLU, Dropout(0.2)
  -> shared Linear(256,128), ReLU, Dropout(0.2)
  -> explicit-mask arithmetic temporal mean (B,128)
  -> Linear(128,1) correspondence logits (B,1)
```

Parameter count: **558,081**. This is an initial configurable architecture, with
no claim of optimality. Hidden widths, ReLU/GELU, dropout and initialization seed
are configurable. Validated AVE dimensions and ten bins are fixed. There is no
BatchNorm or fixed batch-size requirement. Initialization uses an isolated CPU
RNG context and preserves the caller's RNG state. Move the model and feature
tensors to the same CPU/CUDA device and dtype before calling it.

`forward(audio, visual, validity=None, return_probabilities=False,
return_bin_representations=False)` returns `FusionOutput`. Raw logits are always
available; optional probabilities are sigmoid(logits). Optional per-bin hidden
representations are inspection features, **not independently supervised temporal
verification predictions**. Invalid inputs are zeroed before the MLP, and their
hidden representations are excluded from the mean. All-masked pairs, malformed
shapes/masks, nonfinite inputs and nonfinite outputs fail closed. Evaluation and
smoke checks call `.eval()` to disable dropout.

## Training and checkpoint interfaces

`TrainingConfig` rejects unknown fields, invalid seeds/sizes/devices, nonfinite
hyperparameters, unsupported balance policies and non-Forge destinations.
`OptimizerConfig` supports Adam/AdamW with explicit learning rate, weight decay,
betas and epsilon. `seed_everything` seeds Python, NumPy and PyTorch, sets
deterministic algorithm behavior and disables cuDNN benchmarking. Determinism
still depends on hardware, backend versions and supported kernels; metadata must
record them. CUDA is supported when an appropriate PyTorch build/device exists;
an unavailable requested device fails instead of silently falling back.

The objective is mean `binary_cross_entropy_with_logits`. Targets must be finite
floating binary `(B,1)` tensors, separate from inputs, with no gradients and the
same dtype/device as logits. Default class balance is an unweighted mean over
pairs: arbitrary primary class counts are allowed, but complete optimization and
selection datasets must include both target classes. One-class minibatches are
valid. An explicit positive-class weight is permitted only with the matching
balance policy; any ratio must be derived from TRAIN counts, never test data.
No automatic balancing, event-label weighting or resampling occurs.

Before future training, call `validate_split_isolation(train, validation)` and
record its returned source-overlap IDs. It rejects overlapping annotation/sample
identities. It preserves the official AVE split rather than claiming source
disjointness: real source-video overlaps are 10 train–validation, 8 train–test,
and 0 validation–test. These are existing annotation-split limitations, not
permission to borrow rows across splits. Phase 2 rejects cross-split pairs and
same-source substituted negatives.

`make_loader(..., role="train")` requires complete TRAIN primary data and uses a
seeded shuffle. `role="val"` requires complete validation primary data and keeps
canonical order. Both preserve final partial/single-item minibatches. `test` and
challenge roles fail. Worker seeds, CPU defaults, optional pinning and explicit
device transfer are provided. `move_batch` accepts only the three feature keys
and separate targets. It rejects metadata added to the feature dictionary.

Future minibatch orchestration should use:

```python
overlap = validate_split_isolation(train_data, validation_data)
seed_everything(config)
device = resolve_device(config)
model = LateFusionMLP(model_config).to(device)
optimizer = build_optimizer(model, config.optimizer)
loader = make_loader(train_data, config, role="train")
# In the future experiment runner, after online Forge tracking is established:
# optimizer.zero_grad(set_to_none=True)
# features, targets = move_batch(batch, device)
# loss = correspondence_loss(model(**features).logits, targets, config)
# backward_checked(loss, model, max_gradient_norm=config.max_gradient_norm)
# optimizer.step()
```

`backward_checked` checks finite scalar loss, missing/nonfinite gradients and
gradient norm, then clips. Failure clears gradients and raises before an optimizer
step. It does not mutate optimizer state or perform a step. Orchestration must
call `zero_grad` first and step only after the guard succeeds. Validation must
use `.eval()` and inference/no-grad context and remain reserved for epoch/model
selection. Final-test examples are never optimization or selection data.

`CheckpointMetadata` is a versioned JSON metadata schema with strict JSON
round-trip reconstruction and fingerprint verification. It can record:

- Full model/configuration and training/optimizer settings, seeds and Forge destination.
- Full Git revision and a source-code SHA-256 (needed when the working tree differs).
- TRAIN/validation manifest, policy, array and available export-file hashes;
  explicit audio/vision export identities and Python/NumPy/PyTorch versions.
- Official source-overlap evidence, epoch, optimizer step/state-presence declaration,
  selected epoch and bounded finite validation selection metrics.

The stable experiment fingerprint excludes epoch progression and selection
results; it includes configurations, versions, code and data evidence. Changing
hyperparameters/data changes it. The schema rejects final-test provenance and
selection without supporting validation metrics. Actual model/optimizer/RNG
states belong in a future checkpoint payload; this batch writes no checkpoint.

## Original implementation evidence (2026-10-22; before remediation)

Install Phase 4 dependencies in the existing ignored environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-fusion-mlp.txt
```

For CPU-only installation the official PyTorch CPU index can be used separately.
Verified environment: Python 3.11, NumPy 2.4.6, PyTorch 2.14.1+cpu; CUDA unavailable.
Requirements retain repository-style compatible version ranges; experiment
metadata records the actual resolved versions rather than claiming a lockfile.

Pre-change complete suite: **308 tests, 11 errors**. After implementation:
**343 tests: 331 passed, 11 errors, 1 CUDA skip**. The **35 new Phase 4 tests**
have **34 passes and one conditional CUDA skip**. CPU forward/backward, one
synthetic optimizer minibatch, deterministic initialization/loading, BCE behavior,
empty annotations, shuffled rows, input/target isolation, masks, immutable sources,
checkpoint validation and smoke source/held-out boundaries are covered.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_fusion_mlp.py -v
```

The 11 unchanged errors originate in Phase 3 runner tests calling its production
`_repository()` check from `feature/fusion-phase4-mlp`. That check requires
`feature/fusion-phase3-baselines`. It is an upstream test-isolation/branch
portability issue, not a Phase 4 numerical failure. It was left unchanged in the
original implementation batch and corrected in the subsequent audit below.

Real smoke command:

```powershell
.\.venv\Scripts\python.exe -B -m fusion.smoke_mlp `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --integration-root _local/fusion/task28_real --limit 8
```

Result: **PASS**. Original SHA-256 and sizes matched for 28 producer/official
source files and three opaque manifest files. Only the TRAIN manifest was
deserialized. Full structural all-split context was validated because the audited
manifest binds that context; no validation/test embeddings entered the model.
Eight deterministic TRAIN primary pairs (4 original positives, 4 easy negatives)
produced `(8,10,256)` audio, `(8,10,512)` visual and `(8,1)` logits. Logits and
sigmoid probabilities were finite; original row-reference/value mismatches were
zero. Before/after hashes confirmed producer inputs, evidence and manifests were
unchanged. Scores were not printed or interpreted as performance. Safe producer
run IDs were `beats_lora_r8_v1` and `resnet18_v1`; historical producer test metrics
are excluded from smoke output. No real-data backward pass occurred.

Local evidence remains ignored: `_local/fusion/phase4_full_suite.log` and
`_local/fusion/phase4_real_smoke.json`. Source embeddings/manifests and `.venv`
remain ignored, with no tracked local artifacts. At the original implementation
handoff all pre-existing tracked files were unchanged and seven Phase 4 files
were new. The subsequent audit additionally modifies the Phase 3 test file.

## Forge readiness and Git dependencies

Future actual training/evaluation experiments must establish online tracking at:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

Configuration and checkpoint metadata enforce this destination. The future
runner must validate runtime tracking settings/access and require successful
logging; this foundation does not implement or claim that orchestration. Never
log credentials or arbitrary SDK exception text. Every actual experiment must
log architecture/versions, seed/hyperparameters, Git revision, data/manifest
fingerprints, train/validation losses, ROC-AUC, AP, accuracy, F1, checkpoints and
selected epoch, calibration/threshold evidence and separate same-category
challenge results. Synthetic tests and untrained integrity smoke checks create
no W&B run. No authentication or training-readiness claim is inferred from them.

Confirmed GC2 origin: `https://github.com/NUS-AIS-GCs/Audio-Video-Confidence.git`.
Required branch: `feature/fusion-phase4-mlp`. Initial tracked/index/untracked state
was clean. HEAD remains `00838cc3fdec0c4073c410940701b785a11f986c`, the Phase 3
implementation commit. A fresh remote check and fetch of main's history showed:

- Phase 2 merged through PR #13; remote main is
  `1846d9076eea354ddf463469921314c9dfb05a05`.
- Phase 3 tip is `00838cc3fdec0c4073c410940701b785a11f986c` and is **not** an
  ancestor of current remote main. Phase 4 depends on that unmerged implementation.
- Relative to refreshed `origin/main`, this branch has one unique Phase 3 commit
  and lacks main's Phase 2 merge commit (the Phase 2 implementation itself is present).

No working branch switch, integration merge, commit or push was made. The local
`main` branch was not used to claim fresh upstream merge status.

Confirmed limitations after remediation: CUDA
execution unverified on this CPU build; known official source-video overlap;
producer declarations plus byte continuity do not prove cryptographic encoder
checkpoint provenance. No full-model performance claim is made. Stop after
Tasks 4.1–4.3 and their audit; no subsequent real training is authorized here.

## Foundation audit and test-portability remediation

The audit started on the same GC2 branch and HEAD, with seven untracked Phase 4
files and no tracked changes. A fresh pre-remediation complete run reproduced
343 tests, 11 errors and one skip. Every traceback was checked separately: all
11 ended at `fusion/run_baseline_experiments.py:80`, where production
`_repository()` requires `feature/fusion-phase3-baselines`. The original tests
were then loaded directly from `git show HEAD:tests/test_fusion_baseline_experiments.py`
and the same 11 failures were reproduced independently of the new test fixture.
This confirms they predate the uncommitted Phase 4 implementation.

The minimal Phase 3 test correction supplies fixed synthetic `_repository()`
metadata in `RunnerTests.setUp`, with cleanup after each test. One regression
test makes `_git` raise if the synthetic runner queries the actual checkout.
Three separate tests call the real production preflight with mocked Git replies:
the intended Phase 3 branch passes; main, Phase 4 and unrelated branches fail;
wrong origins and dirty tracked/index state still fail. Production preflight,
hash checks, artifact validation, challenge separation and held-out guards remain
unchanged. All 26 Phase 3 experiment-module tests pass on the Phase 4 checkout.

One confirmed Phase 4 defect was corrected: Phase 1 accepts finite float64 vision
exports, but an extreme finite value could overflow to Inf in the adapter's
float32 tensor conversion. A reproducer confirmed finite producer values and an
infinite emitted tensor. The adapter now rejects values outside finite float32
range at construction; a regression covers both signs, unchanged producers and
normal float64 conversion. No clipping, rescaling or source mutation was added.

Independent audit calculations used NumPy layer-by-layer matrix products and
ReLUs, explicit valid-bin means and the final linear head. Batches of 1, 3 and 7
matched the 558,081-parameter default MLP with maximum absolute logit error below
`8.2e-9`. Invalid-bin perturbations left logits exactly unchanged. Eval-mode
dropout was exactly repeatable, while training-mode dropout remained active.
Stable NumPy logistic-loss calculations and analytical derivatives matched
unweighted and positive-weighted BCE, including logits of ±1000; loss absolute
error was below `6e-14`. Independent Euclidean norms and gradient scaling matched
the clipping interface, and a first-layer finite-difference derivative agreed
within `1e-8` absolute error. These synthetic calculations performed no optimizer
steps or real-data training.

The audit also independently checked explicit modality-mask intersection,
immutable public arrays, source overlap reporting, all train/validation/test
loader-role combinations, challenge exclusion, Python/NumPy/PyTorch seed repeats,
AdamW hyperparameters without stepping, checkpoint JSON round-trip and six
tampering cases covering architecture, optimizer settings, versions, manifests,
source hashes and Forge configuration. Existing tests additionally cover empty
annotations, original row identities, shuffled exports, all-masked rejection,
single-item minibatches and finite-gradient failure cleanup. Checkpoint metadata
records declarations/hashes, not proof of encoder checkpoint origin or pretrained
fused weights. Future real-experiment tracking remains the next runner's duty.

Final complete command: `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v`.
Result: **348 tests, 347 passes, zero errors/failures, one unavailable-CUDA skip**.
The 36 Phase 4 tests have 35 passes and that skip. Syntax parsing passed for all
29 fusion/test Python files, `pip check` found no broken requirements, whitespace
and scoped credential-pattern checks passed. The security check is a source and
scope inspection, not a dependency vulnerability-database certification.

The real TRAIN-only smoke was rerun after the adapter fix: eight pairs (four
positive/four easy negative), audio `(8,10,256)`, visual `(8,10,512)`, logits
`(8,1)`, finite untrained outputs, zero source-row/value mismatches and unchanged
hashes for all 28 source files, three manifests and the integration evidence.
Test manifests were hashed as opaque evidence; no final-test scores were read.
There was no real-data optimization, checkpoint save or W&B run.

Files changed in this remediation/audit batch: `tests/test_fusion_baseline_experiments.py`,
`fusion/embedding_data.py`, `tests/test_fusion_mlp.py` and this document. The other
Phase 4 files and all Phase 1/2, teammate and Phase 3 production code/manifests
remain unchanged. Local audit script and evidence stay ignored under `_local/fusion/`:
`audit_phase4_foundation.py`, `phase4_independent_audit.json`,
`phase4_audit_original_head.log`, `phase4_audit_before.log`,
`phase4_audit_full_suite.log` and `phase4_audit_real_smoke.json`.

A fresh remote check confirms main remains `1846d9076eea354ddf463469921314c9dfb05a05`
and Phase 3 remains at `00838cc3fdec0c4073c410940701b785a11f986c`, unmerged into
main. The Phase 4 branch still depends on that Phase 3 implementation. No commit,
push, branch switch, PR creation or merge was performed. No remaining confirmed
foundation correctness defects were found. CUDA verification and official
source-video overlap remain documented limitations. **READY FOR TASKS 4.4–4.6**
means the foundation passed this audit; those tasks were not begun.

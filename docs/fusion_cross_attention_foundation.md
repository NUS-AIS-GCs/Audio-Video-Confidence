# Phase 5 Tasks 5.1–5.3: bidirectional cross-attention foundation

Execution date: 2026-10-24, Asia/Singapore. Scope: architecture, reusable training
interfaces, offline numerical tests and an untrained TRAIN-only integrity smoke.
No full real-data training, threshold fitting, final-test scoring or live W&B run
is part of this batch. Tasks 5.4–5.6 remain future work.

Status after the independent audit and remediation: **FIXED — READY FOR FOCUSED RE-AUDIT.**

## Preflight and dependency

The repository origin is
`https://github.com/NUS-AIS-GCs/Audio-Video-Confidence.git`; the working branch is
`feature/fusion-phase5-cross-attention`, HEAD `38d4171`. The initial tree was
**not completely clean**: `requirements-fusion-training.txt` was already untracked.
Its bytes are preserved, and it is not staged. No GC1 path was accessed.

Remote `git ls-remote` confirmed main at
`467c7568def85d3f9270756ce1ce98d861958c45` and Phase 4 at
`76add86568daed6f7dbb31659f26c020b264962c`. Git ancestry confirms Phase 4 is not
in main, but is included in this branch through merge `38d4171`. Phase 4 remains
an upstream branch/PR dependency for any eventual Phase 5 integration. Main
contains the Phase 3 merge, PR #14. No new PR was created or inspected for status.

The preflight suite passed: 412 tests, one CUDA skip. The adapter, Phase 4 MLP,
training contracts, epoch/prediction functions, experiment runner, tracker,
checkpoint and comparison/study contracts, and Phase 2 manifest/source verifier
were inspected before integration.

## Model and outputs

`fusion.cross_attention_model.BidirectionalCrossAttention` accepts only:

| Input | Required shape and meaning |
|---|---|
| `audio` | floating `(B,10,256)`, original producer features |
| `visual` | floating `(B,10,512)`, original producer features |
| `validity` | explicit boolean `(B,10)` on the same device |

Features must match the model's device and floating dtype. Batch size is arbitrary
and nonempty; singleton batches work. There is no BatchNorm. Configurations retain
the producer dimensions and ten temporal indices, and reject incompatible head
counts, invalid dimensions, nonfinite/out-of-range dropout and unsupported modes.
The residual/LayerNorm architecture requires `d_model >= 2`.

Defaults are `d_model=256`, `num_heads=4`, `attention_dropout=0.1`, residual and
feed-forward `dropout=0.1`, `feedforward_dim=512`, GELU, fixed sinusoidal positions,
masked-mean aggregation, and initialization seed 42. `feedforward_dim=0` disables
the feed-forward updates; `position_encoding="none"` explicitly selects a future
position ablation. Dimensions, heads, dropout, activation and seed are configurable.

Audio and video have separate learned linear projections into `d_model`.
The two branches consume the same original positioned projections in parallel:

* **Audio to video:** Q = positioned audio, K = positioned video, V = positioned video.
* **Video to audio:** Q = positioned video, K = positioned audio, V = positioned audio.

Each branch computes post-norm residual updates:

```text
x = LayerNorm(query + Dropout(MultiheadAttention(query, context, context)))
y = LayerNorm(x + Dropout(Linear(Dropout(GELU(Linear(x))))))
```

The second line is omitted when the feed-forward width is zero. Invalid query
outputs are zeroed after the updates. Branch outputs concatenate at each original
index into `(B,10,2*d_model)`. A validity-weighted temporal mean feeds one linear
correspondence head. There are no event or temporal classification heads.

The frozen `CrossAttentionOutput` object contains raw `logits: (B,1)` and optional
`probabilities: (B,1)` (`sigmoid(logits)`), `bin_representations: (B,10,2*d_model)`,
`audio_to_video_attention` and `video_to_audio_attention`. Each map is
`(B,num_heads,query_bin,key_bin)` and is available via `return_attention_weights=True`.
Representations are not temporal verification scores. Attention weights do not
establish physical causality. Probabilities have no calibration claim.

| Default component | Trainable parameters |
|---|---:|
| Audio and visual projections | 197,120 |
| Two multihead attention modules | 526,336 |
| Four LayerNorm modules | 2,048 |
| Two feed-forward modules | 525,824 |
| Correspondence head | 513 |
| **Total** | **1,251,841** |

The fixed temporal-position buffer is not a trainable parameter.

## Temporal and masking semantics

Position `t` is the existing one-second producer index, from 0 through 9:
`PE(t,2i)=sin(t/10000**(2i/d_model))` and
`PE(t,2i+1)=cos(t/10000**(2i/d_model))`. The same buffer is added to both
projected modalities. Gaps are never removed or renumbered. Position information
is explicit; cross-attention itself has no assumption of temporal order. Attention
can mix any valid bins, with no causal or diagonal-only restriction.

The adapter gathers the original audio and visual producer rows separately and
returns the intersection of their explicit validity masks. This foundation keeps
that conservative contract: an index invalid in either modality is excluded from
both streams as a key, value, query output and aggregation position. It does not
retain separately usable tokens from the other modality at that index. Independently
masked streams would require a future adapter/API revision, not invented event
labels. Missing producer masks use all ten bins as the existing export assumption;
AVE event annotations never determine validity.

Every input value must be finite, including excluded bins. Masked features are
zeroed **before** projection, avoiding overflow from large finite invalid values.
PyTorch `key_padding_mask=~validity` excludes invalid keys/values. Returned maps
explicitly zero invalid rows and columns. All-masked examples reject the entire
batch before attention. Intermediate projections, positioned features, attention
updates/maps, residuals, feed-forward activations, normalization, pooling and final
logits have finite-value guards. Inputs and masks are never modified in place.

Maps are the actual **post-attention-dropout** weights. In evaluation, every valid
query row sums to one and invalid rows sum to zero. Training dropout means valid
row sums need not be one. Invalid-bin invariance in training assumes the same RNG
state, since independent calls otherwise draw different dropout masks.

Initialization uses a fork of the CPU RNG, leaves the caller's RNG unchanged and
does not initialize a CUDA context. Standard `.train()`, `.eval()`, `.to(device)`
and `.double()` work. CPU float32/float64 are verified; CUDA is conditional and was
unavailable locally. Mixed precision/autocast and device-specific reproducibility
have not been validated in this batch.

## Shared training and checkpoint interfaces

`fusion.model_interface.build_model(config)` selects the architecture from strict
`MLPConfig` or `CrossAttentionConfig`. Both models accept the dataset feature schema
and return `.logits`, so the existing `make_loader`, `move_batch`, `train_epoch`,
`predict`, `correspondence_loss`, optimizer construction, seed/device contracts and
gradient safeguards work directly. No epoch loop or evaluation policy is copied.
The index side channel still verifies distinct original-index full-epoch coverage
and keeps the final singleton minibatch. Model features contain only audio, visual
and validity; targets and metadata cannot enter `forward()`.

Targets remain original-versus-substituted pairing provenance. Optimization uses
complete TRAIN primary pairs only; checkpoint selection uses validation primary
only. Challenges require an already frozen model/threshold for diagnostic inference.
Final-test data are rejected for optimization and model selection. The known AVE
official source-video overlap remains the Phase 4 split limitation; annotation
identities cannot overlap or move splits.

Checkpoint metadata accepts the cross-attention configuration only under
`gc2-fusion-cross-attention-checkpoint-v1`. MLP metadata retains the original
`gc2-fusion-checkpoint-v1` schema, serialized fields and experiment fingerprints.
Schema/architecture mismatches fail. The existing safe checkpoint saver/loader
retains hash/size checks, `weights_only=True`, strict state loading and model/epoch/
experiment fingerprint checks; reconstruction now uses the model factory. Both
architectures have JSON metadata and synthetic weight reload tests.

The historical `fusion.train_mlp` experiment configuration and CLI remain MLP-only,
including its comparison names and historical selection policy. Phase 5 does not
reuse MLP-labeled experiment outputs. A Phase 5 experiment entry point and
prespecified real experiment settings belong to Tasks 5.4–5.6.

Keep the following decisions separate in that future entry point:

1. Architecture configuration and architecture selection.
2. Training/optimizer settings, data provenance, seeds and device.
3. Best epoch selection on validation primary data (the existing Phase 4 epoch
   runner uses strictly greater ROC-AUC and earliest exact ties).
4. Frozen threshold fitting only after checkpoint selection.
5. Separate challenge diagnostics after all applicable selection decisions freeze.

No real threshold was fitted or selection metric computed here. Shared numeric
configuration validation does not by itself prespecify a future experiment.

## Forge readiness

`ForgeCrossAttentionTracker` specializes the existing tracker with group
`phase5_bidirectional_cross_attention` and `phase5-cross-attention-*` artifact
names. Phase 4 groups, default run names and artifact names remain unchanged.
The destination stays:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

The subclass inherits the same authenticated readiness-artifact upload/read-back,
online-only enforcement, final remote summary/history/artifact verification, local
integrity callback **before** successful finalization, bounded retries, redacted
failures and idempotent failure cleanup. Tests use mocked SDKs only. No live
authentication, Forge access check or W&B initialization was performed.

## Verification and real smoke

The new CPU-friendly tests cover projections, Q/K/V directions, batches 1/3/7,
config validation, optional feed-forward blocks, deterministic initialization,
train/eval dropout, both-modality/all-parameter gradients, feature/metadata boundaries,
device/dtype checks, explicit masks, mixed validity, key/query exclusion,
normalization, all-masked rejection, invalid-bin invariance and zero invalid input
gradients, finite-value guards, original temporal positions, position ablation,
no batch leakage, independent producer masks and original row references.

A manually specified two-dimensional, one-head attention block with identity
Q/K/V/output projections is checked using independent Python scalar dot products,
exponentials, value sums and residual LayerNorm arithmetic. Weights and outputs
agree to `1e-12` in float64. Synthetic training checks include BCE, one optimizer
step, a complete irregular-batch epoch, duplicate-index rejection, split/challenge
guards, checkpoint JSON/state reloads and mocked Phase 5 Forge finalization order.

Run the required complete suite with:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Initial foundation result: **438 tests run, 436 passed, two CUDA skips, 70.812 seconds**, including 26
new foundation tests. The first targeted run exposed an undersized synthetic
final-test fixture whose requested negative-pair quota exceeded its eligible
pairs; the fixture was expanded and the targeted run passed. No production defect
remains confirmed. Syntax parsing, dependency compatibility, whitespace, Git scope,
new-file sizes and credential-pattern checks passed. The index remains empty.

The real smoke is reproducible without tracking or training:

```powershell
.\.venv\Scripts\python.exe -B -m fusion.smoke_cross_attention `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --integration-root _local/fusion/task28_real --limit 8
```

Observed CPU environment: PyTorch `2.14.1+cpu`, NumPy `2.4.6`, CUDA unavailable.
`pip check` passed without dependency changes. The smoke reused the audited
Phase 2 source verifier: 28 source/reference files and all three opaque manifest
hashes were checked. Only the TRAIN manifest was deserialized and TRAIN primary
features reached the model. All-split array/metadata loading is structural context
required by the existing verifier, not held-out inference.

Results: eight deterministic pairs (four originals, four easy substitutions),
audio `(8,10,256)`, video `(8,10,512)`, boolean mask `(8,10)`, logits `(8,1)` and
both maps `(8,4,10,10)`. All values are finite, masks/normalization are correct,
all eight pairs use ten exported bins, original row-reference mismatches are zero,
and source files/evidence/manifests are unchanged. Source runs are
`beats_lora_r8_v1` and `resnet18_v1`. TRAIN manifest SHA-256 is
`52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f`.
No performance was calculated from these untrained logits.

Ignored local evidence is in `_local/phase5_preflight_tests.log`,
`_local/phase5_full_tests.log`, `_local/phase5_real_smoke.json` and
`_local/phase5_preservation_before.json`. Four historical Phase 4 checkpoints
(initial, E1, E2, E3) successfully reloaded read-only with unchanged recorded model
and experiment fingerprints. A preservation inventory covers 280 protected files
(126,941,543 bytes), including producer exports, original manifests, Phase 3
baselines, Phase 4 checkpoints and historical W&B evidence; the final hash
comparison confirms every inventoried file is unchanged.

## Files and boundaries

Added: `fusion/cross_attention_model.py`, `fusion/model_interface.py`,
`fusion/cross_attention_tracking.py`, `fusion/smoke_cross_attention.py`,
`tests/test_fusion_cross_attention.py` and this document.

Modified: `fusion/training_contracts.py` (architecture-bound checkpoint parsing),
`fusion/train_mlp.py` (checkpoint reconstruction factory), `fusion/mlp_tracking.py`
(overridable names/groups with identical Phase 4 defaults), and `fusion/smoke_mlp.py`
(shared integrity smoke implementation, retaining its public MLP behavior).
The original MLP architecture, embedding adapter, Phase 1–3 code, producer exports,
original manifests and historical experiment files are unchanged.

No commits, pushes, merges, staged files or PR creation are part of this work.
The pre-existing untracked requirements file remains the only unrelated Git item.
CUDA execution and actual online Phase 5 training remain unverified until the
corresponding hardware and authorized later tasks are available.

## Independent-audit remediation: checkpoint binding and minimum dimension

Both confirmed defects were reproduced offline before production changes. The
original saver/loader accepted a sinusoidal model with metadata declaring
`position_encoding="none"`; the loaded buffer retained 76 nonzero entries for
`d_model=8`. The original accepted `d_model=1` model produced identical predictions
for different inputs and zero modality gradients because LayerNorm normalized
each one-dimensional token to a constant.

`CrossAttentionConfig` now rejects `d_model < 2` before any module construction.
Positive head counts, exact divisibility, nonnegative feed-forward width, finite
dropout in `[0,1)` and strict supported configuration types remain enforced.
Dimensions 2, 3 and 8 remain usable in CPU forward/backward tests; defaults remain
256 dimensions and four heads, with the same 1,251,841 parameters.

Before any checkpoint write, `validate_live_model` compares the live model with
a freshly constructed reference from the caller's validated metadata. It checks
the concrete model/configuration types, full configuration equality, module
coverage/order/types, parameter-versus-buffer registration, projection sizes,
attention dimensions/heads/scaling attributes,
attention and residual/feed-forward dropout, activation, normalization settings,
state tensor coverage/shapes/dtypes/finiteness, and canonical fixed buffers.
Changing the live model's `config` attribute cannot conceal mutated modules.

New checkpoint payloads contain a `gc2-fusion-model-binding-v1` record with the
complete model configuration and architecture. Loading reconciles that record
with validated persisted metadata before reconstructing the architecture. It
requires exact state coverage and compatible finite floating tensors, preserves
their dtype, and compares positional buffers exactly with independently regenerated
canonical values before strict state loading. Both sinusoidal and `none` modes
retain a required `(1,10,d_model)` buffer; for `none` it must contain only zeros.
Missing, extra, malformed, wrong-shape or altered buffers are rejected. This
detects the original mismatch even if both configuration records and all ordinary
hashes/fingerprints are made consistent with the incorrect declaration.

The original file hash/size, experiment/model fingerprint, provenance, selected
epoch and optional expected-metadata checks remain, as does
`torch.load(..., weights_only=True)`. Attention checkpoints without the new binding
are rejected. Historical MLP checkpoints retain the legacy no-binding read path;
new MLP checkpoints include and validate the binding. No historical file is rewritten.

Weights alone cannot determine head count, dropout, some activations, or the
initialization seed after training. The supported write path therefore checks
the live module settings and persists their full binding. The load path verifies
that binding and all state-derived facts; it cannot prove historical settings if
someone deliberately rewrites every configuration record and integrity claim
into another internally valid artifact. Legacy MLP files also cannot retroactively
prove settings absent from their original state. These are provenance/trust
limitations, not a claim of cryptographic authentication.

Remediation changes are limited to `fusion/cross_attention_model.py`,
`fusion/model_interface.py`, `fusion/train_mlp.py`, this document and the new
`tests/test_fusion_checkpoint_binding.py`. The 19 new tests use synthetic
TRAIN/validation provenance, temporary outputs and no real AVE data, final-test
split, tracking or training experiment. They cover configuration/live-module
mismatches, schema/weight swaps, missing/malformed states, altered positional
buffers with recomputed integrity claims, valid sinusoidal/zero-position and
float32/float64 round-trips, legacy MLP reloads, safe-load/hash/fingerprint guards,
minimum dimensions and supported finite gradients.

The required full suite passed after remediation: **457 tests run, 455 passed,
two CUDA skips and zero failures/errors**. Independent post-fix probes
confirmed rejection of the original save mismatch before any file write,
rejection of inconsistent positional state despite consistent configuration
records/hashes, rejection of `d_model=1` during configuration validation, and valid
`d_model=2` sinusoidal/zero-position checkpoint round-trips.

The four historical Phase 4 checkpoints (initial, E1, E2, E3) were compared against
the committed original Phase 4 loader: weights, serialized metadata, evaluation
outputs and seeded training-mode outputs are bitwise identical. Their checkpoint
file hashes are unchanged. The 280-file preservation inventory still matches
exactly. The untracked requirements file, Phase 1–3 and producer code, embeddings,
pair manifests, trained evidence, training/index coverage, selection/threshold
policies, Forge lifecycle and existing TRAIN-only smoke implementation were
preserved. No real training, live W&B initialization, commit, push, merge or PR
was performed. No remaining confirmed defect is known within these two fixes.

## Subsequent Tasks 5.4–5.6

The targeted re-audit's LOW integer-versus-float dropout compatibility issue was
resolved using exact numerical equality for validated dropout values, preserving
all strict architecture and state checks. The subsequent one-run real training
batch is documented in [the training report](fusion_cross_attention_training.md)
and [primary comparison](fusion_cross_attention_comparison.md). Statements above
about no real training describe the earlier foundation/remediation batch only.

# Trained temporal AV presence: H2-AVE-presence-v1

GC2 private experimental model. Public prediction gates remain unchanged.
The original BEATs + LoRA, ResNet-18, E3/A2 weights and operating points are not
retrained or replaced. Head 2 has its own trained weights and threshold.

## Supervision and separation

The AVE authors define an audio-visual event as both audible and visible and
provide temporal event boundaries ([paper](https://arxiv.org/abs/1803.08842),
[author repository](https://github.com/YapengTian/AVE-ECCV18)). The existing pinned
author-reference verification is reused, including annotation/split hashes and
the official HDF5 split orders at commit 93f843fed72fb17394df76330df958b0c2dbde76.
Existing frozen exports are checked against the artifact-bound E3 metadata's
input-export hashes, class mappings, sample IDs, shapes and finite values.

Target at second t is the union of original AVE intervals satisfying
`start <= t < end` across annotations of the same video. Same-video annotation
rows are deduplicated, using the lexicographically first sample ID's paired
embeddings. All 18 source videos shared across official splits are removed from
every split. The known empty 0–0 annotation source VWi2ENBuTbw is excluded rather
than invented as a negative example. These are source-ID-disjoint subsets of
the official partitions, not the full official benchmark counts.

| Partition | Unique clips | Seconds | Joint-event positive | Annotated background |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 3,293 | 32,930 | 27,347 | 5,583 |
| Validation | 391 | 3,910 | 3,183 | 727 |
| TEST | 394 | 3,940 | 3,232 | 708 |

Background means no annotated joint AVE event, not exhaustive absence of all
possible sounds/visible sources. Audio-swapped pairs have **unknown joint
temporal labels** and are excluded from supervised training/scored evaluation.
No synthetic targets, E3 scores or event-score-overlap targets are used.

## Model and selection

Input: frozen audio `(10,256)` plus visual `(10,512)` embeddings. TRAIN-only
feature means/stds are checkpoint buffers; scale has a Head-2-specific lower
bound of 1e-6. This changes no existing encoder normalization or parity policy.
The selected architecture projects 768 → 128, applies ReLU/dropout 0.3, a
bidirectional GRU with 64 units per direction, and a 128 → 1 binary classifier.
It returns ten float32 logits, followed by uncalibrated sigmoid scores. The
bidirectional model requires the completed ten-second window.

TRAIN-only weighted binary cross-entropy uses background/positive count ratio.
AdamW: lr 0.001, weight decay 0.01, batch 128 clips, seed 42, at most 40 epochs.
MLP and local-convolution screening failed validation and are preserved in local
run histories. Two GRU widths were subsequently fixed before their evaluation;
all selection used TRAIN/validation only. Acceptance bounds were not lowered:
macro-F1 >=0.70 and recall >=0.60 for each binary class on validation and TEST.

Among validation thresholds meeting the recall constraint, selection maximizes
macro-F1; ties prefer the threshold closest to 0.5, earlier epoch, then smaller
width. Threshold grid: 0.05 to 0.95 in steps of 0.05. Selected epoch: **22**;
Head 2 threshold: **0.25**. The durable selection/checkpoint hashes were written
before TEST features were opened. TEST was evaluated only after that freeze and
was not used to retune architecture, weights, threshold or acceptance criteria.

## Actual per-second evaluation

Precision/recall below refer to the joint-event positive class. Macro-F1 averages
background and presence F1. Confusion matrices pool individual one-second bins.

| Partition | Accuracy | Precision | Recall | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 0.992712 | 0.998822 | 0.992394 | 0.987219 |
| Validation | 0.802046 | 0.903247 | 0.847628 | 0.702732 |
| TEST | 0.802284 | 0.914498 | 0.837252 | 0.706476 |

TEST confusion matrix (rows true background/presence; columns predicted
background/presence): `[[455,253],[526,2706]]`. Background recall is 0.642655;
presence recall is 0.837252. Macro precision/recall: 0.689155 / 0.739954.
The always-presence baseline has **higher accuracy (0.820305)** but lower
macro-F1 (0.450641). The trained head improves balanced classification, not that
baseline's accuracy. The large TRAIN/held-out gap remains a limitation.

Boundary errors match in-window onsets and offsets one-to-one, minimizing
same-direction absolute errors on the one-second annotation grid. TEST has
168 annotated and 226 predicted boundaries: 104 matched, 64 missed, 122 extra.
Matched boundary MAE is **1.711538 seconds**. Validation matched MAE is 1.530973 s.
This is not subsecond localization or a physical sound-causation guarantee.

## Private checkpoint and runtime

Checkpoint: `_local/temporal_presence/run-gru-v1/weights.pt`, 703,161 bytes,
SHA-256 `f957f83cf98f0f11e51217926f4519f93dcf4f04b43fa789e797e822a0dae8a7`.
Private `metadata.json`, `selection.json`, `evaluation.json`, complete histories,
protocols and exact training source snapshots are in the same ignored run tree.
The public supplemental catalog binds weights/metadata, original encoder
identities and Head 2's threshold. It includes no private weights or media.

The optional `Head2` model directory is verified before safe weights-only load.
State, architecture, source and artifact checks remain active at inference. The
existing isolated worker runs Head 2 on the same genuine embeddings used by E3;
signed tickets, authentication, expiry, cancellation and cleanup are unchanged.
Missing optional Head 2 artifacts produce an explicit unavailable state; present
but invalid artifacts fail closed. Nothing silently substitutes a heuristic head.

The evaluated scope is original matched AVE cached embeddings. Historical
cache-to-live numerical compatibility remains unresolved. Arbitrary uploads,
audio swaps and browser-live inputs do not gain temporal ground truth or
independent release approval from this experiment. Their private displayed
scores are genuine model outputs with these limitations, not validated accuracy.

Head 2 uses a deterministic 50-digit Decimal sigmoid before float64 serialization
on both worker and broker. A genuine logit 0.2688511610031128 exposed a one-ULP
Linux/Windows NumPy-exp difference at the exact display validator. The shared
reference fixes that rejection without weakening equality checks, changing
weights/thresholds, or activating the existing historical parity tolerances.

## Functional positive and negative examples

Prespecified first TRAIN clip with both label types: `--bSurT-1Ak`.
Truth/prediction: `[0,0,0,0,0,0,1,1,1,1]`.
Negative/background bin 0: score **0.00004089395465**, predicted background.
Positive/event bin 6: score **0.999993635997**, predicted presence.
These are functional TRAIN examples, not held-out performance claims. No
whole audio-swapped clip was falsely assigned ten negative temporal labels.

## Reproduce without changing existing models

Use CPython 3.11.16 and the existing deployment environment; a separate training
environment can install `requirements-temporal-presence.txt`. Authorized private
exports, reference metadata and the original E3 metadata must already be present.
The checked-in `docs/temporal_presence_protocol.json` is an exact byte copy of
the accepted run's archived protocol, SHA-256
`0e1af0955e948725a0e1f66f0adf5ad7282466db348ab21f74368c782546aa2e`.
The private source snapshot binds the actual selected run.
Use a fresh ignored output directory; never overwrite the existing checkpoint:

```powershell
$gc2Python = '.\_local\phase8_stage1\venv\Scripts\python.exe'
& $gc2Python -B -m fusion.train_temporal_presence --protocol docs/temporal_presence_protocol.json --output _local/temporal_presence/reproduction-v1
```

Provision original artifacts and Head 2 to a fresh **operator-owned** private
directory. On Windows, run provisioning and the supervisor under the same
trusted account; do not loosen permissions on sandbox-owned directories:

```powershell
& $gc2Python -B -m deployment.provision_artifacts --destination _local/temporal_presence/product-models-host --source-root _local/phase8_stage2c/models
& $gc2Python -B -m deployment.temporal_presence --source _local/temporal_presence/run-gru-v1 --destination _local/temporal_presence/product-models-host/Head2
& $gc2Python -B -m deployment.run_product_demo --models _local/temporal_presence/product-models-host --private-dir _local/three_head_demo/private --port 7864
```

Do not start a second supervisor against a running private directory. The
generated login remains in its existing private `login.json`; never place it in
Git, logs or PR text. The product's own model volume is read-only and offline.
Linux provisioning retains the existing UID-1000/group-0 model permissions.

Run `pytest tests/test_temporal_presence.py tests/test_temporal_presence_runtime.py`
and the existing regressions. Genuine runtime tests explicitly skip when private
Head 2 artifacts are absent. For browser acceptance set
`AVAL_PRODUCT_REQUIRE_HEAD2=1`, the local test URL/private directory/output
variables, then run `node tests/integration_product_demo.cjs`. Only Chrome fake
devices are used. Public gates remain closed. No commit, push, merge or deployment
is part of training, provisioning or these tests.

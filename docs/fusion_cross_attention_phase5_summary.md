# Phase 5 final research freeze and PR readiness

Freeze date: **2026-10-25, Asia/Singapore**. Repository:
`NUS-AIS-GCs/Audio-Video-Confidence`; branch
`feature/fusion-phase5-cross-attention`.

**Research findings: FROZEN. Current integration readiness: READY TO COMMIT.**
The [integration follow-up](fusion_cross_attention_integration.md) records the
fresh passing suite, exact checkpoint replay, Windows diagnosis and merged
Phase 4 dependency. The verification and Git sections below preserve the
earlier freeze-session snapshot; the frozen JSON is unchanged.

At that earlier freeze session, a new Windows
Application Control error blocks PyTorch DLL loading in this freeze session.
The required fresh test suite and checkpoint replay cannot currently pass.
The earlier independent audits remain recorded separately from this failed
current verification. No research configuration or result was changed.

The [versioned freeze record](fusion_cross_attention_phase5_freeze.json) contains
full-precision results, selection rules, all four pinned Forge artifact pairs,
checkpoint/model hashes, all 92 local run-file hashes, source/data manifest
bindings, audit status and exact Git scope. Historical evidence was not rewritten.
Earlier documents remain records of their individual execution stages; this
summary consolidates the final decisions and current verification status.

## Objective and architecture

Phase 5 evaluates whether cross-attention improves original-versus-substituted
audio-video pairing prediction over the historical learned MLP and JSD reference.
The positive label means **original pairing provenance**, not proven identity of
the physical sound source. The negative label means substituted pairing.

Inputs are frozen BEATs LoRA audio embeddings `(B,10,256)`, ResNet18 visual
embeddings `(B,10,512)` and an explicit boolean `(B,10)` validity mask.
Ten original aligned one-second bins and producer source-row references are
preserved. Masks use producer validity, not AVE event annotations.

Both modalities project to width 256. The initial architecture computes parallel
audio-query/video-key-value and video-query/audio-key-value attention from the
original positioned projections. Each branch retains the query residual,
LayerNorm and FF512/GELU update. Invalid keys and query outputs are masked.
Masked mean pooling concatenates two 256-dimensional streams before one binary
logit. The model uses four heads and fixed sinusoidal positions.

A1 removes the video-query branch; A2 removes the audio-query branch. Their
single 256-dimensional classifier input includes the query residual and attended
other-modality values. Removed branches have no parameters or placeholder
outputs. A3 retains both directions and the original bin order, with a validated
zero positional buffer. One-way variants save 527,360 parameters; these are not
parameter-matched comparisons.

Implementation and numerical details:
[foundation](fusion_cross_attention_foundation.md),
[initial training](fusion_cross_attention_training.md),
[controlled ablations](fusion_cross_attention_ablations.md).
Phases 1–4 interfaces, checkpoints and research evidence remain available.

## Frozen protocol and findings

All four attention runs used seed 42, AdamW LR 0.0003, weight decay 0.01,
batch 64, maximum 30 epochs, patience 5, width 256/four heads, attention/residual/
feed-forward dropout 0.1, FF512/GELU, unweighted BCEWithLogitsLoss, gradient
clipping at norm 5 and CPU/four threads. There was no scheduler or resumed run.

Each epoch covered all 6,678 distinct TRAIN primary pairs, including the final
22-example batch: 105 optimizer steps. Validation used 804 primary pairs,
balanced 402/402. Epoch selection required strictly greater primary-validation
AUC, retaining the earliest exact tie. After best-checkpoint freeze, threshold
selection maximized primary-validation binary F1, choosing the largest threshold
on a tie. Decisions use stable float64 sigmoid scores `>=` the frozen threshold.

Exactly one initial run and one attempt each of A1/A2/A3 completed. The three
ablation configurations were preregistered before optimization; guards protected
their protocol and source snapshots. Full-precision stored values determine all
rankings. Values below are rounded to nine decimals for readability.

| Model | Parameters | Best/completed epochs | Validation AUC | AP | Binary F1 |
|---|---:|---:|---:|---:|---:|
| Initial bidirectional | 1,251,841 | 9/14 | 0.956473850 | 0.951058238 | 0.902761104 |
| A1 audio → video | 724,481 | 9/14 | 0.954357565 | 0.948288901 | 0.907063197 |
| A2 video → audio | 724,481 | 18/23 | 0.960854929 | 0.958387720 | 0.905263158 |
| A3 without positions | 1,251,841 | 11/16 | 0.959976238 | 0.955512095 | 0.898263027 |
| Historical E3 MLP | 558,081 | 17/22 | 0.970019306 | 0.968596681 | 0.913669065 |
| JSD (non-trainable) | 0 | —/— | 0.969641841 | 0.972805358 | 0.917293233 |

| Model | Precision | Recall | Frozen threshold | TP/TN/FP/FN |
|---|---:|---:|---:|---|
| initial | 0.872389791 | 0.935323383 | 0.0406411536485028 | 376/347/55/26 |
| A1 | 0.903703704 | 0.910447761 | 0.25910291122020657 | 366/363/39/36 |
| A2 | 0.854304636 | 0.962686567 | 0.0026640388978268056 | 387/336/66/15 |
| A3 | 0.896039604 | 0.900497512 | 0.10440009964709925 | 362/360/42/40 |
| E3 | 0.881944444 | 0.947761194 | 0.009579099009671126 | 381/351/51/21 |
| JSD | 0.924242424 | 0.910447761 | 0.32543530977887525 | 366/372/30/36 |

**Selected attention variant: A2. Best learned model by validation ROC-AUC: E3.**
A2 is not the overall winning fusion model. The frozen attention order is
**A2 → A3 → initial → A1**, using AUC, AP, F1 descending and then the fixed
initial/A1/A2/A3 order. No actual metric tie required a secondary rule.

A2 AUC is **0.9608549293334323**: +0.0043810796762455695 versus initial,
−0.00916437711937823 versus E3 and −0.008786911215068915 versus JSD.
A1 has higher F1 than A2. Removing positions improved AUC but reduced F1
relative to initial attention. Every attention variant trails E3 and JSD in
AUC, AP and F1. JSD remains a separate non-trainable reference, with no BCE
training loss or epoch count. All methods share the same ordered 804 validation
identities, targets, source references, positive-class meaning and score direction.

See the [ablation comparison](fusion_cross_attention_ablations.md) for selected
TRAIN/validation BCE, all deltas and interpretation. Historical E3 used its own
prespecified training protocol and is a reference rather than an isolated
architectural control.

## Experiment and checkpoint references

| Run | Existing Forge run | Total optimizer steps | Steps in selected checkpoint |
|---|---|---:|---:|
| initial | [hiyb44br](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/hiyb44br) | 1470 | 945 |
| A1 | [2wbi8en8](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/2wbi8en8) | 1470 | 945 |
| A2 | [6mo5xu4k](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/6mo5xu4k) | 2415 | 1890 |
| A3 | [usp3ura0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/usp3ura0) | 1680 | 1155 |

The freeze record pins both selected-checkpoint and reproducibility-report
artifact names, versions, digests and URLs for each run. All are existing v0
Phase 5 versions. The frozen ablation protocol SHA-256 is
`33cfdb7e5dd8853e847ab1c2b6208c58923f181cd3e8caded4c6e92ad8e599f6`.
The original initial-run preregistration is separately bound by its file hash.

Fresh live Forge checks confirmed all four runs are `finished`, their complete
configurations, final summaries and 14/14/23/16 epoch histories agree with local
evidence, and all **88 artifact files** match local SHA-256/size and remote MD5
manifests. No new run, artifact version or remote modification was made.
The final five-model/JSD comparison remains the identical comparison uploaded
inside A3's report artifact.

Prior independent audits verified safe `weights_only=True` loading, exact
804-logit reproduction for the initial model and every ablation, strict
architecture/configuration/buffer binding and rejection of swapped checkpoints.
The ablation audit also reproduced attention numerics within `3.4e-16`,
checked finite modality gradients and batch isolation, and confirmed identical
historical MLP states, metadata and synthetic outputs against the original loader.

## Freeze-session verification and blocker (historical snapshot)

The previous independent ablation audit passed with scientific warnings:
**478 tests, 476 passes and two CUDA skips**, in 44.008 seconds.
That is a prior audit result, not this freeze session's test outcome.

This task ran the required command:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

Current result: **319 discovery entries, 312 passes, seven module-import
errors**, in 54.514 seconds. PyTorch fails before its tests or checkpoint
verification can execute:

```text
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading torch/lib/c10.dll or one of its dependencies.
```

An import/checkpoint probe outside the sandbox produced the same error.
This is an observed environment execution blocker; no source regression has
been established. No policy, DLL, dependency or model was changed to bypass it.
An authorized administrator must resolve normal PyTorch DLL loading, after
which the full suite and selected-checkpoint replay must be rerun before the
commit-readiness verdict can change.

Checks completed despite that blocker:

- Recomputed AUC/AP/F1, precision, recall, accuracy and confusion counts from
  saved primary-validation predictions for all six references.
- Matched all four local output manifests, selected-checkpoint hashes and all
  92 existing local run files; source snapshots and original manifests match.
- Re-hashed all **280 protected historical files / 126,941,543 bytes** and all
  **23 original attention-run files**, with no changes. This preserves all
  four Phase 4 checkpoint bytes and producer exports.
- Confirmed the existing untracked requirements file SHA-256 remains
  `a01116f4399965d31d989d92ec8d14a07cd518b55208d42a9081efd91f2848ab`.
- Confirmed Phases 1–4 modules are present; dependency consistency, Python
  syntax, whitespace and credential-pattern checks pass. The Git index is empty.
- Confirmed no GC1 access, new training, tuning, additional ablation,
  challenge/test inference or historical evidence rewrite occurred.

Only the two final documentation/reference files were created in this task.
All 21 pre-existing working-tree files remain unchanged.

## Scientific boundaries and final research conclusion

These are single-seed development comparisons. The same primary validation
selects epochs, thresholds and architecture, adding selection optimism. Ten
source-video IDs overlap across the official TRAIN/validation annotation splits.
Capacities differ; topology changes initialization and dropout draw placement.
No causal attribution to attention direction alone or statistical significance
is supported.

Ten one-second bins limit temporal resolution. Class-oriented embedding
representations may fail to distinguish same-category substitutions. Attention
same-category mismatch performance remains unevaluated. No held-out TEST
evaluation was conducted. Pairing provenance and attention maps do not establish
physical source identity or causality; sigmoid outputs lack independent
calibration evidence.

The completed study does not justify extra cross-attention complexity over E3
or JSD on observed primary-validation metrics. A2 is retained as the frozen
attention research variant; E3 remains the best learned validation-AUC model.
No further experiment is part of this freeze.

## Git and eventual PR scope (historical freeze snapshot)

After `git fetch --no-tags origin`, HEAD remains
`38d4171d1b9c6aefab476a87fa8fc144999191d9`.
Fetched main remains `467c7568def85d3f9270756ce1ce98d861958c45`;
Phase 4 remains `76add86568daed6f7dbb31659f26c020b264962c`.
Phase 4 is included here and remains outside main.

HEAD has zero commits behind and two commits ahead of main: the Phase 4 feature
commit and its local merge. The committed HEAD tree equals the Phase 4 tree;
all actual Phase 5 implementation/documentation changes are still uncommitted.
There is no fetched origin tracking reference configured for this local Phase 5
branch. A PR to current main would include Phase 4 changes.

The committed `origin/main...HEAD` diff is **16 files, 4,710 insertions** from
Phase 4. The existing tracked working diff is **six files, +188/−43**.
The Phase 5 review inventory is **22 intended files: six modifications and
16 additions**, listed below; ordinary `git diff` omits the untracked additions.

```text
A docs/fusion_cross_attention_ablations.md
A docs/fusion_cross_attention_comparison.md
A docs/fusion_cross_attention_foundation.md
A docs/fusion_cross_attention_phase5_freeze.json
A docs/fusion_cross_attention_phase5_summary.md
A docs/fusion_cross_attention_training.md
A fusion/cross_attention_ablations.py
A fusion/cross_attention_model.py
A fusion/cross_attention_tracking.py
A fusion/model_interface.py
A fusion/smoke_cross_attention.py
A fusion/train_cross_attention.py
A tests/test_fusion_checkpoint_binding.py
A tests/test_fusion_cross_attention.py
A tests/test_fusion_cross_attention_ablations.py
A tests/test_fusion_cross_attention_training.py
M fusion/embedding_data.py
M fusion/mlp_study.py
M fusion/mlp_tracking.py
M fusion/smoke_mlp.py
M fusion/train_mlp.py
M fusion/training_contracts.py
```

The pre-existing `requirements-fusion-training.txt` is separately preserved
untracked and excluded from that intended staging inventory. No credentials,
raw embeddings or real model weights are included. Earlier uncommitted Phase 5
files were preserved; this task adds only the final summary and freeze record.

Required integration sequence: **Phase 4 → main; synchronize Phase 5 with main;
Phase 5 → main**. No merge, rebase, commit, push, PR or branch-history change
was performed. Fetching refreshed remote references only.

**Historical freeze verdict: findings frozen; verification required before
commit.** Those follow-up checks and the merged upstream dependency are now
recorded in the [integration handoff](fusion_cross_attention_integration.md).

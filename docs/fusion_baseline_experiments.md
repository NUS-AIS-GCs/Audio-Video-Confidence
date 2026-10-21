# Phase 3 Tasks 3.6–3.7: real correspondence baselines

This runner evaluates the four audited non-neural baselines on complete official
AVE train and validation pair partitions. Test pairs are never deserialized,
scored or evaluated. Phase 1/2, Tasks 3.1–3.5 and teammate implementations are
unchanged. No models are trained and no production pair files are rewritten.

## Environment and invocation

Use Python 3.11 and the existing NumPy environment. Experiment dependencies tested
here are `wandb==0.31.0` and the optional pinned-reference reader `h5py==3.16.0`.
Core contract/unit-test imports still need only NumPy and the standard library.
W&B is imported lazily and all unit-test tracking is mocked. A fresh environment
can install the experiment packages in its virtual environment; this workspace
already has W&B and an isolated HDF5 reader under ignored `_local/tools/ave_reference`.
No credential is supplied through the CLI or copied into experiment metadata.

```powershell
$env:PYTHONPATH = (Resolve-Path _local/tools/ave_reference).Path
$env:WANDB_BASE_URL = 'https://api.forge.coreweave.com'
$env:WANDB_ENTITY = 'GC2-AValidate'
$env:WANDB_PROJECT = 'avalidate'
.\.venv\Scripts\python.exe -B -m fusion.run_baseline_experiments run `
  --audio-root _local/embeddings/audio_final `
  --vision-root _local/embeddings/vision_final/resnet18_v1 `
  --official-root _local/datasets/AVE `
  --reference-root _local/datasets/AVE_reference `
  --reference-evidence _local/fusion/ave_reference_verified/verification.json `
  --integration-root _local/fusion/task28_real `
  --output-dir _local/fusion/phase3_real_experiments/run_20261022_01 `
  --seed 42
```

All roots are explicit; no private machine paths are hardcoded in source. The
output must be fresh, Git-ignored, inside this repository and separate from every
input root. Later repetitions must use another fresh output directory. The CLI
requires the GC2 origin, `feature/fusion-phase3-baselines` branch and a clean
tracked-file/index state. Untracked reviewed source is identified in the recorded
working-tree status; the Git commit alone does not identify that code, so every
`fusion/*.py` file also has a recorded SHA-256.

The configurable seed must match the existing pair manifests and Phase 2 report.
It is not permission to regenerate pairs or change the production sampling policy.
No stochastic model, RNG state, epochs, learning rate, loss curve or trained
checkpoint is involved; training-only fields are marked not applicable.

## Verification and held-out boundary

Before scores are computed, the runner requires the existing complete Phase 2
integration report, exact export/metadata inventory coverage, matching byte sizes
and SHA-256, and consistent artifact inventory fingerprint. It verifies all three
production manifest byte hashes, including test **as opaque integrity evidence**.
Only the `train_pairs.json` and `val_pairs.json` documents enter the safe reader.

It freshly runs `verify_reference` against the pinned authors' four reference
files and compares the result with the saved verified-reference evidence.
The Phase 2 `_load_exports`, `_verify_official` and `manifest_from_json` interfaces
then validate arrays, official identities and full typed pair documents. The safe
reader regenerates candidates for comparison in memory; it never replaces files.
Full-bundle structural validation includes held-out export arrays/metadata because
the persisted manifest binds the full source-overlap context. This is distinct
from scoring or evaluating test pairs; no test-pair reader call is made.

Required coverage for each scored split is all original positives, one easy
negative per annotation and one separately partitioned same-category challenge
per annotation, with zero hard negatives. Original IDs must cover every annotation.
Expected counts are 3,339 train / 402 validation annotations, 6,678 / 804 primary
pairs, and 3,339 / 402 separate challenges. Probabilities retain ten bins, 29
classes and original modality-specific row references. Negative indices, mismatched
identities, truncated manifests, wrong splits, altered hashes or missing evidence
fail closed. No quotas are silently reduced.

The runner checks source declarations for audio `beats_lora_r8_v1` /
`beats_iter3` (BEATs+LoRA) and vision `resnet18_v1` (ResNet-18). These are producer
declarations and byte-continuity evidence, not cryptographic checkpoint provenance.
Historical unimodal test measurements in producer notes are not copied into
results or used for selection. Input and source-code hashes are checked again
after computation; changes prevent a successful local completion marker.

## Scoring, fitting and comparison

All prediction features flow through the audited probability adapter and scoring
APIs: top-1 agreement, dot product, cosine and base-2 Jensen–Shannon similarity,
with arithmetic mean across all ten bins. Ground-truth categories, intervals,
pair types and original/substituted targets never become predictive inputs.
Compact output score CSVs keep evaluation metadata in separate columns; they
contain pair-level scores, not raw probability/embedding arrays or media.

Each method fits maximum binary F1 on validation **primary** pairs exclusively,
with exact-rational comparisons and largest-threshold tie resolution. All four
policies are frozen before any operating-point report. Train/validation primary
reports include ranking metrics, confusion/classification metrics, class-specific
score distributions and fixed-bin descriptive reliability. Average precision is
noninterpolated; it is not labeled trapezoidal PR-AUC.

Challenge reports remain separate and contain all-zero provenance targets,
acceptance/rejection counts and rates, score distributions and frozen-policy
context. They do not contribute to primary ROC-AUC, AP, F1, threshold selection
or the winning-method rule.

The prespecified method-selection order is:

1. Highest validation-primary ROC-AUC.
2. Highest validation-primary noninterpolated average precision.
3. Highest validation-primary F1 at its frozen threshold.
4. First method in fixed order: `top1`, `dot_product`, `cosine`, `jensen_shannon`.

The winner is a **validation-selected candidate**, with no final-test conclusion.
The same validation primary pairs fit thresholds and report their resulting
metrics, so those values are not unbiased held-out performance estimates. No
method is selected from challenge results.

## Forge tracking and evidence

The required destination is fixed:

```text
WANDB_BASE_URL=https://api.forge.coreweave.com
WANDB_ENTITY=GC2-AValidate
WANDB_PROJECT=avalidate
```

One online comparison run identifies all four methods through metric prefixes
and a comparison table, grouped as `phase3_correspondence_baselines`. Authentication
uses existing SDK configuration without interactive prompts or printing credentials.
Conflicting destinations, offline mode, missing login, failed server access or
failed upload cannot count as completed tracking. There is no wandb.ai fallback.
SDK exceptions are reduced to nonsecret stage/type information rather than
persisting arbitrary messages, tokens, request headers or environment dumps.

The run logs classification/ranking/challenge metrics, frozen thresholds and
validation reuse. Its immutable evaluation artifact contains all 36 core files
plus their output manifest: configuration/environment/code versions, full input
hash verification, fresh reference evidence, data counts and source overlap,
four score CSVs, complete frozen policies, sensitivity sweeps, all primary and
challenge reports, comparison CSV/JSON and the logged metric dictionary.
W&B run URL, artifact identity/digest and upload status are isolated in
`tracking.json`; `completion.json` records local versus tracking completion.

Core files use deterministic serialization and exclude timestamps/run IDs.
`output_manifest.json` names and hashes exactly the canonical files. Neither SDK
logs nor arbitrary files are added to the artifact. Model exports, generated pair
manifests, checkpoints and raw prediction arrays are not copied into results.

If Forge is unavailable, local verified computations may still complete and all
evidence is preserved with `BLOCKED_WANDB_UPLOAD`; Phase 3 tracking remains
incomplete. Exit codes are 0 complete including upload, 2 local complete but
tracking blocked, and 1 fail-closed input/computation failure. A directory without
`completion.json` is not a complete local experiment.

To retry tracking without fitting or scoring again:

```powershell
.\.venv\Scripts\python.exe -B -m fusion.run_baseline_experiments upload `
  --output-dir _local/fusion/phase3_real_experiments/run_20261022_01
```

This checks the saved completion/manifest digest and every core file's hash
before opening a new run. Only the fixed, canonical in-directory evidence files
can be uploaded; traversal paths, unknown files and corrupted evidence fail.
It changes only operational tracking/completion metadata, not statistical files.

## Limitations and Phase 4 boundary

Original versus substituted pairing is recording provenance, not evidence of
physical causality, visible sources or waveform synchronization. Similarity
scores are not calibrated probabilities; no raw-score ECE/Brier/calibrator is
claimed. Background dominance and same-category swaps may remain difficult.
Official assignments and the known 10 train/validation, 8 train/test and 0
validation/test shared source videos are retained. Report warnings come from
fresh supplied metadata; overlap filtering cannot undo previous unimodal encoder
exposure. Empty annotations retain UNKNOWN semantic status and existing masks.

Before Phase 4, audit the implementation, frozen policies, complete local/remote
evidence and validation reuse; resolve the pending Phase 2 dependency through the
normal review process; predeclare model/validation allocation and supplementary
source-disjoint analyses. Any final test requires a separately authorized step
with frozen method/model/threshold configuration and no refitting from test data.

## Verification and actual results

The existing 286-test suite passed before implementation. New tests use temporary
synthetic files and mocked W&B only. They exercise configuration, input corruption,
reference/completion evidence, full manifests, train/val-only scoring, fitting and
freeze ordering, deterministic method precedence, challenge isolation, validation
reuse, counts, stable output bytes, failed/offline authentication/upload, credential
redaction, exact destinations, upload-only recovery and artifact path confinement.

### Executed experiment

The complete seed-42 experiment ran on 2026-10-22 (Singapore time), from Git
commit `0e67c343e33e1821355bd621a15381a65042d4f4` plus the recorded untracked
Phase 3 source hashes. Local evidence is under
`_local/fusion/phase3_real_experiments/run_20261022_01/`.
Both local computation and mandatory Forge upload completed.

All 37 input roles passed integrity verification. Counts were exactly 3,339 train
and 402 validation annotations; 6,678/804 primary pairs (3,339/402 positive and
the same number negative); and 3,339/402 separate target-zero same-category
challenges. Hard negatives: zero. All 44,892 pair/method scores were finite and
bounded. Original modality row references matched the production pair files.
The official metadata/reference verification covers all 4,143 annotations;
held-out test inputs were used only for structural integrity, never scored.

Primary results (AP is noninterpolated; F1 uses the frozen validation threshold):

| Method | Train ROC-AUC | Train AP | Train F1 | Validation ROC-AUC | Validation AP | Validation F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-1 | 0.999461 | 0.999314 | 0.978599 | 0.921895 | 0.920731 | 0.878825 |
| Dot product | 0.999882 | 0.999879 | 0.987140 | 0.967464 | 0.969230 | 0.910256 |
| Cosine | 0.999894 | 0.999892 | 0.987432 | 0.966238 | 0.968914 | 0.909765 |
| Jensen-Shannon | 0.999948 | 0.999948 | 0.991390 | 0.969642 | 0.972805 | 0.917293 |

| Method | Frozen threshold (`score >= threshold`) |
| --- | ---: |
| Top-1 | 0.1 |
| Dot product | 0.1305835717350063 |
| Cosine | 0.2286503682368508 |
| Jensen-Shannon | 0.32543530977887525 |

Operating metrics below use TP/TN/FP/FN order. Balanced accuracy equals accuracy
for these balanced primary partitions. Complete reports also contain positive
prediction rates, score distributions and all ten descriptive score bins.

| Split | Method | TP/TN/FP/FN | Accuracy | Precision | Recall | Macro-F1 | Specificity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Train | Top-1 | 3338/3194/145/1 | 0.978137 | 0.958369 | 0.999701 | 0.978127 | 0.956574 |
| Train | Dot product | 3339/3252/87/0 | 0.986972 | 0.974606 | 1.000000 | 0.986970 | 0.973944 |
| Train | Cosine | 3339/3254/85/0 | 0.987272 | 0.975175 | 1.000000 | 0.987270 | 0.974543 |
| Train | Jensen-Shannon | 3339/3281/58/0 | 0.991315 | 0.982926 | 1.000000 | 0.991314 | 0.982630 |
| Validation | Top-1 | 359/346/56/43 | 0.876866 | 0.865060 | 0.893035 | 0.876833 | 0.860697 |
| Validation | Dot product | 355/379/23/47 | 0.912935 | 0.939153 | 0.883085 | 0.912858 | 0.942786 |
| Validation | Cosine | 368/363/39/34 | 0.909204 | 0.904177 | 0.915423 | 0.909200 | 0.902985 |
| Validation | Jensen-Shannon | 366/372/30/36 | 0.917910 | 0.924242 | 0.910448 | 0.917906 | 0.925373 |

Separate same-category challenge results, never pooled into primary metrics or
used to choose thresholds/methods:

| Method | Train accepted | Train rejected | Train acceptance | Train rejection | Validation accepted | Validation rejected | Validation acceptance | Validation rejection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-1 | 3299 | 40 | 0.988020 | 0.011980 | 333 | 69 | 0.828358 | 0.171642 |
| Dot product | 3310 | 29 | 0.991315 | 0.008685 | 329 | 73 | 0.818408 | 0.181592 |
| Cosine | 3297 | 42 | 0.987421 | 0.012579 | 341 | 61 | 0.848259 | 0.151741 |
| Jensen-Shannon | 3268 | 71 | 0.978736 | 0.021264 | 339 | 63 | 0.843284 | 0.156716 |

The prespecified rule selects **Jensen-Shannon** by validation-primary ROC-AUC;
no tie-break is needed. This is a validation-selected candidate, with validation
reused to select thresholds and report their F1. Neither validation operating
metrics nor near-perfect training metrics establish independent generalization.
Dot product's higher challenge rejection does not change selection. All methods
accept most same-category substitutions. No final-test conclusion is available;
scores remain uncalibrated similarities and labels remain pairing provenance.
Known source-video overlap warnings (10 train/validation, 8 train/test,
0 validation/test) are retained for both modalities.

### Forge evidence and verification

- [Finished Forge comparison run `14lwfgan`](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/14lwfgan), group `phase3_correspondence_baselines`.
- [Immutable evaluation artifact v0](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase3-correspondence-baseline-evidence/v0): `GC2-AValidate/avalidate/phase3-correspondence-baseline-evidence:v0`.
- Artifact digest: `438a65e3cb4242a721b2f3879ba17067`.
- Phase 2 input inventory fingerprint: `c8ec1448ab79e7c5fcc044678e9375bece168c7f2adfbbe5fa7a57c0ca957949`.
- Core output manifest SHA-256: `c0a33292cd5c376f4c2267bbc91a4845c246c006d581abc9f400430a3d7b9110`.

The artifact contains exactly 36 core files plus their output manifest. A fresh
Forge API read confirmed the run is finished and all 166 summary fields match
the local logged dictionary. Downloaded artifact coverage, every file byte size
and every SHA-256 match local evidence; its digest also matches the upload result.
Downloads are isolated in ignored
`_local/fusion/phase3_real_experiments/remote_verification_run01/`.

Additional local checks independently recomputed all primary confusion counts,
challenge acceptance/rejection counts, pairwise ROC-AUC, grouped noninterpolated
AP, and exhaustive rational-F1 threshold optima with largest-threshold ties.
They match the saved reports. All 11,223 exported pair rows preserve original
identities and row indices. Post-run verification records are operational files
`remote_verification.json` and `independent_verification.json`; they are not added
retroactively to the immutable core manifest/artifact.

The full suite passes **308 tests**, including 22 new offline tests and all 286
previous tests. Syntax/import and whitespace checks pass. The ten audited Phase 3
files retain their pre-task hashes; tracked Phase 1/2 and teammate files are
unchanged. No dependency declarations changed. All results/data remain ignored.
Phase 2 PR #13 remains a declared pending dependency until confirmed merged;
this task does not modify it. Implementation and evidence are ready for the
final Phase 3 audit, with the validation-reuse and challenge limitations above.

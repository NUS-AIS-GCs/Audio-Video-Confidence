# Phase 4 controlled MLP regularization study

Status: **COMPLETE — READY FOR INDEPENDENT IMPROVEMENT AUDIT.**

Three configurations were prescribed before training on 2026-10-24. Each received exactly one real CPU training attempt. All three completed; no variants, replacements, manual threshold adjustments or additional training runs were added. GC1 was not accessed. No commit, push, merge, PR or final-test evaluation occurred.

## Fixed protocol and configuration

All candidates use the original 6,678 TRAIN primary pairs, 804 validation primary pairs and separate 402 validation challenge pairs, with identical original modality row references. BEATs+LoRA audio `(10,256)` and ResNet-18 vision `(10,512)` embeddings, ten-bin validity treatment and correspondence labels are unchanged. Every completed optimization epoch must visit each original TRAIN dataset index exactly once; partial final batches are retained. Targets, categories, pair types, provenance and indices never enter model predictive tensors.

Fixed settings: initialization/training/pair seed 42; AdamW; learning rate 0.001; batch 64; at most 30 epochs; patience 5; betas `(0.9,0.999)`; epsilon `1e-8`; unweighted BCEWithLogitsLoss; gradient-norm clipping at 5; four CPU threads; deterministic algorithms; ReLU; no scheduler. Checkpoints use strictly higher validation-primary ROC-AUC, retaining the earliest epoch on exact ties. The unchanged F1 threshold policy uses validation primary only and chooses the largest threshold on an exact F1 tie.

| Candidate | Hidden widths | Dropout | Weight decay | Parameters |
|---|---|---:|---:|---:|
| Original MLP | 512,256,128 | 0.2 | 0.0001 | 558,081 |
| E1 | 256,128,64 | 0.2 | 0.0001 | 238,081 |
| E2 | 512,256,128 | 0.5 | 0.0001 | 558,081 |
| E3 | 512,256,128 | 0.2 | 0.01 | 558,081 |

The plan was persisted before the first new run in ignored `plan.json`, with its hash subsequently bound into `selection.json`. The group is `phase4_mlp_regularization_study` on CoreWeave Forge, entity `GC2-AValidate`, project `avalidate`. Model and optimizer overrides were already supported by the contracts; minimal Phase 4 changes added study-specific tracking names/group and deferred challenge scoring. No Phase 1–3 production implementation or historical experiment was changed.

## Primary comparison and frozen winner

| Candidate | TRAIN AUC | TRAIN AP | TRAIN F1 | Val AUC | Val AP | Val F1 | Best epoch / epochs |
|---|---:|---:|---:|---:|---:|---:|---|
| original | 1.000000000 | 1.000000000 | 0.995082700 | 0.961745996 | 0.957184990 | 0.906439854 | 15 / 20 |
| E1 | 0.999846263 | 0.999838583 | 0.973611314 | 0.966436474 | 0.962626053 | 0.908878505 | 8 / 13 |
| E2 | 0.999141531 | 0.999144742 | 0.976124213 | 0.961331403 | 0.956874228 | 0.902994012 | 8 / 13 |
| E3 | 0.999996412 | 0.999996412 | 0.994045847 | 0.970019306 | 0.968596681 | 0.913669065 | 17 / 22 |
| Phase 3 Jensen–Shannon | 0.999948426 | 0.999947968 | 0.991389549 | 0.969641841 | 0.972805358 | 0.917293233 | frozen Phase 3 reference |

**Selected MLP: E3.** Selection used full-precision validation-primary ROC-AUC, then AP, then F1, then fixed order original/E1/E2/E3. JSD is a comparator and is not eligible to win the MLP selection. The frozen decision and exact checkpoint/threshold identities were saved before challenge scoring.

Selected ROC-AUC: `0.9700193064528105`; AP: `0.968596681483803`; F1: `0.9136690647482014`; threshold: `0.009579099009671126`.

Versus original MLP: AUC `+0.008273310067`, AP `+0.011411691950`, F1 `+0.007229210556`. Versus JSD: AUC `+0.000377465904`, AP `-0.004208676783`, F1 `-0.003624168335`.

| Candidate | TRAIN − val AUC | TRAIN − val AP | TRAIN − val F1 |
|---|---:|---:|---:|
| original | 0.038254004 | 0.042815010 | 0.088642846 |
| E1 | 0.033409789 | 0.037212530 | 0.064732809 |
| E2 | 0.037810129 | 0.042270513 | 0.073130201 |
| E3 | 0.029977106 | 0.031399730 | 0.080376782 |
| JSD | 0.030306585 | 0.027142610 | 0.074096316 |

These are observed validation differences from a single seed. They do not establish the cause of a generalization gap or prove a population-level regularization effect. Validation primary has now been reused for epoch, threshold and configuration selection, introducing additional model-selection optimism. No unbiased held-out improvement claim is made.

## Checkpoints, thresholds and Forge evidence

### E1

Run: [E1-reduced-256-128-64-seed42](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/rd4am0w8). Selected epoch 8 of 13; frozen threshold `0.07924851944578684`. Optimization plus primary-validation duration: 11.172 seconds, excluding checkpoint/network overhead.

Checkpoint: `_local/fusion/phase4_improvement/study_20261024_seed42_01/E1/best.pt`; SHA-256 `b7e01dad4c58eef6352105e279d3661a90488ffaa3c5a5c3f85a850217874364`.

Model-state fingerprint: `f722f93a17b67f8bb1d554bb65e7908299f9ff5921f8f19299d8b9ba1660765c`. Frozen-threshold file SHA-256: `954c39e27d13c9346af8481ad4d148e0fa100795d6df4e13a4046d875a7c6479`.

Working-tree code fingerprint: `d993ed74b953d81902a271fd14d60831f1c6996364d321c61b2786a5c93412f0`. Device `cpu`; versions `{'cuda_runtime': None, 'numpy': '2.4.6', 'python': '3.11.16', 'torch': '2.14.1+cpu'}`.

- [GC2-AValidate/avalidate/phase4-mlp-selected-checkpoint:v1](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase4-mlp-selected-checkpoint/v1); digest `9efa1ac245077b22cf5f04f849a2b366`.
- [GC2-AValidate/avalidate/phase4-mlp-reproducibility-reports:v1](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase4-mlp-reproducibility-reports/v1); digest `87193d16246c686d8a2e105b6b1a336e`.

Fresh safe checkpoint reconstruction reproduced every selected validation logit exactly. Artifacts, all epoch curves and final summaries were read back before `finish(exit_code=0)`; local integrity checks also preceded successful finalization. Each run uploads its checkpoint and primary reproducibility reports, never raw producer exports. No challenge performance was computed or logged in these three runs.

### E2

Run: [E2-dropout-0p5-seed42](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/z5iy50uh). Selected epoch 8 of 13; frozen threshold `0.05563702508176123`. Optimization plus primary-validation duration: 19.500 seconds, excluding checkpoint/network overhead.

Checkpoint: `_local/fusion/phase4_improvement/study_20261024_seed42_01/E2/best.pt`; SHA-256 `83af330ea4fd5f0adc3a62840662c497b0ae8ae5b1d821d2a9962f4df31f696a`.

Model-state fingerprint: `9ab0d0715aed04dab360fbb4b16678935431195e57c616dd201f250d99966092`. Frozen-threshold file SHA-256: `1d542725cccfd4853e2ffac278bce11263496e7a0304791e3d1ba2c5df2c3685`.

Working-tree code fingerprint: `d993ed74b953d81902a271fd14d60831f1c6996364d321c61b2786a5c93412f0`. Device `cpu`; versions `{'cuda_runtime': None, 'numpy': '2.4.6', 'python': '3.11.16', 'torch': '2.14.1+cpu'}`.

- [GC2-AValidate/avalidate/phase4-mlp-selected-checkpoint:v2](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase4-mlp-selected-checkpoint/v2); digest `98324ce3a7849fd4ddb86ea4decbf7bc`.
- [GC2-AValidate/avalidate/phase4-mlp-reproducibility-reports:v2](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase4-mlp-reproducibility-reports/v2); digest `09ba3ae83b4125208337441c05274987`.

Fresh safe checkpoint reconstruction reproduced every selected validation logit exactly. Artifacts, all epoch curves and final summaries were read back before `finish(exit_code=0)`; local integrity checks also preceded successful finalization. Each run uploads its checkpoint and primary reproducibility reports, never raw producer exports. No challenge performance was computed or logged in these three runs.

### E3

Run: [E3-weight-decay-0p01-seed42](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/runs/c7mvo4sd). Selected epoch 17 of 22; frozen threshold `0.009579099009671126`. Optimization plus primary-validation duration: 37.294 seconds, excluding checkpoint/network overhead.

Checkpoint: `_local/fusion/phase4_improvement/study_20261024_seed42_01/E3/best.pt`; SHA-256 `e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925`.

Model-state fingerprint: `d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025`. Frozen-threshold file SHA-256: `706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad`.

Working-tree code fingerprint: `d993ed74b953d81902a271fd14d60831f1c6996364d321c61b2786a5c93412f0`. Device `cpu`; versions `{'cuda_runtime': None, 'numpy': '2.4.6', 'python': '3.11.16', 'torch': '2.14.1+cpu'}`.

- [GC2-AValidate/avalidate/phase4-mlp-selected-checkpoint:v3](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/model/phase4-mlp-selected-checkpoint/v3); digest `6ff41d5101b152592793ed3cdf924f59`.
- [GC2-AValidate/avalidate/phase4-mlp-reproducibility-reports:v3](https://forge.coreweave.com/wandb/GC2-AValidate/avalidate/artifacts/evaluation/phase4-mlp-reproducibility-reports/v3); digest `697684b091fcd17f6b2ea895ce3812df`.

Fresh safe checkpoint reconstruction reproduced every selected validation logit exactly. Artifacts, all epoch curves and final summaries were read back before `finish(exit_code=0)`; local integrity checks also preceded successful finalization. Each run uploads its checkpoint and primary reproducibility reports, never raw producer exports. No challenge performance was computed or logged in these three runs.

## Actual epoch histories

Epoch losses are minibatch-weighted optimization loss for TRAIN and eval-mode loss for validation. The F1 column below is the fixed 0.5 diagnostic, not a newly fitted epoch threshold. Best epochs are marked. Every completed epoch contains 105 optimizer steps over all 6,678 distinct TRAIN indices.

### E1

| Epoch | TRAIN BCE | Val BCE | Val AUC | Val AP | Val F1 at 0.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.528267008 | 0.387278334 | 0.904111284 | 0.882474677 | 0.831707317 |
| 2 | 0.236374108 | 0.306452703 | 0.942724190 | 0.931112951 | 0.875000000 |
| 3 | 0.157641874 | 0.342950254 | 0.947476548 | 0.940604034 | 0.866666667 |
| 4 | 0.116119857 | 0.466508255 | 0.949153486 | 0.939252251 | 0.835616438 |
| 5 | 0.091542880 | 0.359096860 | 0.951826687 | 0.944386298 | 0.863346105 |
| 6 | 0.068526978 | 0.436699919 | 0.944215490 | 0.941610051 | 0.869123253 |
| 7 | 0.063862118 | 0.446107229 | 0.951158387 | 0.947290582 | 0.824657534 |
| **8** | 0.046791071 | 0.305659585 | 0.966436474 | 0.962626053 | 0.891385768 |
| 9 | 0.044136889 | 0.652977474 | 0.956851316 | 0.951931207 | 0.801724138 |
| 10 | 0.038696591 | 0.441687516 | 0.956368654 | 0.951610754 | 0.880710660 |
| 11 | 0.030005495 | 0.593427864 | 0.953020965 | 0.950267560 | 0.849797023 |
| 12 | 0.043222972 | 0.501243207 | 0.953800648 | 0.947536487 | 0.856770833 |
| 13 | 0.023080748 | 0.514450328 | 0.959512141 | 0.955519748 | 0.860186418 |

### E2

| Epoch | TRAIN BCE | Val BCE | Val AUC | Val AP | Val F1 at 0.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.538843183 | 0.355396411 | 0.927068637 | 0.910258082 | 0.842911877 |
| 2 | 0.212266864 | 0.307796236 | 0.949914606 | 0.940567982 | 0.874358974 |
| 3 | 0.151725078 | 0.402637971 | 0.944506324 | 0.938815026 | 0.846965699 |
| 4 | 0.109327258 | 0.319172954 | 0.956337714 | 0.951209463 | 0.884422111 |
| 5 | 0.087931340 | 0.277959934 | 0.960168065 | 0.955159773 | 0.880710660 |
| 6 | 0.070836543 | 0.467589253 | 0.945632534 | 0.944371781 | 0.851802403 |
| 7 | 0.060142051 | 0.394800951 | 0.954425633 | 0.951921382 | 0.858666667 |
| **8** | 0.049200266 | 0.390648501 | 0.961331403 | 0.956874228 | 0.872395833 |
| 9 | 0.047400908 | 0.359520209 | 0.960805426 | 0.957619104 | 0.874035990 |
| 10 | 0.040658349 | 0.483358333 | 0.952068018 | 0.949165485 | 0.851851852 |
| 11 | 0.038507184 | 0.422862850 | 0.956133512 | 0.952568511 | 0.871927555 |
| 12 | 0.031346831 | 0.598428870 | 0.950650974 | 0.949681235 | 0.841544607 |
| 13 | 0.025383755 | 0.598542133 | 0.961034380 | 0.959061282 | 0.841225627 |

### E3

| Epoch | TRAIN BCE | Val BCE | Val AUC | Val AP | Val F1 at 0.5 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.475319498 | 0.353475035 | 0.932080889 | 0.918874439 | 0.849807445 |
| 2 | 0.187298382 | 0.360565044 | 0.942643746 | 0.933178363 | 0.857516340 |
| 3 | 0.129108794 | 0.406306178 | 0.943448182 | 0.937948431 | 0.843085106 |
| 4 | 0.092990180 | 0.366213759 | 0.951993763 | 0.949975454 | 0.880487805 |
| 5 | 0.065410207 | 0.377619465 | 0.954685528 | 0.949246232 | 0.864795918 |
| 6 | 0.059549384 | 0.505612829 | 0.945032301 | 0.946198201 | 0.835616438 |
| 7 | 0.042955637 | 0.455267038 | 0.952853890 | 0.948195120 | 0.849797023 |
| 8 | 0.037816407 | 0.364582042 | 0.959635900 | 0.956447060 | 0.863753213 |
| 9 | 0.039950902 | 0.392509499 | 0.961424222 | 0.957280555 | 0.884816754 |
| 10 | 0.031372441 | 0.800077335 | 0.960397020 | 0.959548845 | 0.793557833 |
| 11 | 0.026846705 | 0.504396415 | 0.962197718 | 0.960828409 | 0.856753070 |
| 12 | 0.021858139 | 0.646561455 | 0.959784411 | 0.953367721 | 0.821378340 |
| 13 | 0.012862359 | 0.530118773 | 0.967705007 | 0.964363690 | 0.866035183 |
| 14 | 0.024350896 | 0.500561981 | 0.954902106 | 0.945917551 | 0.874524715 |
| 15 | 0.030327150 | 0.381656742 | 0.963874657 | 0.964058483 | 0.875324675 |
| 16 | 0.017054064 | 0.479870765 | 0.966479790 | 0.965454616 | 0.874172185 |
| **17** | 0.007393066 | 0.482346303 | 0.970019306 | 0.968596681 | 0.872139973 |
| 18 | 0.007842713 | 0.600014582 | 0.966801564 | 0.965897028 | 0.861413043 |
| 19 | 0.026832686 | 0.437794075 | 0.960248509 | 0.956294742 | 0.874015748 |
| 20 | 0.013056438 | 0.449392700 | 0.965904309 | 0.964457762 | 0.874345550 |
| 21 | 0.015009741 | 0.509157901 | 0.966968639 | 0.963815761 | 0.852054795 |
| 22 | 0.006992599 | 0.615571067 | 0.963719957 | 0.963694501 | 0.844566713 |

## Exploratory selected-model challenge diagnostic

Only after the global decision was persisted and the chosen checkpoint/threshold frozen, E3 was evaluated on the separate 402 validation challenge pairs: **accepted 360/402; rejected 42/402 (10.447761194%)**. The selected model was not changed afterward.

Original MLP rejection was 61/402 (15.174129%); JSD rejection was 15.671642%. This same-category substitution diagnostic has been examined previously and is exploratory. It is not a model-selection criterion, an independent test estimate or evidence of physical causality. The diagnostic is stored in the local study reports and this document; no fourth W&B run was created and no finished run was amended to add it.

## Integrity, tests and limitations

All original producer/manifests and historical experiment evidence are preserved. The separate 73-file audit inventory retained every SHA-256 value, including original checkpoint `39fccb99c7e06c1d6be3db4bb321a8818932a5c91b23954843836713a9f0ac34`. Original source archive and historical fingerprints remain historical; new runs record the new exact source snapshot. Original runs `14lwfgan` and `de5u89au` were not modified.

The full suite passed before training with 393 tests (392 passes, one unavailable-CUDA skip), including nine focused new study tests. Prior finalization, index-coverage, checkpoint, selection and Phase 3 portability tests remain intact. The final full suite passed with **396 tests: 395 passed, one unavailable-CUDA skip; zero failures/errors** (44.242 seconds). Twelve new offline study tests cover exact overrides, ranking/ties, challenge deferral, failed attempts and the coordinator recovery guard.

Local evidence: `_local/fusion/phase4_improvement/study_20261024_seed42_01/` contains the prespecified plan, attempt ledger, primary comparison, frozen selection, all run histories/checkpoints/manifests, selected diagnostic and preservation inventories. Independent verification recalculates ROC-AUC, noninterpolated AP, confusion counts, F1, best epochs and frozen thresholds from persisted primary scores; validates selected challenge counts; and compares remote histories/artifact content checksums against local files without creating runs.

Known limits: one seed, fixed official annotation split with known source-video overlap, repeated validation reuse, previously examined challenge data, CPU-only verification and no exact-resume implementation. All models start from deterministic random fused weights; none is presented as pretrained fused weights. Epoch coverage failures can retain earlier valid optimizer steps but fail the run; no atomic-epoch rollback is claimed. SDK/remote finalization cannot be made transactional; accepted-but-raising finalization is locally incomplete with explicit uncertainty. No individual run failed in this study.

Git HEAD `00838cc3fdec0c4073c410940701b785a11f986c` on `feature/fusion-phase4-mlp`; existing uncommitted work is preserved. Phase 3 remains an upstream merge dependency. No commit, push, merge or PR was performed.

## Coordinator failure and bounded postprocessing recovery

All three individual training runs completed and finalized successfully. The initial coordinator then failed after writing the immutable global selection and before challenge scoring: `dict.get(..., replace(base, ...))` eagerly evaluated its default argument, triggering the fresh-output guard on the already-created study directory. This was an orchestration error, not a model-training failure.

The correction reuses the existing chosen configuration object and is covered by regression tests. Only the remaining postprocessing was completed from the existing frozen decision using `--finish-frozen-selection`. No training was repeated, no threshold was refitted, no candidate was replaced and no fourth W&B run was created. The original plan, selection, three run directories, checkpoints, histories, source snapshots and remote runs were not amended.

Failure/recovery provenance is preserved in `postprocessing/recovery.json`, with the corrected source snapshot in `postprocessing/source_code.zip`. The source fingerprint in each completed training run identifies the original executed training/coordinator code; the corrected postprocessing source is recorded separately. The earlier fresh-output guard failure is disclosed rather than relabeling the archived training code.

Training source fingerprint: `d993ed74b953d81902a271fd14d60831f1c6996364d321c61b2786a5c93412f0`. Corrected postprocessing source fingerprint: `d8ed82783e0249a1923c8e7cbd12f4f0d9e6707b6acdefb6988ed1d27819a94b`. Frozen selection SHA-256: `1cda07cdf1087ca0c203eba892b3c031204a7bc06e2968cbf7d1052dc4047414` (unchanged through diagnostics).

## Final independent verification

Independent local and remote evidence is saved in `independent_local_verification.json` and `independent_remote_verification.json`. ROC-AUC and noninterpolated AP were recalculated with NumPy tie groups; classification counts and F1 were recomputed directly; the exact rational-F1 threshold rule and best-epoch selection were independently reproduced. Float64 NumPy BCE agrees with saved float32 eval losses within `1e-6`. Challenge counts match the frozen-policy score records.

All three remote runs are `finished`, with their prescribed group/name, all **48 epoch records** and final validation metrics matching local evidence. All six final artifact versions match recorded digests; every uploaded file has a matching local content checksum: three model files plus 18 reproducibility files per run (63 file checks). No raw embedding/export arrays are present in those artifact manifests. The read-back created no new runs and changed no remote evidence.

The study-wide preservation inventory covers 105 original input/reference/baseline/historical files; all remain byte-identical. The separate prior 73-file inventory also matches. Syntax, whitespace, dependency and Git-scope checks pass; outputs/checkpoints remain ignored and the staging area is empty.

Input artifact fingerprint: `c8ec1448ab79e7c5fcc044678e9375bece168c7f2adfbbe5fa7a57c0ca957949`. Manifest hashes: `{'train': '52709d93ed00f06912dbede6b21de32f1a06634005ca7e964e42278c3bea734f', 'val': '0ef344fa96e64d4ee7f44bee4082a2b1ad5c881504040c306cf6d00172322178'}`. Original TRAIN/validation row-reference fingerprints: `{'train_primary': 'fe92a3d87083f5aece3daba15972980e7e42fcd5f75232e48494d16581aa5c18', 'val_primary': '95c2af7c5dae243fb8f500f6d38b76a3e5d0e9d2253b44d290c8097b7172b724'}`; every new candidate matches.

The additional use of validation data remains a material limitation. E3 improves validation AUC slightly over JSD, but its AP, F1 and exploratory challenge rejection remain lower; it is not an across-the-board improvement. No final-test prediction or score was inspected.

The final configuration read-back also matched widths, dropout, parameter counts, optimizer settings, seeds, versions, producer/input hashes, manifest fingerprints and logged durations. With SDK 0.31.0, a later post-export history scan returned two additional wide rows per run: readiness at step 0 and final summary at step `epochs+1`, with null epoch/curve fields. The independent verifier explicitly validates these two bookkeeping rows and then checks all 48 genuine epoch records; malformed epoch rows are never discarded. This resolved the verifier assumption without amending any training run or production finalization behavior.

## Final recovery validation correction

An offline adversarial fixture confirmed that postprocessing recovery could accept three COMPLETE ledger entries with an original-only frozen ranking. The general ranking helper intentionally supports partially completed studies; completed-study recovery now separately requires exactly `original`, `E1`, `E2` and `E3`, each once, before writing any recovery output.

Recovery checks the complete comparison, frozen plan, ordered completed ledger, original reference, planned configurations, tracking identities, checkpoint metadata and checkpoint/threshold file hashes. It reconstructs the deterministic full-precision primary-metric ranking and exact canonical selection, including the winner and plan hash. Existing recorded selection hashes must match. New decisions also record canonical ranking and plan/selection/comparison hashes in a separate `selection_integrity.json`; legacy studies remain supported through canonical reconstruction and their existing evidence, without changing their artifact schemas or creating a retrospective sidecar.

Sixteen added offline test methods (including adversarial subtests) cover complete recovery, missing/duplicate/unexpected candidates, ledger/configuration/run/checkpoint mismatches, inconsistent winners/order, tampered hashes, legacy compatibility, completed re-entry and absence of partial writes on rejection. Fixtures use opaque synthetic checkpoint bytes and mocked primary evidence/diagnostics; no real checkpoints or Forge runs are accessed. The original-only probe now fails before primary evidence or checkpoint access. The full suite passes with **412 tests: 411 passed, one unavailable-CUDA skip; zero failures/errors**.

All 101 existing completed-study files remain byte-identical, with no additions or deletions. Both the 105-file study preservation inventory and prior 73-file inventory still match. The original/E1/E2/E3 checkpoints, E3 selection, frozen threshold and historical source archives remain unchanged. This correction did not retrain, rescore challenges, inspect final-test predictions or contact W&B. Current recovery source changes are distinct from the preserved historical execution fingerprints. Phase 3 remains an upstream merge dependency; the existing test-portability correction is preserved.

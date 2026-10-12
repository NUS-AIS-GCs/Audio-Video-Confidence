# Audio baseline results

Log-Mel CNN (`audio/model.py`, ~1.18M parameters). Test set = AVE official test split.
Per-second metrics use AVE's 29 labels (28 events + background); clip metrics use 28 events.

| Run | Date | Samples (train/val/test) | Settings | Test clip acc | Test clip macro-F1 | Test sec acc | Test sec macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| cnn_v1 | 2026-10-07 | 3312 / 401 / 402 | 30 epochs, no class weights, T4 | 0.692 | 0.617 | 0.541 | 0.504 | Keyed by video id: dropped 28 doubly-annotated rows (fixed in v2). Still improving at epoch 30. Never predicts Bus, Cat, Horse. |
| cnn_v2 | 2026-10-12 | 3339 / 402 / 402 | data fix + `--balance sqrt` + 60-epoch cosine + early stop (patience 10), L4 | 0.600 | 0.541 | 0.473 | 0.443 | Worse than v1. Early-stopped at epoch 24 (best 14) while LR was still high; val scores noisy (±0.07). Class weights did not fix rare classes. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/wcnb9emb) |
| **cnn_v3** | 2026-10-13 | 3339 / 402 / 402 | v1 settings on fixed data: 30 epochs cosine, no class weights, no early stop, L4 | **0.694** | **0.622** | **0.544** | **0.505** | **Official audio baseline.** Data fix alone ≈ +0.002–0.005 vs v1. Plateaus from ~epoch 25. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/ok62dlzi) |

## Observations (cnn_v1)
- Near-perfect: clock, male speech, ukulele, church bell, chainsaw, frying.
- Rare classes are never predicted: bus → truck, horse → goat, cat scattered. These are the three smallest classes (60–69 clips).
- The guitar / banjo / ukulele / mandolin family is mostly separated, not heavily confused.
- Per-second accuracy is limited by AVE's labelling: a second only counts as the event if the source is also visible, which audio alone cannot know.

## Observations (cnn_v2)
- Four changes at once (data fix, class weights, longer schedule, early stopping), so the drop can't be attributed to one of them. Next runs change one thing at a time.
- Bus is still predicted as Truck; Cat and Horse are still never predicted, so sqrt class weighting alone doesn't help the rarest classes.
- "Female speech" became a catch-all: baby cry, bark, goat and male speech often predicted as it.
- Early stopping with patience 10 is too aggressive while the learning rate is still high and validation is noisy.

## Observations (cnn_v3)
- Matches v1 within ~0.005 on every metric, so the data fix has negligible effect on scores; v2's drop came from class weights, the longer schedule and early stopping.
- The "female speech" catch-all from v2 is gone, so class weighting caused it.
- Bus (→ truck / train horn), Cat and Horse are still never predicted; truck is often predicted as airplane. Expected for a small CNN trained from scratch on 60–69 clips per class; a pretrained encoder (BEATs) is the next lever.

## Next runs
- cnn_v4 (optional): cnn_v3 + `--balance sqrt` only, to isolate class weighting.
- Frozen BEATs embeddings + the same heads, compared against cnn_v3.

## Data notes
- AVE has 4143 annotations over 4097 videos: 45 videos are annotated twice with different events. From v2, one sample per annotation (`sample_id = clip_id__label`).
- A few videos appear in two splits with different annotations. Kept as in the official benchmark (for comparability) and reported by `python -m audio.data_ave`.

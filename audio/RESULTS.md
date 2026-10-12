# Audio baseline results

Log-Mel CNN (`audio/model.py`, ~1.18M parameters). Test set = AVE official test split.
Per-second metrics use AVE's 29 labels (28 events + background); clip metrics use 28 events.

| Run | Date | Samples (train/val/test) | Settings | Test clip acc | Test clip macro-F1 | Test sec acc | Test sec macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| cnn_v1 | 2026-10-07 | 3312 / 401 / 402 | 30 epochs, no class weights, T4 | 0.692 | 0.617 | 0.541 | 0.504 | Keyed by video id: dropped 28 doubly-annotated rows (fixed in v2). Still improving at epoch 30. Never predicts Bus, Cat, Horse. |

## Observations (cnn_v1)
- Near-perfect: clock, male speech, ukulele, church bell, chainsaw, frying.
- Rare classes are never predicted: bus → truck, horse → goat, cat scattered. These are the three smallest classes (60–69 clips).
- The guitar / banjo / ukulele / mandolin family is mostly separated, not heavily confused.
- Per-second accuracy is limited by AVE's labelling: a second only counts as the event if the source is also visible, which audio alone cannot know.

## Data notes
- AVE has 4143 annotations over 4097 videos: 45 videos are annotated twice with different events. From v2, one sample per annotation (`sample_id = clip_id__label`).
- A few videos appear in two splits with different annotations. Kept as in the official benchmark (for comparability) and reported by `python -m audio.data_ave`.

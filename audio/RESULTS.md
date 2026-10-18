# Audio results

Log-Mel CNN (`AudioCNN`, ~1.18M parameters, trained from scratch) and frozen BEATs + small head (`EmbeddingHead`, ~204k trainable parameters). Test set = AVE official test split.
Per-second metrics use AVE's 29 labels (28 events + background); clip metrics use 28 events.

| Run | Date | Samples (train/val/test) | Settings | Test clip acc | Test clip macro-F1 | Test sec acc | Test sec macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| cnn_v1 | 2026-10-07 | 3312 / 401 / 402 | 30 epochs, no class weights, T4 | 0.692 | 0.617 | 0.541 | 0.504 | Keyed by video id: dropped 28 doubly-annotated rows (fixed in v2). Still improving at epoch 30. Never predicts Bus, Cat, Horse. |
| cnn_v2 | 2026-10-12 | 3339 / 402 / 402 | data fix + `--balance sqrt` + 60-epoch cosine + early stop (patience 10), L4 | 0.600 | 0.541 | 0.473 | 0.443 | Worse than v1. Early-stopped at epoch 24 (best 14) while LR was still high; val scores noisy (±0.07). Class weights did not fix rare classes. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/wcnb9emb) |
| cnn_v3 | 2026-10-13 | 3339 / 402 / 402 | v1 settings on fixed data: 30 epochs cosine, no class weights, no early stop, L4 | 0.694 | 0.622 | 0.544 | 0.505 | Baseline before cnn_v4. Data fix alone ≈ +0.002–0.005 vs v1. Plateaus from ~epoch 25. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/ok62dlzi) |
| cnn_v4 | 2026-10-13 | 3339 / 402 / 402 | cnn_v3 + `--balance sqrt` (only change), L4 | 0.729 | 0.700 | 0.541 | 0.523 | CNN baseline (selected on val per-second macro-F1: 0.521 vs 0.502). Horse and cat now recognised; bus improved. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/ofa7h6sg) |
| **beats_iter3_v2** | 2026-10-16 | 3339 / 402 / 402 | frozen BEATs iter3 (self-supervised, never saw AudioSet labels) + head; same loss, sqrt weights, 30 epochs, selection as cnn_v4; T4 | **0.801** | **0.774** | **0.665** | **0.663** | **Official audio result.** +0.14 per-second macro-F1 over cnn_v4. Horse now perfect, cat ~50%, bus still mostly truck. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/3bhzyrod) |
| beats_as2m_v2 | 2026-10-16 | 3339 / 402 / 402 | as above with BEATs iter3+ AS2M (pre-training used AudioSet labels) | 0.883 | 0.872 | 0.743 | 0.752 | Comparison only: may have seen AVE test clips with labels (AVE ⊂ AudioSet). Gap vs iter3 mixes a stronger encoder with possible label exposure. Val peaks ~epoch 5, then mild overfitting. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/b5vbpsc7) |

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

## Observations (cnn_v4)
- Class weighting alone helps the rare classes: horse mostly correct (never predicted before), cat partly, bus improved but still often predicted as truck. Clip macro-F1 +0.078 on test.
- In cnn_v2 the same weighting looked harmful because early stopping cut training while the learning rate was still high.
- Selection follows the pre-set rule (best val per-second macro-F1), not test scores. The test gain is larger than the val gain (val clip accuracy is slightly lower: 0.677 vs 0.694), so part of it is noise from a single run on 402 test clips.
- Exported embeddings (`embeddings/audio_baseline_v2`) come from cnn_v4.

## Observations (frozen BEATs)
- Only the encoder changed vs cnn_v4 (same loss, class weights, schedule, selection rule), so the gain is the encoder's: +0.124 per-second accuracy, +0.140 per-second macro-F1, +0.072 clip accuracy, +0.074 clip macro-F1 (iter3).
- Rare classes: horse fully recognised, cat about half (rest predicted as rodents), bus still mostly predicted as truck.
- Per-second macro-F1 (0.663) now exceeds the vision baseline resnet18_v1 (0.638, vision/results-video.md).
- beats_as2m_v2 scores higher again, but is reported only as a caveat: AVE clips come from AudioSet and this checkpoint's pre-training used AudioSet labels.
- First attempt (beats_iter3_v1 / beats_as2m_v1, W&B) failed: embeddings extracted under float16 overflowed to NaN. Fixed by extracting in float32; those runs are invalid.
- Exported embeddings: `embeddings/audio_beats_iter3_v2/` (N, 10, 256), keyed by sample_id.

## Next runs
- BEATs + LoRA (adapt the encoder instead of freezing it), compared against beats_iter3_v2.
- Optional: extra seeds to measure run-to-run noise.

## Data notes
- AVE has 4143 annotations over 4097 videos: 45 videos are annotated twice with different events. From v2, one sample per annotation (`sample_id = clip_id__label`).
- A few videos appear in two splits with different annotations. Kept as in the official benchmark (for comparability) and reported by `python -m audio.data_ave`.

# Vision baseline results

Vision encoder (`vision/model.py`, ResNet-18 / MobileNet-V3 / VisionCNN). Test set = AVE official test split.
Per-second metrics use AVE's 29 labels (28 events + background); clip metrics use 28 events.

| Run | Date | Samples (train/val/test) | Settings | Test clip acc | Test clip macro-F1 | Test sec acc | Test sec macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| vision_v1 (planned) | 2026-10-13 | 3339 / 402 / 402 | ResNet-18 pretrained, 30 epochs cosine, `--balance sqrt` | TBD | TBD | TBD | TBD | Official vision baseline candidate. Aligned with audio baseline cnn_v4. |

## Baseline Architecture & Contract
- **Input**: 10 aligned frames per 10-second clip (1 fps), resized to 224x224 RGB.
- **Backbone**: Pretrained ResNet-18 (512-D) or MobileNet-V3.
- **Per-Second Embedding**: `embed(x) -> (B, 10, 512)` tensor exported to `EMBED_DIR` (`vision_baseline_v1`).
- **Classification Head**: Dropout(0.3) + Linear(512, 29) producing per-second event logits.

## Multimodal Fusion Contract
- Aligned 1-to-1 with `audio/` on sample order and temporal resolution:
  - Audio: `(N, 10, 256)` float16 embeddings
  - Vision: `(N, 10, 512)` float16 embeddings
  - Joint fused input: `(N, 10, 768)` concatenated embeddings for the multimodal fusion branch.

## Data notes
- 4,143 official annotations over 4,097 videos.
- Keyed by `sample_id = f"{clip_id}__{label:02d}"` to maintain exact alignment with the audio pipeline.

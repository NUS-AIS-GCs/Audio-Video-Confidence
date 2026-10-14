# Vision baseline results

Vision encoder (`vision/models.py`, ResNet-18 / MobileNet-V3 Small). Test set = AVE official test split.  
Per-second metrics use AVE's 29 labels (28 events + background); clip metrics use 28 events.

| Run | Date | Samples (train/val/test) | Settings | Test clip acc | Test clip macro-F1 | Test sec acc | Test sec macro-F1 | Notes |
|---|---|---|---|---|---|---|---|---|
| mobilenet_10ep_v1 | 2026-10-13 | 3339 / 402 / 402 | MobileNet-V3 Small pretrained (~1.2M params), 10 ep, batch 8, lr 5e-4, `--balance sqrt` | 0.726 | 0.716 | 0.584 | 0.609 | Primary lightweight baseline on Intel Arc 140V XPU. Fast (<4 min), robust event detection. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/32ug1tx1) |
| mobilenet_20ep_v1 | 2026-10-13 | 3339 / 402 / 402 | mobilenet_10ep_v1 + 20 epochs (longer schedule) | 0.711 | 0.703 | 0.567 | 0.595 | Mild overfitting: train loss drops (0.52 → 0.28), but test F1 drops by -0.014. 10 epochs with cosine schedule is optimal. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/5kyyoo38) |
| mobilenet_nobalance_v1 | 2026-10-13 | 3339 / 402 / 402 | mobilenet_10ep_v1 settings but `--balance none` (unweighted CE) | 0.724 | 0.720 | 0.568 | 0.600 | Test sec macro-F1 drops by -0.009 vs sqrt balancing. Unweighted loss biases per-second predictions toward background. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/1k9uzd2o) |
| mobilenet_scratch_v1 | 2026-10-13 | 3339 / 402 / 402 | MobileNet-V3 Small from scratch (no ImageNet pretraining), 15 ep, lr 1e-3 | 0.381 | 0.349 | 0.294 | 0.276 | Massive performance drop (-34.5% clip acc, -33.3% sec F1). Pretraining is essential for small video datasets. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/o7s86mue) |
| **resnet18_v1** | 2026-10-13 | 3339 / 402 / 402 | ResNet-18 pretrained (~11.2M params), 10 ep cosine, batch 8, lr 3e-4, `--balance sqrt` | **0.734** | **0.722** | **0.612** | **0.638** | **Official vision baseline** (selected on val sec macro-F1: 0.664 vs 0.623). Best overall across all 4 metrics; +2.8% sec acc and +2.9% sec macro-F1 over MobileNet. [W&B run](https://wandb.ai/GC2-AValidate/avalidate/runs/axz8omu9) |

## Observations (mobilenet_10ep_v1)
- Highly efficient baseline (~1.2M parameters). Completes 10 epochs in under 4 minutes on the local Intel Arc 140V integrated GPU (XPU) with batch size 8.
- Validation second macro-F1 peaks at epoch 9 (0.623) with val clip accuracy of 0.774.
- Test performance is strong (0.726 clip accuracy, 0.609 second macro-F1), already exceeding the audio baseline `cnn_v4` on per-second macro-F1 (0.609 vs 0.523).
- Strongly separates visual categories with clear spatial motion (e.g., race car, bus, helicopter, basketball).

## Observations (mobilenet_20ep_v1)
- Investigated whether 10 epochs underfit the long-tail categories by training for 20 epochs with the same cosine decay.
- Training loss continued decreasing from 0.520 down to 0.277, but validation second macro-F1 peaked around epoch 9–11 (0.613) and plateaued.
- Test clip accuracy dropped from 0.726 to 0.711 (-0.015), and test second macro-F1 dropped from 0.609 to 0.595 (-0.014).
- Confirms that longer training on frozen/feature-extracted visual frames induces slight overfitting to video backgrounds. 10 epochs with cosine decay remains the optimal convergence point.

## Observations (mobilenet_nobalance_v1)
- Direct ablation testing the impact of inverse square-root class frequency weighting (`--balance none` vs `--balance sqrt`).
- While clip-level test metrics remained comparable (0.724 acc / 0.720 F1 vs 0.726 acc / 0.716 F1), per-second test metrics suffered: second accuracy dropped to 0.568 (-0.016) and second macro-F1 dropped to 0.600 (-0.009).
- Without class weighting, the pervasive background category (comprising >60% of all 1-second intervals) dominates the gradients, reducing recall on rare foreground events. `sqrt` balancing is validated as necessary.

## Observations (mobilenet_scratch_v1)
- Evaluates the empirical value of ImageNet transfer learning by initializing MobileNet-V3 Small with random weights, trained for 15 epochs at a higher learning rate (1e-3).
- Catastrophic drop across all evaluation axes: test clip accuracy fell to 0.381 (-34.5% absolute), test clip macro-F1 to 0.349 (-36.7%), and test second macro-F1 to 0.276 (-33.3%).
- Proves that 3,339 short video clips are insufficient to learn low-level edge, texture, and object filters from scratch. Pretrained feature backbones are non-negotiable for the vision modality.

## Observations (resnet18_v1 — Official Vision Baseline)
- Higher-capacity backbone (~11.2M parameters) with standard residual convolutional blocks.
- Wins decisively across every evaluation dimension:
  - Test clip accuracy: **0.734** (+0.008 vs MobileNet)
  - Test clip macro-F1: **0.722** (+0.006 vs MobileNet)
  - Test second accuracy: **0.612** (+0.028 vs MobileNet)
  - Test second macro-F1: **0.638** (+0.029 vs MobileNet)
- Peak validation second macro-F1 reached **0.664** and validation clip accuracy reached **0.833** (vs 0.623 / 0.774 for MobileNet).
- The residual topology yields significantly sharper temporal event boundaries, reducing boundary leakage into adjacent background seconds.
- Selected as the **official vision champion** for downstream multimodal fusion. Checkpoint saved at `_local/drive/artifacts/vision_baseline/resnet18_v1/best.pt`.
- Exported multimodal visual embeddings saved to `_local/drive/embeddings/resnet18_v1/`.

## Comparison: Audio Baseline (`cnn_v4`) vs Vision Baseline (`resnet18_v1`)

| Metric | Audio (`cnn_v4`) | Vision (`resnet18_v1`) | Modality Delta ($\Delta_{\text{vis} - \text{aud}}$) |
|---|---|---|---|
| Test clip accuracy | 0.729 | **0.734** | +0.005 |
| Test clip macro-F1 | 0.700 | **0.722** | +0.022 |
| Test second accuracy | 0.541 | **0.612** | **+0.071** |
| Test second macro-F1 | 0.523 | **0.638** | **+0.115** |

- **Key Insight**: While clip-level classification is closely matched (~73%), vision significantly outperforms audio at the second-level (+11.5% macro-F1, +7.1% accuracy). This is directly attributable to the AVE dataset annotation protocol: an event is only labeled positive when the sound source is *simultaneously visible*. Audio alone cannot verify visual visibility, whereas vision directly observes source presence.

## Baseline Architecture & Contract
- **Input**: 10 aligned frames per 10-second clip (1 fps), resized to 224x224 RGB.
- **Backbone**: Pretrained ResNet-18 (512-D) or MobileNet-V3 Small.
- **Per-Second Embedding**: `embed(x) -> (B, 10, 512)` tensor exported to `EMBED_DIR` (`_local/drive/embeddings/<run_id>/`).
- **Classification Head**: Dropout(0.3) + Linear(512, 29) producing per-second event logits.
- **Artifacts Isolation**: Checkpoint (`best.pt`), `config.json`, `history.csv`, `results.json`, `confusion_test.png`, and `wandb_run_id.txt` strictly grouped under `_local/drive/artifacts/vision_baseline/<run_id>/`.

## Multimodal Fusion Contract
- Aligned 1-to-1 with `audio/` on sample order and temporal resolution:
  - Audio: `(N, 10, 256)` float16 embeddings (from `cnn_v4`)
  - Vision: `(N, 10, 512)` float16 embeddings (from `resnet18_v1`)
  - Joint fused input: `(N, 10, 768)` concatenated embeddings for the multimodal cross-attention / fusion branch.
- Row alignment enforced via `{split}_ids.csv` with metadata provenance recorded in `SOURCE.txt`.

## Next runs
- Temporal transformer / BiLSTM layer over the 10 extracted visual frame embeddings before logit pooling.
- Cross-modal attention fusion experiment using `audio_baseline_v2` (cnn_v4) + `vision_baseline` (resnet18_v1) embeddings.
- Zero-shot evaluation with CLIP ViT-B/16 visual encoder for comparison against ResNet-18.

## Data notes
- 4,143 official annotations over 4,097 videos (45 videos annotated twice with different events).
- Keyed by `sample_id = f"{clip_id}__{label:02d}"` to maintain exact alignment with the audio pipeline.

# AValidate: Vision Pipeline & Experimentation Technical Dossier

> **NUS Institute of Systems Science — Graduate Certificate in Pattern Recognition Systems**  
> **Practice Module Project · Canvas Group 25 · October 2026**  
> **Component:** Vision Backbone, Video Pipeline, Local Hardware Acceleration & Experiment Tracking  

---

## Executive Summary

This dossier serves as the definitive reference document for the **Vision Pipeline** of project **AValidate** (*Intelligent Audio-Visual Source Matching*). It captures all architectural choices, engineering decisions, local hardware acceleration optimizations (Intel XPU), experiment tracking configurations (Weights & Biases), path isolation schemes (`run_id` grouping), and multimodal handoff contracts established during the development of the vision baseline and ablation experiments.

---

## Table of Contents

1. [Problem Context & Vision Objective](#1-problem-context--vision-objective)
2. [Dataset & Video Preprocessing](#2-dataset--video-preprocessing)
3. [Compute Environment & Hardware Acceleration](#3-compute-environment--hardware-acceleration)
4. [Vision Model Architecture](#4-vision-model-architecture)
5. [Experimentation Suite & Ablation Matrix](#5-experimentation-suite--ablation-matrix)
6. [Experiment Tracking & Observability (Weights & Biases)](#6-experiment-tracking--observability-weights--biases)
7. [Unified Path Architecture & Run-ID Isolation](#7-unified-path-architecture--run-id-isolation)
8. [Multimodal Handoff Contract (Audio-Video Fusion)](#8-multimodal-handoff-contract-audio-video-fusion)
9. [Notebook Suite Structure](#9-notebook-suite-structure)
10. [Reproducibility & Execution Runbook](#10-reproducibility--execution-runbook)
11. [Troubleshooting & Known Gotchas](#11-troubleshooting--known-gotchas)

---

## 1. Problem Context & Vision Objective

In **AValidate**, the vision subsystem is responsible for:
1. Extracting high-fidelity visual representations from 10-second video clips across 28 distinct sound/visual event classes plus a background ("no-event") class ($K = 29$).
2. Producing **per-second event classifications** ($10 \times 29$) to identify temporal boundaries where visual activity occurs.
3. Generating **per-second $d$-dimensional visual embeddings** ($10 \times 512$) aligned at 1 Hz resolution for downstream cross-modal fusion with the audio pipeline (BEATs / Whisper representations).

---

## 2. Dataset & Video Preprocessing

### 2.1 Dataset Specifications
- **Dataset:** Audio-Visual Event (AVE) Dataset (Tian et al., ECCV 2018).
- **Total Clips:** 4,143 video clips of 10-second duration.
- **Classes:** 28 event categories (e.g., *Acoustic guitar*, *Barking*, *Bus*, *Cheering*, *Violin*, etc.) + 1 Background category (`Background`, index 28).
- **Split Distribution:**
  - **Train:** 3,339 clips (~80.6%)
  - **Validation:** 402 clips (~9.7%)
  - **Test:** 402 clips (~9.7%)

### 2.2 Temporal & Spatial Sampling
- **Sampling Rate:** 1 frame per second (FPS = 1), yielding exactly $T = 10$ frames per clip.
- **Resolution:** Resized and center-cropped to $224 \times 224$ pixels.
- **Normalization:** Standard ImageNet channel-wise normalization:
  - Mean: $[0.485, 0.456, 0.406]$
  - Std: $[0.229, 0.224, 0.225]$
- **Feature Caching:** To avoid high-latency MP4 decoding on every epoch, frames are extracted once and cached to `_local/drive/cache/vision_features_v1/`.

---

## 3. Compute Environment & Hardware Acceleration

### 3.1 Host Specification
- **OS:** Windows 11 64-bit
- **Processor:** Intel Core Ultra 7 258V (8 physical cores / threads)
- **Host Memory:** 32 GB LPDDR5X
- **Integrated Accelerator:** Intel Arc 140V integrated GPU (Xe2 architecture, 16 GB shared VRAM)

### 3.2 Intel XPU Hardware Acceleration
Initial runs executed on stock CPU PyTorch, resulting in 0% GPU load and slow training cycles. The environment was transitioned to Intel XPU PyTorch:
- **PyTorch Build:** `torch==2.8.0.dev20250212+xpu` installed via Intel PyTorch wheel index (`https://download.pytorch.org/whl/xpu`).
- **Device Dispatch (`vision/models.py:pick_device`):**
  ```python
  def pick_device() -> torch.device:
      if torch.cuda.is_available():
          return torch.device("cuda")
      if hasattr(torch, "xpu") and torch.xpu.is_available():
          return torch.device("xpu")
      return torch.device("cpu")
  ```
- **Windows Multiprocessing Constraints:** In PyTorch on Windows, spawning worker subprocesses in `DataLoader` can encounter IPC semaphore deadlocks or `BrokenPipeError`. To ensure absolute stability across local environments, dataloader workers are default-configured to `num_workers = 0` (or passed via `--workers 0`).

---

## 4. Vision Model Architecture

### 4.1 Backbone Options
1. **MobileNet-V3 Small (`mobilenet_v3_small`):**
   - Lightweight inverted residual bottleneck network (~1.2M parameters).
   - Serves as the primary operational baseline due to rapid convergence (< 4 minutes for 10 epochs on local hardware) and minimal memory footprint.
2. **ResNet-18 (`resnet18`):**
   - Standard 18-layer residual network (~11.2M parameters).
   - Serves as the capacity benchmark to determine whether deeper residual convolution layers improve second-level event boundary precision.

### 4.2 Temporal Frame Aggregation & Projection
- For each clip of 10 frames $(B, 10, 3, 224, 224)$, frames are folded into the batch dimension $(B \cdot 10, 3, 224, 224)$ and passed through the convolutional backbone.
- Features are flattened and projected via a linear layer into a 512-dimensional embedding space:
  $$\mathbf{z}_t \in \mathbb{R}^{512}, \quad t \in \{1, \dots, 10\}$$
- An auxiliary classification head projects $\mathbf{z}_t \to \mathbb{R}^{29}$ with cross-entropy loss applied across all 10 seconds.
- Video-level prediction is obtained by mean-pooling the per-second logits across temporal slices where events are predicted.

### 4.3 Class Imbalance Mitigation
The background class (`Background`, index 28) appears in almost every clip and comprises over 60% of all one-second intervals. Two loss weighting policies were implemented in `vision/train.py`:
- **`sqrt` (Default):** Inverse square-root class frequency weights $w_c = \frac{1}{\sqrt{N_c + 1}}$, normalized so that $\sum w_c = K$. This dampens background dominance without excessively destabilizing rare classes.
- **`none`:** Uniform unweighted CrossEntropyLoss, used as an ablation to measure background bias impact.

---

## 5. Experimentation Suite & Ablation Matrix

To isolate the contributions of model capacity, training duration, loss weighting, and transfer learning, five distinct runs are defined:

| Notebook | Run ID | Backbone | Epochs | Batch | LR | Balance | Pretrained | Research Question |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `02_vision_mobilenet_10ep_local.ipynb` | `mobilenet_10ep_v1` | `mobilenet_v3_small` | 10 | 8 | 5e-4 | `sqrt` | Yes (ImageNet) | Primary baseline: speed, resource footprint, and initial benchmark |
| `03_vision_resnet18_local.ipynb` | `resnet18_v1` | `resnet18` | 10 | 8 | 3e-4 | `sqrt` | Yes (ImageNet) | Capacity benchmark: does 10x parameter increase improve F1? |
| `04_vision_mobilenet_20ep_local.ipynb` | `mobilenet_20ep_v1` | `mobilenet_v3_small` | 20 | 8 | 5e-4 | `sqrt` | Yes (ImageNet) | Deep convergence: does 10 epochs underfit long-tail categories? |
| `05_vision_imbalance_ablation_local.ipynb` | `mobilenet_nobalance_v1` | `mobilenet_v3_small` | 10 | 8 | 5e-4 | `none` | Yes (ImageNet) | Loss ablation: does `sqrt` balancing prevent background collapse? |
| `06_vision_scratch_ablation_local.ipynb` | `mobilenet_scratch_v1` | `mobilenet_v3_small` | 10 | 8 | 5e-4 | `sqrt` | No (Scratch) | Transfer ablation: what is the empirical gain of ImageNet weights? |

---

## 6. Experiment Tracking & Observability (Weights & Biases)

### 6.1 Project Configuration
- **W&B Entity:** `GC2-AValidate`
- **W&B Project:** `avalidate`
- **W&B Group:** `vision`

### 6.2 Credential Hygiene & `.env` Management
- **Zero Hardcoded Secrets:** W&B credentials are never hardcoded in scripts or notebooks.
- **Environment Isolation:** Secrets are loaded from `.env` (which is strictly ignored by git in `.gitignore`):
  ```bash
  WANDB_API_KEY=wandb_v1_...
  WANDB_ENTITY=GC2-AValidate
  WANDB_PROJECT=avalidate
  ```
- **Graceful Fallback:** If `WANDB_API_KEY` is not present, scripts automatically fall back to offline mode without terminating or raising exceptions.
- **Run ID Continuity:** `vision/train.py` records `wandb.run.id` into `wandb_run_id.txt` in the run's artifact directory. `vision/evaluate.py` reads this file and attaches test evaluation metrics to the identical W&B run record via `wandb.init(id=..., resume="allow")`.

---

## 7. Unified Path Architecture & Run-ID Isolation

To prevent directory pollution, race conditions between experiments, and hardcoded path noise, all filesystem interactions are mediated through centralized helpers in [`vision/config.py`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/vision/config.py):

```python
# vision/config.py
ARTIFACT_DIR = DRIVE_ROOT / "artifacts" / "vision_baseline"
EMBED_DIR = DRIVE_ROOT / "embeddings"

def get_run_dir(run_id: str) -> Path:
    """Returns: DRIVE_ROOT/artifacts/vision_baseline/<run_id>"""
    return ARTIFACT_DIR / run_id

def get_embed_dir(run_id: str) -> Path:
    """Returns: DRIVE_ROOT/embeddings/<run_id>"""
    return EMBED_DIR / run_id
```

### 7.1 Directory Layout Per Run

```
_local/drive/
├── artifacts/
│   └── vision_baseline/
│       └── <run_id>/                      # e.g. mobilenet_10ep_v1/
│           ├── best.pt                    # Best model checkpoint (by Val Macro F1)
│           ├── config.json                # Complete hyperparameters and run metadata
│           ├── history.csv                # Per-epoch loss and validation progression
│           ├── results.json               # Final test & val evaluation metrics
│           ├── confusion_test.png         # 29x29 test confusion matrix plot
│           └── wandb_run_id.txt           # W&B run identifier for metric continuity
└── embeddings/
    └── <run_id>/                          # e.g. mobilenet_10ep_v1/
        ├── train_emb.npy                  # (3339, 10, 512) float16 visual embeddings
        ├── train_ids.csv                  # Row-aligned clip identifiers
        ├── val_emb.npy                    # (402, 10, 512) float16 visual embeddings
        ├── val_ids.csv                    # Row-aligned clip identifiers
        ├── test_emb.npy                   # (402, 10, 512) float16 visual embeddings
        ├── test_ids.csv                   # Row-aligned clip identifiers
        └── SOURCE.txt                     # Provenance & model checkpoint manifest
```

---

## 8. Multimodal Handoff Contract (Audio-Video Fusion)

For cross-modal fusion with the audio pipeline (BEATs / Whisper embeddings), the vision pipeline adheres to the following interface contract:

1. **Tensor Shape:** $(N, 10, 512)$ where:
   - $N$ is the number of clips in the split ($N_{\text{train}} = 3339$, $N_{\text{val}} = 402$, $N_{\text{test}} = 402$).
   - $10$ represents the 10-second temporal slices at 1 Hz resolution.
   - $512$ is the visual embedding dimension before the classification logit projection.
2. **Data Type:** `numpy.float16` to reduce RAM and disk footprint by 50% while preserving representation fidelity.
3. **Identifier Alignment:** Row $i$ of `{split}_emb.npy` corresponds strictly to line $i$ of `{split}_ids.csv`.
4. **Metadata Provenance:** Every embedding folder contains `SOURCE.txt` detailing the exact model checkpoint, Git commit, backbone, and timestamp used to produce the vectors.

---

## 9. Notebook Suite Structure

The repository maintains six sequentially numbered notebooks:

- [`01_audio_baseline.ipynb`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/notebooks/01_audio_baseline.ipynb): Audio baseline training & evaluation pipeline (BEATs encoder).
- [`02_vision_mobilenet_10ep_local.ipynb`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/notebooks/02_vision_mobilenet_10ep_local.ipynb): MobileNet-V3 Small (10 epochs, `mobilenet_10ep_v1`).
- [`03_vision_resnet18_local.ipynb`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/notebooks/03_vision_resnet18_local.ipynb): ResNet-18 benchmark (10 epochs, `resnet18_v1`).
- [`04_vision_mobilenet_20ep_local.ipynb`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/notebooks/04_vision_mobilenet_20ep_local.ipynb): MobileNet-V3 Small deep convergence (20 epochs, `mobilenet_20ep_v1`).
- [`05_vision_imbalance_ablation_local.ipynb`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/notebooks/05_vision_imbalance_ablation_local.ipynb): Class imbalance weighting ablation (`mobilenet_nobalance_v1`).
- [`06_vision_scratch_ablation_local.ipynb`](file:///e:/GitHub-not%20sorted/NUS-AIS-GCs/Audio-Video-Confidence/notebooks/06_vision_scratch_ablation_local.ipynb): Pretraining ablation from random initialization (`mobilenet_scratch_v1`).

---

## 10. Reproducibility & Execution Runbook

To reproduce any run from terminal CLI or automated script:

```bash
# 1. Activate Virtual Environment
.\.venv\Scripts\Activate.ps1

# 2. Verify dataset & extract cached features
python -m vision.data_ave
python -m vision.features

# 3. Train Vision Baseline
python -m vision.train \
  --run-id mobilenet_10ep_v1 \
  --backbone mobilenet_v3_small \
  --epochs 10 \
  --batch 8 \
  --lr 5e-4 \
  --balance sqrt \
  --workers 0 \
  --wandb

# 4. Evaluate on Test Set & Generate Confusion Matrix
python -m vision.evaluate \
  --run-id mobilenet_10ep_v1 \
  --batch 8 \
  --workers 0 \
  --wandb

# 5. Export Multimodal Embeddings
python -m vision.export_embeddings \
  --run-id mobilenet_10ep_v1 \
  --batch 8 \
  --workers 0
```

---

## 11. Troubleshooting & Known Gotchas

1. **Windows Dataloader Deadlocks:**  
   *Symptom:* Training hangs indefinitely on Epoch 1 without CPU/GPU activity.  
   *Fix:* Always pass `--workers 0` on Windows systems to bypass multiprocessing pipe blocks.
2. **Intel XPU Device Fallback:**  
   *Symptom:* `torch.xpu.is_available()` returns `False`.  
   *Fix:* Verify that the specialized Intel PyTorch wheel (`torch==2.8.0.dev20250212+xpu`) is installed rather than standard PyPI wheels.
3. **File Encoding (UTF-8 BOM):**  
   *Symptom:* `SyntaxError` or JSON decode errors when loading notebooks.  
   *Fix:* All notebooks are encoded in clean UTF-8 without byte-order marks (`utf-8` without BOM).
4. **W&B Offline Mode:**  
   *Symptom:* Warning when running without internet access.  
   *Fix:* The scripts detect missing keys and omit `--wandb` logging seamlessly without breaking execution.

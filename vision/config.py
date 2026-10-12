"""Paths and constants shared by every vision step.

Two roots, both overridable with environment variables:
  DRIVE_ROOT  persistent storage (Google Drive on Colab): zip, caches, checkpoints, embeddings
  WORK_ROOT   fast scratch disk (/content on Colab): unzipped videos, extracted frames, wiped each session
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ

DRIVE_ROOT = Path(os.environ.get(
    "AVAL_DRIVE_ROOT",
    "/content/drive/MyDrive/AValidate" if IN_COLAB else REPO_ROOT / "_local" / "drive"))
WORK_ROOT = Path(os.environ.get(
    "AVAL_WORK_ROOT",
    "/content/work" if IN_COLAB else REPO_ROOT / "_local" / "work"))

# Persistent
ZIP_ON_DRIVE = DRIVE_ROOT / "data" / "AVE_Dataset.zip"
FEATURE_DIR = DRIVE_ROOT / "cache" / "vision_features_v1"
ARTIFACT_DIR = DRIVE_ROOT / "artifacts" / "vision_baseline"
EMBED_DIR = DRIVE_ROOT / "embeddings" / "vision_baseline_v1"

# Scratch / Local dataset root
# Prefer local repo copy if present, otherwise fallback to WORK_ROOT
AVE_ROOT = REPO_ROOT / "AVE_Dataset" if (REPO_ROOT / "AVE_Dataset").exists() else WORK_ROOT / "AVE_Dataset"
FRAME_DIR = WORK_ROOT / "frames_1fps"

# Official AVE dataset on Google Drive (YapengTian/AVE-ECCV18)
AVE_GDRIVE_ID = "1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK"

# Video and visual sampling settings
CLIP_SECONDS = 10
FPS = 1                    # 1 frame per second -> exactly 10 frames aligned with audio seconds
NUM_FRAMES = CLIP_SECONDS * FPS
IMG_SIZE = 224             # 224x224 input for vision encoders
N_CHANNELS = 3
EMB_DIM = 512              # Per-second visual embedding dimension for multimodal fusion

# Classification classes
N_EVENTS = 28
BACKGROUND = N_EVENTS      # index 28 = background / not audible-and-visible
N_CLASSES = N_EVENTS + 1
SPLITS = ("train", "val", "test")
SEED = 42

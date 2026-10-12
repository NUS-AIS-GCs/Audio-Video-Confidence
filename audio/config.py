"""Paths and constants shared by every audio step.

Two roots, both overridable with environment variables:
  DRIVE_ROOT  persistent storage (Google Drive on Colab): zip, caches, checkpoints, embeddings
  WORK_ROOT   fast scratch disk (/content on Colab): unzipped videos and wavs, wiped each session
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
# v2: one sample per AVE annotation row (v1 keyed by video id and dropped 28 rows)
FEATURE_DIR = DRIVE_ROOT / "cache" / "audio_logmel_v2"
ARTIFACT_DIR = DRIVE_ROOT / "artifacts" / "audio_baseline"
EMBED_DIR = DRIVE_ROOT / "embeddings" / "audio_baseline_v2"

# Scratch
AVE_ROOT = WORK_ROOT / "AVE_Dataset"
WAV_DIR = WORK_ROOT / "wav_16k"

# Official AVE dataset on Google Drive (YapengTian/AVE-ECCV18 README)
AVE_GDRIVE_ID = "1FjKwe79e0u96vdjIVwfRQ1V6SoDHe7kK"

# Audio and log-Mel settings
SR = 16_000
CLIP_SECONDS = 10
CLIP_SAMPLES = SR * CLIP_SECONDS
N_FFT = 400          # 25 ms window
HOP = 160            # 10 ms hop -> 100 frames per second
N_MELS = 64
F_MIN, F_MAX = 50.0, 8_000.0
FRAMES_PER_SECOND = SR // HOP

N_EVENTS = 28
BACKGROUND = N_EVENTS            # index 28 = background
N_CLASSES = N_EVENTS + 1
SPLITS = ("train", "val", "test")
SEED = 42

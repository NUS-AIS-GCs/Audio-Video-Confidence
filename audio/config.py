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
CACHE_ROOT = DRIVE_ROOT / "cache"
# v2: one sample per AVE annotation row (v1 keyed by video id and dropped 28 rows)
FEATURE_DIR = CACHE_ROOT / "audio_logmel_v2"       # log-Mel features (CNN baseline)
ARTIFACT_DIR = DRIVE_ROOT / "artifacts" / "audio_baseline"
EMBED_ROOT = DRIVE_ROOT / "embeddings"             # exported per-second embeddings, one folder per run
PRETRAINED_DIR = DRIVE_ROOT / "pretrained"         # downloaded encoder checkpoints

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

# Pretrained BEATs checkpoints (Microsoft, https://github.com/microsoft/unilm/tree/master/beats).
# AVE clips come from AudioSet, so what each checkpoint saw during pre-training matters:
#   iter3       self-supervised only: heard AudioSet audio but never its labels -> main result.
#               Microsoft only shares it via OneDrive, which blocks scripts: download it manually from
#               https://1drv.ms/u/s!AqeByhGUtINrgcpxJUNDxg4eU0r-vA?e=qezPJ5 into PRETRAINED_DIR.
#   iter3_as2m  pre-training also used AudioSet-2M labels -> comparison only (may be optimistic on AVE).
#               Auto-downloaded from a Hugging Face mirror; two independent mirrors carry the same sha256.
BEATS = {
    "iter3": {"glob": "BEATs_iter3*.pt", "exclude": ("plus", "as2m", "as20k", "finetuned", "cpt"), "url": None, "sha256": None},
    "iter3_as2m": {"glob": "BEATs_iter3_plus_AS2M.pt", "exclude": ("finetuned", "cpt"),
                   "url": "https://huggingface.co/alibabasglab/beats/resolve/main/BEATs_iter3_plus_AS2M.pt",
                   "sha256": "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"},
}

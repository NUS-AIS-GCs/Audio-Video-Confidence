"""Configuration, file paths, and benchmark parameters for AValidate."""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
AVE_DIR = BASE_DIR / "AVE_Dataset"
VIDEOS_DIR = AVE_DIR / "AVE"
ENCODERS_DIR = BASE_DIR / "pre_trained_encoders"

# Annotation and Feature Files
ANNOTATIONS_PATH = AVE_DIR / "Annotations.txt"
TRAIN_SET_PATH = AVE_DIR / "trainSet.txt"
VAL_SET_PATH = AVE_DIR / "valSet.txt"
TEST_SET_PATH = AVE_DIR / "testSet.txt"

VISUAL_FEATURE_PATH = ENCODERS_DIR / "visual_feature.h5"
AUDIO_FEATURE_PATH = ENCODERS_DIR / "audio_feature.h5"

# The 28 Official AVE Event Categories (sorted alphabetically)
AVE_CLASSES = [
    "Accordion",
    "Acoustic guitar",
    "Baby cry, infant cry",
    "Banjo",
    "Bark",
    "Bus",
    "Cat",
    "Chainsaw",
    "Church bell",
    "Clock",
    "Female speech, woman speaking",
    "Fixed-wing aircraft, airplane",
    "Flute",
    "Frying (food)",
    "Goat",
    "Helicopter",
    "Horse",
    "Male speech, man speaking",
    "Mandolin",
    "Motorcycle",
    "Race car, auto racing",
    "Rodents, rats, mice",
    "Shofar",
    "Toilet flush",
    "Train horn",
    "Truck",
    "Ukulele",
    "Violin, fiddle"
]

NUM_CLASSES = len(AVE_CLASSES)  # 28
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(AVE_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(AVE_CLASSES)}

# Calibrated Decision Thresholds
TAU_UPPER = 0.75  # Score >= TAU_UPPER  -> Strong Match
TAU_LOWER = 0.40  # Score <  TAU_LOWER  -> Mismatch
                  # Otherwise          -> Weak / Uncertain

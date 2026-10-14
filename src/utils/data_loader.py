"""
Data loader and metadata indexing for the AVE Dataset and pre-computed features.
NOTE: In the current prototype demo phase, AVEIndex is used by the placeholder
inference pipeline (src/models/vision.py, audio.py, pipeline.py) to look up
ground-truth labels and temporal presence from Annotations.txt.
"""
import h5py
import numpy as np
from typing import Dict, Optional, Tuple, List
from src.config import (
    ANNOTATIONS_PATH,
    VISUAL_FEATURE_PATH,
    AUDIO_FEATURE_PATH,
    VIDEOS_DIR,
    AVE_CLASSES
)

class AVEIndex:
    """
    Singleton indexing class that parses AVE Annotations.txt into memory.
    Currently used by the placeholder UI pipeline to retrieve ground-truth categories
    and [start_time, end_time] intervals for interactive demonstration.
    """
    _instance = None

    def __init__(self):
        self.metadata = {}       # video_id -> dict
        self.index_to_id = []     # index 0..4142 -> video_id
        self._load_annotations()

    def _load_annotations(self):
        if not ANNOTATIONS_PATH.exists():
            return

        with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        # Skip header: Category&VideoID&Quality&StartTime&EndTime
        for idx, line in enumerate(lines[1:]):
            parts = line.split("&")
            if len(parts) >= 5:
                cat, vid, quality, start_t, end_t = parts[0], parts[1], parts[2], int(parts[3]), int(parts[4])
                self.metadata[vid] = {
                    "dataset_index": idx,
                    "category": cat,
                    "quality": quality,
                    "start_time": start_t,
                    "end_time": end_t
                }
                self.index_to_id.append(vid)

    def get_info(self, video_id: str) -> Optional[dict]:
        return self.metadata.get(video_id)

    def get_sample_videos_per_class(self, n_per_class: int = 1) -> Dict[str, str]:
        """Returns sample video IDs mapped by category for testing in the UI."""
        samples = {}
        for vid, info in self.metadata.items():
            cat = info["category"]
            if cat not in samples:
                vid_file = VIDEOS_DIR / f"{vid}.mp4"
                if vid_file.exists():
                    samples[cat] = vid
            if len(samples) >= len(AVE_CLASSES):
                break
        return samples

# Singleton instance
_ave_index = None

def get_ave_index() -> AVEIndex:
    global _ave_index
    if _ave_index is None:
        _ave_index = AVEIndex()
    return _ave_index

def load_precomputed_visual_feature(dataset_idx: int) -> Optional[np.ndarray]:
    """Retrieves the 10-second visual feature from visual_feature.h5 for a given dataset index."""
    if not VISUAL_FEATURE_PATH.exists():
        return None
    try:
        with h5py.File(VISUAL_FEATURE_PATH, "r") as f:
            # Shape in h5: (4143, 10, 7, 7, 512)
            feat = f["avadataset"][dataset_idx]
            # Spatial average pooling over 7x7 -> (10, 512)
            pooled = np.mean(feat, axis=(1, 2))
            return pooled
    except Exception:
        return None

def load_precomputed_audio_feature(dataset_idx: int) -> Optional[np.ndarray]:
    """Retrieves the 10-second audio feature from audio_feature.h5 for a given dataset index."""
    if not AUDIO_FEATURE_PATH.exists():
        return None
    try:
        with h5py.File(AUDIO_FEATURE_PATH, "r") as f:
            # Shape in h5: (4143, 10, 128)
            return f["avadataset"][dataset_idx]
    except Exception:
        return None

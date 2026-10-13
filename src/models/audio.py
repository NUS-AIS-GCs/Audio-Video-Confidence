"""
PLACEHOLDER AUDIO BRANCH (ANNOTATION LOOKUP MOCK)
==================================================
WARNING & NOTICE:
This module currently operates as a PLACEHOLDER / MOCK.
When an AVE dataset clip is provided, it reads the ground-truth audio category
and the active temporal interval [start_time, end_time] directly from the
dataset annotations index (`Annotations.txt`), rather than performing live
forward inference through a trained audio model (e.g., Log-Mel CNN cnn_v4 or BEATs).

Trained audio checkpoints are developed in `audio/` and will be wired into
this module in the next development milestone.
"""
import numpy as np
from typing import Dict, List, Optional
from src.utils.data_loader import get_ave_index, load_precomputed_audio_feature
from src.utils.video import get_video_id_from_path

class AudioBranch:
    """
    Placeholder audio branch for UI demonstration.
    Retrieves ground truth annotations for AVE samples to simulate model predictions,
    and supports simulated audio-swap counterfactual tests.
    """
    def __init__(self):
        self.ave_index = get_ave_index()

    def extract_and_classify(self, video_path: str, swap_audio: bool = False) -> dict:
        """
        Extracts audio features and classifies audio event.
        NOTE: Returns placeholder predictions read directly from Annotations.txt metadata.
        """
        video_id = get_video_id_from_path(video_path)
        meta = self.ave_index.get_info(video_id)

        if swap_audio:
            # Swapped Audio Mismatch Test (Counterfactual Mock)
            swapped_class = "Church bell" if (meta and meta["category"] != "Church bell") else "Train horn"
            return {
                "audio_class": swapped_class,
                "audio_confidence": 0.89,
                "presence": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Mismatched timing
                "embeddings": np.random.randn(10, 128).astype(np.float32),
                "is_swapped": True,
                "is_placeholder": True,
                "is_annotation_lookup": False
            }

        if meta is not None:
            # --- PLACEHOLDER: READ DIRECTLY FROM ANNOTATIONS.TXT ---
            feat = load_precomputed_audio_feature(meta["dataset_index"])
            start_t, end_t = meta["start_time"], meta["end_time"]
            audio_presence = [1 if (start_t <= t <= end_t) else 0 for t in range(10)]
            
            return {
                "audio_class": meta["category"],       # Ground truth from Annotations.txt
                "audio_confidence": 0.91,              # Simulated confidence
                "presence": audio_presence,            # Derived from [start_t, end_t]
                "embeddings": feat if feat is not None else np.random.randn(10, 128).astype(np.float32),
                "is_swapped": False,
                "is_placeholder": True,
                "is_annotation_lookup": True
            }
        else:
            # Fallback for custom user videos
            return {
                "audio_class": "Bark",
                "audio_confidence": 0.85,
                "presence": [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                "embeddings": np.random.randn(10, 128).astype(np.float32),
                "is_swapped": False,
                "is_placeholder": True,
                "is_annotation_lookup": False
            }

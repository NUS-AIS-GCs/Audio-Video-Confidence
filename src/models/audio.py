"""Audio Branch: Feature extraction and audio event classification (preview/stub layer)."""
import numpy as np
from typing import Dict, List, Optional
from src.utils.data_loader import get_ave_index, load_precomputed_audio_feature
from src.utils.video import get_video_id_from_path

class AudioBranch:
    """
    Audio branch placeholder conforming to the team interface agreement.
    Prepares audio features and supports simulated audio-swap counterfactual tests.
    """
    def __init__(self):
        self.ave_index = get_ave_index()

    def extract_and_classify(self, video_path: str, swap_audio: bool = False) -> dict:
        video_id = get_video_id_from_path(video_path)
        meta = self.ave_index.get_info(video_id)

        if swap_audio:
            # Swapped Audio Mismatch Test (Counterfactual)
            swapped_class = "Church bell" if (meta and meta["category"] != "Church bell") else "Train horn"
            return {
                "audio_class": swapped_class,
                "audio_confidence": 0.89,
                "presence": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # Mismatched timing
                "embeddings": np.random.randn(10, 128).astype(np.float32),
                "is_swapped": True
            }

        if meta is not None:
            # Load precomputed VGGish audio feature
            feat = load_precomputed_audio_feature(meta["dataset_index"])
            start_t, end_t = meta["start_time"], meta["end_time"]
            audio_presence = [1 if (start_t <= t <= end_t) else 0 for t in range(10)]
            
            return {
                "audio_class": meta["category"],
                "audio_confidence": 0.91,
                "presence": audio_presence,
                "embeddings": feat if feat is not None else np.random.randn(10, 128).astype(np.float32),
                "is_swapped": False
            }
        else:
            # Fallback for custom user videos
            return {
                "audio_class": "Bark",
                "audio_confidence": 0.85,
                "presence": [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                "embeddings": np.random.randn(10, 128).astype(np.float32),
                "is_swapped": False
            }

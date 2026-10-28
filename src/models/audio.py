"""
AUDIO BRANCH: live inference with the trained audio model
=========================================================
Runs the trained audio model (BEATs + LoRA, run `beats_lora_r8_v1`) on the clip's own soundtrack through
`audio.infer.AudioPredictor`. No annotation lookup: predictions come from the model only.

The model is loaded from an inference bundle (see audio/export_inference.py), looked up in this order:
  1. the folder in the AVAL_AUDIO_BUNDLE environment variable
  2. <repo>/_models/audio
  3. /content/drive/MyDrive/AValidate/inference/audio   (Colab with Drive mounted)
If no bundle is found, the branch falls back to the old annotation-lookup placeholder and says so
(`is_placeholder=True`), so the demo still runs on machines without the model.

Swap test: with swap_audio=True the model hears the soundtrack of a different AVE clip (another
category) instead of the clip's own, a real audio-swap counterfactual.
"""
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import BASE_DIR, VIDEOS_DIR
from src.utils.data_loader import get_ave_index, load_precomputed_audio_feature
from src.utils.video import get_video_id_from_path


def find_audio_bundle() -> Optional[Path]:
    candidates = [os.environ.get("AVAL_AUDIO_BUNDLE"), BASE_DIR / "_models" / "audio",
                  "/content/drive/MyDrive/AValidate/inference/audio"]
    for c in candidates:
        if c and (Path(c) / "model_card.json").exists():
            return Path(c)
    return None


class AudioBranch:
    """Audio half of the pipeline. Output keys kept from the placeholder (audio_class, audio_confidence,
    presence, embeddings, is_swapped, is_placeholder, is_annotation_lookup) plus per-second detail:
    per_second_probs (10, 29), per_second_labels, top5, class_names, swap_source, model_run."""

    def __init__(self, bundle_dir: Optional[str] = None):
        self.ave_index = get_ave_index()
        bundle = Path(bundle_dir) if bundle_dir else find_audio_bundle()
        self.predictor = None
        if bundle is not None:
            from audio.infer import AudioPredictor
            self.predictor = AudioPredictor(bundle)
            self.model_run = self.predictor.card["run_id"]
        else:
            warnings.warn("No audio inference bundle found (set AVAL_AUDIO_BUNDLE or place it in _models/audio). "
                          "Audio branch is falling back to the annotation-lookup placeholder.")
            self.model_run = None

    # ---- donor clip for the swap test -------------------------------------------------------------
    def _swap_donor(self, video_path: str) -> Optional[Path]:
        """A different AVE clip of another category, chosen deterministically from the clip id."""
        own = self.ave_index.get_info(get_video_id_from_path(video_path))
        own_cat = own["category"] if own else None
        pool = sorted((cat, VIDEOS_DIR / f"{vid}.mp4") for cat, vid in self.ave_index.get_sample_videos_per_class(1).items()
                      if cat != own_cat and (VIDEOS_DIR / f"{vid}.mp4").exists())
        if not pool:
            return None
        key = sum(map(ord, get_video_id_from_path(video_path)))
        return Path(pool[key % len(pool)][1])

    def extract_and_classify(self, video_path: str, swap_audio: bool = False) -> dict:
        if self.predictor is None:
            return self._placeholder(video_path, swap_audio)

        source = Path(video_path)
        if swap_audio:
            donor = self._swap_donor(video_path)
            if donor is None:
                raise RuntimeError(f"Swap test needs other AVE videos in {VIDEOS_DIR}; none found.")
            source = donor
        pred = self.predictor.predict_video(source)
        return {
            "audio_class": pred.clip_label,
            "audio_confidence": pred.clip_confidence,
            "presence": pred.presence,
            "embeddings": pred.embeddings,                 # (10, 256), same as the offline export
            "per_second_probs": pred.probs,                # (10, 29), columns = class_names
            "per_second_labels": pred.second_labels,
            "top5": pred.top_k(5),
            "class_names": pred.class_names,
            "is_swapped": swap_audio,
            "swap_source": get_video_id_from_path(str(source)) if swap_audio else None,
            "model_run": self.model_run,
            "is_placeholder": False,
            "is_annotation_lookup": False,
        }

    # ---- fallback: the original placeholder, only when no model bundle is available ---------------
    def _placeholder(self, video_path: str, swap_audio: bool) -> dict:
        meta = self.ave_index.get_info(get_video_id_from_path(video_path))
        if swap_audio:
            swapped_class = "Church bell" if (meta and meta["category"] != "Church bell") else "Train horn"
            return {"audio_class": swapped_class, "audio_confidence": 0.89, "presence": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    "embeddings": np.random.randn(10, 128).astype(np.float32), "is_swapped": True,
                    "is_placeholder": True, "is_annotation_lookup": False, "model_run": None}
        if meta is not None:
            feat = load_precomputed_audio_feature(meta["dataset_index"])
            start_t, end_t = meta["start_time"], meta["end_time"]
            return {"audio_class": meta["category"], "audio_confidence": 0.91,
                    "presence": [1 if (start_t <= t <= end_t) else 0 for t in range(10)],
                    "embeddings": feat if feat is not None else np.random.randn(10, 128).astype(np.float32),
                    "is_swapped": False, "is_placeholder": True, "is_annotation_lookup": True, "model_run": None}
        return {"audio_class": "Bark", "audio_confidence": 0.85, "presence": [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
                "embeddings": np.random.randn(10, 128).astype(np.float32), "is_swapped": False,
                "is_placeholder": True, "is_annotation_lookup": False, "model_run": None}

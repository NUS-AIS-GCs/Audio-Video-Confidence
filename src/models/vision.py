"""
VISION BRANCH: live inference with the trained vision model
===========================================================
Runs the trained vision model (ResNet-18, run `resnet18_v1`) on the clip's frames through
`vision.infer.VisionPredictor`. No annotation lookup: predictions come from the model only.

The model is loaded from an inference bundle (see vision/export_inference.py), looked up in this order:
  1. the folder in the AVAL_VISION_BUNDLE environment variable
  2. <repo>/_models/vision
  3. /content/drive/MyDrive/AValidate/inference/vision   (Colab with Drive mounted)
If no bundle is found, the branch falls back to the old annotation-lookup placeholder and says so
(`is_placeholder=True`), so the demo still runs on machines without the model weights.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
import warnings

import numpy as np

from src.config import AVE_CLASSES, BASE_DIR
from src.utils.data_loader import get_ave_index, load_precomputed_visual_feature
from src.utils.video import compute_visual_activity, get_video_id_from_path


def find_vision_bundle() -> Optional[Path]:
    candidates = [
        os.environ.get("AVAL_VISION_BUNDLE"),
        BASE_DIR / "_models" / "vision",
        Path("/content/drive/MyDrive/AValidate/inference/vision"),
    ]
    for c in candidates:
        if c and (Path(c) / "model_card.json").exists():
            return Path(c)
    return None


class VisionBranch:
    """Vision half of the pipeline. Output keys kept from the placeholder (top_class, top_confidence,
    presence, embeddings, is_benchmark_sample, is_placeholder, is_annotation_lookup) plus per-second detail:
    per_second_probs (10, 29), per_second_labels, top5, class_names, model_run."""

    def __init__(self, bundle_dir: Optional[str] = None):
        self.ave_index = get_ave_index()
        self.classes = AVE_CLASSES
        bundle = Path(bundle_dir) if bundle_dir else find_vision_bundle()
        self.predictor = None
        if bundle is not None:
            from vision.infer import VisionPredictor
            self.predictor = VisionPredictor(bundle)
            self.model_run = self.predictor.card["run_id"]
        else:
            warnings.warn("No vision inference bundle found (set AVAL_VISION_BUNDLE or place it in _models/vision). "
                          "Vision branch is falling back to the annotation-lookup placeholder.")
            self.model_run = None

    def extract_and_classify(self, frames: List[np.ndarray], video_path: str) -> dict:
        video_id = get_video_id_from_path(video_path)
        meta = self.ave_index.get_info(video_id)

        if self.predictor is not None:
            vpath = Path(video_path)
            if vpath.exists():
                pred = self.predictor.predict_video(vpath)
            else:
                arr = np.stack(frames, axis=0).astype(np.uint8)
                pred = self.predictor.predict_frames(arr)

            return {
                "embeddings": pred.embeddings,              # (10, 512), same as offline export
                "top_class": pred.clip_label,
                "top_confidence": pred.clip_confidence,
                "top5": pred.top_k(5),
                "presence": pred.presence,
                "per_second_probs": pred.probs,             # (10, 29), columns = class_names
                "per_second_labels": pred.second_labels,
                "class_names": pred.class_names,
                "is_benchmark_sample": (meta is not None),
                "is_placeholder": False,
                "is_annotation_lookup": False,
                "model_run": self.model_run,
            }

        return self._placeholder(frames, video_path)

    # ---- fallback: the original placeholder, only when no model bundle is available ---------------
    def _placeholder(self, frames: List[np.ndarray], video_path: str) -> dict:
        video_id = get_video_id_from_path(video_path)
        meta = self.ave_index.get_info(video_id)

        if meta is not None:
            precomputed = load_precomputed_visual_feature(meta["dataset_index"])
            embeddings = precomputed if precomputed is not None else self._generate_fallback_embeddings(frames)
            ground_truth_class = meta["category"]
            start_t, end_t = meta["start_time"], meta["end_time"]
            visual_presence = [1 if (start_t <= t <= end_t) else 0 for t in range(10)]
            top_class = ground_truth_class
            top_conf = 0.88
            top5 = self._build_top5(top_class, top_conf)
            is_annotation_lookup = True
        else:
            embeddings = self._generate_fallback_embeddings(frames)
            visual_presence = compute_visual_activity(frames)
            top_class = "Bark"
            top_conf = 0.82
            top5 = self._build_top5(top_class, top_conf)
            is_annotation_lookup = False

        return {
            "embeddings": embeddings,
            "top_class": top_class,
            "top_confidence": top_conf,
            "top5": top5,
            "presence": visual_presence,
            "is_benchmark_sample": (meta is not None),
            "is_placeholder": True,
            "is_annotation_lookup": is_annotation_lookup,
            "model_run": None,
        }

    def _generate_fallback_embeddings(self, frames: List[np.ndarray]) -> np.ndarray:
        emb_list = []
        for frame in frames:
            small = frame[::4, ::4, :]
            hist = np.histogram(small, bins=512, range=(0, 256))[0].astype(np.float32)
            norm = np.linalg.norm(hist) + 1e-6
            emb_list.append(hist / norm)
        return np.array(emb_list, dtype=np.float32)

    def _build_top5(self, top_class: str, top_conf: float) -> List[Dict[str, float]]:
        remaining_prob = 1.0 - top_conf
        alt_classes = [c for c in self.classes if c != top_class]
        rng = np.random.default_rng(seed=abs(hash(top_class)) % (2**32))
        chosen_alts = rng.choice(alt_classes, size=4, replace=False)
        alt_weights = rng.dirichlet(np.ones(4)) * remaining_prob
        top5 = [{"class": top_class, "confidence": float(top_conf)}]
        for c, w in zip(chosen_alts, alt_weights):
            top5.append({"class": str(c), "confidence": float(w)})
        return sorted(top5, key=lambda x: x["confidence"], reverse=True)

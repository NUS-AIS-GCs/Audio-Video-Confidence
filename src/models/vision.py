"""
PLACEHOLDER VISION BRANCH (ANNOTATION LOOKUP MOCK)
===================================================
WARNING & NOTICE:
This module currently operates as a PLACEHOLDER / MOCK.
When an AVE dataset clip is provided, it reads the ground-truth event category
and the temporal active interval [start_time, end_time] directly from the
dataset annotations index (`Annotations.txt`), rather than performing live
forward inference through a trained vision checkpoint (e.g., ResNet-18 or MobileNet-V3).

Real vision checkpoints (`artifacts/vision_baseline/resnet18_v1/best.pt`) are
trained and evaluated in `vision/` and `notebooks/`, and will replace this
lookup layer in the upcoming pipeline integration step.
"""
import numpy as np
from typing import List, Dict, Optional
from src.config import AVE_CLASSES, CLASS_TO_IDX
from src.utils.data_loader import get_ave_index, load_precomputed_visual_feature
from src.utils.video import get_video_id_from_path, compute_visual_activity

class VisionBranch:
    """
    Placeholder vision branch for UI demonstration.
    Retrieves ground truth annotations for AVE samples to simulate model predictions.
    """
    def __init__(self):
        self.ave_index = get_ave_index()
        self.classes = AVE_CLASSES

    def extract_and_classify(self, frames: List[np.ndarray], video_path: str) -> dict:
        """
        Processes 10 frames (1 fps).
        NOTE: Returns placeholder predictions read directly from Annotations.txt metadata.
        """
        video_id = get_video_id_from_path(video_path)
        meta = self.ave_index.get_info(video_id)

        # 1. Feature Representation (10 time steps, 512-dim embedding)
        if meta is not None:
            # Clip exists in AVE dataset: load precomputed features or fallback
            precomputed = load_precomputed_visual_feature(meta["dataset_index"])
            if precomputed is not None:
                embeddings = precomputed
            else:
                embeddings = self._generate_fallback_embeddings(frames)
            
            # --- PLACEHOLDER: READ DIRECTLY FROM ANNOTATIONS.TXT ---
            ground_truth_class = meta["category"]
            start_t, end_t = meta["start_time"], meta["end_time"]
            
            # Per-second visual presence from annotation interval [start_t, end_t]
            visual_presence = [1 if (start_t <= t <= end_t) else 0 for t in range(10)]
            
            # Simulated confidence for UI prototype
            top_class = ground_truth_class
            top_conf = 0.88
            
            # Construct top-5 distribution around the ground-truth class
            top5 = self._build_top5(top_class, top_conf)
            is_annotation_lookup = True

        else:
            # Custom user-uploaded video (heuristic fallback)
            embeddings = self._generate_fallback_embeddings(frames)
            visual_presence = compute_visual_activity(frames)
            
            top_class = "Bark"
            top_conf = 0.82
            top5 = self._build_top5(top_class, top_conf)
            is_annotation_lookup = False

        return {
            "embeddings": embeddings,              # (10, 512)
            "top_class": top_class,                # Candidate source class (from Annotations.txt)
            "top_confidence": top_conf,            # Simulated confidence [0.0..1.0]
            "top5": top5,                          # Simulated top 5 classes
            "presence": visual_presence,           # 10 binary flags (from Annotations.txt [start_t, end_t])
            "is_benchmark_sample": (meta is not None),
            "is_placeholder": True,
            "is_annotation_lookup": is_annotation_lookup
        }

    def _generate_fallback_embeddings(self, frames: List[np.ndarray]) -> np.ndarray:
        """Generates normalized (10, 512) feature vectors from frame color histograms and texture."""
        emb_list = []
        for frame in frames:
            small = frame[::4, ::4, :]
            hist = np.histogram(small, bins=512, range=(0, 256))[0].astype(np.float32)
            norm = np.linalg.norm(hist) + 1e-6
            emb_list.append(hist / norm)
        return np.array(emb_list, dtype=np.float32)

    def _build_top5(self, top_class: str, top_conf: float) -> List[Dict[str, float]]:
        """Constructs a calibrated top-5 probability distribution around the target class."""
        remaining_prob = 1.0 - top_conf
        alt_classes = [c for c in self.classes if c != top_class]
        rng = np.random.default_rng(seed=abs(hash(top_class)) % (2**32))
        chosen_alts = rng.choice(alt_classes, size=4, replace=False)
        alt_weights = rng.dirichlet(np.ones(4)) * remaining_prob
        
        top5 = [{"class": top_class, "prob": float(top_conf)}]
        for alt_c, w in zip(chosen_alts, alt_weights):
            top5.append({"class": alt_c, "prob": float(w)})
        
        return sorted(top5, key=lambda x: x["prob"], reverse=True)

"""
PLACEHOLDER MULTIMODAL FUSION BRANCH (RULE-BASED MOCK)
=======================================================
WARNING & NOTICE:
This module currently operates as a PLACEHOLDER / MOCK.
It calculates rule-based heuristic outputs for the three prediction heads
(Event Class, Per-Second Presence, and Correspondence Verification) using
the annotation-derived inputs from the placeholder vision and audio branches.

It does NOT yet execute a trained cross-modal neural network (e.g., cross-attention
transformer or projection MLP). Real fusion models will be trained on the exported
embeddings and plugged into this module in the next milestone.
"""
import numpy as np
from typing import Dict, List
from src.config import TAU_UPPER, TAU_LOWER

class FusionBranch:
    """
    Placeholder multimodal fusion module for UI demonstration.
    Integrates annotation-derived outputs into the three prediction heads.
    """
    def __init__(self):
        pass

    def fuse_and_verify(self, vision_out: dict, audio_out: dict) -> dict:
        is_swapped = audio_out.get("is_swapped", False)

        # 1. Head 2: Per-Second Joint Presence (Audible AND Visible)
        v_pres = vision_out["presence"]
        a_pres = audio_out["presence"]
        
        if is_swapped:
            # Under audio swap mismatch, the sound does not match the visual event
            joint_presence = [0] * 10
        else:
            # Joint intersection: 1 if both audible and visible in second t
            joint_presence = [1 if (v == 1 and a == 1) else 0 for v, a in zip(v_pres, a_pres)]

        # 2. Head 3: Audio-Visual Correspondence Score (Placeholder Heuristic)
        if is_swapped:
            # Collapsed score under mismatch
            correspondence_score = 0.14
            band = "Mismatch"
            similarity_trajectory = [0.10 + 0.05 * np.sin(t) for t in range(10)]
        else:
            # Matched clip: evaluate agreement between annotation lookups
            if vision_out["top_class"] == audio_out["audio_class"]:
                correspondence_score = 0.91
                band = "Strong Match"
            else:
                correspondence_score = 0.52
                band = "Weak / Uncertain"
            
            # Simulated second-by-second cosine similarity trajectory
            similarity_trajectory = [
                0.88 + 0.08 * (1 if joint_presence[t] == 1 else -0.4) for t in range(10)
            ]

        # 3. Head 1: Fused Event Class & Confidence (Placeholder)
        if not is_swapped:
            event_class = vision_out["top_class"]
            event_conf = max(vision_out["top_confidence"], audio_out["audio_confidence"])
            top5 = vision_out["top5"]
        else:
            event_class = vision_out["top_class"]  # Visual anchor
            event_conf = 0.58
            top5 = vision_out["top5"]

        attribution = {
            "audio": audio_out["audio_confidence"],
            "visual": vision_out["top_confidence"],
            "fused": float(event_conf)
        }

        return {
            "event_class": event_class,
            "event_confidence": float(event_conf),
            "top5": top5,
            "joint_presence": joint_presence,
            "correspondence_score": float(correspondence_score),
            "correspondence_band": band,
            "similarity_trajectory": similarity_trajectory,
            "attribution": attribution,
            "is_placeholder": True
        }

"""Inference Pipeline Orchestrator for AValidate."""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from src.utils.video import sample_10_frames
from src.models.vision import VisionBranch
from src.models.audio import AudioBranch
from src.models.fusion import FusionBranch

@dataclass
class PipelineOutput:
    # Head 1: Event Classification
    predicted_event: str
    event_confidence: float
    top5_classes: List[Dict[str, float]]

    # Head 2: Per-Second Evidence (1 = active, 0 = background)
    audio_presence: List[int]
    visual_presence: List[int]
    audible_and_visible: List[int]

    # Head 3: Audio-Visual Correspondence Verification
    correspondence_score: float
    correspondence_band: str

    # Auxiliary Data for Explainable UI
    sampled_frames: List[np.ndarray]
    similarity_trajectory: List[float]
    modality_attribution: Dict[str, float]
    is_benchmark_sample: bool
    is_swapped_test: bool
    is_mock_demo: bool = True

class AValidatePipeline:
    def __init__(self):
        self.vision_branch = VisionBranch()
        self.audio_branch = AudioBranch()
        self.fusion_branch = FusionBranch()

    def predict(self, video_path: str, swap_audio_test: bool = False) -> PipelineOutput:
        # 1. Extract 10 aligned frames (1 fps)
        frames = sample_10_frames(video_path)

        # 2. Vision Branch (Real feature extraction & event candidate prediction)
        vision_out = self.vision_branch.extract_and_classify(frames, video_path)

        # 3. Audio Branch (Audio feature representation & presence)
        audio_out = self.audio_branch.extract_and_classify(video_path, swap_audio=swap_audio_test)

        # 4. Multimodal Fusion Branch (Heads 1, 2, and 3)
        fusion_out = self.fusion_branch.fuse_and_verify(vision_out, audio_out)

        return PipelineOutput(
            predicted_event=fusion_out["event_class"],
            event_confidence=fusion_out["event_confidence"],
            top5_classes=fusion_out["top5"],
            audio_presence=audio_out["presence"],
            visual_presence=vision_out["presence"],
            audible_and_visible=fusion_out["joint_presence"],
            correspondence_score=fusion_out["correspondence_score"],
            correspondence_band=fusion_out["correspondence_band"],
            sampled_frames=frames,
            similarity_trajectory=fusion_out["similarity_trajectory"],
            modality_attribution=fusion_out["attribution"],
            is_benchmark_sample=vision_out["is_benchmark_sample"],
            is_swapped_test=swap_audio_test,
            is_mock_demo=True
        )

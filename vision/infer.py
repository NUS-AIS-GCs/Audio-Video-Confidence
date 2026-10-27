"""Live vision inference for one clip, from a self-contained inference bundle.

The predictor applies exactly the training-time preprocessing, so for an AVE clip its output equals that clip's
row in the offline export (`embeddings/resnet18_v1` or `vision_final/resnet18_v1`, up to float16 storage rounding).
A fusion model trained on the exports can therefore take live video without changes.

Bundle (written by `python -m vision.export_inference`):
  model_card.json   run id, model config, scores, checksums
  weights.pt        trained weights (ResNet-18 backbone + pooling + classification head)
  class_names.json  29 names: 28 AVE events alphabetically, then "background"
  environment.json  library versions the bundle was built with

Usage:
  p = VisionPredictor("/path/to/inference/vision")
  out = p.predict_video("clip.mp4")      # or p.predict_frames(frames_10x224x224x3)
  out.probs (10, 29), out.embeddings (10, 512), out.clip_label, out.second_labels
"""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

from vision import config as C
from vision.model import VisionModel

# Standard ImageNet normalization for vision backbones
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


@dataclass
class VisionPrediction:
    probs: np.ndarray            # (10, 29) per-second class probabilities, columns = class_names
    embeddings: np.ndarray       # (10, 512) per-second embedding (fusion input)
    class_names: list            # 29 names, last = "background"

    @property
    def second_labels(self) -> List[str]:
        """Most likely label for each second (may be 'background')."""
        return [self.class_names[i] for i in self.probs.argmax(1)]

    @property
    def presence(self) -> List[int]:
        """1 where the most likely label of the second is an event (not background)."""
        return [int(i != C.BACKGROUND) for i in self.probs.argmax(1)]

    @property
    def clip_distribution(self) -> np.ndarray:
        """Clip-level event distribution over the 28 events: mean per-second probabilities, background
        excluded, renormalised. Its argmax is the clip label used in evaluation."""
        p = self.probs[:, : C.N_EVENTS].mean(0)
        total = p.sum()
        return p / total if total > 0 else p

    @property
    def clip_label(self) -> str:
        return self.class_names[int(self.clip_distribution.argmax())]

    @property
    def clip_confidence(self) -> float:
        return float(self.clip_distribution.max())

    def top_k(self, k: int = 5) -> List[dict]:
        d = self.clip_distribution
        return [{"class": self.class_names[i], "prob": float(d[i])} for i in np.argsort(-d)[:k]]


def frames_from_video(
    video_path: Union[str, Path],
    seconds: int = C.CLIP_SECONDS,
    size: int = C.IMG_SIZE,
) -> np.ndarray:
    """Extract exactly `seconds` RGB frames at 1 fps (224x224), matching vision/extract_frames.py."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((seconds, size, size, 3), dtype=np.uint8)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    last_valid = np.zeros((size, size, 3), dtype=np.uint8)

    for sec in range(seconds):
        target_frame = min(int(sec * fps), max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != (size, size):
                frame_rgb = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
            last_valid = frame_rgb
        else:
            frames.append(last_valid.copy())

    cap.release()
    return np.stack(frames, axis=0).astype(np.uint8)


class VisionPredictor:
    def __init__(self, bundle_dir: Union[str, Path], device: Optional[str] = None):
        from vision.evaluate import pick_device
        self.dir = Path(bundle_dir)
        self.card = json.loads((self.dir / "model_card.json").read_text(encoding="utf-8"))
        self.class_names = json.loads((self.dir / "class_names.json").read_text(encoding="utf-8"))
        self.device = torch.device(device) if device else pick_device()
        self.cfg = self.card["config"]
        weights = torch.load(self.dir / "weights.pt", map_location="cpu")["model"]

        self.model = VisionModel(
            n_classes=C.N_CLASSES,
            emb_dim=self.cfg.get("emb_dim", C.EMB_DIM),
            backbone=self.cfg.get("backbone", "resnet18"),
            pretrained=False,
        )
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_frames(self, raw_frames: np.ndarray) -> VisionPrediction:
        """raw_frames: (10, 224, 224, 3) uint8 RGB array in [0..255]."""
        if raw_frames.shape != (C.CLIP_SECONDS, C.IMG_SIZE, C.IMG_SIZE, 3):
            raise ValueError(f"Expected shape (10, 224, 224, 3), got {raw_frames.shape}")

        # Convert uint8 (10, 224, 224, 3) -> float32 (1, 10, 3, 224, 224) in [0..1]
        x = (raw_frames.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)  # (10, 3, 224, 224)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD

        t_x = torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, 10, 3, 224, 224)
        logits, emb = self.model(t_x)

        probs = torch.softmax(logits.float(), -1)[0].cpu().numpy()
        embeddings = emb[0].float().cpu().numpy()

        return VisionPrediction(probs=probs, embeddings=embeddings, class_names=self.class_names)

    def predict_video(self, video_path: Union[str, Path]) -> VisionPrediction:
        frames = frames_from_video(video_path)
        return self.predict_frames(frames)


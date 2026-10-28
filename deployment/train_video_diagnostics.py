"""Offline extraction for SHA-bound, explicitly permitted TRAIN media.

This is not a secure upload decoder. OpenCV is called only on approved local
artifacts; the web validation, resource limits and readiness gate are unchanged.
Actual timestamps remain evidence and are never relabeled as exact bin starts.
"""
import hashlib
import math

import cv2
import numpy as np

from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact


def sample_train_frames(source):
    if type(source) is not Artifact:
        raise ContractError("explicit source Artifact required")
    source.verify()
    cap = cv2.VideoCapture(source.path)
    try:
        if not cap.isOpened():
            raise ContractError("TRAIN video cannot be opened; no black-frame substitution")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count < 1:
            raise ContractError("invalid video metadata; FPS fallback forbidden")
        evidence = {"fps": fps, "frame_count": count, "duration_from_count_fps": count / fps,
                    "backend": cap.getBackendName(), "frames": [], "repairs_applied": False}
        frames, indices = [], []
        for sec in range(10):
            target = int(sec * fps)  # Original producer truncation, never round(fps).
            if target >= count or target in indices:
                raise ContractError("clip requires clamping or duplicate frame indices")
            if not cap.set(cv2.CAP_PROP_POS_FRAMES, target):
                raise ContractError("frame seek failed")
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise ContractError("invalid frame; repeat/black substitution forbidden")
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            achieved = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
            if not math.isfinite(timestamp) or achieved != target:
                raise ContractError("frame position unverifiable")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (224, 224):
                rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)
            indices.append(target)
            evidence["frames"].append({"intended_second": sec, "source_frame_index": target,
                "timestamp_seconds": timestamp, "timestamp_error_seconds": timestamp-sec,
                "valid": True, "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
        return np.stack(frames), evidence
    finally:
        cap.release()
        source.verify()

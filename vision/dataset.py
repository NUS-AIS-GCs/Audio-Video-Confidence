"""PyTorch dataset for AVE video frames, with temporal alignment and augmentations."""
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vision import config as C

# Standard ImageNet normalization for vision backbones
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class AVEVisual(Dataset):
    """One item = one clip: x (10, 3, 224, 224), y (10,), yclip ()."""

    def __init__(
        self,
        split: str,
        augment: bool = False,
        indices=None,
        use_cached_array: bool = True
    ):
        self.split = split
        self.augment = augment
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

        ids_path = C.FEATURE_DIR / f"{split}_ids.csv"
        y_path = C.FEATURE_DIR / f"{split}_y.npy"
        yclip_path = C.FEATURE_DIR / f"{split}_yclip.npy"
        x_path = C.FEATURE_DIR / f"{split}_X.npy"

        if not y_path.exists() or not ids_path.exists():
            raise RuntimeError(
                f"Features not found for split '{split}' in {C.FEATURE_DIR}. "
                f"Run `python -m vision.features` first."
            )

        self.df = pd.read_csv(ids_path)
        self.y = np.load(y_path)
        self.yclip = np.load(yclip_path)

        # Check if single monolithic X array is available
        self.use_monolithic_X = use_cached_array and x_path.exists()
        if self.use_monolithic_X:
            self.X = np.load(x_path, mmap_mode="r")
        else:
            self.X = None

        self.idx = np.arange(len(self.y)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.idx)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x: (10, 3, 224, 224)
        # Consistent horizontal flip across the entire 10s video sequence
        if torch.rand(1) < 0.5:
            x = torch.flip(x, dims=[-1])

        # Random brightness / contrast scaling
        if torch.rand(1) < 0.5:
            factor = float(0.8 + 0.4 * torch.rand(1))
            x = torch.clamp(x * factor, 0.0, 1.0)

        # Random temporal frame cutout (spec-augment equivalent for video frames)
        if torch.rand(1) < 0.3:
            drop_sec = int(torch.randint(0, 10, (1,)))
            x[drop_sec] = 0.0

        return x

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = self.idx[i]
        row = self.df.iloc[j]

        if self.use_monolithic_X:
            raw_frames = np.asarray(self.X[j], dtype=np.uint8)
        else:
            frame_path = C.FRAME_DIR / f"{row.clip_id}.npy"
            if frame_path.exists():
                raw_frames = np.load(frame_path)
            else:
                raw_frames = np.zeros((C.CLIP_SECONDS, C.IMG_SIZE, C.IMG_SIZE, C.N_CHANNELS), dtype=np.uint8)

        # Convert (10, 224, 224, 3) uint8 [0..255] -> (10, 3, 224, 224) float32 [0..1]
        x = torch.from_numpy(raw_frames).permute(0, 3, 1, 2).float() / 255.0

        if self.augment:
            x = self._augment(x)

        # Normalize with ImageNet stats
        x = (x - self.mean) / self.std

        return x, torch.from_numpy(self.y[j]), torch.tensor(self.yclip[j])

"""PyTorch dataset over the cached log-Mel arrays."""
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from audio import config as C


def load_arrays(split: str):
    X = np.load(C.FEATURE_DIR / f"{split}_X.npy", mmap_mode="r")
    y = np.load(C.FEATURE_DIR / f"{split}_y.npy")
    yclip = np.load(C.FEATURE_DIR / f"{split}_yclip.npy")
    return X, y, yclip


def norm_stats():
    s = json.loads((C.FEATURE_DIR / "norm_stats.json").read_text())
    mean = torch.tensor(s["mean"]).view(1, -1, 1)   # broadcast over (seconds, mels, frames)
    std = torch.tensor(s["std"]).view(1, -1, 1)
    return mean, std, s["class_names"]


class AVEAudio(Dataset):
    """One item = one clip: x (10, 1, 64, 100), y (10,), yclip ()."""

    def __init__(self, split: str, augment: bool = False, indices=None):
        self.X, self.y, self.yclip = load_arrays(split)
        self.idx = np.arange(len(self.y)) if indices is None else np.asarray(indices)
        self.mean, self.std, _ = norm_stats()
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        # Hide a random band of frequencies and a random stretch of time in each second.
        for s in range(x.shape[0]):
            if torch.rand(1) < 0.5:
                f = int(torch.randint(0, 9, (1,)))
                f0 = int(torch.randint(0, C.N_MELS - f + 1, (1,)))
                x[s, f0:f0 + f, :] = 0
            if torch.rand(1) < 0.5:
                t = int(torch.randint(0, 11, (1,)))
                t0 = int(torch.randint(0, C.FRAMES_PER_SECOND - t + 1, (1,)))
                x[s, :, t0:t0 + t] = 0
        return x

    def __getitem__(self, i):
        j = self.idx[i]
        x = (torch.from_numpy(np.asarray(self.X[j], dtype=np.float32)) - self.mean) / self.std
        if self.augment:
            x = self._spec_augment(x)
        return x.unsqueeze(1), torch.from_numpy(self.y[j]), torch.tensor(self.yclip[j])

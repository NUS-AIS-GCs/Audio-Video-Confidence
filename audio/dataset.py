"""PyTorch dataset over cached per-second audio features.

Two kinds of cache share one layout ({split}_X.npy, _y, _yclip, _ids.csv, norm_stats.json):
  log-Mel   X (N, 10, 64, 100)  -> items (10, 1, 64, 100), for the CNN baseline
  embedding X (N, 10, D)        -> items (10, D), e.g. frozen BEATs (D = 768)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from audio import config as C


def feature_dir(name: str = None) -> Path:
    """Cache folder by name ('audio_logmel_v2', 'beats_iter3', ...); default = the log-Mel baseline."""
    return C.CACHE_ROOT / name if name else C.FEATURE_DIR


def load_arrays(split: str, fdir: Path = None):
    fdir = fdir or C.FEATURE_DIR
    X = np.load(fdir / f"{split}_X.npy", mmap_mode="r")
    y = np.load(fdir / f"{split}_y.npy")
    yclip = np.load(fdir / f"{split}_yclip.npy")
    return X, y, yclip


def norm_stats(fdir: Path = None):
    """(mean, std, class_names) shaped to broadcast over one item of this cache."""
    fdir = fdir or C.FEATURE_DIR
    s = json.loads((fdir / "norm_stats.json").read_text())
    mean, std = torch.tensor(s["mean"]), torch.tensor(s["std"])
    if np.load(fdir / "train_X.npy", mmap_mode="r").ndim == 4:      # log-Mel: one value per mel bin
        mean, std = mean.view(1, -1, 1), std.view(1, -1, 1)
    return mean, std, s["class_names"]


class AVEAudio(Dataset):
    """One item = one sample: x (see module doc), y (10,), yclip ()."""

    def __init__(self, split: str, augment: bool = False, indices=None, fdir: Path = None):
        self.fdir = fdir or C.FEATURE_DIR
        self.X, self.y, self.yclip = load_arrays(split, self.fdir)
        self.is_logmel = self.X.ndim == 4
        self.idx = np.arange(len(self.y)) if indices is None else np.asarray(indices)
        self.mean, self.std, _ = norm_stats(self.fdir)
        self.augment = augment and self.is_logmel      # SpecAugment only applies to spectrograms

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
        if self.is_logmel:
            x = x.unsqueeze(1)
        return x, torch.from_numpy(self.y[j]), torch.tensor(self.yclip[j])

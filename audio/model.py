"""Log-Mel CNN baseline: each 1-second spectrogram -> 256-D embedding -> 29 per-second logits."""
import torch
from torch import nn

from audio import config as C


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class AudioCNN(nn.Module):
    def __init__(self, n_classes: int = C.N_CLASSES, emb_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128), _block(128, emb_dim))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, n_classes))
        self.emb_dim = emb_dim

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 10, 1, 64, 100) -> (B, 10, emb_dim): the per-second embedding shared with the team."""
        b, s = x.shape[:2]
        z = self.pool(self.features(x.flatten(0, 1))).flatten(1)
        return z.view(b, s, -1)

    def forward(self, x: torch.Tensor):
        emb = self.embed(x)
        return self.head(emb), emb          # logits (B, 10, 29), emb (B, 10, 256)

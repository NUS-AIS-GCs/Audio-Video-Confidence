"""Vision baseline model: each 1-second video frame -> 512-D embedding -> 29 per-second logits.

Matches audio/model.py:
  embed(x)   -> (B, 10, emb_dim) per-second embedding contract for multimodal fusion
  forward(x) -> (logits (B, 10, 29), emb (B, 10, emb_dim))
"""
import torch
from torch import nn
import torchvision.models as tv_models

from vision import config as C


def _simple_conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )


class VisionCNN(nn.Module):
    """Lightweight pure PyTorch CNN backbone for fast training and baseline comparisons."""
    def __init__(self, emb_dim: int = C.EMB_DIM):
        super().__init__()
        self.features = nn.Sequential(
            _simple_conv_block(3, 32),    # 224 -> 112
            _simple_conv_block(32, 64),   # 112 -> 56
            _simple_conv_block(64, 128),  # 56 -> 28
            _simple_conv_block(128, 256), # 28 -> 14
            _simple_conv_block(256, emb_dim), # 14 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 3, 224, 224) -> (N, emb_dim)
        return self.pool(self.features(x)).flatten(1)


class VisionModel(nn.Module):
    """AValidate Vision Branch: processes 10 frames per clip and outputs aligned representations."""
    def __init__(
        self,
        n_classes: int = C.N_CLASSES,
        emb_dim: int = C.EMB_DIM,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.backbone_name = backbone

        if backbone == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
            m = tv_models.resnet18(weights=weights)
            in_dim = m.fc.in_features  # 512
            m.fc = nn.Identity()
            self.encoder = m
            if emb_dim != in_dim:
                self.proj = nn.Linear(in_dim, emb_dim)
            else:
                self.proj = nn.Identity()
        elif backbone == "mobilenet_v3_small":
            weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = tv_models.mobilenet_v3_small(weights=weights)
            in_dim = m.classifier[0].in_features
            m.classifier = nn.Identity()
            self.encoder = m
            self.proj = nn.Linear(in_dim, emb_dim)
        elif backbone == "cnn":
            self.encoder = VisionCNN(emb_dim=emb_dim)
            self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose 'resnet18', 'mobilenet_v3_small', or 'cnn'.")

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_classes)
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, 10, 3, 224, 224) -> (B, 10, emb_dim): per-second visual embedding."""
        b, s = x.shape[:2]
        # Flatten batch and seconds dimensions to pass through 2D backbone
        flat_x = x.flatten(0, 1)
        feat = self.encoder(flat_x)
        emb = self.proj(feat)
        return emb.view(b, s, -1)

    def forward(self, x: torch.Tensor):
        emb = self.embed(x)
        return self.head(emb), emb   # logits (B, 10, 29), emb (B, 10, 512)

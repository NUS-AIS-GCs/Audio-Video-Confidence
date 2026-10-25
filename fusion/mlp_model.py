"""Shared per-bin learned fusion and one pair-level correspondence logit."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .contracts import ContractError


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "relu"
    dropout: float = 0.2
    initialization_seed: int = 42
    audio_dim: int = 256
    visual_dim: int = 512
    temporal_bins: int = 10
    aggregation: str = "masked_mean"

    def __post_init__(self):
        if (type(self.hidden_dims) is not tuple or not self.hidden_dims
                or any(type(n) is not int or n < 1 for n in self.hidden_dims)):
            raise ContractError("hidden_dims must be a nonempty tuple of positive integers")
        if self.activation not in ("relu", "gelu"):
            raise ContractError("activation must be relu or gelu")
        if (type(self.dropout) not in (float, int) or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ContractError("dropout must be finite in [0,1)")
        if type(self.initialization_seed) is not int or not 0 <= self.initialization_seed < 2**32:
            raise ContractError("initialization_seed must be an unsigned 32-bit integer")
        if any(type(getattr(self, k)) is not int for k in ("audio_dim", "visual_dim", "temporal_bins")):
            raise ContractError("feature dimensions must be integers")
        if (self.audio_dim, self.visual_dim, self.temporal_bins) != (256, 512, 10):
            raise ContractError("validated AVE input dimensions are 256, 512 and ten bins")
        if self.aggregation != "masked_mean":
            raise ContractError("only masked mean aggregation is supported")


@dataclass(frozen=True)
class FusionOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor | None = None
    # Representations, NOT independently supervised temporal verification scores.
    bin_representations: torch.Tensor | None = None


class LateFusionMLP(nn.Module):
    def __init__(self, config: MLPConfig = MLPConfig()):
        super().__init__()
        if type(config) is not MLPConfig:
            raise ContractError("expected MLPConfig")
        config.__post_init__()
        self.config = config
        # Build on CPU inside a fork: deterministic initialization without
        # advancing the caller's CPU RNG or initializing any CUDA context.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.initialization_seed)
            layers = []
            width = config.audio_dim + config.visual_dim
            for hidden in config.hidden_dims:
                layers.extend((nn.Linear(width, hidden, device="cpu", dtype=torch.float32),
                               nn.ReLU() if config.activation == "relu" else nn.GELU(),
                               nn.Dropout(config.dropout)))
                width = hidden
            self.bin_mlp = nn.Sequential(*layers)
            self.correspondence_head = nn.Linear(width, 1, device="cpu", dtype=torch.float32)

    def forward(self, audio, visual, validity=None, *, return_probabilities=False,
                return_bin_representations=False):
        if type(return_probabilities) is not bool or type(return_bin_representations) is not bool:
            raise ContractError("output switches must be booleans")
        parameter = next(self.parameters())
        for value, dim in ((audio, 256), (visual, 512)):
            if (not isinstance(value, torch.Tensor) or value.ndim != 3
                    or tuple(value.shape[1:]) != (10, dim) or value.shape[0] < 1
                    or not value.is_floating_point()):
                raise ContractError(f"expected nonempty floating (B,10,{dim}) tensor")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ContractError("features must match model device and floating dtype")
            if not torch.isfinite(value).all():
                raise ContractError("features must be finite, including masked bins")
        if audio.shape[0] != visual.shape[0]:
            raise ContractError("modality batch sizes differ")
        if validity is None:
            validity = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        if (not isinstance(validity, torch.Tensor) or validity.dtype != torch.bool
                or validity.shape != audio.shape[:2] or validity.device != audio.device):
            raise ContractError("validity must be boolean (B,10) on the feature device")
        if not validity.any(dim=1).all():
            raise ContractError("all-masked pairs cannot be aggregated")
        joined = torch.cat((audio, visual), dim=-1)
        # Zero BEFORE the MLP: large finite invalid inputs cannot overflow hidden
        # activations or gradients, and cannot affect any valid representation.
        joined = joined.masked_fill(~validity.unsqueeze(-1), 0)
        bins = self.bin_mlp(joined).masked_fill(~validity.unsqueeze(-1), 0)
        pooled = bins.sum(dim=1) / validity.sum(dim=1, keepdim=True)
        logits = self.correspondence_head(pooled)
        if not torch.isfinite(bins).all() or not torch.isfinite(logits).all():
            raise ContractError("nonfinite model output")
        return FusionOutput(logits, logits.sigmoid() if return_probabilities else None,
                            bins if return_bin_representations else None)

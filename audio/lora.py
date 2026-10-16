"""LoRA (Low-Rank Adaptation, Hu et al. 2021) in a few lines, so it's clear what it does.

A frozen linear layer computes y = W x. LoRA keeps W frozen and adds a small trainable correction:
    y = W x + (alpha / r) * B (A x)
A is (r x in), B is (out x r), with r tiny (e.g. 8) compared with in/out (768). B starts at zero, so at the
start the layer behaves exactly like the original; training only moves the small A and B.
"""
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))       # zero: no change at the start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale

    # Some code (e.g. BEATs' initialisation) reads .weight / .bias of the original layer.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def inject_lora(model: nn.Module, targets=("q_proj", "v_proj"), r: int = 8, alpha: float = 16.0,
                dropout: float = 0.1) -> int:
    """Freeze every parameter of `model`, then wrap each nn.Linear whose attribute name is in `targets`.
    Returns the number of layers wrapped."""
    for p in model.parameters():
        p.requires_grad = False
    wrapped = 0
    for module in list(model.modules()):
        for name in targets:
            child = getattr(module, name, None)
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    return wrapped

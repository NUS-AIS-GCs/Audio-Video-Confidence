"""Bidirectional correspondence attention over the ten original one-second bins.

The shared validity mask is the adapter's intersection of producer masks. It
excludes both streams at that index; it is never an AVE event annotation.
Attention is unrestricted across valid bins, and is not evidence of causality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .contracts import ContractError


@dataclass(frozen=True)
class CrossAttentionConfig:
    d_model: int = 256
    num_heads: int = 4
    attention_dropout: float = 0.1
    dropout: float = 0.1
    feedforward_dim: int = 512  # zero disables each branch's feed-forward block
    activation: str = "gelu"
    position_encoding: str = "sinusoidal"  # "none" is an explicit future ablation
    initialization_seed: int = 42
    audio_dim: int = 256
    visual_dim: int = 512
    temporal_bins: int = 10
    aggregation: str = "masked_mean"
    architecture: str = "bidirectional_cross_attention"

    def __post_init__(self):
        if type(self.d_model) is not int or self.d_model < 2:
            raise ContractError("d_model must be an integer >= 2 for the LayerNorm architecture")
        for name in ("d_model", "num_heads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads:
            raise ContractError("d_model must be divisible by num_heads")
        if type(self.feedforward_dim) is not int or self.feedforward_dim < 0:
            raise ContractError("feedforward_dim must be a nonnegative integer")
        for name in ("attention_dropout", "dropout"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 1:
                raise ContractError(f"{name} must be finite in [0,1)")
        if self.activation not in ("relu", "gelu"):
            raise ContractError("activation must be relu or gelu")
        if self.position_encoding not in ("sinusoidal", "none"):
            raise ContractError("position_encoding must be sinusoidal or none")
        if type(self.initialization_seed) is not int or not 0 <= self.initialization_seed < 2**32:
            raise ContractError("initialization_seed must be an unsigned 32-bit integer")
        if (any(type(getattr(self, k)) is not int for k in ("audio_dim", "visual_dim", "temporal_bins"))
                or (self.audio_dim, self.visual_dim, self.temporal_bins) != (256, 512, 10)):
            raise ContractError("validated AVE input dimensions are 256, 512 and ten bins")
        if self.aggregation != "masked_mean":
            raise ContractError("unsupported aggregation")
        self._validate_topology()

    @property
    def attention_direction(self):
        # A property, not a new serialized field: historical configuration and
        # checkpoint fingerprints remain byte-for-byte compatible.
        return "bidirectional"

    def _validate_topology(self):
        if self.architecture != "bidirectional_cross_attention":
            raise ContractError("unsupported architecture")


@dataclass(frozen=True)
class UnidirectionalAttentionConfig(CrossAttentionConfig):
    architecture: str = "unidirectional_cross_attention"
    attention_direction: str = "audio_to_video"

    def _validate_topology(self):
        if (self.architecture != "unidirectional_cross_attention"
                or self.attention_direction not in ("audio_to_video", "video_to_audio")):
            raise ContractError("unidirectional architecture requires an explicit supported attention direction")


@dataclass(frozen=True)
class CrossAttentionOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor | None = None
    bin_representations: torch.Tensor | None = None
    # (B, heads, query_bin, key_bin), after attention dropout. In eval, each
    # valid query sums to one. Invalid queries AND keys are exactly zero.
    audio_to_video_attention: torch.Tensor | None = None
    video_to_audio_attention: torch.Tensor | None = None


def _finite(value, stage):
    if not torch.isfinite(value).all():
        raise ContractError(f"nonfinite {stage}")
    return value


class CrossAttentionBranch(nn.Module):
    """One directional update; both branches consume the original projections."""

    def __init__(self, config):
        super().__init__()
        kw = {"device": "cpu", "dtype": torch.float32}
        self.attention = nn.MultiheadAttention(config.d_model, config.num_heads,
                                              dropout=config.attention_dropout,
                                              batch_first=True, **kw)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.attention_norm = nn.LayerNorm(config.d_model, **kw)
        self.feedforward = None
        if config.feedforward_dim:
            self.feedforward = nn.Sequential(
                nn.Linear(config.d_model, config.feedforward_dim, **kw),
                nn.GELU() if config.activation == "gelu" else nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.feedforward_dim, config.d_model, **kw),
                nn.Dropout(config.dropout))
            self.feedforward_norm = nn.LayerNorm(config.d_model, **kw)

    def forward(self, query, context, validity):
        update, weights = self.attention(query, context, context,
                                         key_padding_mask=~validity,
                                         need_weights=True, average_attn_weights=False)
        _finite(update, "attention update")
        _finite(weights, "attention weights")
        stream = self.attention_norm(_finite(query + self.residual_dropout(update), "attention residual"))
        _finite(stream, "attention normalization")
        if self.feedforward is not None:
            hidden = stream
            for layer in self.feedforward:
                hidden = _finite(layer(hidden), "feed-forward activation")
            stream = self.feedforward_norm(_finite(stream + hidden, "feed-forward residual"))
            _finite(stream, "feed-forward normalization")
        stream = stream.masked_fill(~validity.unsqueeze(-1), 0)
        weights = weights.masked_fill(~validity[:, None, :, None], 0)
        weights = weights.masked_fill(~validity[:, None, None, :], 0)
        return stream, weights


class _CrossAttentionFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.__post_init__()
        self.config = config
        # Match Phase 4: initialize on CPU without advancing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.initialization_seed)
            kw = {"device": "cpu", "dtype": torch.float32}
            self.audio_projection = nn.Linear(256, config.d_model, **kw)
            self.visual_projection = nn.Linear(512, config.d_model, **kw)
            if config.attention_direction in ("bidirectional", "audio_to_video"):
                self.audio_to_video = CrossAttentionBranch(config)
            if config.attention_direction in ("bidirectional", "video_to_audio"):
                self.video_to_audio = CrossAttentionBranch(config)
            streams = 2 if config.attention_direction == "bidirectional" else 1
            self.correspondence_head = nn.Linear(streams * config.d_model, 1, **kw)
        # PE(t,2i)=sin(t/10000**(2i/d)); PE(t,2i+1)=cos(...).
        # Original indices are never compressed when some bins are masked.
        positions = torch.arange(10, device="cpu", dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, config.d_model, 2, device="cpu", dtype=torch.float32)
                              * (-math.log(10000.0) / config.d_model))
        encoding = torch.zeros(10, config.d_model, device="cpu", dtype=torch.float32)
        if config.position_encoding == "sinusoidal":
            encoding[:, 0::2] = torch.sin(positions * frequency)
            encoding[:, 1::2] = torch.cos(positions * frequency[:config.d_model // 2])
        self.register_buffer("temporal_positions", encoding.unsqueeze(0))

    def forward(self, audio, visual, validity, *, return_probabilities=False,
                return_bin_representations=False, return_attention_weights=False):
        if any(type(v) is not bool for v in (return_probabilities, return_bin_representations,
                                             return_attention_weights)):
            raise ContractError("output switches must be booleans")
        parameter = next(self.parameters())
        for value, dim in ((audio, 256), (visual, 512)):
            if (not isinstance(value, torch.Tensor) or value.ndim != 3
                    or tuple(value.shape[1:]) != (10, dim) or value.shape[0] < 1
                    or not value.is_floating_point()):
                raise ContractError(f"expected nonempty floating (B,10,{dim}) tensor")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ContractError("features must match model device and floating dtype")
            _finite(value, "features, including masked bins")
        if audio.shape[0] != visual.shape[0]:
            raise ContractError("modality batch sizes differ")
        if (not isinstance(validity, torch.Tensor) or validity.dtype != torch.bool
                or validity.shape != audio.shape[:2] or validity.device != audio.device):
            raise ContractError("validity must be explicit boolean (B,10) on the feature device")
        if not validity.any(dim=1).all():
            raise ContractError("all-masked pairs cannot attend or aggregate")
        invalid = ~validity.unsqueeze(-1)
        # Mask before projection so even enormous finite invalid inputs cannot
        # overflow, affect valid bins, or receive nonzero input gradients.
        a = _finite(self.audio_projection(audio.masked_fill(invalid, 0)), "audio projection")
        v = _finite(self.visual_projection(visual.masked_fill(invalid, 0)), "visual projection")
        a = _finite(a + self.temporal_positions, "positioned audio").masked_fill(invalid, 0)
        v = _finite(v + self.temporal_positions, "positioned video").masked_fill(invalid, 0)
        streams, av_weights, va_weights = [], None, None
        if self.config.attention_direction in ("bidirectional", "audio_to_video"):
            attended_a, av_weights = self.audio_to_video(a, v, validity)
            streams.append(attended_a)
        if self.config.attention_direction in ("bidirectional", "video_to_audio"):
            attended_v, va_weights = self.video_to_audio(v, a, validity)
            streams.append(attended_v)
        # A one-way stream already contains the query residual AND attended
        # context values. No fake missing-branch tensors or passive stream.
        bins = _finite(torch.cat(streams, dim=-1), "fused bins")
        pooled = _finite(bins.sum(dim=1) / validity.sum(dim=1, keepdim=True), "masked mean")
        logits = _finite(self.correspondence_head(pooled), "correspondence logits")
        return CrossAttentionOutput(
            logits, logits.sigmoid() if return_probabilities else None,
            bins if return_bin_representations else None,
            av_weights if return_attention_weights else None,
            va_weights if return_attention_weights else None)


class BidirectionalCrossAttention(_CrossAttentionFusion):
    def __init__(self, config: CrossAttentionConfig = CrossAttentionConfig()):
        if type(config) is not CrossAttentionConfig:
            raise ContractError("expected CrossAttentionConfig")
        super().__init__(config)


class UnidirectionalCrossAttention(_CrossAttentionFusion):
    def __init__(self, config: UnidirectionalAttentionConfig):
        if type(config) is not UnidirectionalAttentionConfig:
            raise ContractError("expected UnidirectionalAttentionConfig")
        super().__init__(config)

"""Architecture selection, separate from optimizer and evaluation policies.

Both models accept only audio/visual/validity features and return .logits (B,1).
The existing loader, BCE, guarded epoch and prediction functions are shared.
The historical Phase 4 experiment CLI remains restricted to MLPConfig.
"""
from collections import OrderedDict
from dataclasses import asdict
import math

import torch
from torch import nn

from .contracts import ContractError
from .cross_attention_model import (
    BidirectionalCrossAttention, CrossAttentionConfig,
    UnidirectionalAttentionConfig, UnidirectionalCrossAttention,
)
from .mlp_model import LateFusionMLP, MLPConfig


def build_model(config):
    if type(config) is MLPConfig:
        return LateFusionMLP(config)
    if type(config) is CrossAttentionConfig:
        return BidirectionalCrossAttention(config)
    if type(config) is UnidirectionalAttentionConfig:
        return UnidirectionalCrossAttention(config)
    raise ContractError("unsupported model configuration")


def model_config_from_dict(value, schema):
    if type(value) is not dict:
        raise ContractError("model configuration must be an object")
    fields = dict(value)
    if schema == "gc2-fusion-checkpoint-v1":
        fields["hidden_dims"] = tuple(fields["hidden_dims"])
        return MLPConfig(**fields)
    if schema == "gc2-fusion-cross-attention-checkpoint-v1":
        if fields.get("architecture") == "unidirectional_cross_attention":
            if "attention_direction" not in fields:
                raise ContractError("unidirectional checkpoint requires explicit attention_direction")
            return UnidirectionalAttentionConfig(**fields)
        return CrossAttentionConfig(**fields)
    raise ContractError("unknown checkpoint schema")


def model_binding(config):
    """Persist claims that weights alone cannot distinguish (heads/dropout/etc.)."""
    if type(config) not in (MLPConfig, CrossAttentionConfig, UnidirectionalAttentionConfig):
        raise ContractError("unsupported model configuration")
    config.__post_init__()
    return {"schema": "gc2-fusion-model-binding-v1",
            "architecture": ("late_fusion_mlp" if type(config) is MLPConfig else config.architecture),
            "config": asdict(config)}


def validate_model_binding(binding, config):
    if type(binding) is not dict or set(binding) != {"schema", "architecture", "config"}:
        raise ContractError("missing or malformed checkpoint model binding")
    schema = ("gc2-fusion-checkpoint-v1" if type(config) is MLPConfig
              else "gc2-fusion-cross-attention-checkpoint-v1")
    try:
        persisted = model_config_from_dict(binding["config"], schema)
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError("malformed checkpoint model binding configuration") from exc
    if persisted != config or binding != model_binding(config):
        raise ContractError("checkpoint model binding disagrees with metadata")


def validate_model_state(reference, state):
    """Check exact state coverage/shapes and regenerate fixed buffers independently.

    The reference MUST be freshly constructed from the persisted configuration,
    not the live model whose buffers might have been altered. Return a reference
    with the state's floating dtype so strict loading never silently casts weights.
    """
    expected = reference.state_dict()
    if type(state) not in (dict, OrderedDict) or set(state) != set(expected):
        raise ContractError("checkpoint state dictionary coverage differs from architecture")
    dtype = None
    for name, value in state.items():
        if (type(value) is not torch.Tensor or value.layout != torch.strided
                or value.device.type == "meta" or not value.is_floating_point()
                or value.shape != expected[name].shape or not torch.isfinite(value).all()):
            raise ContractError(f"malformed or nonfinite checkpoint state tensor: {name}")
        if dtype is None:
            dtype = value.dtype
        elif value.dtype != dtype:
            raise ContractError("checkpoint floating state tensors must share one dtype")
    reference = reference.to(dtype=dtype)
    for name, canonical in reference.named_buffers():
        if not torch.equal(state[name].detach().cpu(), canonical.detach().cpu()):
            raise ContractError(f"checkpoint buffer disagrees with declared configuration: {name}")
    return reference


def validate_live_model(model, config):
    """Reconcile the live module graph, hyperparameters, tensors and fixed buffers.

    Comparing model.config alone would miss replaced modules or a forged config
    attribute. Only the two concrete supported model types are accepted.
    """
    reference = build_model(config)
    if (type(model) is not type(reference) or type(getattr(model, "config", None)) is not type(config)
            or model.config != config):
        raise ContractError("live model architecture/configuration disagrees with metadata")
    model.config.__post_init__()
    actual_modules, expected_modules = dict(model.named_modules()), dict(reference.named_modules())
    if tuple(actual_modules) != tuple(expected_modules):
        raise ContractError("live model module coverage/order disagrees with configuration")
    if (set(dict(model.named_parameters())) != set(dict(reference.named_parameters()))
            or set(dict(model.named_buffers())) != set(dict(reference.named_buffers()))):
        raise ContractError("live model parameter/buffer coverage disagrees with configuration")
    attributes = {
        nn.Linear: ("in_features", "out_features"),
        nn.MultiheadAttention: ("embed_dim", "num_heads", "head_dim", "dropout", "batch_first",
                                "kdim", "vdim", "_qkv_same_embed_dim", "add_zero_attn"),
        nn.LayerNorm: ("normalized_shape", "eps", "elementwise_affine"),
        nn.Dropout: ("p", "inplace"), nn.ReLU: ("inplace",), nn.GELU: ("approximate",),
    }
    for name, canonical in expected_modules.items():
        actual = actual_modules[name]
        if type(actual) is not type(canonical):
            raise ContractError(f"live model module type disagrees with configuration: {name}")
        for attribute in attributes.get(type(canonical), ()):
            value, expected = getattr(actual, attribute), getattr(canonical, attribute)
            dropout = ((type(canonical) is nn.Dropout and attribute == "p")
                       or (type(canonical) is nn.MultiheadAttention and attribute == "dropout"))
            invalid = (type(value) not in (int, float) or not math.isfinite(value)
                       or not 0 <= value < 1 or value != expected) if dropout else (
                           type(value) is not type(expected) or value != expected)
            if invalid:
                raise ContractError(f"live model setting disagrees with configuration: {name}.{attribute}")
    validate_model_state(reference, model.state_dict())

"""Deterministic semantic agreement proxies. Higher always means more compatible.

All 29 predicted classes, including background, participate. Equal bin indices
are compared; neither stream synchrony nor recording causality is established.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .contracts import ContractError, PROBABILITY_ATOL

METHODS = ("top1", "dot_product", "cosine", "jensen_shannon")
AGGREGATION = "mean_all_10_bins"


def _probabilities(audio, visual):
    result = []
    for value in (audio, visual):
        value = np.asarray(value)
        if (value.ndim != 3 or value.shape[1:] != (10, 29)
                or value.dtype not in (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))):
            raise ContractError("expected floating probabilities of shape (N,10,29)")
        value = value.astype(np.float64, copy=True)
        if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ContractError("probabilities must be finite and in [0,1]")
        mass = value.sum(axis=-1, keepdims=True)
        if np.any((mass != 0) & (np.abs(mass - 1) > PROBABILITY_ATOL)):
            raise ContractError("nonzero probability vectors must sum to one within Phase 1 tolerance")
        # Exact zero vectors are an explicit low-level abstention, never a
        # distribution invented from labels. Real artifact contracts reject them.
        np.divide(value, mass, out=value, where=mass != 0)
        result.append(value)
    a, v = result
    if a.shape != v.shape:
        raise ContractError("modality probability shapes differ")
    usable = (a.sum(axis=-1) > 0) & (v.sum(axis=-1) > 0)
    return a, v, usable


def _score(a, v, usable, method):
    if method == "top1":
        # NumPy argmax resolves ties to the lowest column index.
        result = (a.argmax(axis=-1) == v.argmax(axis=-1)).astype(np.float64)
    elif method == "dot_product":
        result = np.sum(a * v, axis=-1)
    elif method == "cosine":
        dot = np.sum(a * v, axis=-1)
        norm = np.sqrt(np.sum(a * a, axis=-1) * np.sum(v * v, axis=-1))
        result = np.divide(dot, norm, out=np.zeros_like(dot), where=norm > 0)
    else:
        mid = (a + v) / 2

        def entropy(p):
            logs = np.log2(p, out=np.zeros_like(p), where=p > 0)
            return -np.sum(p * logs, axis=-1)

        # Equivalent entropy form avoids division by a midpoint rounded to zero
        # for subnormal probabilities. Zero log terms contribute exactly zero.
        result = 1 - (entropy(mid) - (entropy(a) + entropy(v)) / 2)
    # Any abstaining side scores zero, including two zero vectors. No false
    # perfect agreement from argmax(zeros), cosine(0,0), or JSD(0,0).
    return np.where(usable, np.clip(result, 0, 1), 0)


def top1_agreement(audio, visual):
    """Per-bin top-1 equality; ties choose the first vocabulary column."""
    return _score(*_probabilities(audio, visual), "top1")


def dot_product_agreement(audio, visual):
    """Per-bin dot product of normalized predictions (not chance corrected)."""
    return _score(*_probabilities(audio, visual), "dot_product")


def cosine_similarity(audio, visual):
    """Per-bin cosine; zero-norm comparisons return zero."""
    return _score(*_probabilities(audio, visual), "cosine")


def jensen_shannon_similarity(audio, visual):
    """1 minus base-2 JSD (not its square root); zero terms contribute zero."""
    return _score(*_probabilities(audio, visual), "jensen_shannon")


@dataclass(frozen=True)
class BaselineScores:
    per_bin: Mapping[str, np.ndarray]
    pair_scores: Mapping[str, np.ndarray]
    aggregation: str = AGGREGATION


def score_probabilities(audio, visual) -> BaselineScores:
    """Score predictions alone; aggregate by arithmetic mean over all ten bins.

    No activity weighting, annotation masking, foreground selection, thresholds,
    fitting, RNG, or data-dependent aggregation is used.
    """
    a, v, usable = _probabilities(audio, visual)
    bins, means = {}, {}
    for name in METHODS:
        bins[name] = _score(a, v, usable, name)
        means[name] = bins[name].mean(axis=1)
        bins[name].flags.writeable = False
        means[name].flags.writeable = False
    return BaselineScores(MappingProxyType(bins), MappingProxyType(means))


__all__ = ["METHODS", "AGGREGATION", "BaselineScores", "score_probabilities",
           "top1_agreement", "dot_product_agreement", "cosine_similarity",
           "jensen_shannon_similarity"]

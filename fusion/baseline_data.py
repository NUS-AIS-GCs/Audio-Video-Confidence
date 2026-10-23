"""Read-only Phase 2 row selection; metadata never enters probability features."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .contracts import Bundle, ContractError, SPLITS
from .pair_contracts import PairRecord, PairType, pair_order_key
from .pair_generator import GeneratedManifest, validate_generated_manifest


@dataclass(frozen=True)
class ProbabilityFeatures:
    """Only predictions, in matching bin order; no annotations or identifiers."""

    audio: np.ndarray
    visual: np.ndarray

    def __post_init__(self):
        for name in ("audio", "visual"):
            value = np.array(getattr(self, name), copy=True, subok=False)
            if value.ndim != 3 or value.shape[1:] != (10, 29):
                raise ContractError("probability features must have shape (N,10,29)")
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        if self.audio.shape != self.visual.shape:
            raise ContractError("audio and visual feature shapes must match")


@dataclass(frozen=True)
class ProbabilityPartition:
    split: str
    partition: str
    pairs: tuple[PairRecord, ...]
    features: ProbabilityFeatures

    def __post_init__(self):
        if self.split not in SPLITS or self.partition not in ("primary", "challenge"):
            raise ContractError("unknown probability split or partition")
        if type(self.pairs) is not tuple or type(self.features) is not ProbabilityFeatures:
            raise ContractError("expected immutable pairs and ProbabilityFeatures")
        if len(self.pairs) != len(self.features.audio) or self.features.audio.shape != self.features.visual.shape:
            raise ContractError("pair and feature row counts differ")
        keys = [pair_order_key(p) for p in self.pairs]
        if keys != sorted(keys) or len(set(p.identity for p in self.pairs)) != len(keys):
            raise ContractError("partition pairs must be unique and canonically ordered")
        for pair in self.pairs:
            challenge = pair.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
            if pair.split != self.split or challenge != (self.partition == "challenge"):
                raise ContractError("pair split or type conflicts with partition")


@dataclass(frozen=True)
class PairProbabilities:
    primary: ProbabilityPartition
    challenge: ProbabilityPartition


def extract_pair_probabilities(
    manifest: GeneratedManifest, audio: Bundle, vision: Bundle, *, split: str,
    limit_per_partition: int | None = None,
) -> PairProbabilities:
    """Validate the complete manifest/bundles before selecting original rows.

    Optional limits take a canonical prefix of each partition, after validation;
    they do not change the production manifest or certify a reduced manifest.
    Phase 1 validates numeric artifacts; Phase 2 validates identity/provenance.
    Loaded bundles must retain their original producer row order. No annotation
    mask, true class, pair target or pair type affects the extracted values.
    """
    if split not in SPLITS:
        raise ContractError("unknown split")
    if limit_per_partition is not None and (
        not isinstance(limit_per_partition, Integral)
        or isinstance(limit_per_partition, bool) or limit_per_partition < 0
    ):
        raise ContractError("limit_per_partition must be a nonnegative integer or None")
    if audio.modality != "audio" or vision.modality != "vision":
        raise ContractError("expected audio and vision bundles in that order")
    validate_generated_manifest(manifest, audio, vision)
    if manifest.config.split != split:
        raise ContractError("requested split conflicts with manifest split")
    ap = audio.splits[split].probabilities
    vp = vision.splits[split].probabilities
    if ap is None or vp is None:
        raise ContractError("both modalities require predicted probabilities")

    def select(pairs, name):
        chosen = pairs if limit_per_partition is None else pairs[:limit_per_partition]
        ai = np.array([p.audio.row_index for p in chosen], dtype=np.intp)
        vi = np.array([p.visual.row_index for p in chosen], dtype=np.intp)
        for bundle, indices in ((audio, ai), (vision, vi)):
            valid = bundle.splits[split].input_validity
            if valid is not None and not valid[indices].all():
                raise ContractError("all-ten-bin baseline rejects explicitly invalid selected input bins")
        return ProbabilityPartition(split, name, chosen, ProbabilityFeatures(ap[ai], vp[vi]))

    return PairProbabilities(select(manifest.primary_pairs, "primary"),
                             select(manifest.challenge_pairs, "challenge"))


__all__ = ["ProbabilityFeatures", "ProbabilityPartition", "PairProbabilities",
           "extract_pair_probabilities"]

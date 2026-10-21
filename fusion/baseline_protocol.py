"""Bind score records to validated Phase 2 data without changing earlier APIs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from numbers import Integral

import numpy as np

from .baseline_metrics import PairScoreRecord, evaluate_records
from .baselines import METHODS
from .contracts import Bundle, ContractError, SPLITS
from .pair_contracts import _digest, _text, _validate_exact, split_seed
from .pair_generator import GeneratedManifest, GenerationProvenance, validate_generated_manifest

PROTOCOL_VERSION = "gc2-operating-points-v1"


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False) + "\n"


def _fingerprint(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScoreProvenance:
    """Full available Phase 2 overlap context; binary identity is a declaration."""

    reference: str
    generation: GenerationProvenance
    source_overlaps: tuple[tuple[str, str, tuple[str, ...]], ...]
    missing_audio_metadata_splits: tuple[str, ...]
    missing_visual_metadata_splits: tuple[str, ...]
    artifact_inventory_fingerprint: str | None = None

    def __post_init__(self):
        _text(self.reference, "score source reference")
        _validate_exact(self.generation, GenerationProvenance, "generation provenance")
        if self.artifact_inventory_fingerprint is not None:
            _digest(self.artifact_inventory_fingerprint, "declared artifact inventory")
        if type(self.source_overlaps) is not tuple:
            raise ContractError("source overlaps must be an immutable tuple")
        keys = []
        for item in self.source_overlaps:
            if type(item) is not tuple or len(item) != 3:
                raise ContractError("invalid source overlap record")
            modality, split_pair, ids = item
            if modality not in ("audio", "vision") or split_pair not in ("train&val", "train&test", "val&test"):
                raise ContractError("invalid source overlap scope")
            if (type(ids) is not tuple or any(type(s) is not str or not s for s in ids)
                    or ids != tuple(sorted(set(ids)))):
                raise ContractError("overlap source IDs must be unique sorted text")
            keys.append((modality, split_pair))
        if keys != sorted(set(keys)):
            raise ContractError("source overlaps must have unique canonical scopes")
        for missing in (self.missing_audio_metadata_splits, self.missing_visual_metadata_splits):
            if (type(missing) is not tuple or any(s not in SPLITS for s in missing)
                    or len(set(missing)) != len(missing)):
                raise ContractError("invalid missing metadata splits")


def _records(records, *, split, partition):
    if split not in ("train", "val"):
        raise ContractError("this implementation does not accept test score records")
    if type(records) not in (list, tuple):
        raise ContractError("score records must use an ordinary ordered list or tuple")
    records = tuple(records)
    # Class-owned checks in the audited API validate targets, finite bounded
    # scores, unique identities, canonical ordering and separate partitions.
    try:
        evaluate_records(records, split=split, partition=partition)
    except AttributeError as exc:
        raise ContractError("score record is missing required fields") from exc
    return records


@dataclass(frozen=True)
class ScoreDataset:
    """Immutable evaluation snapshot; use validate_score_dataset at ingestion."""

    split: str
    partition: str
    records: tuple[PairScoreRecord, ...]
    provenance: ScoreProvenance

    def __post_init__(self):
        if type(self.records) is not tuple:
            raise ContractError("dataset records must be an immutable tuple")
        _records(self.records, split=self.split, partition=self.partition)
        _validate_exact(self.provenance, ScoreProvenance, "score provenance")
        source = self.provenance.generation.source_provenance
        if source.split_seed != split_seed(source.root_seed, self.split):
            raise ContractError("dataset split conflicts with generation source provenance")
        if any(r.pair.provenance != source for r in self.records):
            raise ContractError("score records conflict with generation source provenance")

    @property
    def fingerprint(self):
        _validate_exact(self, ScoreDataset, "score dataset")
        return _fingerprint({"protocol": PROTOCOL_VERSION, "dataset": self})


def validate_score_dataset(
    records, manifest: GeneratedManifest, audio: Bundle, vision: Bundle, *,
    split: str, partition: str, source_reference: str,
    artifact_inventory_fingerprint: str | None = None,
) -> ScoreDataset:
    """Require exact partition coverage and Phase 2 row/provenance validation.

    No predictions are fitted/recomputed here. Scores must come from the audited
    scoring path. Metadata/score fingerprints do not authenticate binary exports;
    fresh artifact/reference checks remain the Phase 2 caller's responsibility.
    """
    records = _records(records, split=split, partition=partition)
    validation = validate_generated_manifest(manifest, audio, vision)
    if manifest.config.split != split:
        raise ContractError("manifest split conflicts with evaluation split")
    expected = manifest.primary_pairs if partition == "primary" else manifest.challenge_pairs
    if tuple(r.pair for r in records) != expected:
        raise ContractError("score records must cover exactly the validated manifest partition")
    context = ScoreProvenance(
        source_reference, manifest.provenance,
        tuple(sorted((o.modality, o.split_pair, o.clip_ids) for o in validation.source_overlaps)),
        validation.missing_audio_metadata_splits, validation.missing_visual_metadata_splits,
        artifact_inventory_fingerprint,
    )
    return ScoreDataset(split, partition, records, context)


def _dataset_values(dataset, method):
    _validate_exact(dataset, ScoreDataset, "score dataset")
    if type(method) is not str or method not in METHODS:
        raise ContractError("unsupported baseline method")
    i = METHODS.index(method)
    return (np.array([r.target for r in dataset.records], dtype=np.int64),
            np.array([r.scores[i][1] for r in dataset.records], dtype=np.float64))


def _warnings(*datasets):
    messages = {"Scores measure semantic agreement, not physical causality or waveform synchronization."}
    for dataset in datasets:
        context = dataset.provenance
        messages.add("Source-disjoint fusion evaluation cannot erase prior unimodal encoder exposure.")
        for modality, scope, ids in context.source_overlaps:
            if ids:
                messages.add(f"{modality}: {len(ids)} shared source videos across {scope}; official splits retained.")
        for modality, missing in (("audio", context.missing_audio_metadata_splits),
                                  ("vision", context.missing_visual_metadata_splits)):
            if missing:
                messages.add(f"{modality}: overlap context unavailable for {','.join(missing)}; no zero-overlap claim.")
    return tuple(sorted(messages))


__all__ = ["PROTOCOL_VERSION", "ScoreProvenance", "ScoreDataset", "validate_score_dataset"]

"""Phase 7 correspondence contracts, with explicit exploratory/archive trust.

Caller-created PredictionSets are unverified. Reserved model attribution requires
the restricted archived TRAIN admission path; no model execution or fitting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType

import numpy as np

from .contracts import ContractError
from .baseline_protocol import _plain
from .pair_contracts import PairRecord, PairType, _digest, _text, _validate_exact, pair_order_key

SCORE_KINDS = ("semantic_similarity", "uncalibrated_sigmoid", "calibrated_probability")


def canonical_json(value) -> str:
    """Strict JSON: deterministic keys, no NaN/Infinity or process metadata."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _finite_float(value, name, *, bounded=True):
    if (type(value) is not float or not np.isfinite(value)
            or (bounded and not 0 <= value <= 1)):
        raise ContractError(f"{name}: expected finite Python float" + (" in [0,1]" if bounded else ""))


@dataclass(frozen=True)
class FrozenOperatingPoint:
    model_id: str
    artifact_sha256: str
    threshold: float
    score_kind: str
    evidence_reference: str
    evidence_sha256: str
    validation_score_fingerprint: str
    objective: str = "maximum_validation_primary_binary_f1"
    tie_break: str = "largest_threshold"
    fitting_split: str = "val"
    fitting_partition: str = "primary"
    decision_rule: str = "score >= threshold"
    positive_target: int = 1
    calibration_reference: str | None = None
    calibration_sha256: str | None = None

    def __post_init__(self):
        for name in ("model_id", "evidence_reference", "objective"):
            _text(getattr(self, name), name)
        for name in ("artifact_sha256", "evidence_sha256", "validation_score_fingerprint"):
            _digest(getattr(self, name), name)
        # Legacy policies may use nextafter(1,+inf) to predict no positives.
        _finite_float(self.threshold, "threshold", bounded=False)
        if (self.score_kind not in SCORE_KINDS or self.fitting_split != "val"
                or self.fitting_partition != "primary" or self.tie_break != "largest_threshold"
                or self.decision_rule != "score >= threshold"
                or type(self.positive_target) is not int or self.positive_target != 1):
            raise ContractError("unsupported score semantics, threshold provenance or positive direction")
        if self.score_kind == "calibrated_probability":
            _text(self.calibration_reference, "calibration_reference")
            _digest(self.calibration_sha256, "calibration_sha256")
        elif self.calibration_reference is not None or self.calibration_sha256 is not None:
            raise ContractError("calibration evidence requires calibrated_probability semantics")


@dataclass(frozen=True)
class CorrespondencePrediction:
    pair: PairRecord
    target: int
    score: float

    def __post_init__(self):
        _validate_exact(self.pair, PairRecord, "pair")
        if (type(self.target) is not int or self.target not in (0, 1)
                or self.target != self.pair.original_pair_target):
            raise ContractError("target must be integer 0/1 and match original-pair provenance")
        _finite_float(self.score, "score")


@dataclass(frozen=True)
class EvaluationProvenance:
    split: str
    partition: str
    pair_manifest_sha256: str
    score_source_sha256: str
    score_source_reference: str
    source_overlap_sha256: str
    source_overlap_reference: str

    def __post_init__(self):
        # Check scope before examining records; no TEST/challenge opt-in exists.
        if self.split not in ("train", "val") or self.partition != "primary":
            raise ContractError("Phase 7 foundation permits TRAIN/validation primary only")
        for name in ("pair_manifest_sha256", "score_source_sha256", "source_overlap_sha256"):
            _digest(getattr(self, name), name)
        for name in ("score_source_reference", "source_overlap_reference"):
            _text(getattr(self, name), name)


@dataclass(frozen=True)
class PredictionSet:
    model_id: str
    artifact_sha256: str
    score_kind: str
    provenance: EvaluationProvenance
    records: tuple[CorrespondencePrediction, ...]

    def __post_init__(self):
        _validate_exact(self.provenance, EvaluationProvenance, "evaluation provenance")
        _text(self.model_id, "model_id")
        if type(self.model_id) is not str:
            raise ContractError("model identity must be plain text")
        _digest(self.artifact_sha256, "artifact_sha256")
        if self.score_kind not in SCORE_KINDS:
            raise ContractError("unknown score interpretation")
        if type(self.records) is not tuple or not self.records:
            raise ContractError("evaluation requires a nonempty tuple of predictions")
        identities = set()
        for record in self.records:
            _validate_exact(record, CorrespondencePrediction, "prediction")
            if (record.pair.split != self.provenance.split
                    or record.pair.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE):
                raise ContractError("mixed splits or challenge records are forbidden")
            if record.pair.identity in identities:
                raise ContractError("duplicate sample identity")
            identities.add(record.pair.identity)

    def ordered_records(self):
        _validate_exact(self, PredictionSet, "prediction set")
        return tuple(sorted(self.records, key=lambda r: pair_order_key(r.pair)))

    @property
    def fingerprint(self):
        return fingerprint({"model_id": self.model_id, "artifact_sha256": self.artifact_sha256,
                            "score_kind": self.score_kind, "provenance": asdict(self.provenance),
                            "records": [asdict(r) for r in self.ordered_records()]})


# Read-only pins copied from byte-verified historical evidence. Importing this
# module never opens those paths, imports PyTorch, or loads a selected checkpoint.
_P3 = "_local/fusion/phase3_real_experiments/run_20261022_01"
_E3 = "_local/fusion/phase4_improvement/study_20261024_seed42_01/E3"
_A2 = "_local/fusion/phase5_ablations/study_20261025_seed42_01/A2"
FROZEN_MODELS = MappingProxyType({
    "JSD": FrozenOperatingPoint(
        "JSD", "c0a33292cd5c376f4c2267bbc91a4845c246c006d581abc9f400430a3d7b9110",
        0.32543530977887525, "semantic_similarity", _P3 + "/policies/jensen_shannon.json",
        "f57a8e78e70fe2e1194a449e9edf4e10de823d8b396e47893491da47b70f0f80",
        "3515a5744419544824617ef9bb4eb1180abedbb3f9cc03314c616ef8184e0c69",
        objective="maximum_validation_binary_f1"),
    "E3": FrozenOperatingPoint(
        "E3", "e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925",
        0.009579099009671126, "uncalibrated_sigmoid", _E3 + "/frozen_operating_point.json",
        "706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad",
        "0bc8e7ec7cec2cf3305aa1a36d49235b330a7545e190addfddfa2eb3579dc94f"),
    "A2": FrozenOperatingPoint(
        "A2", "0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12",
        0.0026640388978268056, "uncalibrated_sigmoid", _A2 + "/frozen_operating_point.json",
        "16994e04c6a8421ec8bc22c1fac33691a4a0228ba544f6e57b0730f98c95b384",
        "38e98df8d767b0916833de3e06dea0cd9179a391e5b2e9fbf2e0bf7ee9a08f52"),
})


def validate_binding(dataset, operating_point: FrozenOperatingPoint):
    # Late import keeps the numerical/exploratory path independent of file I/O.
    from .evaluation_admission import resolve_predictions

    dataset, trust = resolve_predictions(dataset)
    _validate_exact(operating_point, FrozenOperatingPoint, "frozen operating point")
    if (dataset.model_id, dataset.artifact_sha256, dataset.score_kind) != (
            operating_point.model_id, operating_point.artifact_sha256, operating_point.score_kind):
        raise ContractError("model/artifact/score semantics do not match frozen threshold")
    if dataset.model_id in FROZEN_MODELS and operating_point != FROZEN_MODELS[dataset.model_id]:
        raise ContractError("historical model operating point differs from frozen evidence")
    return dataset, trust


def align_prediction_sets(datasets: tuple) -> dict:
    """Align by full pair identity, never by position or score; reject subsets.

    Equal identities alone are insufficient: labels, source rows, source video
    IDs, sampling policy/provenance and pair-manifest/overlap evidence must agree.
    """
    if type(datasets) is not tuple or len(datasets) < 2:
        raise ContractError("alignment requires at least two prediction sets")
    from .evaluation_admission import resolve_predictions

    resolved = [resolve_predictions(dataset) for dataset in datasets]
    if len({trust["level"] for _, trust in resolved}) != 1:
        raise ContractError("cannot mix authenticated and exploratory comparisons")
    datasets = tuple(dataset for dataset, _ in resolved)
    if len({d.model_id for d in datasets}) != len(datasets):
        raise ContractError("duplicate model identity")
    ordered = [d.ordered_records() for d in datasets]
    reference = [(r.pair, r.target) for r in ordered[0]]
    p = datasets[0].provenance
    for dataset, records in zip(datasets[1:], ordered[1:]):
        q = dataset.provenance
        if ([(r.pair, r.target) for r in records] != reference
                or (q.split, q.partition, q.pair_manifest_sha256, q.source_overlap_sha256)
                != (p.split, p.partition, p.pair_manifest_sha256, p.source_overlap_sha256)):
            raise ContractError("cross-model sample/label/source/split provenance mismatch")
    return {"trust_level": resolved[0][1]["level"],
            "authenticated_model_comparison": resolved[0][1]["level"] == "authenticated_archived_train",
            "identities": [list(r.pair.identity) for r in ordered[0]],
            "targets": [r.target for r in ordered[0]],
            "models": {d.model_id: {"scores": [r.score for r in records],
                                     "prediction_fingerprint": d.fingerprint}
                       for d, records in sorted(zip(datasets, ordered), key=lambda item: item[0].model_id)}}

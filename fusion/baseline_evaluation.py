"""Frozen train/val reports and descriptive reliability, without raw-score ECE."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline_metrics import MethodEvaluation, ScoreSummary, evaluate_records, summarize_scores
from .baseline_protocol import (
    PROTOCOL_VERSION, ScoreDataset, ScoreProvenance, _dataset_values, _json, _warnings,
)
from .baseline_thresholds import (
    ClassificationMetrics, FrozenThresholdPolicy, OperatingPoint, ThresholdPolicy,
    _objective, classification_metrics, threshold_candidates,
)
from .contracts import ContractError
from .pair_contracts import _validate_exact

SCORE_BIN_EDGES = tuple(i / 10 for i in range(11))
SCORE_MEANING = "semantic_similarity_not_a_probability"


@dataclass(frozen=True)
class ScoreBin:
    lower: float
    upper: float
    upper_inclusive: bool
    count: int
    positive_count: int
    negative_count: int
    mean_score: float | None
    observed_original_pair_frequency: float | None


def score_reliability(dataset: ScoreDataset, *, method: str) -> tuple[ScoreBin, ...]:
    """Ten fixed [i/10,(i+1)/10) bins; the final bin includes 1.

    Observed original-pair frequency is descriptive, not a calibrated probability
    estimate. No ECE, Brier score, isotonic regression or Platt scaling is used.
    """
    y, s = _dataset_values(dataset, method)
    groups = np.minimum(np.searchsorted(SCORE_BIN_EDGES, s, side="right") - 1, 9)
    bins = []
    for i, (lower, upper) in enumerate(zip(SCORE_BIN_EDGES[:-1], SCORE_BIN_EDGES[1:])):
        selected = groups == i
        count, positive = int(selected.sum()), int(y[selected].sum())
        bins.append(ScoreBin(lower, upper, i == 9, count, positive, count - positive,
                             float(s[selected].mean()) if count else None,
                             float(positive / count) if count else None))
    return tuple(bins)


@dataclass(frozen=True)
class ThresholdSensitivity:
    method: str
    source_fingerprint: str
    objective: str
    selected_threshold: float
    equally_optimal_thresholds: tuple[float, ...]
    validation_operating_points: tuple[OperatingPoint, ...]
    interpretation: str = "Validation fitting sweep only; not resampling stability or held-out generalization."


def threshold_sensitivity(policy: ThresholdPolicy) -> ThresholdSensitivity:
    _validate_exact(policy, ThresholdPolicy, "threshold policy")
    points = threshold_candidates(policy.source, method=policy.method)
    best = _objective(policy.selection)
    tied = tuple(point.threshold for point in points if _objective(point) == best)
    return ThresholdSensitivity(policy.method, policy.source_fingerprint, policy.objective,
                                policy.threshold, tied, points)


@dataclass(frozen=True)
class ReportContext:
    protocol_version: str
    split: str
    partition: str
    method: str
    threshold: float
    objective: str
    tie_break: str
    frozen_policy_fingerprint: str
    fitting_source_fingerprint: str
    evaluation_source_fingerprint: str
    fitting_provenance: ScoreProvenance
    evaluation_provenance: ScoreProvenance
    validation_count: int
    validation_positive_count: int
    validation_negative_count: int
    fitting_pair_reuse_count: int
    fitting_source_video_reuse_count: int
    validation_reuse_note: str
    warnings: tuple[str, ...]
    score_meaning: str = SCORE_MEANING


def _context(dataset, frozen):
    _validate_exact(frozen, FrozenThresholdPolicy, "frozen threshold policy")
    _validate_exact(dataset, ScoreDataset, "score dataset")
    policy = frozen.policy
    source = policy.source
    pair_ids = {record.identity for record in source.records}

    def clips(data):
        return {s.record.clip_id for r in data.records for s in (r.pair.audio, r.pair.visual)}

    reuse = sum(record.identity in pair_ids for record in dataset.records)
    source_reuse = len(clips(source) & clips(dataset))
    note = ("Validation diagnostics reuse validation for threshold selection; no independent generalization claim."
            if dataset.split == "val" else "Training-split diagnostics; not held-out evaluation.")
    return ReportContext(
        PROTOCOL_VERSION, dataset.split, dataset.partition, policy.method, policy.threshold,
        policy.objective, policy.tie_break, frozen.policy_fingerprint, policy.source_fingerprint,
        dataset.fingerprint, source.provenance, dataset.provenance,
        policy.validation_count, policy.validation_positive_count, policy.validation_negative_count,
        reuse, source_reuse, note, _warnings(source, dataset),
    )


@dataclass(frozen=True)
class PrimaryEvaluation:
    context: ReportContext
    classification: ClassificationMetrics
    threshold_independent: MethodEvaluation
    score_bins: tuple[ScoreBin, ...]


def evaluate_primary(dataset: ScoreDataset, frozen: FrozenThresholdPolicy) -> PrimaryEvaluation:
    """Apply a frozen validation policy to one train/val primary partition."""
    context = _context(dataset, frozen)
    if dataset.partition != "primary":
        raise ContractError("primary evaluation cannot include challenge pairs")
    y, s = _dataset_values(dataset, context.method)
    classification = classification_metrics(y, s, threshold=context.threshold)
    ranking = evaluate_records(dataset.records, split=dataset.split, partition="primary")
    method = next(item for item in ranking.methods if item.method == context.method)
    return PrimaryEvaluation(context, classification, method, score_reliability(dataset, method=context.method))


@dataclass(frozen=True)
class ChallengeEvaluation:
    context: ReportContext
    count: int
    positive_count: int
    negative_count: int
    classified_as_original: int
    rejected: int
    false_acceptance_rate: float | None
    rejection_rate: float | None
    score_distribution: ScoreSummary
    score_bins: tuple[ScoreBin, ...]


def evaluate_challenge(dataset: ScoreDataset, frozen: FrozenThresholdPolicy) -> ChallengeEvaluation:
    """Separate same-category false acceptance using a validation-primary policy.

    Challenge labels must all be provenance target 0. Empty rates are undefined.
    No challenge ROC-AUC/AP/F1 is produced or pooled with primary examples.
    """
    context = _context(dataset, frozen)
    if dataset.partition != "challenge":
        raise ContractError("challenge evaluation requires a separate challenge partition")
    y, s = _dataset_values(dataset, context.method)
    if np.any(y != 0):
        raise ContractError("challenge provenance targets must all be zero")
    accepted, count = int(np.sum(s >= context.threshold)), len(s)
    return ChallengeEvaluation(context, count, 0, count, accepted, count - accepted,
                               float(accepted / count) if count else None,
                               float((count - accepted) / count) if count else None,
                               summarize_scores(s), score_reliability(dataset, method=context.method))


def evaluation_report_to_json(report: PrimaryEvaluation | ChallengeEvaluation) -> str:
    """Canonical output serialization, not an untrusted-report ingestion API."""
    if type(report) not in (PrimaryEvaluation, ChallengeEvaluation):
        raise ContractError("expected a primary or challenge evaluation output")
    return _json(report)


def frozen_policy_to_json(policy: FrozenThresholdPolicy) -> str:
    """Persist the frozen policy with complete validation selection evidence."""
    _validate_exact(policy, FrozenThresholdPolicy, "frozen threshold policy")
    return _json(policy)


__all__ = ["SCORE_BIN_EDGES", "SCORE_MEANING", "ScoreBin", "ThresholdSensitivity", "ReportContext",
           "PrimaryEvaluation", "ChallengeEvaluation", "score_reliability", "threshold_sensitivity",
           "evaluate_primary", "evaluate_challenge", "evaluation_report_to_json", "frozen_policy_to_json"]

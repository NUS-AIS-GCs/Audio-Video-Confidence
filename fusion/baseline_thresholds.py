"""Validation-primary-only operating points; similarities are not probabilities."""
from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from numbers import Real

import numpy as np

from .baseline_protocol import PROTOCOL_VERSION, ScoreDataset, _dataset_values, _fingerprint
from .baselines import METHODS
from .contracts import ContractError
from .pair_contracts import _digest, _validate_exact

OBJECTIVE = "maximum_validation_binary_f1"
TIE_BREAK = "largest_threshold"


def _finite_threshold(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ContractError("threshold must be a finite real number, not a boolean")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ContractError("threshold is not a finite representable number") from exc
    if not np.isfinite(result):
        raise ContractError("threshold must be finite")
    return result


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self):
        if any(type(getattr(self, f.name)) is not int or getattr(self, f.name) < 0 for f in fields(self)):
            raise ContractError("confusion counts must be nonnegative Python integers")


def _metric_values(counts):
    _validate_exact(counts, ConfusionCounts, "confusion counts")
    tp, tn, fp, fn = counts.tp, counts.tn, counts.fp, counts.fn
    positive, negative = tp + fn, tn + fp
    n = positive + negative

    def zero_ratio(a, b):
        return float(a / b) if b else 0.0

    recall, specificity = zero_ratio(tp, positive), zero_ratio(tn, negative)
    f1, negative_f1 = zero_ratio(2 * tp, 2 * tp + fp + fn), zero_ratio(2 * tn, 2 * tn + fp + fn)
    return dict(
        count=n, positive_count=positive, negative_count=negative,
        accuracy=float((tp + tn) / n) if n else None,
        precision=zero_ratio(tp, tp + fp), recall=recall, f1=f1,
        macro_f1=float((f1 + negative_f1) / 2),
        balanced_accuracy=float((recall + specificity) / 2) if positive and negative else None,
        specificity=specificity, positive_prediction_rate=float((tp + fp) / n) if n else None,
    )


@dataclass(frozen=True)
class ClassificationMetrics:
    """Zero denominators give 0, except empty accuracy/PPR and one-class BA."""

    counts: ConfusionCounts
    count: int
    positive_count: int
    negative_count: int
    accuracy: float | None
    precision: float
    recall: float
    f1: float
    macro_f1: float
    balanced_accuracy: float | None
    specificity: float
    positive_prediction_rate: float | None

    def __post_init__(self):
        for name, expected in _metric_values(self.counts).items():
            value = getattr(self, name)
            if type(value) is not type(expected) or value != expected:
                raise ContractError(f"classification metric {name} conflicts with confusion counts")


def _metrics(counts):
    return ClassificationMetrics(counts, **_metric_values(counts))


def classification_metrics(targets, scores, *, threshold) -> ClassificationMetrics:
    """Pure numeric utility for synthetic checks or approved train/val use.

    Prediction is score >= threshold. Scores must be finite in [0,1]; targets
    are binary integers (booleans excluded). Arrays alone carry no split proof;
    provenance-aware reporting requires a ScoreDataset and frozen policy.
    """
    threshold = _finite_threshold(threshold)
    y, s = np.asarray(targets), np.asarray(scores)
    if (y.ndim != 1 or s.ndim != 1 or y.shape != s.shape
            or (y.size and y.dtype.kind not in "iu") or s.dtype.kind not in "fiu"
            or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all()
            or np.any((s < 0) | (s > 1))):
        raise ContractError("expected aligned binary integer targets and finite scores in [0,1]")
    # Keep NumPy scalar promotion from rounding the cutoff to a lower score dtype.
    s = s.astype(np.float64, copy=False)
    pred, positive = s >= threshold, y == 1
    counts = ConfusionCounts(int(np.sum(pred & positive)), int(np.sum(~pred & ~positive)),
                             int(np.sum(pred & ~positive)), int(np.sum(~pred & positive)))
    return _metrics(counts)


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    metrics: ClassificationMetrics

    def __post_init__(self):
        if type(self.threshold) is not float:
            raise ContractError("operating point threshold must be a canonical float")
        _finite_threshold(self.threshold)
        _validate_exact(self.metrics, ClassificationMetrics, "classification metrics")


def _validation_values(dataset, method, *, both_classes):
    y, s = _dataset_values(dataset, method)
    if dataset.split != "val" or dataset.partition != "primary":
        raise ContractError("threshold fitting requires validation primary records only")
    if not len(y):
        raise ContractError("threshold fitting requires nonempty validation primary records")
    if both_classes and (not np.any(y == 1) or not np.any(y == 0)):
        raise ContractError("threshold selection requires both correspondence target classes")
    return y, s


def _sweep(y, s):
    """O(N log N), one point for each realizable prediction partition."""
    order = np.argsort(s, kind="stable")
    ordered_s, ordered_y = s[order], y[order]
    starts = np.r_[0, np.flatnonzero(ordered_s[1:] != ordered_s[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(y)])
    pos = np.add.reduceat(ordered_y, starts)
    neg = sizes - pos
    p, n = int(y.sum()), int(len(y) - y.sum())
    below_p, below_n = np.cumsum(pos) - pos, np.cumsum(neg) - neg
    result = []
    for i, start in enumerate(starts):
        fn, tn = int(below_p[i]), int(below_n[i])
        result.append(OperatingPoint(float(ordered_s[start]), _metrics(ConfusionCounts(p - fn, tn, n - tn, fn))))
    result.append(OperatingPoint(float(np.nextafter(float(ordered_s[-1]), np.inf)),
                                 _metrics(ConfusionCounts(0, n, 0, p))))
    return tuple(result)


def threshold_candidates(dataset: ScoreDataset, *, method: str) -> tuple[OperatingPoint, ...]:
    """Ascending distinct validation scores plus nextafter(max,+inf).

    The smallest observed score predicts all-positive; the final finite sentinel
    predicts all-negative, even when max == 1. No tied scores are split. This
    covers all U+1 distinct >= prediction partitions, without redundant cutoffs.
    One-class candidates are inspectable, but cannot produce a fitted policy.
    """
    y, s = _validation_values(dataset, method, both_classes=False)
    return _sweep(y, s)


def _objective(point):
    counts = point.metrics.counts
    denominator = 2 * counts.tp + counts.fp + counts.fn
    return Fraction(2 * counts.tp, denominator) if denominator else Fraction(0)


def _select(points):
    # Exact rational comparison prevents floating roundoff from breaking F1 ties.
    return max(points, key=lambda point: (_objective(point), point.threshold))


@dataclass(frozen=True)
class ThresholdPolicy:
    method: str
    threshold: float
    objective: str
    tie_break: str
    validation_count: int
    validation_positive_count: int
    validation_negative_count: int
    selection: OperatingPoint
    candidate_count: int
    source_fingerprint: str
    source: ScoreDataset
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self):
        y, s = _validation_values(self.source, self.method, both_classes=True)
        if (self.objective != OBJECTIVE or type(self.objective) is not str
                or self.tie_break != TIE_BREAK or type(self.tie_break) is not str
                or self.protocol_version != PROTOCOL_VERSION):
            raise ContractError("unsupported threshold policy objective, tie rule or protocol")
        _digest(self.source_fingerprint, "threshold source fingerprint")
        if self.source_fingerprint != self.source.fingerprint:
            raise ContractError("threshold source fingerprint does not match validation evidence")
        points = _sweep(y, s)
        _validate_exact(self.selection, OperatingPoint, "threshold selection result")
        if type(self.threshold) is not float or self.threshold != self.selection.threshold:
            raise ContractError("threshold conflicts with selection result")
        if self.selection != _select(points):
            raise ContractError("threshold selection is not the deterministic validation optimum")
        for name, expected in (("validation_count", len(y)), ("validation_positive_count", int(y.sum())),
                               ("validation_negative_count", int(len(y) - y.sum())), ("candidate_count", len(points))):
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ContractError(f"threshold policy {name} conflicts with validation evidence")

    @property
    def fingerprint(self):
        _validate_exact(self, ThresholdPolicy, "threshold policy")
        return _fingerprint(self)

    @property
    def provenance(self):
        return self.source.provenance


def select_threshold(dataset: ScoreDataset, *, method: str) -> ThresholdPolicy:
    """Maximize validation binary F1; among exact ties choose largest threshold.

    Empty/one-class sources fail explicitly instead of supplying an arbitrary
    deployment threshold. No train, test, or challenge fitting is supported.
    """
    y, s = _validation_values(dataset, method, both_classes=True)
    points = _sweep(y, s)
    chosen = _select(points)
    return ThresholdPolicy(method, chosen.threshold, OBJECTIVE, TIE_BREAK,
                           len(y), int(y.sum()), int(len(y) - y.sum()), chosen,
                           len(points), dataset.fingerprint, dataset)


def select_thresholds(dataset: ScoreDataset) -> tuple[ThresholdPolicy, ...]:
    return tuple(select_threshold(dataset, method=method) for method in METHODS)


@dataclass(frozen=True)
class FrozenThresholdPolicy:
    """Explicit freeze step; it grants no permission to run final-test metrics."""

    policy: ThresholdPolicy
    policy_fingerprint: str
    protocol_version: str = PROTOCOL_VERSION

    def __post_init__(self):
        _validate_exact(self.policy, ThresholdPolicy, "threshold policy")
        _digest(self.policy_fingerprint, "frozen policy fingerprint")
        if self.protocol_version != PROTOCOL_VERSION or self.policy_fingerprint != self.policy.fingerprint:
            raise ContractError("frozen threshold policy differs from its validated fingerprint")


def freeze_threshold(policy: ThresholdPolicy) -> FrozenThresholdPolicy:
    _validate_exact(policy, ThresholdPolicy, "threshold policy")
    return FrozenThresholdPolicy(policy, policy.fingerprint)


__all__ = ["OBJECTIVE", "TIE_BREAK", "ConfusionCounts", "ClassificationMetrics", "OperatingPoint",
           "ThresholdPolicy", "FrozenThresholdPolicy", "classification_metrics", "threshold_candidates",
           "select_threshold", "select_thresholds", "freeze_threshold"]

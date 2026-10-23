"""Threshold-independent evaluation records; no test evaluation in this phase."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baseline_data import ProbabilityPartition
from .baselines import AGGREGATION, METHODS, score_probabilities
from .contracts import ContractError
from .pair_contracts import PairRecord, PairType, pair_order_key


@dataclass(frozen=True)
class PairScoreRecord:
    """Metadata and evaluation targets live here, outside predictive inputs.

    The original Phase 2 pair preserves both sample/source IDs, original row
    indices, split, exact type, provenance target and unknown semantic status.
    """

    pair: PairRecord
    partition: str
    scores: tuple[tuple[str, float], ...]
    aggregation: str = AGGREGATION

    def __post_init__(self):
        pair_order_key(self.pair)  # Reuses exact Phase 2 record validation.
        challenge = self.pair.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
        if self.partition != ("challenge" if challenge else "primary"):
            raise ContractError("score record partition conflicts with pair type")
        if self.aggregation != AGGREGATION:
            raise ContractError("unsupported temporal aggregation")
        if (type(self.scores) is not tuple or len(self.scores) != len(METHODS)
                or any(type(item) is not tuple or len(item) != 2 for item in self.scores)
                or tuple(name for name, _ in self.scores) != METHODS):
            raise ContractError("scores must contain every baseline in canonical order")
        for _, value in self.scores:
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, np.floating))
                    or not np.isfinite(value) or not 0 <= value <= 1):
                raise ContractError("baseline scores must be finite floating values in [0,1]")

    @property
    def identity(self):
        return self.pair.identity

    @property
    def target(self):
        return self.pair.original_pair_target

    @property
    def split(self):
        return self.pair.split

    @property
    def pair_type(self):
        return self.pair.pair_type


def make_score_records(partition: ProbabilityPartition) -> tuple[PairScoreRecord, ...]:
    """Compute scores using only features; attach pair metadata afterwards."""
    if type(partition) is not ProbabilityPartition:
        raise ContractError("expected ProbabilityPartition")
    ProbabilityPartition.__post_init__(partition)
    scores = score_probabilities(partition.features.audio, partition.features.visual)
    return tuple(PairScoreRecord(pair, partition.partition,
                                 tuple((name, float(scores.pair_scores[name][i])) for name in METHODS))
                 for i, pair in enumerate(partition.pairs))


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    undefined_reason: str | None = None


@dataclass(frozen=True)
class ScoreSummary:
    count: int
    minimum: float | None
    q25: float | None
    median: float | None
    q75: float | None
    maximum: float | None
    mean: float | None
    std: float | None


def summarize_scores(values) -> ScoreSummary:
    """Population std (ddof=0), linear-interpolated quartiles, no hist bins."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ContractError("score summary requires a finite one-dimensional array")
    if not len(values):
        return ScoreSummary(0, None, None, None, None, None, None, None)
    q = np.quantile(values, [0, .25, .5, .75, 1], method="linear")
    return ScoreSummary(len(values), *(float(x) for x in q),
                        float(values.mean()), float(values.std(ddof=0)))


def ranking_metrics(targets, scores) -> tuple[MetricValue, MetricValue]:
    """Return ROC-AUC and noninterpolated average precision, grouping ties.

    AUC = P(s_positive > s_negative) + 0.5 P(tie).
    AP = sum over distinct descending scores of delta_recall * precision.
    AP is a stepwise PR integral, not trapezoidal PR-AUC. Both metrics are
    explicitly undefined for empty or one-class input in this contract.
    This numeric utility is for synthetic verification or approved train/val
    evaluation; callers must retain split provenance via evaluate_records.
    """
    y, s = np.asarray(targets), np.asarray(scores)
    if (y.ndim != 1 or s.ndim != 1 or y.shape != s.shape
            or (y.size and y.dtype.kind not in "iu") or s.dtype.kind not in "fiu"
            or not np.isin(y, [0, 1]).all() or not np.isfinite(s).all()):
        raise ContractError("ranking metrics require aligned binary integer targets and finite scores")
    positives = int(y.sum())
    negatives = len(y) - positives
    if not positives or not negatives:
        reason = "empty_partition" if not len(y) else "only_one_target_class"
        return MetricValue(None, reason), MetricValue(None, reason)
    order = np.argsort(s, kind="stable")
    ordered_s, ordered_y = s[order], y[order]
    starts = np.r_[0, np.flatnonzero(ordered_s[1:] != ordered_s[:-1]) + 1]
    ends = np.r_[starts[1:], len(y)]
    pos = np.add.reduceat(ordered_y.astype(np.int64), starts)
    neg = ends - starts - pos
    neg_below = np.cumsum(neg) - neg
    auc = np.sum(pos * (neg_below + .5 * neg)) / (positives * negatives)
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    ap = np.sum((pos[::-1] / positives) * (tp / (tp + fp)))
    return MetricValue(float(auc)), MetricValue(float(ap))


@dataclass(frozen=True)
class MethodEvaluation:
    method: str
    roc_auc: MetricValue
    average_precision: MetricValue
    all_scores: ScoreSummary
    positive_scores: ScoreSummary
    negative_scores: ScoreSummary


@dataclass(frozen=True)
class PartitionEvaluation:
    split: str
    partition: str
    count: int
    positive_count: int
    negative_count: int
    pair_type_counts: tuple[tuple[str, int], ...]
    methods: tuple[MethodEvaluation, ...]


def evaluate_records(records, *, split: str, partition: str) -> PartitionEvaluation:
    """Evaluate exactly one train/val partition; reject test or mixed inputs.

    Split/partition remain explicit even for empty partitions. No threshold,
    calibration, classification decision or experiment logging is performed.
    """
    if split not in ("train", "val"):
        raise ContractError("Phase 3 Tasks 3.1-3.3 do not permit test metrics")
    if partition not in ("primary", "challenge"):
        raise ContractError("unknown evaluation partition")
    if not isinstance(records, (tuple, list)):
        raise ContractError("records must be an ordered tuple or list")
    for record in records:
        if type(record) is not PairScoreRecord:
            raise ContractError("expected PairScoreRecord")
        PairScoreRecord.__post_init__(record)
        if record.split != split or record.partition != partition:
            raise ContractError("mixed evaluation splits or partitions")
    identities = [r.identity for r in records]
    keys = [pair_order_key(r.pair) for r in records]
    if len(set(identities)) != len(identities) or keys != sorted(keys):
        raise ContractError("score records must be unique and canonically ordered")
    y = np.array([r.target for r in records], dtype=np.int64)
    methods = []
    for i, name in enumerate(METHODS):
        s = np.array([r.scores[i][1] for r in records], dtype=np.float64)
        auc, ap = ranking_metrics(y, s)
        methods.append(MethodEvaluation(name, auc, ap, summarize_scores(s),
                                        summarize_scores(s[y == 1]), summarize_scores(s[y == 0])))
    counts = tuple((kind.value, sum(r.pair_type is kind for r in records)) for kind in PairType)
    return PartitionEvaluation(split, partition, len(y), int(y.sum()), int(len(y) - y.sum()),
                               counts, tuple(methods))


__all__ = ["PairScoreRecord", "MetricValue", "ScoreSummary", "MethodEvaluation",
           "PartitionEvaluation", "make_score_records", "summarize_scores",
           "ranking_metrics", "evaluate_records"]

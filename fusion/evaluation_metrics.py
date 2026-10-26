"""Deterministic Phase 7 metrics over already-produced correspondence scores.

No file readers, training, threshold search, tracker, or final-evaluation CLI.
"""
from dataclasses import asdict

import numpy as np

from .baseline_evaluation import SCORE_BIN_EDGES
from .baseline_metrics import MetricValue, ranking_metrics
from .baseline_protocol import _plain
from .baseline_thresholds import classification_metrics
from .evaluation_contracts import FrozenOperatingPoint, validate_binding


def _curves(y, s):
    """Descending tied-score groups, with explicit all-negative origin.

    The origin uses JSON null (cutoff above maximum), never Infinity. Both
    curves are unavailable on one-class data, matching legacy ranking policy.
    PR includes (recall=0, precision=1); AP uses steps, not trapezoidal PR area.
    """
    if len(np.unique(y)) < 2:
        unavailable = {"points": [], "undefined_reason": "only_one_target_class"}
        return {"roc": dict(unavailable), "precision_recall": dict(unavailable)}
    order = np.argsort(-s, kind="stable")
    scores, targets = s[order], y[order]
    ends = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(y) - 1]
    tp = np.cumsum(targets, dtype=np.int64)[ends]
    fp = ends + 1 - tp
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    roc = [{"threshold": None, "above_maximum": True, "fpr": 0.0, "tpr": 0.0}]
    pr = [{"threshold": None, "above_maximum": True, "recall": 0.0, "precision": 1.0}]
    for end, t, f in zip(ends, tp, fp):
        cutoff = float(scores[end])
        roc.append({"threshold": cutoff, "above_maximum": False,
                    "fpr": float(f / negatives), "tpr": float(t / positives)})
        pr.append({"threshold": cutoff, "above_maximum": False,
                   "recall": float(t / positives), "precision": float(t / (t + f))})
    return {"roc": {"points": roc, "undefined_reason": None},
            "precision_recall": {"points": pr, "undefined_reason": None}}


def _reliability(y, s, score_kind):
    groups = np.minimum(np.searchsorted(SCORE_BIN_EDGES, s, side="right") - 1, 9)
    bins = []
    for i, (lower, upper) in enumerate(zip(SCORE_BIN_EDGES[:-1], SCORE_BIN_EDGES[1:])):
        selected = groups == i
        count = int(selected.sum())
        bins.append({"lower": lower, "upper": upper, "upper_inclusive": i == 9,
                     "count": count, "positive_count": int(y[selected].sum()),
                     "mean_score": float(s[selected].mean()) if count else None,
                     "observed_positive_frequency": float(y[selected].mean()) if count else None})
    semantic = score_kind == "semantic_similarity"
    brier = MetricValue(None, "score_not_a_probability") if semantic else MetricValue(float(np.mean((s - y) ** 2)))
    interpretation = {
        "semantic_similarity": "Descriptive score bins only; similarity is not a probability.",
        "uncalibrated_sigmoid": "Uncalibrated sigmoid probability estimates; Brier/bins are diagnostics, not calibration proof.",
        "calibrated_probability": "Probability diagnostics; calibration is a caller-supplied evidence claim, not independently verified here.",
    }[score_kind]
    return {"score_kind": score_kind, "interpretation": interpretation,
            "brier_score": asdict(brier), "bins": bins}


def evaluate_predictions(dataset, operating_point: FrozenOperatingPoint) -> dict:
    """Apply an explicit frozen model-specific threshold; never optimize it.

    Classification reuses Phase 3's zero_division=0 values. Undefined reasons
    accompany those values; one-class AUC/AP remain null with legacy reasons.
    """
    dataset, trust = validate_binding(dataset, operating_point)
    records = dataset.ordered_records()
    y = np.array([r.target for r in records], dtype=np.int64)
    s = np.array([r.score for r in records], dtype=np.float64)
    classification = classification_metrics(y, s, threshold=operating_point.threshold)
    auc, ap = ranking_metrics(y, s)
    counts = classification.counts
    zero_reasons = {
        "precision": "no_positive_predictions" if counts.tp + counts.fp == 0 else None,
        "recall": "no_positive_targets" if counts.tp + counts.fn == 0 else None,
        "f1": "no_positive_targets_or_predictions" if 2 * counts.tp + counts.fp + counts.fn == 0 else None,
        "macro_f1": "undefined_class_f1_zero_filled" if (
            2 * counts.tp + counts.fp + counts.fn == 0 or 2 * counts.tn + counts.fp + counts.fn == 0) else None,
    }
    metrics = {name: asdict(MetricValue(getattr(classification, name), zero_reasons.get(name)))
               for name in ("accuracy", "precision", "recall", "f1", "macro_f1")}
    metrics.update(roc_auc=asdict(auc), average_precision=asdict(ap))
    return _plain({
        "schema": "gc2-phase7-correspondence-evaluation-v1",
        "trust": trust,
        "task": "original_vs_substituted_pairing_provenance",
        "positive_target": 1, "positive_meaning": "original audio-video pairing provenance",
        "negative_target": 0, "negative_meaning": "substituted audio-video pairing provenance",
        "score_direction": "higher means more likely original pairing",
        "causal_source_identity_claim": False,
        "operating_point": asdict(operating_point), "provenance": asdict(dataset.provenance),
        "prediction_fingerprint": dataset.fingerprint,
        "count": classification.count,
        "support": {"positive": classification.positive_count, "negative": classification.negative_count},
        "confusion_matrix": {"labels": [0, 1], "rows": "true", "columns": "predicted",
                             "values": [[counts.tn, counts.fp], [counts.fn, counts.tp]],
                             "counts": asdict(counts)},
        "metrics": metrics, "curves": _curves(y, s),
        "reliability": _reliability(y, s, dataset.score_kind),
        "undefined_policy": "classification zero denominators => 0 with reason; fixed [0,1] macro; one-class AUC/AP => null",
        "evaluation_note": ("TRAIN diagnostic; not held-out performance." if dataset.provenance.split == "train"
                            else "Development validation reused for model/epoch/threshold selection; not held-out performance."),
        "provenance_note": trust["interpretation"],
        "samples": [{"identity": list(r.pair.identity), "pair": asdict(r.pair),
                     "target": r.target, "score": r.score,
                     "predicted_target": int(r.score >= operating_point.threshold)} for r in records],
    })

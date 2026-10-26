"""Headless display of already-computed metrics; no fitting or score evaluation.

These private drawing helpers confer no authenticity. The reporting writer is
the public artifact boundary and always builds its data from fresh admissions.
"""
from __future__ import annotations

import importlib
from pathlib import Path

from .contracts import ContractError

PNG_NAMES = ("roc_curve.png", "precision_recall_curve.png", "confusion_matrices.png",
             "reliability_diagnostics.png", "metrics_table.png")
COLORS = {"JSD": "#16768a", "E3": "#b4552b", "A2": "#7058a3"}
METRICS = ("accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "average_precision")


def _load_backend():
    try:
        mpl = importlib.import_module("matplotlib")
        figure = importlib.import_module("matplotlib.figure").Figure
        canvas = importlib.import_module("matplotlib.backends.backend_agg").FigureCanvasAgg
        return mpl, figure, canvas
    except ImportError as exc:
        raise ContractError("Matplotlib is required for PNG reports; install the existing plotting dependency") from exc
    except OSError as exc:
        raise ContractError("Matplotlib could not be loaded; verify installation and runtime access") from exc


def _format_metric(metric):
    if metric["value"] is None:
        return "undefined: " + metric["undefined_reason"]
    value = f"{metric['value']:.6f}"
    return value + (" *" if metric.get("undefined_reason") else "")


def _draw_curve(ax, models, *, curve):
    roc = curve == "roc"
    metric = "roc_auc" if roc else "average_precision"
    for model in models:
        points = model["curves"][curve]["points"]
        label = model["model_id"] + ": " + _format_metric(model["metrics"][metric])
        # Observed coordinates only. No smoothing, resampling or interpolation.
        ax.plot([p["fpr" if roc else "recall"] for p in points],
                [p["tpr" if roc else "precision"] for p in points],
                linestyle="None", marker=".", markersize=3, color=COLORS[model["model_id"]], label=label)
    missing = [m["model_id"] + ": " + m["curves"][curve]["undefined_reason"]
               for m in models if not m["curves"][curve]["points"]]
    if missing:
        ax.text(0.03, 0.03, "Unavailable curves:\n" + "\n".join(missing), transform=ax.transAxes,
                fontsize=8, va="bottom", bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"})
    ax.set(xlabel="False positive rate" if roc else "Recall (positive pairing class)",
           ylabel="True positive rate (recall)" if roc else "Precision (positive pairing class)",
           xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right" if roc else "lower left", fontsize=8)


def _draw_confusion(ax, model, count):
    matrix = model["confusion_matrix"]
    if matrix["labels"] != [0, 1] or matrix["rows"] != "true" or matrix["columns"] != "predicted":
        raise ContractError("unsupported confusion matrix orientation")
    values = matrix["values"]
    ax.imshow(values, interpolation="nearest", origin="upper", cmap="Blues", vmin=0, vmax=count)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, str(values[row][column]), ha="center", va="center", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0 substituted", "1 original"],
           yticklabels=["0 substituted", "1 original"], xlabel="Predicted label", ylabel="True label",
           title=model["model_id"] + " | threshold=" + repr(model["threshold_provenance"]["threshold"]))


def _draw_reliability(ax, model):
    diagnostic = model["reliability"]
    bins = [b for b in diagnostic["bins"] if b["count"]]
    ax.plot([b["mean_score"] for b in bins], [b["observed_positive_frequency"] for b in bins],
            linestyle="None", marker="o", markersize=5, color=COLORS[model["model_id"]], label="Occupied bins")
    for bin_data in bins:
        ax.annotate(str(bin_data["count"]), (bin_data["mean_score"], bin_data["observed_positive_frequency"]),
                    xytext=(3, 4), textcoords="offset points", fontsize=7)
    semantic = diagnostic["score_kind"] == "semantic_similarity"
    if not semantic:
        ax.plot([0, 1], [0, 1], "--", color="#999999", linewidth=0.8, label="y=x reference")
    ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.08),
           xlabel="Mean JSD similarity (not probability)" if semantic else "Mean uncalibrated sigmoid estimate",
           ylabel="Observed original-pair frequency",
           title=model["model_id"] + (": descriptive score bins" if semantic else ": uncalibrated diagnostic"))
    ax.text(0.02, 0.97, "Bin labels: sample count", transform=ax.transAxes, va="top", fontsize=7)
    ax.grid(alpha=0.2)
    ax.legend(loc="lower right", fontsize=7)


def _draw_table(ax, report):
    models = report["models"]
    rows = [[_format_metric(m["metrics"][name]) for m in models] for name in METRICS]
    rows.append([_format_metric(m["reliability"]["brier_score"]) for m in models])
    rows.extend([[str(report["support"][name])] * len(models) for name in ("positive", "negative")])
    rows.append([repr(m["threshold_provenance"]["threshold"]) for m in models])
    labels = list(METRICS) + ["Brier diagnostic", "Positive support", "Negative support", "Frozen threshold"]
    ax.axis("off")
    table = ax.table(cellText=rows, rowLabels=labels, colLabels=[m["model_id"] for m in models],
                     loc="center", cellLoc="center", colWidths=[0.27] * len(models))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    ax.text(0, 0.01, "* Undefined denominator: historical zero-filled value. JSON retains full precision and reasons.",
            transform=ax.transAxes, fontsize=8)


def _render_pngs(report, directory: Path):
    """Internal renderer; caller must already have authenticated report data."""
    mpl, Figure, Canvas = _load_backend()
    # Ignore external style files while restoring the caller's style afterwards.
    with mpl.rc_context(rc=mpl.rcParamsDefault):
        mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                             "axes.labelsize": 9, "figure.facecolor": "white", "savefig.facecolor": "white",
                             "text.usetex": False, "path.simplify": False})
        for name in PNG_NAMES:
            fig = Figure(figsize=(12.8, 5.8) if name in ("confusion_matrices.png", "reliability_diagnostics.png")
                         else (10.5, 6.5), dpi=120)
            Canvas(fig)
            if name in ("roc_curve.png", "precision_recall_curve.png"):
                _draw_curve(fig.subplots(), report["models"], curve="roc" if name == "roc_curve.png" else "precision_recall")
            elif name == "confusion_matrices.png":
                for ax, model in zip(fig.subplots(1, 3), report["models"]):
                    _draw_confusion(ax, model, report["count"])
            elif name == "reliability_diagnostics.png":
                for ax, model in zip(fig.subplots(1, 3), report["models"]):
                    _draw_reliability(ax, model)
            else:
                _draw_table(fig.subplots(), report)
            title = {"roc_curve.png": "ROC: observed threshold coordinates",
                     "precision_recall_curve.png": "Precision-recall: observed threshold coordinates (AP, not trapezoidal PR-AUC)",
                     "confusion_matrices.png": "Confusion matrices: raw counts, fixed label order [0,1]",
                     "reliability_diagnostics.png": "Reliability diagnostics; no independent calibration claim",
                     "metrics_table.png": "Binary correspondence metrics and frozen operating points"}[name]
            fig.suptitle(title, fontsize=12)
            footer = (f"TRAIN primary | authenticated archived evidence | N={report['count']} "
                      f"(+{report['support']['positive']}/-{report['support']['negative']})\n"
                      "Training diagnostics, not held-out generalization; pairing provenance, not physical causality. "
                      "Shared source videos: TRAIN/val 10, TRAIN/TEST 8.\n"
                      "Pair manifest SHA-256: " + report["alignment"]["pair_manifest_sha256"])
            fig.text(0.5, 0.015, footer, ha="center", va="bottom", fontsize=7)
            fig.tight_layout(rect=(0.02, 0.13, 0.99, 0.92))
            fig.canvas.print_png(str(directory / name), metadata={"Software": "GC2 AValidate Phase 7 reporting"})
            fig.clear()
    return {"matplotlib": mpl.__version__, "backend": "FigureCanvasAgg", "font": "DejaVu Sans", "dpi": 120}

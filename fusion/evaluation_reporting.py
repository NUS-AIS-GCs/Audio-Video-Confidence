"""Authenticated TRAIN-only comparison and transactional PNG/JSON reporting.

No caller-provided metric dictionaries or scores enter the public writer. No
model selection, training, threshold fitting, validation or held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .contracts import ContractError
from .evaluation_admission import AuthenticatedPredictionSet, admit_archived_train_predictions
from .evaluation_contracts import FROZEN_MODELS, PredictionSet, align_prediction_sets, canonical_json, fingerprint
from .evaluation_metrics import evaluate_predictions
from .evaluation_plots import PNG_NAMES, _render_pngs
from .pair_contracts import _validate_exact

MODEL_ORDER = ("JSD", "E3", "A2")
SCHEMA = "gc2-phase7-authenticated-train-report-v1"
OUTPUT_BASE = Path("_local/phase7_reporting")
LIMITATIONS = (
    "TRAIN diagnostics are not held-out generalization results; no statistical significance claim.",
    "Official source-video overlap: TRAIN/validation 10, TRAIN/TEST 8, validation/TEST 0; splits retained.",
    "Source-disjoint fusion filtering cannot undo prior encoder exposure; pairs sharing sources are dependent.",
    "Epoch, threshold and architecture selection reused development validation; single-seed research evidence.",
    "E3/A2 sigmoid scores are uncalibrated estimates; Brier and reliability bins are diagnostics, not calibration proof.",
    "JSD is semantic similarity, not a probability; its bins are descriptive and Brier is unavailable.",
    "Authenticated pairing provenance does not prove physical sound-source causality or temporal/event-head readiness.",
)


def _scope(split, partition):
    if split != "train" or partition != "primary":
        raise ContractError("production reporting supports authenticated TRAIN primary only")


def build_train_comparison(admissions: tuple, *, split="train", partition="primary") -> dict:
    """Authenticate and align every model before any metric calculation."""
    _scope(split, partition)
    if type(admissions) is not tuple or len(admissions) != len(MODEL_ORDER):
        raise ContractError("comparison requires exactly JSD, E3 and A2 admissions")
    if any(type(a) is not AuthenticatedPredictionSet for a in admissions):
        raise ContractError("comparison rejects exploratory/mixed inputs; authenticated admissions required")
    try:
        for admitted in admissions:
            _validate_exact(admitted.dataset, PredictionSet, "comparison prediction set")
        names = tuple(a.dataset.model_id for a in admissions)
    except AttributeError as exc:
        raise ContractError("incomplete authenticated comparison admission") from exc
    if len(set(names)) != len(names) or set(names) != set(MODEL_ORDER):
        raise ContractError("unknown or duplicate model identities")
    ordered = tuple(sorted(admissions, key=lambda a: MODEL_ORDER.index(a.dataset.model_id)))
    aligned = align_prediction_sets(ordered)  # Full fresh authentication, checkpoint and row checks.
    if not aligned["authenticated_model_comparison"]:
        raise ContractError("unverified comparisons cannot produce authoritative artifacts")
    evaluated = [evaluate_predictions(a, FROZEN_MODELS[a.dataset.model_id]) for a in ordered]
    models = []
    for result in evaluated:
        name = result["operating_point"]["model_id"]
        point = dict(result["operating_point"])
        # Relative machine-local evidence references are also omitted: portable
        # artifacts identify sources by semantic role and authenticated digests.
        point.pop("evidence_reference")
        point.pop("calibration_reference")
        models.append({"model_id": name,
                       "artifact": {"role": "baseline_output_manifest" if name == "JSD" else "selected_checkpoint",
                                    "sha256": point.pop("artifact_sha256")},
                       "threshold_provenance": point,
                       "prediction_fingerprint": result["prediction_fingerprint"],
                       "provenance": {"score_archive_sha256": result["provenance"]["score_source_sha256"],
                                      "source_overlap_evidence_sha256": result["provenance"]["source_overlap_sha256"]},
                       "metrics": result["metrics"], "curves": result["curves"],
                       "confusion_matrix": result["confusion_matrix"], "reliability": result["reliability"]})
    first = evaluated[0]
    # Recheck evidence after all computations, before any artifact can be written.
    align_prediction_sets(ordered)
    return {"schema": SCHEMA, "evidence_authenticity": "authenticated_archived_train",
            "split": "train", "partition": "primary",
            "cohort": "Frozen original-versus-cross-category-substituted pairing provenance; TRAIN diagnostic only",
            "model_order": list(MODEL_ORDER), "model_selection_performed": False,
            "count": first["count"], "support": first["support"],
            "positive_target": 1, "positive_meaning": first["positive_meaning"],
            "negative_target": 0, "negative_meaning": first["negative_meaning"],
            "alignment": {"identity_sha256": fingerprint(aligned["identities"]),
                          "pair_manifest_sha256": first["provenance"]["pair_manifest_sha256"],
                          "identity_count": len(aligned["identities"])},
            "producer_versions": {"audio": "beats_lora_r8_v1 / beats_iter3", "vision": "resnet18_v1"},
            "models": models, "undefined_policy": first["undefined_policy"],
            "scientific_limitations": list(LIMITATIONS)}


def _output_path(repository_root, output_directory):
    root = Path(repository_root).resolve()
    relative = Path(output_directory)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError("report output must be relative to the ignored local reporting directory")
    base = (root / OUTPUT_BASE).resolve()
    output = (root / relative).resolve()
    if (not base.is_relative_to(root) or output == base or not output.is_relative_to(base)
            or output.exists()):
        raise ContractError("report output must be a fresh child of _local/phase7_reporting")
    return base, output


def write_train_comparison(admissions: tuple, *, repository_root, output_directory,
                           split="train", partition="primary") -> dict:
    """Build from admissions only; publish a complete report without overwrites.

    PNG rendering failure leaves no destination report. Only a new owned staging
    directory is cleaned up; historical evidence paths are never output targets.
    """
    _scope(split, partition)
    base, output = _output_path(repository_root, output_directory)
    report = build_train_comparison(admissions, split=split, partition=partition)
    base.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=base)).resolve()
    try:
        rendering = _render_pngs(report, staging)
        report["rendering"] = rendering
        (staging / "report.json").write_text(canonical_json(report) + "\n", encoding="utf-8", newline="\n")
        expected = set(PNG_NAMES) | {"report.json"}
        if {p.name for p in staging.iterdir()} != expected:
            raise ContractError("renderer produced missing or unexpected report artifacts")
        manifest = {"schema": SCHEMA + "-files", "files": []}
        for name in sorted(expected):
            content = (staging / name).read_bytes()
            manifest["files"].append({"name": name, "sha256": hashlib.sha256(content).hexdigest(),
                                      "size_bytes": len(content)})
        (staging / "manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
        final_alignment = align_prediction_sets(admissions)
        if (fingerprint(final_alignment["identities"]) != report["alignment"]["identity_sha256"]
                or any(final_alignment["models"][m["model_id"]]["prediction_fingerprint"]
                       != m["prediction_fingerprint"] for m in report["models"])):
            raise ContractError("authenticated inputs changed during report rendering")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise ContractError("report destination appeared during rendering; refusing overwrite")
        staging.rename(output)
        return manifest
    except Exception as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("report rendering or publication failed; no complete report published") from exc
    finally:
        if staging.exists():
            # Verify the resolved target remains the fresh directory we created.
            if not staging.is_relative_to(base) or not staging.name.startswith(".staging-"):
                raise ContractError("unsafe staging cleanup target")
            shutil.rmtree(staging)


def generate_train_report(*, repository_root, output_directory, split="train", partition="primary") -> dict:
    """Production interface: fixed three-model cohort, no caller score inputs."""
    _scope(split, partition)
    _output_path(repository_root, output_directory)
    admissions = tuple(admit_archived_train_predictions(name, repository_root=repository_root) for name in MODEL_ORDER)
    return write_train_comparison(admissions, repository_root=repository_root, output_directory=output_directory,
                                  split=split, partition=partition)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Authenticated TRAIN-primary comparison; no held-out reader")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = generate_train_report(repository_root=args.repository_root, output_directory=args.output_directory)
    except ContractError as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps({"status": "COMPLETE", "artifacts": len(manifest["files"]) + 1,
                      "scope": "authenticated TRAIN primary only"}, sort_keys=True))


if __name__ == "__main__":
    main()

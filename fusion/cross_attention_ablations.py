"""Exactly three frozen architectural ablations; no challenge/test inference."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import csv
import io
import math
from pathlib import Path
import sys
from typing import ClassVar

from .artifact_integration import _file_entry, _inventory, _json
from .baseline_tracking import TrackingUnavailable, safe_failure
from .contracts import ContractError
from .cross_attention_model import CrossAttentionConfig, UnidirectionalAttentionConfig
from .mlp_study import primary_evidence, row_references
from .mlp_tracking import ForgeMLPTracker
from .model_interface import build_model, model_config_from_dict
from .train_cross_attention import (
    CrossAttentionExperimentConfig, SCHEMA as INITIAL_SCHEMA, _prepare_verified_inputs,
    comparison, freeze_run, initial_training, repository,
)
from .train_mlp import (
    _portable, _read, _run_experiment, _source_archive, _source_inventory,
    _verify_output_manifest,
)
from .train_mlp import _write
from .train_cross_attention import _verify_rows
from .training_contracts import _fingerprint
from .verify_ave_reference import _output_path

GROUP = "phase5_cross_attention_ablations"
SCHEMA = "gc2-cross-attention-ablation-study-v1"
RUN_SCHEMA = "gc2-cross-attention-ablation-experiment-v1"
LABELS = ("A1", "A2", "A3")
ORDER = ("initial", *LABELS)
NAMES = {"A1": "phase5_ablation_a1_audio_to_video_seed42",
         "A2": "phase5_ablation_a2_video_to_audio_seed42",
         "A3": "phase5_ablation_a3_no_position_seed42"}
NOTICE = ("Epochs, thresholds and attention variants are selected on the same primary development validation. "
          "Additional selection introduces optimism; one seed does not establish significance or generalization. "
          "Comparisons are not parameter-matched. Challenge/test diagnostics are excluded.")


def variant_model(label):
    if label == "A1":
        return UnidirectionalAttentionConfig(attention_direction="audio_to_video")
    if label == "A2":
        return UnidirectionalAttentionConfig(attention_direction="video_to_audio")
    if label == "A3":
        return CrossAttentionConfig(position_encoding="none")
    raise ContractError("exactly A1, A2 and A3 are supported")


@dataclass(frozen=True)
class AblationExperimentConfig(CrossAttentionExperimentConfig):
    model_config_types: ClassVar[tuple] = (CrossAttentionConfig, UnidirectionalAttentionConfig)
    tracking_groups: ClassVar[tuple] = (GROUP,)
    tracking_group: str = GROUP
    variant: str = "A1"

    def __post_init__(self):
        super().__post_init__()
        self.validate_ablation()

    def validate_ablation(self):
        if (self.variant not in LABELS or self.model != variant_model(self.variant)
                or self.training != initial_training(self.training.device) or self.patience != 5
                or self.cpu_threads != 4 or self.run_name != NAMES[self.variant]):
            raise ContractError("ablation configuration differs from the fixed architectural inventory")

    @property
    def fingerprint(self):
        self.validate_ablation()
        return super().fingerprint


class ForgeAblationTracker(ForgeMLPTracker):
    group = GROUP
    allowed_groups = (GROUP,)

    def __init__(self, output, label):
        if label not in LABELS:
            raise ContractError("invalid ablation tracker label")
        self.label = label
        self.artifact_prefix = "phase5-ablation-" + label.lower()
        self.run_prefix = self.artifact_prefix
        super().__init__(output)

    def start(self, config):
        try:
            model = model_config_from_dict(config["model"], "gc2-fusion-cross-attention-checkpoint-v1")
            if model != variant_model(self.label) or config.get("wandb_name") != NAMES[self.label]:
                raise ValueError("variant mismatch")
        except (KeyError, TypeError, ValueError):
            raise TrackingUnavailable("tracking configuration differs from the frozen ablation") from None
        return super().start(config)


def configurations(base, study):
    fields = {name: getattr(base, name) for name in (
        "audio_root", "vision_root", "official_root", "reference_root", "reference_evidence",
        "integration_root", "baseline_root", "mlp_reference_root")}
    return {label: AblationExperimentConfig(**fields, output_dir=study / label, variant=label,
            model=variant_model(label), training=base.training, patience=5, cpu_threads=4,
            run_name=NAMES[label]) for label in LABELS}


def rank_attention(rows):
    if {r["label"] for r in rows} != set(ORDER) or len(rows) != len(ORDER):
        raise ContractError("ranking requires the original and exactly three ablations")
    for row in rows:
        for key in ("val_roc_auc", "val_average_precision", "val_f1"):
            if type(row[key]) not in (float, int) or not math.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ContractError("ranking requires full-precision primary metrics")
    return sorted(rows, key=lambda r: (-r["val_roc_auc"], -r["val_average_precision"], -r["val_f1"], ORDER.index(r["label"])))


def _pending_row(label, config, train, val):
    root = config.output_dir
    selected, policy, history = (_read(root / name) for name in (
        "selection.json", "frozen_operating_point.json", "history.json"))
    row = {"label": label, "parameter_count": sum(p.numel() for p in build_model(config.model).parameters()),
           "selected_epoch": selected["best_epoch"], "epochs_executed": len(history),
           "frozen_threshold": policy["threshold"], "model": asdict(config.model), "training": asdict(config.training)}
    for split, report in (("train", train), ("val", val)):
        for key in ("loss", "roc_auc", "average_precision", "f1", "precision", "recall", "accuracy", "counts"):
            row[f"{split}_{key}"] = report[key]
    return row


def table(initial, completed, current, e3, jsd):
    rows = [initial, *(completed[label] if label in completed else current
                       for label in LABELS)]
    if any(row is None for row in rows):
        raise ContractError("full comparison requires every variant")
    ranked = rank_attention(rows)
    keys = ("parameter_count", "selected_epoch", "epochs_executed", "frozen_threshold", "model",
            "train_loss", "val_loss", "val_roc_auc", "val_average_precision", "val_f1",
            "val_precision", "val_recall", "val_accuracy", "val_counts")
    projected = [{"label": row["label"], **{key: row[key] for key in keys}} for row in (*rows, e3)]
    references = {"initial": initial, "E3": e3, "JSD": jsd}
    return {"schema": "gc2-cross-attention-ablation-comparison-v1", "models": projected,
            "jsd_reference": jsd, "attention_ranking": [r["label"] for r in ranked],
            "selected_attention_variant": ranked[0]["label"],
            "ranking_rule": "validation ROC-AUC descending, AP descending, F1 descending, fixed initial/A1/A2/A3 order",
            "differences": {r["label"]: {name: {key: r[key]-reference[key]
                for key in ("val_roc_auc", "val_average_precision", "val_f1")}
                for name, reference in references.items()} for r in rows},
            "selection_optimism": NOTICE, "challenge_evaluated": False, "final_test_scored": False}


def run_study(base, initial_root, study_root, *, data_factory=_prepare_verified_inputs,
              tracker_factory=ForgeAblationTracker, progress=print):
    initial_root, study = Path(initial_root).resolve(), Path(study_root).resolve()
    _output_path(study, (base.audio_root, base.vision_root, base.integration_root, initial_root, base.mlp_reference_root))
    git = repository()
    variants = configurations(base, study)
    initial = primary_evidence("initial", initial_root, schema=INITIAL_SCHEMA, include_details=True)
    e3 = primary_evidence("E3", base.mlp_reference_root, include_details=True)
    data = data_factory(base)
    if data.challenge is not None:
        raise ContractError("study must not construct a challenge dataset")
    for split, dataset in (("train", data.train), ("val", data.validation)):
        _verify_rows(initial_root / f"{split}_primary_scores.csv", dataset)
        if dataset.provenance.manifest_sha256 != initial["manifest_sha256"][split]:
            raise ContractError("ablation and initial manifests differ")
    if data.summary["artifact_inventory_fingerprint"] != initial["artifact_inventory_fingerprint"]:
        raise ContractError("ablation source inventory differs from audited initial run")
    jsd = dict(data.baseline["jsd"])
    jsd.update(label="JSD", parameter_count=0, selected_epoch=None, epochs_executed=None,
               train_loss=None, val_loss=None)
    for split in ("train", "val"):
        primary = _read(base.baseline_root / f"reports/jensen_shannon_{split}_primary.json")
        for key in ("precision", "recall", "accuracy", "counts"):
            jsd[f"{split}_{key}"] = primary["classification"][key]
    jsd["frozen_threshold"] = _read(base.baseline_root / "policies/jensen_shannon.json")["policy"]["threshold"]
    # Include every original initial-run report/checkpoint in before/after guards.
    for entry in _read(initial_root / "output_manifest.json")["files"]:
        data.paths[f"phase5_initial/{entry['role']}"] = initial_root / entry["role"]
    for name in ("output_manifest.json", "completion.json", "tracking.json"):
        data.paths[f"phase5_initial/{name}"] = initial_root / name
    data.before = _inventory(data.paths)
    data.baseline = {"initial": initial, "e3": e3, "jsd": jsd}
    source_paths, source_before = _source_inventory()
    plan = {"schema": SCHEMA, "git": git, "real_experiments": 3, "no_restarts": True,
            "fixed_order": list(LABELS), "initial_reference": initial, "e3_reference": e3,
            "jsd_reference": jsd, "data": data.summary,
            "data_provenance": [asdict(data.train.provenance), asdict(data.validation.provenance)],
            "input_hashes": _portable(data.before), "source_files": _portable(source_before),
            "epoch_selection": "strictly greater primary-validation AUC; earliest exact tie",
            "threshold_selection": "once after freeze; primary-validation max F1; largest-threshold tie",
            "variant_selection": "AUC descending, AP descending, F1 descending, fixed initial/A1/A2/A3 order",
            "not_parameter_matched": True, "challenge_evaluation": False, "test_inference": False,
            "selection_optimism": NOTICE,
            "experiments": [{"label": label, "model": asdict(cfg.model), "training": asdict(cfg.training),
                "patience": cfg.patience, "cpu_threads": cfg.cpu_threads, "pair_seed": cfg.pair_seed,
                "output_dir": str(cfg.output_dir), "run_name": cfg.run_name, "tracking_group": GROUP,
                "parameter_count": sum(p.numel() for p in build_model(cfg.model).parameters()),
                "configuration_fingerprint": cfg.fingerprint} for label, cfg in variants.items()]}
    study.mkdir(parents=True, exist_ok=False)
    _write(study, "protocol.json", plan)
    _source_archive(study, source_paths)
    plan_sha = _file_entry("plan", study / "protocol.json")["sha256"]
    def guard():
        if (_file_entry("plan", study / "protocol.json")["sha256"] != plan_sha
                or _inventory(source_paths) != source_before):
            raise ContractError("frozen study protocol or source code changed")
        for cfg, entry in zip(variants.values(), plan["experiments"]):
            if cfg.fingerprint != entry["configuration_fingerprint"]:
                raise ContractError("frozen variant configuration changed")
    guard()
    completed, attempts = {}, []
    _write(study, "state.json", {"status": "PREREGISTERED", "protocol_sha256": plan_sha})
    try:
        for label, config in variants.items():
            guard()
            attempt = {"label": label, "status": "ATTEMPTING", "attempt_number": 1}
            attempts.append(attempt); _write(study, "attempts.json", attempts)
            def freeze(cfg, run_config):
                return freeze_run(cfg, run_config, registration_extra={
                    "architecture_ablation_or_search": True, "ablation_label": label,
                    "study_protocol_sha256": plan_sha, "frozen_study_protocol": plan}, external_guard=guard)
            def compare(train, val, challenge, references):
                report = comparison(train, val, challenge, references)
                if label == "A3":
                    report["study_primary_comparison"] = table(initial, completed,
                        _pending_row(label, config, train, val), e3, jsd)
                return report
            progress(f"{label}: beginning the single prescribed attempt, {plan['experiments'][LABELS.index(label)]['parameter_count']:,} parameters.")
            _run_experiment(config, tracker_factory=lambda output: tracker_factory(output, label), progress=progress,
                repository=repository, data_factory=lambda _: data, comparison_factory=compare,
                schema=RUN_SCHEMA, freeze_run=freeze)
            row = primary_evidence(label, config.output_dir, schema=RUN_SCHEMA, include_details=True)
            for split in ("train", "val"):
                if row["row_reference_sha256"][f"{split}_primary"] != initial["row_reference_sha256"][f"{split}_primary"]:
                    raise ContractError("completed variant source references differ from original")
            completed[label] = row
            attempt.update(status="COMPLETE", run_url=row["run_url"], artifacts=row["artifacts"],
                           checkpoint_sha256=row["checkpoint_sha256"], selected_epoch=row["selected_epoch"])
            _write(study, "attempts.json", attempts)
        guard()
        result = table(initial, completed, None, e3, jsd)
        uploaded = _read(variants["A3"].output_dir / "baseline_comparison.json")["study_primary_comparison"]
        if _fingerprint(result) != _fingerprint(uploaded):
            raise ContractError("final comparison differs from the comparison verified in A3's Forge artifact")
        if _inventory(data.paths) != data.before:
            raise ContractError("historical inputs or experiment evidence changed")
        _write(study, "primary_comparison.json", result)
        columns = ("label", "parameter_count", "selected_epoch", "epochs_executed", "train_loss", "val_loss",
                   "val_roc_auc", "val_average_precision", "val_f1", "val_precision", "val_recall", "frozen_threshold")
        stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows({k: row[k] for k in columns} for row in result["models"])
        _write(study, "primary_comparison.csv", stream.getvalue())
        final = {"status": "COMPLETE", "protocol_sha256": plan_sha, "real_experiments": 3,
                 "selected_attention_variant": result["selected_attention_variant"],
                 "attention_ranking": result["attention_ranking"], "challenge_evaluated": False,
                 "test_scored": False, "original_inputs_and_initial_run_unchanged": True,
                 "comparison_uploaded_in": completed["A3"]["artifacts"]}
        _write(study, "state.json", final); _write(study, "completion.json", final)
        return final
    except BaseException as exc:
        if attempts and attempts[-1]["status"] == "ATTEMPTING":
            attempts[-1]["status"] = "FAILED"; _write(study, "attempts.json", attempts)
        _write(study, "state.json", {"status": "FAILED", "completed": list(completed), "no_automatic_retry": True,
                                   "failure": safe_failure("ablation study", exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio-root", "vision-root", "official-root", "reference-root", "reference-evidence",
                 "integration-root", "baseline-root", "mlp-reference-root", "initial-root", "study-root"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        base = CrossAttentionExperimentConfig(**{name: getattr(args, name) for name in (
            "audio_root", "vision_root", "official_root", "reference_root", "reference_evidence",
            "integration_root", "baseline_root", "mlp_reference_root")},
            output_dir=args.study_root / "unused_initial_template", training=initial_training(args.device))
        result = run_study(base, args.initial_root, args.study_root, progress=lambda message: print(message, flush=True))
        print(_json(result), end="")
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(safe_failure("controlled attention ablations", exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exactly three prespecified MLP runs, primary-only selection, then one diagnostic."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import io
import math
from pathlib import Path
import sys

import torch

from .artifact_integration import _file_entry, _inventory
from .baseline_tracking import safe_failure
from .contracts import ContractError
from .mlp_model import LateFusionMLP, MLPConfig
from .mlp_tracking import ForgeMLPTracker, STUDY_GROUP
from .mlp_training import FrozenOperatingPoint, evaluate_frozen, model_fingerprint, predict
from .train_mlp import (
    MLPExperimentConfig, ROOT, _read, _repository, _score_csv, _source_archive,
    _source_inventory, _verify_output_manifest, _write, load_selected_checkpoint,
    prepare_inputs, run_experiment,
)
from .training_contracts import OptimizerConfig, TrainingConfig, _fingerprint

SCHEMA = "gc2-mlp-regularization-study-v1"
VARIANTS = (
    ("E1", (256, 128, 64), .2, .0001, "E1-reduced-256-128-64-seed42"),
    ("E2", (512, 256, 128), .5, .0001, "E2-dropout-0p5-seed42"),
    ("E3", (512, 256, 128), .2, .01, "E3-weight-decay-0p01-seed42"),
)
ORDER = ("original", "E1", "E2", "E3")
SELECTION_NOTICE = (
    "The same validation-primary data select epochs, thresholds and configuration. "
    "This additional model selection introduces optimism; these are not unbiased held-out estimates. "
    "Challenge diagnostics are excluded from selection and remain exploratory after previous examination."
)


def experiment_configs(base):
    """Reject protocol drift; only the three prescribed overrides may vary."""
    expected_training = TrainingConfig(batch_size=64, epochs=30, selection_metric="roc_auc")
    if (type(base) is not MLPExperimentConfig or base.model != MLPConfig()
            or base.training != expected_training or (base.patience, base.cpu_threads, base.pair_seed) != (5, 4, 42)):
        raise ContractError("study requires the unchanged seed-42 CPU reference protocol")
    return tuple((label, replace(
        base, output_dir=base.output_dir / label,
        model=replace(base.model, hidden_dims=widths, dropout=dropout),
        training=replace(base.training, optimizer=replace(base.training.optimizer, weight_decay=decay)),
        evaluate_challenge=False, tracking_group=STUDY_GROUP, run_name=name,
    )) for label, widths, dropout, decay, name in VARIANTS)


def rank_candidates(rows):
    """Full-precision primary metrics only; JSD is a comparator, not an MLP candidate."""
    labels = [row["label"] for row in rows]
    if (not rows or len(set(labels)) != len(labels) or "original" not in labels
            or any(label not in ORDER for label in labels)):
        raise ContractError("invalid controlled MLP candidate inventory")
    for row in rows:
        if any(type(row[k]) not in (float, int) or not math.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in ("val_roc_auc", "val_average_precision", "val_f1")):
            raise ContractError("selection requires finite primary metrics in [0,1]")
    return sorted(rows, key=lambda r: (-r["val_roc_auc"], -r["val_average_precision"],
                                       -r["val_f1"], ORDER.index(r["label"])))


def row_references(output, name):
    rows = csv.DictReader(io.StringIO((output / f"{name}_scores.csv").read_text(encoding="utf-8")))
    fields = ("split", "partition", "audio_sample_id", "visual_sample_id", "audio_row", "visual_row", "target")
    return _fingerprint([tuple(row[key] for key in fields) for row in rows])


def primary_evidence(label, output):
    """Project persisted evidence onto primary metrics; never access challenge scores."""
    _verify_output_manifest(output)
    complete, tracking = _read(output / "completion.json"), _read(output / "tracking.json")
    if complete["status"] != "COMPLETE" or tracking["status"] != "COMPLETE" or not tracking["remote_summary_verified"]:
        raise ContractError("only verified completed runs may enter selection")
    if _file_entry("manifest", output / "output_manifest.json")["sha256"] != complete["output_manifest_sha256"]:
        raise ContractError("completed run output manifest identity changed")
    config, final = _read(output / "config.json"), _read(output / "final_metrics.json")
    history = _read(output / "history.json")
    model, metadata = load_selected_checkpoint(output)
    policy = FrozenOperatingPoint(**_read(output / "frozen_operating_point.json"))
    if (model_fingerprint(model) != policy.model_sha256 or metadata.selected_epoch != policy.selected_epoch
            or final["frozen_threshold"] != policy.threshold or final["selected_epoch"] != policy.selected_epoch
            or config["parameter_count"] != sum(p.numel() for p in model.parameters())):
        raise ContractError("selected checkpoint, threshold and primary evidence differ")
    row = {"label": label, "output_dir": str(output), "parameter_count": config["parameter_count"],
           "epochs_executed": len(history), "selected_epoch": policy.selected_epoch,
           "frozen_threshold": policy.threshold,
           "checkpoint_sha256": _file_entry("checkpoint", output / "best.pt")["sha256"],
           "threshold_sha256": _file_entry("threshold", output / "frozen_operating_point.json")["sha256"],
           "model_sha256": policy.model_sha256,
           "training_validation_seconds": sum(r["duration_seconds"] for r in history),
           "run_url": tracking["run_url"], "artifacts": tracking["artifacts"],
           "artifact_inventory_fingerprint": config["data"]["artifact_inventory_fingerprint"],
           "manifest_sha256": config["data"]["manifest_sha256"],
           "row_reference_sha256": {n: row_references(output, n) for n in ("train_primary", "val_primary")}}
    for split in ("train", "val"):
        primary = final[f"{split}_primary"]
        for key in ("roc_auc", "average_precision", "f1", "loss", "accuracy"):
            row[f"{split}_{key}"] = primary[key]
    row["train_minus_val_gap"] = {key: row[f"train_{key}"]-row[f"val_{key}"]
                                  for key in ("roc_auc", "average_precision", "f1")}
    return row


def preflight(base, reference_output):
    configs = experiment_configs(base)
    repository = _repository()
    reference = primary_evidence("original", reference_output)
    if any(abs(reference[key]-value) > 1e-12 for key, value in (
        ("val_roc_auc", .961745996386228), ("val_average_precision", .9571849895337526), ("val_f1", .9064398541919806))):
        raise ContractError("original MLP primary reference differs from audited values")
    data = prepare_inputs(replace(base, evaluate_challenge=False))
    for key in ("artifact_inventory_fingerprint", "manifest_sha256"):
        if reference[key] != data.summary[key]:
            raise ContractError("study and original MLP inputs or manifests differ")
    access = ForgeMLPTracker(base.output_dir).preflight_access()
    return configs, repository, reference, data, access


def verify_preservation(paths, before):
    if _inventory(paths) != before:
        raise ContractError("historical evidence or original inputs changed during the study")


def chosen_config(base, configs, label):
    # Do not reconstruct a configuration after its fresh output has been
    # created. dict.get's default argument would be evaluated even for E1-E3.
    if label == "original":
        return base
    if label not in dict(configs):
        raise ContractError("chosen label is outside the fixed study inventory")
    return dict(configs)[label]


def _selection_integrity(study, ranked):
    """Separate digests preserve the existing plan/selection schemas."""
    return {"schema": SCHEMA, "ranking_sha256": _fingerprint(ranked),
            **{f"{key}_sha256": _file_entry(key, study / name)["sha256"] for key, name in (
                ("plan", "plan.json"), ("selection", "selection.json"),
                ("comparison", "primary_comparison.json"))}}


def _candidate_index(records, labels, description):
    if (type(records) is not list or len(records) != len(labels)
            or any(type(row) is not dict or type(row.get("label")) is not str for row in records)):
        raise ContractError(f"invalid {description} inventory")
    ids = [row["label"] for row in records]
    if len(set(ids)) != len(ids) or set(ids) != set(labels):
        raise ContractError(f"incomplete, duplicated or unexpected {description} candidates")
    return {row["label"]: row for row in records}


def _validate_recovery(study, plan, selection, attempts):
    """Validate the complete decision and all references before recovery writes.

    Legacy studies have no integrity sidecar: reconstruct their exact canonical
    selection from the persisted comparison, plan and ledger. Honor any existing
    recorded selection digest as well; never repair historical evidence here.
    """
    planned = _candidate_index(plan["experiments"], ORDER[1:], "planned experiment")
    ledger = _candidate_index(attempts, ORDER[1:], "completed experiment ledger")
    frozen = _candidate_index(selection["ranked_candidates"], ORDER, "frozen ranking")
    comparison = _read(study / "primary_comparison.json")
    compared = _candidate_index(comparison["candidates"], ORDER, "primary comparison")
    if (plan["schema"] != SCHEMA or plan["fixed_order"] != list(ORDER)
            or plan["ranking"] != ["val_roc_auc", "val_average_precision", "val_f1", "fixed_order"]
            or [a["label"] for a in attempts] != list(ORDER[1:])
            or any(a["status"] != "COMPLETE" for a in attempts)
            or selection["configuration_selection_frozen"] is not True
            or selection["challenge_used_for_selection"] is not False):
        raise ContractError("recovery requires the fixed plan, frozen decision and three complete attempts")
    if _fingerprint(compared["original"]) != _fingerprint(plan["original"]):
        raise ContractError("original candidate differs from the immutable planned reference")
    ranked = rank_candidates(comparison["candidates"])
    expected_comparison = {"candidates": [compared[label] for label in ORDER], "baseline": plan["baseline"],
        "delta_vs_jsd": {label: {key: compared[label][key]-plan["baseline"][key]
                        for key in ("val_roc_auc", "val_average_precision", "val_f1")} for label in ORDER}}
    if _fingerprint(comparison) != _fingerprint(expected_comparison):
        raise ContractError("primary comparison differs from the planned baseline or complete candidate records")
    expected_selection = {"schema": SCHEMA, "winner": ranked[0], "ranked_candidates": ranked,
        "configuration_selection_frozen": True, "challenge_used_for_selection": False,
        "plan_sha256": _file_entry("plan", study / "plan.json")["sha256"],
        "attempts": attempts, "selection_notice": plan["selection_notice"]}
    if (_fingerprint(selection["ranked_candidates"]) != _fingerprint(ranked)
            or _fingerprint(selection) != _fingerprint(expected_selection)):
        raise ContractError("frozen ranking/selection fingerprint differs from comparison, plan or ledger")
    integrity = study / "selection_integrity.json"
    if integrity.exists() and _read(integrity) != _selection_integrity(study, ranked):
        raise ContractError("stored ranking/selection integrity hashes differ")
    selection_sha = _file_entry("selection", study / "selection.json")["sha256"]
    for name, key in (("postprocessing/recovery.json", "frozen_selection_sha256"),
                      ("selected_val_challenge.json", "selection_sha256")):
        evidence = study / name
        if evidence.exists() and _read(evidence)[key] != selection_sha:
            raise ContractError("frozen selection differs from recorded integrity evidence")
    for label in ORDER:
        row = frozen[label]
        reference = plan["original"] if label == "original" else planned[label]
        output = Path(reference["output_dir"]).resolve()
        if Path(row["output_dir"]).resolve() != output:
            raise ContractError("candidate output differs from the frozen plan")
        if label != "original":
            if output != study / label or Path(ledger[label]["output_dir"]).resolve() != output:
                raise ContractError("candidate, ledger and planned run directories differ")
            config, tracking, metadata = (_read(output / name) for name in (
                "config.json", "tracking.json", "checkpoint_metadata.json"))
            for planned_key, config_key in (("model", "model"), ("training", "training"),
                    ("patience", "patience"), ("cpu_threads", "cpu_threads"), ("pair_seed", "pair_seed"),
                    ("parameter_count", "parameter_count"), ("tracking_group", "wandb_group"), ("run_name", "wandb_name")):
                if _fingerprint(config[config_key]) != _fingerprint(reference[planned_key]):
                    raise ContractError("candidate configuration differs from the frozen plan")
            if any(_fingerprint(metadata[key]) != _fingerprint(config[key]) for key in ("model", "training")):
                raise ContractError("checkpoint configuration differs from the planned run")
            if (tracking["run_url"] != row["run_url"] or tracking["group"] != reference["tracking_group"]
                    or not isinstance(tracking["run_id"], str) or not tracking["run_id"]
                    or row["run_url"].rstrip("/").rsplit("/", 1)[-1] != tracking["run_id"]
                    or _fingerprint(tracking["artifacts"]) != _fingerprint(row["artifacts"])):
                raise ContractError("candidate run/artifact references differ from recorded tracking")
            for key in ("run_id", "run_url", "checkpoint_sha256"):
                expected = tracking["run_id"] if key == "run_id" else row[key]
                if key in ledger[label] and ledger[label][key] != expected:
                    raise ContractError("candidate run/checkpoint differs from the experiment ledger")
        for name, key in (("best.pt", "checkpoint_sha256"), ("frozen_operating_point.json", "threshold_sha256")):
            if _file_entry(name, output / name)["sha256"] != row[key]:
                raise ContractError("candidate checkpoint/threshold differs from the frozen ranking")
        if _fingerprint(primary_evidence(label, output)) != _fingerprint(row):
            raise ContractError("frozen candidate differs from verified completed experiment evidence")


def selected_challenge(study, winner, config, *, data_factory=prepare_inputs):
    """The persisted decision is a prerequisite, never a consequence of diagnostics."""
    selection_path = study / "selection.json"
    selection_hash = _file_entry("selection", selection_path)["sha256"]
    frozen = _read(selection_path)
    if frozen["winner"] != winner or not frozen["configuration_selection_frozen"]:
        raise ContractError("challenge requires the exact persisted frozen configuration decision")
    output = Path(winner["output_dir"])
    for name, key in (("best.pt", "checkpoint_sha256"), ("frozen_operating_point.json", "threshold_sha256")):
        if _file_entry(name, output / name)["sha256"] != winner[key]:
            raise ContractError("chosen checkpoint/threshold changed after configuration selection")
    model, _ = load_selected_checkpoint(output)
    policy = FrozenOperatingPoint(**_read(output / "frozen_operating_point.json"))
    data = data_factory(config)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        predictions = predict(model, data.challenge, config.training, frozen_policy=policy)
    finally:
        torch.set_num_threads(old_threads)
    result = evaluate_frozen(predictions, policy)
    if _file_entry("selection", selection_path)["sha256"] != selection_hash:
        raise ContractError("configuration selection changed during diagnostics")
    result.update(exploratory=True, previously_examined=True, selection_changed=False,
                  selected_label=winner["label"], selection_sha256=selection_hash,
                  original_mlp_rejection_count=61, original_mlp_count=402,
                  jsd_rejection_rate=.15671641791044777, final_test_scored=False)
    _write(study, "selected_val_challenge_scores.csv", _score_csv(predictions, data.challenge))
    _write(study, "selected_val_challenge.json", result)
    return result


def run_study(base, reference_output, *, experiment_runner=run_experiment, progress=print):
    configs, repository, reference, data, access = preflight(base, reference_output)
    study = base.output_dir
    study.mkdir(parents=True, exist_ok=False)
    source_paths, source_before = _source_inventory()
    preservation_paths = {**data.paths, **{f"original/{p.relative_to(reference_output).as_posix()}": p
                            for p in reference_output.rglob("*") if p.is_file()}}
    preservation_before = _inventory(preservation_paths)
    _source_archive(study, source_paths)
    plan = {"schema": SCHEMA, "git": repository, "authentication": access, "max_new_training_runs": 3,
            "retry_failed_runs": False, "challenge_before_configuration_selection": False,
            "fixed_order": list(ORDER), "ranking": ["val_roc_auc", "val_average_precision", "val_f1", "fixed_order"],
            "selection_notice": SELECTION_NOTICE, "code_sha256": _fingerprint(source_before),
            "original": reference, "baseline": data.baseline,
            "experiments": [{"label": label, "output_dir": str(c.output_dir), "model": asdict(c.model),
                             "training": asdict(c.training), "patience": c.patience, "cpu_threads": c.cpu_threads,
                             "pair_seed": c.pair_seed, "tracking_group": c.tracking_group, "run_name": c.run_name,
                             "evaluate_challenge": c.evaluate_challenge,
                             "parameter_count": sum(p.numel() for p in LateFusionMLP(c.model).parameters())}
                            for label, c in configs]}
    _write(study, "plan.json", plan)
    plan_hash = _file_entry("plan", study / "plan.json")["sha256"]
    _write(study, "preservation_before.json", preservation_before)
    rows, attempts = [reference], []
    for label, config in configs:
        # Record each attempt before creating its run; no hidden retry/resume.
        verify_preservation(preservation_paths, preservation_before)
        if _inventory(source_paths) != source_before or _file_entry("plan", study / "plan.json")["sha256"] != plan_hash:
            raise ContractError("prespecified study plan or code changed")
        attempt = {"label": label, "status": "ATTEMPTING", "output_dir": str(config.output_dir)}
        attempts.append(attempt)
        _write(study, "attempts.json", attempts)
        try:
            progress(f"{label}: starting the single prescribed attempt ({config.run_name}).")
            experiment_runner(config, progress=progress)
            row = primary_evidence(label, config.output_dir)
            for key in ("artifact_inventory_fingerprint", "manifest_sha256", "row_reference_sha256"):
                if row[key] != reference[key]:
                    raise ContractError("candidate and original use different source rows or manifests")
            rows.append(row)
            attempt["status"] = "COMPLETE"
        except Exception as exc:
            attempt.update(status="FAILED", blocker=safe_failure(f"{label} prescribed attempt", exc))
            progress(attempt["blocker"])
        _write(study, "attempts.json", attempts)
    verify_preservation(preservation_paths, preservation_before)
    ranked = rank_candidates(rows)
    winner = ranked[0]
    selection = {"schema": SCHEMA, "winner": winner, "ranked_candidates": ranked,
                 "configuration_selection_frozen": True, "challenge_used_for_selection": False,
                 "plan_sha256": plan_hash, "attempts": attempts, "selection_notice": SELECTION_NOTICE}
    # Persist comparison and the checkpoint/threshold identities BEFORE any
    # challenge scoring. JSD is a reference only and cannot win MLP selection.
    _write(study, "primary_comparison.json", {"candidates": rows, "baseline": data.baseline,
           "delta_vs_jsd": {row["label"]: {key: row[key]-data.baseline[key]
                             for key in ("val_roc_auc", "val_average_precision", "val_f1")} for row in rows}})
    _write(study, "selection.json", selection)
    _write(study, "selection_integrity.json", _selection_integrity(study, ranked))
    progress(f"Configuration selection frozen from primary metrics: {winner['label']}.")
    diagnostic_config = chosen_config(base, configs, winner["label"])
    challenge = selected_challenge(study, winner, diagnostic_config)
    verify_preservation(preservation_paths, preservation_before)
    if _inventory(source_paths) != source_before:
        raise ContractError("source changed during the study")
    _write(study, "preservation_after.json", {"unchanged": True, "files": preservation_before})
    result = {"schema": SCHEMA, "status": "COMPLETE" if all(a["status"] == "COMPLETE" for a in attempts) else "PARTIAL",
              "attempts": attempts, "selected_label": winner["label"], "challenge": challenge,
              "no_final_test_scored": True, "historical_evidence_unchanged": True,
              "new_training_run_attempts": len(attempts), "challenge_remote_logging": "local study report only; no fourth run"}
    _write(study, "result.json", result)
    return result


def finish_frozen_study(study):
    """Postprocessing recovery only: no trainer, optimizer or W&B init calls.

    A previously frozen decision and three completed single attempts are
    mandatory. Historical plans, selections and run artifacts stay immutable.
    """
    study = Path(study).resolve()
    if (study / "result.json").exists() or (study / "selected_val_challenge.json").exists():
        raise ContractError("postprocessing already exists; do not repeat diagnostics")
    plan, selection, attempts = (_read(study / name) for name in ("plan.json", "selection.json", "attempts.json"))
    try:
        _validate_recovery(study, plan, selection, attempts)
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("malformed frozen recovery evidence") from None
    before = _read(study / "preservation_before.json")
    paths = {entry["role"]: Path(entry["path"]) for entry in before}
    verify_preservation(paths, before)
    winner = selection["winner"]
    saved = _read(Path(winner["output_dir"]) / "config.json")
    model = MLPConfig(**{**saved["model"], "hidden_dims": tuple(saved["model"]["hidden_dims"])})
    optimizer = saved["training"]["optimizer"]
    training = TrainingConfig(**{**saved["training"], "optimizer": OptimizerConfig(
        **{**optimizer, "betas": tuple(optimizer["betas"])})})
    # The unused output is only required by the fresh-output config contract;
    # it is never created and no run_experiment call is made here.
    config = MLPExperimentConfig(
        ROOT / "_local/embeddings/audio_final", ROOT / "_local/embeddings/vision_final/resnet18_v1",
        ROOT / "_local/datasets/AVE", ROOT / "_local/datasets/AVE_reference",
        ROOT / "_local/fusion/ave_reference_verified/verification.json", ROOT / "_local/fusion/task28_real",
        ROOT / "_local/fusion/phase3_real_experiments/run_20261022_01", study / "unused_diagnostic_output",
        model=model, training=training, evaluate_challenge=False, tracking_group=STUDY_GROUP)
    audit = study / "postprocessing"
    audit.mkdir(exist_ok=False)
    source_paths, source_before = _source_inventory()
    _source_archive(audit, source_paths)
    _write(audit, "recovery.json", {"failure": "eager dict.get default reconstructed an existing fresh-output configuration",
           "stage": "after global selection freeze; before challenge scoring", "training_retried": False,
           "new_wandb_run_created": False, "frozen_selection_sha256": _file_entry("selection", study / "selection.json")["sha256"],
           "original_training_code_sha256": saved["code_sha256"], "postprocessing_source": source_before})
    challenge = selected_challenge(study, winner, config)
    verify_preservation(paths, before)
    if _inventory(source_paths) != source_before:
        raise ContractError("postprocessing source changed during diagnostic evaluation")
    _write(study, "preservation_after.json", {"unchanged": True, "files": before})
    result = {"schema": SCHEMA, "status": "COMPLETE", "attempts": attempts, "selected_label": winner["label"],
              "challenge": challenge, "no_final_test_scored": True, "historical_evidence_unchanged": True,
              "new_training_run_attempts": 3, "postprocessing_recovered": True,
              "failure_evidence": "postprocessing/recovery.json",
              "challenge_remote_logging": "local study report only; no fourth run"}
    _write(study, "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, default=ROOT / "_local/fusion/phase4_training/run_20261023_seed42_01")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finish-frozen-selection", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.finish_frozen_selection:
            if args.preflight_only:
                raise ContractError("recovery and preflight modes are mutually exclusive")
            result = finish_frozen_study(args.output_dir)
            print(f"COMPLETE: postprocessing recovered for frozen {result['selected_label']}; no training or W&B run created.")
            return 0
        base = MLPExperimentConfig(
            ROOT / "_local/embeddings/audio_final", ROOT / "_local/embeddings/vision_final/resnet18_v1",
            ROOT / "_local/datasets/AVE", ROOT / "_local/datasets/AVE_reference",
            ROOT / "_local/fusion/ave_reference_verified/verification.json", ROOT / "_local/fusion/task28_real",
            ROOT / "_local/fusion/phase3_real_experiments/run_20261022_01", args.output_dir,
            evaluate_challenge=False, tracking_group=STUDY_GROUP)
        if args.preflight_only:
            _, _, _, data, access = preflight(base, args.reference_run.resolve())
            print(f"PREFLIGHT PASS: {access['status']}; train={len(data.train)}, val={len(data.validation)}, challenge={len(data.challenge)}; no run created.")
        else:
            result = run_study(base, args.reference_run.resolve(), progress=lambda s: print(s, flush=True))
            print(f"{result['status']}: selected {result['selected_label']}; exactly {result['new_training_run_attempts']} attempts.")
            return 0 if result["status"] == "COMPLETE" else 1
    except KeyboardInterrupt:
        print("INTERRUPTED: preserve study evidence; do not rerun or replace attempts.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(safe_failure("controlled MLP study", exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

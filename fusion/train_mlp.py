"""Verified Phase 4 MLP training, frozen validation and mandatory online Forge evidence."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import torch

from .artifact_integration import _check_hashes, _file_entry, _inventory, _json, _load_exports, _read_json, _verify_official
from .baseline_tracking import TrackingUnavailable, safe_failure
from .contracts import ContractError, OFFICIAL_COUNTS
from .embedding_data import EmbeddingDataset
from .mlp_model import LateFusionMLP, MLPConfig
from .mlp_tracking import ForgeMLPTracker, GROUP, STUDY_GROUP
from .mlp_training import (
    EarlyStopping, evaluate_frozen, fit_operating_point, model_fingerprint,
    predict, primary_metrics, train_epoch,
)
from .run_baseline_experiments import EXPORTS, _full_manifest, _verified_core_files, _verify_inputs
from .training_contracts import (
    CheckpointMetadata, OptimizerConfig, TrainingConfig, _fingerprint,
    build_optimizer, make_loader, resolve_device, seed_everything, validate_split_isolation,
)
from .verify_ave_reference import REFERENCE_HASHES, _output_path

ROOT = Path(__file__).resolve().parents[1]
REUSE_NOTICE = ("Validation primary pairs select the epoch and operating threshold and report their resulting metrics; "
                "these are not unbiased held-out estimates. Sigmoid outputs have no independent calibration evidence.")
SCHEMA = "gc2-real-mlp-experiment-v1"


@dataclass(frozen=True)
class MLPExperimentConfig:
    audio_root: Path
    vision_root: Path
    official_root: Path
    reference_root: Path
    reference_evidence: Path
    integration_root: Path
    baseline_root: Path
    output_dir: Path
    model: MLPConfig = MLPConfig()
    training: TrainingConfig = TrainingConfig(batch_size=64, epochs=30, selection_metric="roc_auc")
    patience: int = 5
    cpu_threads: int = 4
    pair_seed: int = 42
    evaluate_challenge: bool = True
    tracking_group: str = GROUP
    run_name: str | None = None

    def __post_init__(self):
        for name in ("audio_root", "vision_root", "official_root", "reference_root", "reference_evidence",
                     "integration_root", "baseline_root", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ContractError(f"{name} requires an explicit path")
            object.__setattr__(self, name, Path(value).resolve())
        if type(self.model) is not MLPConfig or type(self.training) is not TrainingConfig:
            raise ContractError("strict model/training configuration required")
        self.model.__post_init__(); self.training.__post_init__()
        if self.training.selection_metric != "roc_auc":
            raise ContractError("this prespecified experiment selects validation-primary ROC-AUC")
        for name in ("patience", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")
        if type(self.pair_seed) is not int or not 0 <= self.pair_seed < 2**64:
            raise ContractError("invalid original pair seed")
        if type(self.evaluate_challenge) is not bool or self.tracking_group not in (GROUP, STUDY_GROUP):
            raise ContractError("invalid challenge policy or tracking group")
        if self.run_name is not None and (not isinstance(self.run_name, str) or not self.run_name.strip()):
            raise ContractError("run name must be a nonempty string")
        inputs = (self.audio_root, self.vision_root, self.official_root, self.reference_root,
                  self.reference_evidence.parent, self.integration_root, self.baseline_root)
        _output_path(self.output_dir, inputs)

    @property
    def seed(self):
        # Reuse the audited Phase 3 input verifier's pair-seed interface; the
        # stochastic model/training seed remains an independent recorded field.
        return self.pair_seed


def _read(path):
    return _read_json(Path(path).read_text(encoding="utf-8"))


def _write(output, name, value):
    path = output / name
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes((value if isinstance(value, str) else _json(value)).encode("utf-8"))
    temporary.replace(path)


def _repository():
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    origin = git("remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") not in (
        "https://github.com/NUS-AIS-GCs/Audio-Video-Confidence", "git@github.com:NUS-AIS-GCs/Audio-Video-Confidence"):
        raise ContractError("required GC2 origin mismatch")
    if git("branch", "--show-current") != "feature/fusion-phase4-mlp":
        raise ContractError("required branch is feature/fusion-phase4-mlp")
    return {"repository": "NUS-AIS-GCs/Audio-Video-Confidence", "commit": git("rev-parse", "HEAD"),
            "branch": "feature/fusion-phase4-mlp", "working_tree": git("status", "--porcelain").splitlines(),
            "dirty_source_policy": "record exact source hashes and source snapshot; preserve audited uncommitted changes"}


def _source_inventory():
    paths = {str(p.relative_to(ROOT)).replace("\\", "/"): p for p in sorted((ROOT / "fusion").glob("*.py"))}
    for name in ("requirements-fusion.txt", "requirements-fusion-mlp.txt", "requirements-fusion-training.txt"):
        path = ROOT / name
        if path.is_file():
            paths[name] = path
    return paths, _inventory(paths)


def _portable(inventory):
    return [{k: v for k, v in entry.items() if k != "path"} for entry in inventory]


def _source_archive(output, paths):
    with zipfile.ZipFile(output / "source_code.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(paths.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


@dataclass
class VerifiedData:
    train: EmbeddingDataset
    validation: EmbeddingDataset
    challenge: EmbeddingDataset
    paths: dict
    before: list
    summary: dict
    baseline: dict


def _baseline_evidence(config, report, paths):
    root = config.baseline_root
    complete = _read(root / "completion.json")
    if complete.get("local_experiment_status") != "COMPLETE":
        raise ContractError("Phase 3 comparison evidence is incomplete")
    entry = _file_entry("phase3/output_manifest.json", root / "output_manifest.json")
    if complete.get("output_manifest_sha256") != entry["sha256"]:
        raise ContractError("Phase 3 completion/manifest fingerprint mismatch")
    files = _verified_core_files(root)
    for name in [*files, "output_manifest.json", "completion.json"]:
        paths[f"phase3/{name}"] = root / name
    verification = _read(root / "input_verification.json")
    if verification["artifact_inventory_fingerprint"] != report["artifact_inventory_fingerprint"]:
        raise ContractError("Phase 3 and MLP use different artifact inventories")
    previous = {e["role"]: e for e in verification["inputs"]}
    required = {e["role"]: e for e in report["artifacts"]}
    required.update({f"manifest/{name}": e for name, e in report["outputs"].items()})
    for role, evidence in required.items():
        if role not in previous or any(previous[role][key] != evidence[key] for key in ("sha256", "size_bytes")):
            raise ContractError("Phase 3 and MLP source/manifest bytes differ")
    if config.evaluate_challenge:
        comparison = _read(root / "comparison.json")
        if comparison["selected_validation_candidate"] != "jensen_shannon":
            raise ContractError("audited Phase 3 selected reference is not Jensen-Shannon")
        baseline = next(r for r in comparison["rows"] if r["method"] == "jensen_shannon")
    else:
        # Read primary reports only. Previously viewed challenge diagnostics are
        # excluded from the controlled configuration comparison entirely.
        baseline = {"method": "jensen_shannon"}
        for split in ("train", "val"):
            primary = _read(root / f"reports/jensen_shannon_{split}_primary.json")
            baseline.update({f"{split}_roc_auc": primary["threshold_independent"]["roc_auc"]["value"],
                             f"{split}_average_precision": primary["threshold_independent"]["average_precision"]["value"],
                             f"{split}_f1": primary["classification"]["f1"]})
    expected = {"val_roc_auc": .9696418405485012, "val_average_precision": .9728053582663733,
                "val_f1": .9172932330827067}
    if config.evaluate_challenge:
        expected["val_challenge_rejection"] = .15671641791044777
    if any(not np.isclose(baseline[key], value, rtol=0, atol=1e-12) for key, value in expected.items()):
        raise ContractError("Phase 3 reference evidence differs from prespecified audited values")
    return {**baseline, "verified_same_inputs_and_pairing": True,
            "output_manifest_sha256": entry["sha256"]}


def prepare_inputs(config):
    # Read-only audited integrity helpers; no Phase 3 runner/preflight/tracker,
    # scoring API or final-test manifest reader is invoked.
    report, reference, paths, before = _verify_inputs(config)
    expected = {e["role"]: e["sha256"] for e in report["artifacts"]}
    expected.update({f"manifest/{n}": e["sha256"] for n, e in report["outputs"].items()})
    expected.update({f"reference/{n}": h for n, h in REFERENCE_HASHES.items()})
    _check_hashes(before, expected)
    baseline = _baseline_evidence(config, report, paths)
    before = _inventory(paths)
    _check_hashes(before, expected)
    roots = {"audio": config.audio_root, "vision": config.vision_root}
    with _load_exports(roots, OFFICIAL_COUNTS) as loaded:
        _verify_official(*loaded, config.official_root, OFFICIAL_COUNTS)
        enriched = []
        for bundle in loaded:
            if any(name not in (bundle.provenance.get("source") or "") for name in (
                ("beats_lora_r8_v1", "beats_iter3") if bundle.modality == "audio" else ("resnet18_v1",))):
                raise ContractError("required producer export declaration mismatch")
            files = {name: {**entry, "sha256": expected[f"{bundle.modality}/{name}"]}
                     for name, entry in bundle.provenance["files"].items()}
            enriched.append(replace(bundle, provenance={**bundle.provenance, "files": files}))
        audio, vision = enriched
        data = {}
        for split in ("train", "val"):
            document = (config.integration_root / f"{split}_pairs.json").read_text(encoding="utf-8")
            manifest = _full_manifest(document, split, audio, vision,
                                      report["artifact_inventory_fingerprint"], config.pair_seed)
            data[split] = EmbeddingDataset(manifest, audio, vision, split=split,
                                          artifact_fingerprint=report["artifact_inventory_fingerprint"])
            if split == "val":
                challenge = EmbeddingDataset(manifest, audio, vision, split="val", partition="challenge",
                                             artifact_fingerprint=report["artifact_inventory_fingerprint"])
        overlap = validate_split_isolation(data["train"], data["val"])
        if (len(data["train"]), len(data["val"]), len(challenge)) != (6678, 804, 402):
            raise ContractError("official full train/validation/challenge inventory mismatch")
        summary = {"artifact_inventory_fingerprint": report["artifact_inventory_fingerprint"],
                   "train_primary": {"count": len(data["train"]), "positives": int(data["train"].targets.sum())},
                   "val_primary": {"count": len(data["val"]), "positives": int(data["val"].targets.sum())},
                   "val_challenge": {"count": len(challenge), "positives": 0},
                   "train_validation_source_overlap": list(overlap), "all_source_overlap": report["leakage"],
                   "manifest_sha256": {s: d.provenance.manifest_sha256 for s, d in data.items()},
                   "official_reference": reference,
                   "held_out_scope": "all-split structural metadata/opaque hashes only; no test pair reader or model inference"}
    if _inventory(paths) != before:
        raise ContractError("verified inputs changed during dataset construction")
    return VerifiedData(data["train"], data["val"], challenge, paths, before, summary, baseline)


def save_checkpoint(output, model, optimizer, metadata):
    """Only our own fresh ignored output is written; payload uses safe-load types."""
    metadata.__post_init__()
    path = output / "best.pt"
    temporary = output / "best.pt.tmp"
    torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(), "epoch": metadata.epoch,
                "optimizer_steps": metadata.optimizer_steps,
                "model_sha256": model_fingerprint(model),
                "experiment_fingerprint": metadata.experiment_fingerprint}, temporary)
    temporary.replace(path)
    _write(output, "checkpoint_metadata.json", metadata.to_dict())
    evidence = _file_entry("best.pt", path)
    _write(output, "checkpoint_integrity.json", evidence)
    return evidence


def load_selected_checkpoint(output, *, expected_metadata=None):
    """Load only a verified file produced by this run, never an arbitrary path.

    Explicit weights_only=True forbids arbitrary pickle execution. This is not
    a resume API: optimizer state is recorded, but no RNG/loader position restore
    is implemented or claimed.
    """
    metadata = CheckpointMetadata.from_dict(_read(output / "checkpoint_metadata.json"))
    if expected_metadata is not None and metadata != expected_metadata:
        raise ContractError("checkpoint metadata differs from selected epoch evidence")
    expected = _read(output / "checkpoint_integrity.json")
    actual = _file_entry("best.pt", output / "best.pt")
    if any(actual[k] != expected[k] for k in ("role", "sha256", "size_bytes")):
        raise ContractError("checkpoint hash/size changed before loading")
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = LateFusionMLP(metadata.model)
    model.load_state_dict(payload["model_state"], strict=True)
    if (payload["epoch"] != metadata.selected_epoch or payload["optimizer_steps"] != metadata.optimizer_steps
            or payload["experiment_fingerprint"] != metadata.experiment_fingerprint
            or payload["model_sha256"] != model_fingerprint(model)):
        raise ContractError("checkpoint payload conflicts with metadata or model fingerprint")
    return model.eval(), metadata


def _score_csv(predictions, dataset):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(("split", "partition", "audio_sample_id", "visual_sample_id", "audio_row", "visual_row",
                     "target", "logit", "probability"))
    for pair, y, z, probability in zip(dataset.pairs, predictions.targets, predictions.logits, predictions.probabilities):
        writer.writerow((dataset.split, dataset.partition, pair.audio.record.sample_id, pair.visual.record.sample_id,
                         pair.audio.row_index, pair.visual.row_index, int(y), float(z), float(probability)))
    return stream.getvalue()


def _verify_output_manifest(output):
    manifest = _read(output / "output_manifest.json")
    if manifest.get("schema") != SCHEMA:
        raise ContractError("output evidence schema mismatch")
    names = [entry["role"] for entry in manifest["files"]]
    if len(set(names)) != len(names) or names != sorted(names):
        raise ContractError("duplicate or unordered output evidence roles")
    for entry in manifest["files"]:
        path = (output / entry["role"]).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ContractError("output evidence path escape")
        actual = _file_entry(entry["role"], path)
        if any(actual[key] != entry[key] for key in ("sha256", "size_bytes")):
            raise ContractError("saved report/checkpoint bytes differ from the reproducibility manifest")
    return manifest


def _comparison(train, validation, challenge, baseline):
    mlp = {"val_roc_auc": validation["roc_auc"], "val_average_precision": validation["average_precision"],
           "val_f1": validation["f1"]}
    if challenge is not None:
        mlp["val_challenge_rejection"] = challenge["rejection_rate"]
    deltas = {key: value-baseline[key] for key, value in mlp.items()}
    return {"baseline": baseline, "mlp": mlp, "delta_mlp_minus_baseline": deltas,
            "metric_verdict": {k: "improves" if v > 0 else "underperforms" if v < 0 else "ties" for k, v in deltas.items()},
            "train_minus_val_gap": {"mlp_roc_auc": train["roc_auc"]-validation["roc_auc"],
                "baseline_roc_auc": baseline["train_roc_auc"]-baseline["val_roc_auc"],
                "mlp_average_precision": train["average_precision"]-validation["average_precision"],
                "baseline_average_precision": baseline["train_average_precision"]-baseline["val_average_precision"],
                "mlp_f1": train["f1"]-validation["f1"], "baseline_f1": baseline["train_f1"]-baseline["val_f1"]},
            "validation_reuse": REUSE_NOTICE, "no_final_test_conclusion": True}


def run_experiment(config: MLPExperimentConfig, *, tracker_factory=ForgeMLPTracker, progress=print):
    if type(config) is not MLPExperimentConfig:
        raise ContractError("expected MLPExperimentConfig")
    git = _repository()
    source_paths, source_before = _source_inventory()
    code_sha256 = _fingerprint(_portable(source_before))
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=False)
    tracker = tracker_factory(output)
    state = {"schema": SCHEMA, "status": "RUNNING", "stage": "VERIFYING_INPUTS",
             "optimization_started": False, "completed_epochs": 0, "optimizer_steps_recorded": 0,
             "exact_resume_supported": False}
    _write(output, "state.json", state)
    history, selected_metadata = [], None
    old_threads = torch.get_num_threads()
    try:
        data = prepare_inputs(config)
        torch.set_num_threads(config.cpu_threads)
        seed_everything(config.training)
        device = resolve_device(config.training)
        model = LateFusionMLP(config.model).to(device)
        optimizer = build_optimizer(model, config.training.optimizer)
        run_config = {"schema": SCHEMA, "model": asdict(config.model), "training": asdict(config.training),
                      "parameter_count": sum(p.numel() for p in model.parameters()),
                      "initial_weights": "deterministic random initialization; no pretrained fused weights",
                      "pair_seed": config.pair_seed, "patience": config.patience, "cpu_threads": config.cpu_threads,
                      "selection_rule": "strictly greater validation-primary ROC-AUC; exact ties retain earliest epoch",
                      "epoch_diagnostic_threshold": .5, "scheduler": "none", "exact_resume_supported": False,
                      "git": git, "code_sha256": code_sha256, "source_files": _portable(source_before),
                      "exports": EXPORTS, "data": data.summary,
                      "input_hashes": _portable(data.before), "validation_reuse": REUSE_NOTICE,
                      "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                            "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda},
                      "device": str(device), "wandb_group": config.tracking_group, "wandb_name": config.run_name,
                      "challenge_evaluation": "after within-run freeze" if config.evaluate_challenge else "deferred until study selection",
                      "configuration_fingerprint": _fingerprint({"model": asdict(config.model),
                          "training": asdict(config.training), "patience": config.patience,
                          "cpu_threads": config.cpu_threads, "pair_seed": config.pair_seed,
                          "code_sha256": code_sha256, "data": data.summary})}
        _write(output, "config.json", run_config)
        _write(output, "input_verification.json", {"inputs": _portable(data.before), "data": data.summary})
        _source_archive(output, source_paths)
        state["stage"] = "FORGE_ONLINE_WRITE_PREFLIGHT"
        _write(output, "state.json", state)
        progress("Verified full TRAIN/validation primary inventories, separate challenge and original input hashes.")
        tracking = tracker.start(run_config)
        _write(output, "tracking.json", tracking)
        progress("Authenticated online Forge writes verified; beginning the single prespecified experiment.")
        # Guard sources again immediately before optimization, after network waits.
        if _inventory(data.paths) != data.before or _inventory(source_paths) != source_before:
            raise ContractError("input files or source code changed before optimization")
        loader = make_loader(data.train, config.training, role="train")
        selection = EarlyStopping(config.patience)
        overlap = validate_split_isolation(data.train, data.validation)
        for epoch in range(1, config.training.epochs + 1):
            state.update(stage="OPTIMIZING", optimization_started=True)
            _write(output, "state.json", state)
            started = time.perf_counter()
            training = train_epoch(model, loader, optimizer, config.training)
            state["stage"] = "VALIDATING_PRIMARY"
            validation_predictions = predict(model, data.validation, config.training)
            validation = primary_metrics(validation_predictions, threshold=.5)
            improved = selection.observe(epoch, validation["roc_auc"])
            steps = state["optimizer_steps_recorded"] + training["optimizer_steps"]
            row = {"epoch": epoch, "train_loss": training["loss"], "val_loss": validation["loss"],
                   "val_roc_auc": validation["roc_auc"], "val_average_precision": validation["average_precision"],
                   "val_accuracy_at_0_5": validation["accuracy"], "val_f1_at_0_5": validation["f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "gradient_norm_mean": training["gradient_norm_mean"],
                   "gradient_norm_max": training["gradient_norm_max"],
                   "duration_seconds": time.perf_counter()-started,
                   "optimizer_steps": steps, "selected": improved, "stale_epochs": selection.stale_epochs}
            history.append(row)
            _write(output, "history.json", history)
            if improved:
                selected_metadata = CheckpointMetadata(
                    model=config.model, training=config.training, git_commit=git["commit"], code_sha256=code_sha256,
                    data=(data.train.provenance, data.validation.provenance),
                    export_identities=(("audio", "beats_lora_r8_v1 / beats_iter3"), ("vision", "resnet18_v1")),
                    software_versions=(("python", platform.python_version()), ("numpy", np.__version__),
                                       ("torch", str(torch.__version__))), source_overlap=overlap,
                    epoch=epoch, optimizer_steps=steps, optimizer_state_present=True, selected_epoch=epoch,
                    validation_metrics=tuple((key, float(validation[key])) for key in
                                             ("loss", "roc_auc", "average_precision", "accuracy", "f1")))
                save_checkpoint(output, model, optimizer, selected_metadata)
                # Compare to the exact in-memory selected evaluation logits,
                # avoiding any CPU/CUDA kernel difference in this identity check.
                _write(output, "selected_validation_reference.json", {
                    "logits": validation_predictions.logits.tolist(), "model_sha256": validation_predictions.model_sha256,
                    "metrics_at_0_5": validation})
            state.update(completed_epochs=epoch, optimizer_steps_recorded=steps, stage="LOGGING_EPOCH")
            _write(output, "state.json", state)
            tracker.log_epoch(row)
            progress(f"Epoch {epoch:02d}: train loss={row['train_loss']:.6f}, val loss={row['val_loss']:.6f}, "
                     f"val AUC={row['val_roc_auc']:.6f}, AP={row['val_average_precision']:.6f}, "
                     f"selected={improved}, elapsed={row['duration_seconds']:.1f}s")
            if selection.should_stop:
                break
        state["stage"] = "RESTORING_AND_FREEZING_SELECTED_MODEL"
        _write(output, "state.json", state)
        selected, restored_metadata = load_selected_checkpoint(output, expected_metadata=selected_metadata)
        selected = selected.to(device).eval()
        selected.requires_grad_(False)
        validation_predictions = predict(selected, data.validation, config.training)
        reference = _read(output / "selected_validation_reference.json")
        expected_logits = np.array(reference["logits"], dtype=validation_predictions.logits.dtype)
        if (not np.array_equal(expected_logits, validation_predictions.logits)
                or reference["model_sha256"] != model_fingerprint(selected)):
            raise ContractError("fresh checkpoint instance did not reproduce identical selected validation logits")
        reloaded_metrics = primary_metrics(validation_predictions)
        if reloaded_metrics != reference["metrics_at_0_5"]:
            raise ContractError("selected checkpoint metrics differ after reload")
        reload_evidence = {"identical_all_validation_logits": True, "count": len(expected_logits),
                           "model_sha256": model_fingerprint(selected), "safe_load_weights_only": True,
                           "selected_epoch": restored_metadata.selected_epoch, "exact_resume_supported": False}
        _write(output, "checkpoint_reload.json", reload_evidence)
        policy = fit_operating_point(validation_predictions, selected_epoch=selection.best_epoch)
        _write(output, "frozen_operating_point.json", asdict(policy))
        _write(output, "selection.json", {"best_epoch": selection.best_epoch, "best_auc": selection.best_auc,
               "epochs_executed": len(history), "patience": config.patience,
               "tie_policy": "earliest epoch on exact validation-primary AUC tie", "min_delta": 0,
               "stop_reason": "patience_exhausted" if selection.should_stop else "maximum_epochs",
               "challenge_used_for_selection": False, "threshold_fit_count": 1,
               "validation_reuse": REUSE_NOTICE})
        state["stage"] = "FROZEN_TRAIN_VALIDATION_AND_CHALLENGE_REPORTS"
        _write(output, "state.json", state)
        train_predictions = predict(selected, data.train, config.training)
        train_report = evaluate_frozen(train_predictions, policy)
        val_report = evaluate_frozen(validation_predictions, policy)
        challenge_predictions = challenge_report = None
        if config.evaluate_challenge:
            challenge_predictions = predict(selected, data.challenge, config.training, frozen_policy=policy)
            challenge_report = evaluate_frozen(challenge_predictions, policy)
        comparison = _comparison(train_report, val_report, challenge_report, data.baseline)
        final = {"selected_epoch": selection.best_epoch, "epochs_executed": len(history),
                 "train_primary": train_report, "val_primary": val_report,
                 "model_sha256": policy.model_sha256, "frozen_threshold": policy.threshold,
                 "validation_reuse": REUSE_NOTICE, "final_test_scored": False}
        if config.evaluate_challenge:
            final["val_challenge"] = challenge_report
        _write(output, "final_metrics.json", final)
        _write(output, "baseline_comparison.json", comparison)
        scores = [("train_primary", train_predictions, data.train), ("val_primary", validation_predictions, data.validation)]
        if config.evaluate_challenge:
            scores.append(("val_challenge", challenge_predictions, data.challenge))
        for name, predictions, dataset in scores:
            _write(output, f"{name}_scores.csv", _score_csv(predictions, dataset))
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(history)
        _write(output, "history.csv", stream.getvalue())
        if _inventory(data.paths) != data.before or _inventory(source_paths) != source_before:
            raise ContractError("producer inputs, manifests, baseline evidence or source code changed during training")
        _write(output, "post_verification.json", {"original_input_hashes_unchanged": True,
                                                "source_code_unchanged": True, "final_test_scored": False})
        names = sorted(p.name for p in output.iterdir() if p.is_file() and p.name not in ("state.json", "tracking.json"))
        manifest = {"schema": SCHEMA, "files": _portable(_inventory({n: output/n for n in names}))}
        _write(output, "output_manifest.json", manifest)
        _verify_output_manifest(output)
        manifest_sha = _file_entry("output_manifest.json", output / "output_manifest.json")["sha256"]
        scalars = {"final/selected_epoch": selection.best_epoch, "final/epochs_executed": len(history),
                   "final/frozen_threshold": policy.threshold,
                   "final/val_roc_auc": val_report["roc_auc"], "final/val_average_precision": val_report["average_precision"],
                   "final/val_f1": val_report["f1"], "final/val_accuracy": val_report["accuracy"],
                   "final/val_loss": val_report["loss"], "final/train_loss": train_report["loss"],
                   "final/training_validation_duration_seconds": sum(r["duration_seconds"] for r in history),
                   "final/checkpoint_sha256": _read(output / "checkpoint_integrity.json")["sha256"],
                   **{f"final/val_{key}": count for key, count in val_report["counts"].items()}}
        if config.evaluate_challenge:
            scalars.update({"final/val_challenge_rejection": challenge_report["rejection_rate"],
                            "final/val_challenge_false_acceptance": challenge_report["false_acceptance_rate"]})
        state["stage"] = "UPLOADING_AND_VERIFYING_FORGE_EVIDENCE"
        _write(output, "state.json", state)
        def verify_before_finalize():
            if _inventory(data.paths) != data.before or _inventory(source_paths) != source_before:
                raise ContractError("inputs or source code changed during final artifact upload")
            _verify_output_manifest(output)
            if _file_entry("output_manifest.json", output / "output_manifest.json")["sha256"] != manifest_sha:
                raise ContractError("output manifest changed during final artifact upload")

        tracking = tracker.finish(
            scalars, [*(output/n for n in names), output / "output_manifest.json"], history, manifest_sha,
            before_finalize=verify_before_finalize)
        if tracking.get("status") != "COMPLETE":
            raise TrackingUnavailable("Forge tracker did not confirm complete remote evidence")
        _write(output, "tracking.json", tracking)
        # The completion marker is published only after successful uploads,
        # remote evidence and local input/output identity checks, then SDK success.
        state.update(status="COMPLETE", stage="COMPLETE", tracking_status="COMPLETE",
                     selected_epoch=selection.best_epoch, output_manifest_sha256=manifest_sha)
        _write(output, "state.json", state)
        _write(output, "completion.json", state)
        progress(f"COMPLETE: selected epoch {selection.best_epoch}; Forge evidence uploaded and read back.")
        return {"status": "COMPLETE", "final": final, "comparison": comparison, "tracking": tracking}
    except BaseException as exc:
        tracker.abort()
        blocker = str(exc) if type(exc) is TrackingUnavailable else safe_failure(state["stage"], exc)
        state.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED", blocker=blocker,
                     tracking_status="INCOMPLETE")
        _write(output, "state.json", state)
        _write(output, "tracking.json", {**tracker.metadata, "status": "INCOMPLETE", "blocker": blocker})
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise ContractError(blocker) from None
    finally:
        torch.set_num_threads(old_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio-root", "vision-root", "official-root", "reference-root", "reference-evidence",
                 "integration-root", "baseline-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=.2)
    args = parser.parse_args(argv)
    try:
        config = MLPExperimentConfig(**{k: getattr(args, k) for k in (
            "audio_root", "vision_root", "official_root", "reference_root", "reference_evidence",
            "integration_root", "baseline_root", "output_dir", "patience", "cpu_threads", "pair_seed")},
            model=MLPConfig(dropout=args.dropout, initialization_seed=args.seed),
            training=TrainingConfig(seed=args.seed, batch_size=args.batch_size, epochs=args.epochs,
                                    selection_metric="roc_auc", device=args.device,
                                    optimizer=OptimizerConfig(learning_rate=args.learning_rate, weight_decay=args.weight_decay)))
        run_experiment(config, progress=lambda message: print(message, flush=True))
    except KeyboardInterrupt:
        print("INTERRUPTED: failed-run evidence preserved; no completion published.", file=sys.stderr)
        return 130
    except Exception as exc:
        # No arbitrary SDK/input exception text, environment dump or credential.
        print(safe_failure("Phase 4 experiment", exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

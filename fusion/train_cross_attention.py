"""One preregistered Phase 5 experiment; TRAIN/val only, no challenge inference."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import ClassVar

import numpy as np
import torch

from .artifact_integration import _annotation_lines, _check_hashes, _export_paths, _file_entry, _inventory, _json
from .baseline_tracking import safe_failure
from .contracts import ContractError, OFFICIAL_COUNTS, Record, load_bundle
from .cross_attention_model import CrossAttentionConfig
from .cross_attention_tracking import ForgeCrossAttentionTracker, GROUP
from .embedding_data import EmbeddingDataset
from .mlp_study import primary_evidence
from .train_mlp import (
    ROOT, REUSE_NOTICE, VerifiedData, _baseline_evidence, _portable, _read,
    _run_experiment, _verify_output_manifest, _write,
)
from .training_contracts import OptimizerConfig, TrainingConfig, _fingerprint, validate_split_isolation
from .verify_ave_reference import REFERENCE_HASHES, _output_path

SCHEMA = "gc2-real-cross-attention-experiment-v1"


def initial_training(device="cpu"):
    return TrainingConfig(seed=42, batch_size=64, epochs=30, device=device, selection_metric="roc_auc",
                          optimizer=OptimizerConfig(learning_rate=.0003, weight_decay=.01))


@dataclass(frozen=True)
class CrossAttentionExperimentConfig:
    model_config_types: ClassVar[tuple] = (CrossAttentionConfig,)
    tracking_groups: ClassVar[tuple] = (GROUP,)
    audio_root: Path
    vision_root: Path
    official_root: Path
    reference_root: Path
    reference_evidence: Path
    integration_root: Path
    baseline_root: Path
    mlp_reference_root: Path
    output_dir: Path
    model: CrossAttentionConfig = CrossAttentionConfig()
    training: TrainingConfig = initial_training()
    patience: int = 5
    cpu_threads: int = 4
    pair_seed: int = 42
    evaluate_challenge: bool = False
    tracking_group: str = GROUP
    run_name: str = "phase5_cross_attention_initial_seed42"

    def __post_init__(self):
        names = ("audio_root", "vision_root", "official_root", "reference_root", "reference_evidence",
                 "integration_root", "baseline_root", "mlp_reference_root", "output_dir")
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ContractError("explicit experiment paths are required")
            object.__setattr__(self, name, Path(value).resolve())
        if type(self.model) not in self.model_config_types or type(self.training) is not TrainingConfig:
            raise ContractError("strict attention model and training configurations required")
        self.model.__post_init__(); self.training.__post_init__()
        if (self.training.selection_metric != "roc_auc" or self.evaluate_challenge is not False
                or self.tracking_group not in self.tracking_groups or self.pair_seed != 42):
            raise ContractError("Phase 5 requires primary AUC selection, seed-42 manifests and no challenge evaluation")
        for name in ("patience", "cpu_threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ContractError(f"{name} must be a positive integer")
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise ContractError("explicit run name required")
        _output_path(self.output_dir, tuple(getattr(self, name) if name != "reference_evidence"
                                          else self.reference_evidence.parent for name in names[:-1]))

    @property
    def fingerprint(self):
        # Fresh-output validation belongs to construction, not repeated reads
        # after the output directory exists. Do still revalidate numeric contracts.
        if (type(self.model) not in self.model_config_types or type(self.training) is not TrainingConfig
                or self.evaluate_challenge is not False or self.tracking_group not in self.tracking_groups or self.pair_seed != 42):
            raise ContractError("invalid Phase 5 experiment configuration")
        self.model.__post_init__(); self.training.__post_init__()
        return _fingerprint({key: str(value) if isinstance(value, Path) else value
                             for key, value in asdict(self).items()})


def repository():
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()
    if git("remote", "get-url", "origin").removesuffix(".git").rstrip("/") != "https://github.com/NUS-AIS-GCs/Audio-Video-Confidence":
        raise ContractError("required GC2 repository mismatch")
    if git("branch", "--show-current") != "feature/fusion-phase5-cross-attention":
        raise ContractError("required Phase 5 branch mismatch")
    dependency = git("rev-parse", "feature/fusion-phase4-mlp")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", dependency, "HEAD"]).returncode:
        raise ContractError("Phase 4 dependency is not in current branch")
    return {"repository": "NUS-AIS-GCs/Audio-Video-Confidence", "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"), "phase4_dependency": dependency,
            "working_tree": git("status", "--porcelain").splitlines(),
            "dirty_source_policy": "preserve existing work; exact source hashes and source archive"}


def _verify_rows(path, dataset):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(dataset):
        raise ContractError("historical primary row count differs")
    for row, pair in zip(rows, dataset.pairs):
        if (row["split"] != dataset.split or row["partition"] != "primary"
                or row["audio_sample_id"] != pair.audio.record.sample_id
                or row["visual_sample_id"] != pair.visual.record.sample_id
                or int(row["audio_row"]) != pair.audio.row_index or int(row["visual_row"]) != pair.visual.row_index
                or int(row["target"]) != pair.original_pair_target):
            raise ContractError("historical primary source rows or targets differ")


def prepare_inputs(config):
    """Hash held-out files opaquely; parse producer labels/arrays for TRAIN/val only.

    The pinned original reference verification is rebound by exact byte hashes.
    No final-test annotation, HDF5 order, manifest or prediction parser runs.
    Challenge pair records are structurally regenerated, never feature-gathered.
    """
    if (config.model != CrossAttentionConfig() or config.training != initial_training(config.training.device)
            or config.patience != 5):
        raise ContractError("production entry point permits only the prespecified initial experiment")
    return _prepare_verified_inputs(config)


def _prepare_verified_inputs(config):
    """Shared immutable TRAIN/val input and reference verification, no optimization."""
    report = _read(config.integration_root / "validation.json")
    if (report.get("schema") != "gc2-artifact-integration-v1" or report.get("status") != "COMPLETE"
            or report.get("binary_validation") != "VALIDATED_LOCAL_FILES"
            or report.get("official_metadata") != "MATCHES_SUPPLIED_OFFICIAL_REFERENCE_FILES" or report.get("seed") != 42):
        raise ContractError("complete original Phase 2 evidence required")
    paths, missing = _export_paths({"audio": config.audio_root, "vision": config.vision_root})
    if missing:
        raise ContractError("producer exports missing")
    paths.update({entry["role"]: config.official_root / entry["role"].split("/", 1)[1]
                  for entry in report["artifacts"] if entry["role"].startswith("official/")})
    expected = {entry["role"]: entry["sha256"] for entry in report["artifacts"]}
    inventory = _inventory(paths)
    if set(paths) != set(expected):
        raise ContractError("original source inventory coverage differs")
    _check_hashes(inventory, expected)
    if (any(e["size_bytes"] != next(old["size_bytes"] for old in report["artifacts"] if old["role"] == e["role"])
            for e in inventory) or hashlib.sha256(_json(report["artifacts"]).encode()).hexdigest()
            != report["artifact_inventory_fingerprint"]):
        raise ContractError("source inventory sizes/fingerprint differ")
    for split in ("train", "val", "test"):
        name = f"{split}_pairs.json"
        paths[f"manifest/{name}"] = config.integration_root / name
        entry = _file_entry(f"manifest/{name}", paths[f"manifest/{name}"])
        if any(entry[k] != report["outputs"][name][k] for k in ("size_bytes", "sha256")):
            raise ContractError("original manifest bytes changed")
        expected[entry["role"]] = entry["sha256"]
    reference = _read(config.reference_evidence)
    if reference.get("status") != "PASS" or reference.get("schema") != "gc2-ave-reference-v1":
        raise ContractError("pinned official-reference evidence missing")
    for role, evidence in reference["inputs"].items():
        side, name = role.split("/", 1)
        root = config.reference_root if side == "author" else config.official_root
        entry = _file_entry(f"reference/{role}", root / name)
        if any(entry[k] != evidence[k] for k in ("size_bytes", "sha256")):
            raise ContractError("official reference bytes changed")
        if side == "author" and entry["sha256"] != REFERENCE_HASHES[name]:
            raise ContractError("pinned author reference mismatch")
        paths[entry["role"]] = root / name
    if {r.split("/", 1)[1] for r in reference["inputs"] if r.startswith("author/")} != set(REFERENCE_HASHES):
        raise ContractError("pinned reference coverage missing")
    paths.update({"evidence/phase2_validation.json": config.integration_root / "validation.json",
                  "evidence/reference_verification.json": config.reference_evidence})
    jsd = _baseline_evidence(config, report, paths)
    _verify_output_manifest(config.mlp_reference_root)
    e3 = primary_evidence("E3", config.mlp_reference_root)
    if any(not np.isclose(e3[key], expected_value, rtol=0, atol=5e-10) for key, expected_value in (
            ("val_roc_auc", .970019306), ("val_average_precision", .968596681), ("val_f1", .913669065))):
        raise ContractError("historical E3 primary metrics differ from audited reference")
    for entry in _read(config.mlp_reference_root / "output_manifest.json")["files"]:
        paths[f"phase4_e3/{entry['role']}"] = config.mlp_reference_root / entry["role"]
    for name in ("output_manifest.json", "completion.json", "tracking.json"):
        paths[f"phase4_e3/{name}"] = config.mlp_reference_root / name
    before = _inventory(paths)
    bundles = []
    try:
        for modality, root in (("audio", config.audio_root), ("vision", config.vision_root)):
            bundle = load_bundle(root, modality, splits=("train", "val"), expected_counts=OFFICIAL_COUNTS)
            bundles.append(bundle)
            required = ("beats_lora_r8_v1", "beats_iter3") if modality == "audio" else ("resnet18_v1",)
            if any(name not in (bundle.provenance.get("source") or "") for name in required):
                raise ContractError("producer declaration mismatch")
            for split, name in (("train", "trainSet.txt"), ("val", "valSet.txt")):
                rows = []
                for category, clip, _, start, end in _annotation_lines(config.official_root / name):
                    label = bundle.class_names.index(category)
                    rows.append(Record(f"{clip}__{label:02d}", clip, category, int(start), int(end), label))
                if ({r.sample_id: r for r in rows} != {r.sample_id: r for r in bundle.metadata[split]}
                        or len(rows) != OFFICIAL_COUNTS[split]):
                    raise ContractError("TRAIN/val official source records differ")
                source = bundle.splits[split]
                dim = 256 if modality == "audio" else 512
                if (source.embeddings.shape != (OFFICIAL_COUNTS[split], 10, dim)
                        or str(source.embeddings.dtype) != "float16"):
                    raise ContractError("producer embedding schema changed")
            files = {name: {**entry, "sha256": expected[f"{modality}/{name}"]}
                     for name, entry in bundle.provenance["files"].items()}
            bundles[-1] = replace(bundle, provenance={**bundle.provenance, "files": files})
        audio, vision = bundles
        data = {}
        for split in ("train", "val"):
            name = f"{split}_pairs.json"
            data[split] = EmbeddingDataset.from_scoped_json(
                (config.integration_root / name).read_text(encoding="utf-8"), audio, vision, split=split,
                expected_manifest_sha256=report["outputs"][name]["sha256"],
                artifact_fingerprint=report["artifact_inventory_fingerprint"])
            if len(data[split]) != 2 * OFFICIAL_COUNTS[split] or int(data[split].targets.sum()) != OFFICIAL_COUNTS[split]:
                raise ContractError("complete primary pairing distribution required")
            if (data[split].provenance.manifest_sha256 != e3["manifest_sha256"][split]
                    or report["artifact_inventory_fingerprint"] != e3["artifact_inventory_fingerprint"]):
                raise ContractError("Phase 5 and E3 use different original inputs/manifests")
            _verify_rows(config.mlp_reference_root / f"{split}_primary_scores.csv", data[split])
            _verify_rows(config.baseline_root / f"scores/{split}_primary.csv", data[split])
        overlap = validate_split_isolation(data["train"], data["val"])
        summary = {"artifact_inventory_fingerprint": report["artifact_inventory_fingerprint"],
                   "manifest_sha256": {s: d.provenance.manifest_sha256 for s, d in data.items()},
                   "train_primary": {"count": len(data["train"]), "positives": int(data["train"].targets.sum())},
                   "val_primary": {"count": len(data["val"]), "positives": int(data["val"].targets.sum())},
                   "validation_challenge": {"manifest_count": 402, "features_gathered": False, "evaluated": False},
                   "train_validation_source_overlap": list(overlap),
                   "held_out_scope": "opaque integrity hashes only; no test labels, arrays, pairs or predictions parsed",
                   "scoped_manifest_context": "all-split leakage context opaque; every pair/local field regenerated exactly",
                   "original_row_references_match_e3_and_jsd": True}
    finally:
        for bundle in bundles:
            for source in bundle.splits.values():
                for name in ("embeddings", "probabilities", "input_validity", "timestamps"):
                    mapping = getattr(getattr(source, name), "_mmap", None)
                    if mapping is not None:
                        mapping.close()
    if before != _inventory(paths):
        raise ContractError("inputs changed while constructing scoped datasets")
    return VerifiedData(data["train"], data["val"], None, paths, before, summary, {"e3": e3, "jsd": jsd})


def freeze_run(config, run_config, *, registration_extra=None, external_guard=None):
    initial = config.fingerprint
    run_fingerprint = _fingerprint(run_config)
    config_hash = _file_entry("config", config.output_dir / "config.json")["sha256"]
    registration = {"schema": "gc2-cross-attention-preregistration-v1", "max_real_attempts": 1,
                    "architecture_ablation_or_search": False, "challenge_evaluation": False,
                    "test_access": "opaque integrity hashes only", "configuration_fingerprint": initial,
                    "run_configuration_fingerprint": run_fingerprint,
                    "configuration": run_config,
                    "threshold_policy": "once after checkpoint freeze; maximum primary-validation F1, largest-threshold ties",
                    "epoch_f1_policy": "fixed diagnostic threshold 0.5; no epoch threshold optimization"}
    if registration_extra is not None:
        registration.update(registration_extra)
    _write(config.output_dir, "preregistration.json", registration)
    plan_hash = _file_entry("plan", config.output_dir / "preregistration.json")["sha256"]
    def guard():
        if external_guard is not None:
            external_guard()
        if (config.fingerprint != initial or _fingerprint(run_config) != run_fingerprint
                or _file_entry("config", config.output_dir / "config.json")["sha256"] != config_hash
                or _file_entry("plan", config.output_dir / "preregistration.json")["sha256"] != plan_hash):
            raise ContractError("frozen experiment configuration changed")
    guard()
    return guard


def comparison(train, validation, challenge, references):
    if challenge is not None:
        raise ContractError("initial experiment prohibits challenge diagnostics")
    current = {"val_roc_auc": validation["roc_auc"], "val_average_precision": validation["average_precision"],
               "val_f1": validation["f1"]}
    return {"cross_attention": current, "references": references,
            "delta_cross_attention_minus_reference": {
                name: {key: current[key] - reference[key] for key in current}
                for name, reference in references.items()},
            "definitions": "primary pairing provenance; higher means original; AP noninterpolated; binary F1; probability >= frozen model-specific threshold",
            "challenge_evaluated": False, "final_test_scored": False,
            "validation_reuse": REUSE_NOTICE, "one_seed_statistical_superiority_claim": False}


def run_experiment(config, *, tracker_factory=ForgeCrossAttentionTracker, progress=print, data_factory=None):
    if type(config) is not CrossAttentionExperimentConfig:
        raise ContractError("expected CrossAttentionExperimentConfig")
    config.fingerprint
    return _run_experiment(config, tracker_factory=tracker_factory, progress=progress, repository=repository,
                           data_factory=data_factory or prepare_inputs, comparison_factory=comparison,
                           schema=SCHEMA, freeze_run=freeze_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio-root", "vision-root", "official-root", "reference-root", "reference-evidence",
                 "integration-root", "baseline-root", "mlp-reference-root", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        config = CrossAttentionExperimentConfig(**{key: getattr(args, key) for key in (
            "audio_root", "vision_root", "official_root", "reference_root", "reference_evidence", "integration_root",
            "baseline_root", "mlp_reference_root", "output_dir")}, training=initial_training(args.device))
        run_experiment(config, progress=lambda message: print(message, flush=True))
    except KeyboardInterrupt:
        print("INTERRUPTED: failed-run evidence preserved; no completion published.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(safe_failure("Phase 5 experiment", exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

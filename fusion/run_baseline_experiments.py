"""Verified train/validation baselines and mandatory Forge tracking; never test scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifact_integration import (
    OFFICIAL_FILES, _check_hashes, _export_paths, _file_entry, _inventory, _json, _load_exports,
    _read_json, _verify_official, manifest_from_json,
)
from .baseline_data import extract_pair_probabilities
from .baseline_evaluation import (
    evaluate_challenge, evaluate_primary, evaluation_report_to_json,
    frozen_policy_to_json, threshold_sensitivity,
)
from .baseline_metrics import make_score_records
from .baseline_protocol import validate_score_dataset
from .baseline_thresholds import freeze_threshold, select_thresholds
from .baseline_tracking import BASE_URL, ENTITY, GROUP, PROJECT, ForgeTracker, TrackingUnavailable, safe_failure
from .baselines import AGGREGATION, METHODS
from .contracts import OFFICIAL_COUNTS, ContractError
from .pair_contracts import PairType, SamplingPolicy
from .verify_ave_reference import REFERENCE_HASHES, _output_path, verify_reference

ROOT = Path(__file__).resolve().parents[1]
SCORED_SPLITS = ("train", "val")
SCHEMA = "gc2-real-baselines-v1"
SELECTION_RULE = "validation ROC-AUC, then noninterpolated AP, then frozen-threshold F1, then fixed method order"
REUSE_NOTICE = "Validation primary pairs select thresholds and report metrics; estimates are not unbiased held-out performance."
EXPORTS = {"audio": {"model": "BEATs+LoRA", "run": "beats_lora_r8_v1", "features": "beats_iter3"},
           "vision": {"model": "ResNet-18", "run": "resnet18_v1"}}


@dataclass(frozen=True)
class ExperimentConfig:
    audio_root: Path
    vision_root: Path
    official_root: Path
    reference_root: Path
    reference_evidence: Path
    integration_root: Path
    output_dir: Path
    seed: int = 42

    def __post_init__(self):
        SamplingPolicy(self.seed)
        for name in ("audio_root", "vision_root", "official_root", "reference_root", "reference_evidence",
                     "integration_root", "output_dir"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ContractError(f"{name}: explicit nonempty path required")
            object.__setattr__(self, name, Path(value).resolve())
        inputs = (self.audio_root, self.vision_root, self.official_root, self.reference_root,
                  self.reference_evidence.parent, self.integration_root)
        _output_path(self.output_dir, inputs)


def _git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _repository():
    origin = _git("remote", "get-url", "origin")
    if origin.rstrip("/").removesuffix(".git") not in (
        "https://github.com/NUS-AIS-GCs/Audio-Video-Confidence", "git@github.com:NUS-AIS-GCs/Audio-Video-Confidence"
    ):
        raise ContractError("repository origin is not the required GC2 project")
    if _git("branch", "--show-current") != "feature/fusion-phase3-baselines":
        raise ContractError("required branch is feature/fusion-phase3-baselines")
    if _git("diff", "--name-only") or _git("diff", "--cached", "--name-only"):
        raise ContractError("tracked working tree/index must be clean")
    return {"repository": "NUS-AIS-GCs/Audio-Video-Confidence", "commit": _git("rev-parse", "HEAD"),
            "branch": "feature/fusion-phase3-baselines", "working_tree": _git("status", "--porcelain").splitlines(),
            "phase2_dependency": "PR #13 pending until independently confirmed merged"}


def _portable_inventory(entries):
    return [{key: value for key, value in item.items() if key != "path"} for item in entries]


def _read(path):
    return _read_json(path.read_text(encoding="utf-8"))


def _verify_inputs(config):
    report_path = config.integration_root / "validation.json"
    report = _read(report_path)
    if (report.get("schema") != "gc2-artifact-integration-v1" or report.get("status") != "COMPLETE"
            or report.get("binary_validation") != "VALIDATED_LOCAL_FILES"
            or report.get("official_metadata") != "MATCHES_SUPPLIED_OFFICIAL_REFERENCE_FILES"
            or report.get("seed") != config.seed):
        raise ContractError("missing/incompatible Phase 2 completion evidence")
    roots = {"audio": config.audio_root, "vision": config.vision_root}
    paths, missing = _export_paths(roots)
    if missing:
        raise ContractError("required producer exports are missing")
    paths.update({f"official/{name}": config.official_root / name for name in OFFICIAL_FILES})
    artifacts = _inventory(paths)
    expected = {entry["role"]: entry["sha256"] for entry in report["artifacts"]}
    if set(expected) != {e["role"] for e in artifacts}:
        raise ContractError("Phase 2 inventory coverage differs from current inputs")
    _check_hashes(artifacts, expected)
    context_entries = []
    for entry in artifacts:
        old = next(item for item in report["artifacts"] if item["role"] == entry["role"])
        if entry["size_bytes"] != old["size_bytes"]:
            raise ContractError("Phase 2 input size mismatch")
        # Rebind the original integration's shape/dtype inventory context while
        # allowing portable current input roots. Arrays are freshly validated below.
        context_entries.append(old)
    if hashlib.sha256(_json(context_entries).encode()).hexdigest() != report["artifact_inventory_fingerprint"]:
        raise ContractError("Phase 2 artifact inventory fingerprint is inconsistent")
    for split in ("train", "val", "test"):
        name = f"{split}_pairs.json"
        entry = _file_entry(f"manifest/{name}", config.integration_root / name)
        if any(entry[k] != report["outputs"][name][k] for k in ("sha256", "size_bytes")):
            raise ContractError(f"production {split} manifest hash/size mismatch")
        paths[entry["role"]] = config.integration_root / name
    fresh = verify_reference(config.official_root, config.reference_root)
    if fresh != _read(config.reference_evidence):
        raise ContractError("fresh pinned reference verification differs from saved verified evidence")
    for name in REFERENCE_HASHES:
        paths[f"reference/{name}"] = config.reference_root / name
    paths.update({"evidence/reference_verification.json": config.reference_evidence,
                  "evidence/phase2_validation.json": report_path})
    return report, fresh, paths, _inventory(paths)


def _full_manifest(text, split, audio, vision, context, seed):
    raw = _read_json(text)
    if raw.get("config", {}).get("split") != split or split not in SCORED_SPLITS:
        raise ContractError("manifest is not the requested train/validation split")
    result = manifest_from_json(text, audio, vision, expected_artifact_fingerprint=context)
    n = OFFICIAL_COUNTS[split]
    counts = {kind: sum(p.pair_type is kind for p in result.primary_pairs + result.challenge_pairs) for kind in PairType}
    originals = {p.audio.record.sample_id for p in result.primary_pairs if p.pair_type is PairType.ORIGINAL_POSITIVE}
    if (result.policy.seed != seed or counts != dict(zip(PairType, (n, n, 0, n)))
            or originals != {r.sample_id for r in audio.splits[split].records}):
        raise ContractError("full original/easy/challenge coverage and matching seed are required")
    return result


def rank_methods(rows):
    if len(rows) != len(METHODS) or {r["method"] for r in rows} != set(METHODS):
        raise ContractError("comparison requires exactly the four baseline methods")
    for row in rows:
        if any(isinstance(row[k], bool) or not isinstance(row[k], (int, float)) or not np.isfinite(row[k]) or not 0 <= row[k] <= 1
               for k in ("val_roc_auc", "val_average_precision", "val_f1")):
            raise ContractError("primary selection metrics must be defined finite values in [0,1]")
    return tuple(sorted(rows, key=lambda r: (-r["val_roc_auc"], -r["val_average_precision"],
                                             -r["val_f1"], METHODS.index(r["method"]))))


def _score_csv(dataset):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["split", "partition", "audio_sample_id", "visual_sample_id", "audio_row", "visual_row",
                     "target", "pair_type", *METHODS])
    for row in dataset.records:
        writer.writerow([dataset.split, dataset.partition, row.pair.audio.record.sample_id,
                         row.pair.visual.record.sample_id, row.pair.audio.row_index, row.pair.visual.row_index,
                         row.target, row.pair_type.value, *(format(s, ".17g") for _, s in row.scores)])
    return stream.getvalue()


def _compute(config, report, audio, vision, progress):
    datasets, documents = {}, {}
    for split in SCORED_SPLITS:
        progress(f"Validating complete {split} manifest through Phase 2 reader.")
        manifest = _full_manifest((config.integration_root / f"{split}_pairs.json").read_text(encoding="utf-8"),
                                  split, audio, vision, report["artifact_inventory_fingerprint"], config.seed)
        data = extract_pair_probabilities(manifest, audio, vision, split=split)
        progress(f"Scoring complete {split} primary and separate challenge partitions.")
        for partition in ("primary", "challenge"):
            ds = validate_score_dataset(make_score_records(getattr(data, partition)), manifest, audio, vision,
                split=split, partition=partition, source_reference=f"production-{split}-pairs/full-{partition}",
                artifact_inventory_fingerprint=report["artifact_inventory_fingerprint"])
            datasets[split, partition] = ds
            documents[f"scores/{split}_{partition}.csv"] = _score_csv(ds)
    policies = select_thresholds(datasets["val", "primary"])
    # Freeze every method before any primary or challenge operating-point report.
    frozen = tuple(freeze_threshold(policy) for policy in policies)
    progress("All four validation-primary thresholds selected and frozen.")
    rows, logged, reports = [], {}, {}
    for policy in frozen:
        method = policy.policy.method
        documents[f"policies/{method}.json"] = frozen_policy_to_json(policy)
        documents[f"sensitivity/{method}.json"] = _json(threshold_sensitivity(policy.policy))
        row = {"method": method, "threshold": policy.policy.threshold}
        for split in SCORED_SPLITS:
            progress(f"Producing {method} {split} primary/challenge reports.")
            primary = evaluate_primary(datasets[split, "primary"], policy)
            challenge = evaluate_challenge(datasets[split, "challenge"], policy)
            documents[f"reports/{method}_{split}_primary.json"] = evaluation_report_to_json(primary)
            documents[f"reports/{method}_{split}_challenge.json"] = evaluation_report_to_json(challenge)
            classification = primary.classification
            row.update({f"{split}_roc_auc": primary.threshold_independent.roc_auc.value,
                        f"{split}_average_precision": primary.threshold_independent.average_precision.value,
                        f"{split}_f1": classification.f1,
                        f"{split}_challenge_rejection": challenge.rejection_rate})
            for name, value in vars(classification).items():
                if name == "counts":
                    for count, number in vars(value).items():
                        logged[f"{method}/{split}/{count}"] = number
                elif value is not None:
                    logged[f"{method}/{split}/{name}"] = value
            logged.update({f"{method}/{split}/roc_auc": row[f"{split}_roc_auc"],
                           f"{method}/{split}/average_precision_noninterpolated": row[f"{split}_average_precision"],
                           f"{method}/{split}/challenge_count": challenge.count,
                           f"{method}/{split}/challenge_acceptance": challenge.false_acceptance_rate,
                           f"{method}/{split}/challenge_rejection": challenge.rejection_rate})
            reports[method, split] = primary
        logged[f"{method}/frozen_threshold"] = policy.policy.threshold
        rows.append(row)
    ranked = rank_methods(rows)
    winner = ranked[0]["method"]
    comparison = {"rule": SELECTION_RULE, "fixed_method_order": METHODS,
                  "selected_validation_candidate": winner, "validation_reuse": REUSE_NOTICE,
                  "no_final_test_conclusion": True, "rows": rows}
    documents["comparison.json"] = _json(comparison)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    documents["comparison.csv"] = stream.getvalue()
    logged["comparison/selected_validation_candidate"] = winner
    logged["validation_reuse"] = REUSE_NOTICE
    metadata = {"annotation_counts": {s: len(audio.metadata[s]) for s in SCORED_SPLITS},
                "pair_counts": {s: {p: len(datasets[s, p].records) for p in ("primary", "challenge")} for s in SCORED_SPLITS},
                "score_dataset_fingerprints": {f"{s}/{p}": d.fingerprint for (s, p), d in datasets.items()},
                "source_overlap": datasets["val", "primary"].provenance.source_overlaps,
                "pair_generation": {s: datasets[s, "primary"].provenance.generation for s in SCORED_SPLITS},
                "warnings": reports[winner, "val"].context.warnings, "all_scores_finite": True}
    return documents, rows, logged, comparison, metadata


def _environment():
    versions = {}
    for name in ("numpy", "wandb", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed in active import path"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def _source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "fusion").glob("*.py"))}


def _write(output, name, text):
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))


def _expected_files():
    return sorted(["comparison.json", "comparison.csv", "config.json", "environment.json",
                   "reference_verification.json", "input_verification.json", "dataset_summary.json", "wandb_metrics.json"]
                  + [f"scores/{s}_{p}.csv" for s in SCORED_SPLITS for p in ("primary", "challenge")]
                  + [f"{folder}/{m}.json" for m in METHODS for folder in ("policies", "sensitivity")]
                  + [f"reports/{m}_{s}_{p}.json" for m in METHODS for s in SCORED_SPLITS for p in ("primary", "challenge")])


def _verified_core_files(output):
    manifest = _read(output / "output_manifest.json")
    names = [entry["role"] for entry in manifest["files"]]
    if manifest.get("schema") != SCHEMA or names != _expected_files():
        raise ContractError("saved evidence must contain exactly the canonical experiment files")
    for entry in manifest["files"]:
        path = (output / entry["role"]).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ContractError("artifact path escapes the ignored experiment directory")
        actual = _file_entry(entry["role"], path)
        if any(actual[k] != entry[k] for k in ("sha256", "size_bytes")):
            raise ContractError("saved experiment evidence hash mismatch; upload refused")
    return names


def _upload(output, tracker, core_files, metrics, rows):
    if core_files != _verified_core_files(output):
        raise ContractError("artifact file list differs from verified output evidence")
    digest = hashlib.sha256((output / "output_manifest.json").read_bytes()).hexdigest()
    try:
        tracking = tracker.upload(metrics, rows, [*(output / name for name in core_files), output / "output_manifest.json"], digest)
    except Exception as exc:
        tracker.abort()
        tracking = dict(tracker.metadata, status="BLOCKED_WANDB_UPLOAD",
                        blocker=str(exc) if type(exc) is TrackingUnavailable else safe_failure("Forge upload", exc))
    _write(output, "tracking.json", _json(tracking))
    return tracking


def run_experiments(config: ExperimentConfig, *, tracker_factory=ForgeTracker, progress=None):
    progress = progress or (lambda message: None)
    ExperimentConfig.__post_init__(config)
    git = _repository()
    report, reference, input_paths, before = _verify_inputs(config)
    progress("Input hashes, production manifests and pinned AVE reference evidence verified.")
    source_hashes = _source_hashes()
    def input_name(path):
        return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)

    run_config = {"schema": SCHEMA, "seed": int(config.seed), "methods": METHODS, "aggregation": AGGREGATION,
                  "inputs": {name: input_name(getattr(config, name)) for name in
                             ("audio_root", "vision_root", "official_root", "reference_root", "reference_evidence", "integration_root")},
                  "scored_splits": SCORED_SPLITS, "exports": EXPORTS, "git": git, "environment": _environment(),
                  "source_code_sha256": source_hashes, "selection_rule": SELECTION_RULE,
                  "validation_reuse": REUSE_NOTICE, "threshold_objective": "maximum_validation_binary_f1",
                  "learning_rate": "not applicable", "epochs": "not applicable", "loss_curves": "not applicable",
                  "trained_checkpoint": "not applicable", "wandb": {"base_url": BASE_URL, "entity": ENTITY,
                                                                        "project": PROJECT, "group": GROUP}}
    config.output_dir.mkdir(parents=True, exist_ok=False)
    tracker = tracker_factory(config.output_dir)
    tracking_blocker = None
    try:
        tracker.start(run_config)
        progress("Online CoreWeave Forge comparison run started.")
    except Exception as exc:
        tracker.abort()
        tracking_blocker = str(exc) if type(exc) is TrackingUnavailable else safe_failure("Forge initialization", exc)
        progress("Forge unavailable; continuing local evaluation with BLOCKED_WANDB_UPLOAD status.")
    try:
        with _load_exports({"audio": config.audio_root, "vision": config.vision_root}, OFFICIAL_COUNTS) as (audio, vision):
            _verify_official(audio, vision, config.official_root, OFFICIAL_COUNTS)
            if (any(value not in (audio.provenance.get("source") or "") for value in ("beats_lora_r8_v1", "beats_iter3"))
                    or "resnet18_v1" not in (vision.provenance.get("source") or "")):
                raise ContractError("producer source declarations differ from required BEATs+LoRA/ResNet-18 exports")
            documents, rows, metrics, comparison, metadata = _compute(config, report, audio, vision, progress)
        if _inventory(input_paths) != before or _source_hashes() != source_hashes:
            raise ContractError("source inputs or code changed during experiment; success evidence withheld")
        documents.update({"config.json": _json(run_config), "environment.json": _json(run_config["environment"]),
                          "reference_verification.json": _json(reference),
                          "input_verification.json": _json({"inputs": _portable_inventory(before),
                              "artifact_inventory_fingerprint": report["artifact_inventory_fingerprint"],
                              "source_hashes_unchanged": True, "test_scope": "structural hashes/metadata only; no test pairs scored"}),
                          "dataset_summary.json": _json(metadata), "wandb_metrics.json": _json(metrics)})
        for name, text in documents.items():
            _write(config.output_dir, name, text)
        files = sorted(documents)
        manifest = {"schema": SCHEMA, "files": _portable_inventory(
            [_file_entry(name, config.output_dir / name) for name in files])}
        _write(config.output_dir, "output_manifest.json", _json(manifest))
        if tracking_blocker is None:
            progress("Local evidence complete; uploading verified immutable evidence to Forge.")
            tracking = _upload(config.output_dir, tracker, files, metrics, rows)
        else:
            tracking = dict(tracker.metadata, status="BLOCKED_WANDB_UPLOAD", blocker=tracking_blocker)
            _write(config.output_dir, "tracking.json", _json(tracking))
        completion = {"schema": SCHEMA, "local_experiment_status": "COMPLETE", "tracking_status": tracking["status"],
                      "selected_validation_candidate": comparison["selected_validation_candidate"],
                      "output_manifest_sha256": hashlib.sha256((config.output_dir / "output_manifest.json").read_bytes()).hexdigest()}
        _write(config.output_dir, "completion.json", _json(completion))
        return completion
    except BaseException:
        tracker.abort()
        raise


def upload_saved(output, *, tracker_factory=ForgeTracker):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--quiet", "--no-index", str(output)], capture_output=True).returncode:
        raise ContractError("saved experiment must be inside an ignored repository directory")
    completion = _read(output / "completion.json")
    if completion.get("local_experiment_status") != "COMPLETE":
        raise ContractError("no complete local experiment available for upload")
    digest = hashlib.sha256((output / "output_manifest.json").read_bytes()).hexdigest()
    if digest != completion["output_manifest_sha256"]:
        raise ContractError("saved output manifest fingerprint mismatch")
    files = _verified_core_files(output)
    tracker = tracker_factory(output)
    try:
        tracker.start(_read(output / "config.json"))
        tracking = _upload(output, tracker, files, _read(output / "wandb_metrics.json"), _read(output / "comparison.json")["rows"])
    except Exception as exc:
        tracker.abort()
        tracking = dict(tracker.metadata, status="BLOCKED_WANDB_UPLOAD",
                        blocker=str(exc) if type(exc) is TrackingUnavailable else safe_failure("Forge upload", exc))
        _write(output, "tracking.json", _json(tracking))
    completion["tracking_status"] = tracking["status"]
    _write(output, "completion.json", _json(completion))
    return completion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    for name in ("audio-root", "vision-root", "official-root", "reference-root", "reference-evidence", "integration-root", "output-dir"):
        run.add_argument(f"--{name}", required=True, type=Path)
    run.add_argument("--seed", type=int, default=42)
    upload = commands.add_parser("upload")
    upload.add_argument("--output-dir", required=True, type=Path)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    try:
        result = run_experiments(ExperimentConfig(**args), progress=lambda m: print(m, flush=True)) if command == "run" else upload_saved(args["output_dir"])
        print(_json(result), end="")
        return 0 if result["tracking_status"] == "COMPLETE" else 2
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"Experiment failed closed ({type(exc).__name__}); no complete result claimed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

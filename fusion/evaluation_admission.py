"""Restricted, byte-authenticated historical TRAIN score admission.

The immutable Phase 5 freeze is the root of trust. Its preregistration pins E3's
archive manifest; A2 and JSD manifests are pinned directly. No caller can supply
expected hashes or change the permitted cohort. No encoders/checkpoints execute.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from .contracts import ContractError, Record, _check_record
from .evaluation_contracts import (
    CorrespondencePrediction, EvaluationProvenance, FROZEN_MODELS, PredictionSet,
)
from .pair_contracts import (
    PairRecord, PairSource, PairType, SamplingPolicy, SamplingProvenance, _validate_exact,
    pair_order_key,
)

# SHA-256 of the already committed HEAD:docs/fusion_cross_attention_phase5_freeze.json,
# verified against Git before remediation. This is code-owned, never an API input.
_FREEZE_PATH = "docs/fusion_cross_attention_phase5_freeze.json"
_FREEZE_SHA256 = "4fcbd1537ab3e5574ee6d0d06d7d3a2787431741dd14cc6c161577b0a689f9e5"
_P3 = "_local/fusion/phase3_real_experiments/run_20261022_01"
_E3 = "_local/fusion/phase4_improvement/study_20261024_seed42_01/E3"
_A2 = "_local/fusion/phase5_ablations/study_20261025_seed42_01/A2"
_PAIRS = "_local/fusion/task28_real/train_pairs.json"
_EXPORTS = {"audio": "_local/embeddings/audio_final",
            "vision": "_local/embeddings/vision_final/resnet18_v1"}
_PRODUCERS = [["audio", "beats_lora_r8_v1 / beats_iter3"], ["vision", "resnet18_v1"]]


def _read_pinned(root, relative, expected, files):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ContractError("archive evidence path leaves repository root")
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"missing authenticated archive evidence: {relative}") from exc
    if hashlib.sha256(content).hexdigest() != expected:
        raise ContractError(f"authenticated archive hash mismatch: {relative}")
    files[relative] = expected
    return content


def _entries(document):
    result = {}
    for entry in document["files"]:
        if entry["role"] in result:
            raise ContractError("duplicate archive manifest role")
        result[entry["role"]] = entry["sha256"]
    return result


def _csv(content):
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ContractError("missing or duplicate CSV columns")
    rows = list(reader)
    if not rows or any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ContractError("empty or malformed archive CSV")
    return rows


def _source_records(content):
    records = []
    for row in _csv(content):
        record = Record(row["sample_id"], row["clip_id"], row["category"],
                        int(row["start"]), int(row["end"]), int(row["sample_id"].rsplit("__", 1)[1]))
        _check_record(record, "authenticated TRAIN source")
        records.append(record)
    if len({r.sample_id for r in records}) != len(records):
        raise ContractError("duplicate authenticated source identity")
    return tuple(records)


def _primary_pairs(document, source_records, frozen_data):
    if (document["schema"] != "gc2-pair-manifest-v1" or document["config"]["split"] != "train"
            or document["artifact_fingerprint"] != frozen_data["artifact_inventory_fingerprint"]):
        raise ContractError("unauthorized pair-manifest source or split")
    policy = SamplingPolicy(**document["policy"])
    provenance = SamplingProvenance(**document["provenance"]["source_provenance"])
    result = []
    for raw in document["primary_pairs"]:
        sources = []
        for role, modality in (("audio", "audio"), ("visual", "vision")):
            value = raw[role]
            source = PairSource(value["split"], Record(**value["record"]), value["row_index"])
            rows = source_records[modality]
            if (source.split != "train" or source.row_index >= len(rows)
                    or source.record != rows[source.row_index]):
                raise ContractError("source identity/row differs from authenticated TRAIN exporter metadata")
            sources.append(source)
        pair = PairRecord(*sources, PairType(raw["pair_type"]), policy, provenance)
        if (pair.pair_type is PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE
                or list(pair.identity) != raw["identity"]
                or type(raw["original_pair_target"]) is not int
                or pair.original_pair_target != raw["original_pair_target"]):
            raise ContractError("conflicting primary identity, cohort or target")
        result.append(pair)
    keys = [pair_order_key(p) for p in result]
    if (keys != sorted(keys) or len(set(p.identity for p in result)) != len(result)
            or len(result) != frozen_data["train_primary"]["count"]
            or sum(p.original_pair_target for p in result) != frozen_data["train_primary"]["positives"]):
        raise ContractError("duplicate, missing or reordered primary pair identities")
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class AuthenticatedPredictionSet:
    """Loader-only evidence handle; ordinary construction/replace cannot grant trust.

    Frozen nested data is bound to an admission fingerprint. Full admission is
    repeated at every evaluation/alignment; success is never a reusable flag.
    """
    _dataset: PredictionSet
    _root: Path
    _fingerprint: str
    _files: tuple[tuple[str, str], ...]

    def __init__(self, *args, **kwargs):
        raise ContractError("authenticated predictions require admit_archived_train_predictions")

    @property
    def dataset(self):
        """Inspection only. Copies are ordinary unverified PredictionSets."""
        return self._dataset


def admit_archived_train_predictions(model_id: str, *, repository_root) -> AuthenticatedPredictionSet:
    """Authenticate one frozen TRAIN primary archive; no score computation.

    Fixed logical paths and a fixed root digest grant no TEST/validation/challenge
    admission. Relocation is supported, but replacement hashes/policies are not.
    """
    if type(model_id) is not str or model_id not in FROZEN_MODELS:
        raise ContractError("unknown archived model; choose E3, A2 or JSD")
    root = Path(repository_root).resolve()
    files = {}

    def read(relative, digest):
        return _read_pinned(root, relative, digest, files)

    try:
        freeze = json.loads(read(_FREEZE_PATH, _FREEZE_SHA256))
        a2_run = next(run for run in freeze["phase5_runs"] if run["label"] == "A2")
        a2_hashes = a2_run["evidence_sha256"]
        preregistration = json.loads(read(_A2 + "/preregistration.json", a2_hashes["preregistration.json"]))
        inputs = preregistration["configuration"]["input_hashes"]
        input_hashes = {entry["role"]: entry["sha256"] for entry in inputs}
        if len(input_hashes) != len(inputs):
            raise ContractError("duplicate frozen producer evidence role")
        selected = FROZEN_MODELS[model_id]
        frozen_model = next(m for m in freeze["models"] if m["label"] == model_id)
        if frozen_model["frozen_threshold"] != selected.threshold:
            raise ContractError("frozen checkpoint/threshold declaration conflicts with root evidence")

        directory = {"JSD": _P3, "E3": _E3, "A2": _A2}[model_id]
        manifest_sha = {"JSD": frozen_model.get("output_manifest_sha256"),
                        "E3": input_hashes["phase4_e3/output_manifest.json"],
                        "A2": a2_hashes["output_manifest.json"]}[model_id]
        archive = json.loads(read(directory + "/output_manifest.json", manifest_sha))
        entries = _entries(archive)
        score_name = "scores/train_primary.csv" if model_id == "JSD" else "train_primary_scores.csv"
        score_bytes = read(directory + "/" + score_name, entries[score_name])
        operating_name = "policies/jensen_shannon.json" if model_id == "JSD" else "frozen_operating_point.json"
        if entries[operating_name] != selected.evidence_sha256:
            raise ContractError("archive threshold belongs to a different frozen model")
        operating = json.loads(read(directory + "/" + operating_name, selected.evidence_sha256))
        if model_id == "JSD":
            if manifest_sha != selected.artifact_sha256:
                raise ContractError("JSD producer manifest differs from frozen evidence")
            configuration = json.loads(read(directory + "/config.json", entries["config.json"]))
            if configuration["exports"] != {
                    "audio": {"features": "beats_iter3", "model": "BEATs+LoRA", "run": "beats_lora_r8_v1"},
                    "vision": {"model": "ResNet-18", "run": "resnet18_v1"}}:
                raise ContractError("incorrect JSD producer version")
            if (operating["policy"]["threshold"] != selected.threshold
                    or operating["policy"]["source_fingerprint"] != selected.validation_score_fingerprint):
                raise ContractError("JSD threshold provenance mismatch")
        else:
            if entries["best.pt"] != selected.artifact_sha256:
                raise ContractError("archive declares an incorrect frozen checkpoint")
            read(directory + "/best.pt", selected.artifact_sha256)  # Opaque bytes, no torch.load.
            metadata_hash = (a2_hashes["checkpoint_metadata.json"] if model_id == "A2"
                             else input_hashes["phase4_e3/checkpoint_metadata.json"])
            if entries["checkpoint_metadata.json"] != metadata_hash:
                raise ContractError("checkpoint producer metadata mismatch")
            metadata = json.loads(read(directory + "/checkpoint_metadata.json", metadata_hash))
            if (metadata["export_identities"] != _PRODUCERS or metadata["model"] != frozen_model["model"]
                    or metadata["selected_epoch"] != frozen_model["selected_epoch"]
                    or operating["selected_epoch"] != metadata["selected_epoch"]
                    or operating["threshold"] != selected.threshold
                    or operating["validation_prediction_fingerprint"] != selected.validation_score_fingerprint):
                raise ContractError("checkpoint, threshold or producer version mismatch")
            train_metadata = next(d for d in metadata["data"] if d["split"] == "train")
            if train_metadata["manifest_sha256"] != freeze["data"]["manifest_sha256"]["train"]:
                raise ContractError("checkpoint TRAIN cohort mismatch")

        train_sha = freeze["data"]["manifest_sha256"]["train"]
        document = json.loads(read(_PAIRS, train_sha))
        train_provenance = next(d for d in freeze["data_provenance"] if d["split"] == "train")
        export_hashes = dict(train_provenance["input_export_hashes"])
        source_records = {}
        for modality, export_root in _EXPORTS.items():
            for name in ("SOURCE.txt", "README.md", "class_names.json", "train_ids.csv"):
                content = read(export_root + "/" + name, export_hashes[modality + "/" + name])
                if name == "train_ids.csv":
                    source_records[modality] = _source_records(content)
        pairs = _primary_pairs(document, source_records, freeze["data"])
        rows = _csv(score_bytes)
        if len(rows) != len(pairs):
            raise ContractError("prediction archive has missing or extra sample identities")
        records = []
        for row, pair in zip(rows, pairs):
            if ((row["split"], row["partition"], row["audio_sample_id"], row["visual_sample_id"])
                    != ("train", "primary", pair.audio.record.sample_id, pair.visual.record.sample_id)
                    or int(row["audio_row"]) != pair.audio.row_index
                    or int(row["visual_row"]) != pair.visual.row_index
                    or row["target"] != str(pair.original_pair_target)):
                raise ContractError("score ordering, identity, source row or target differs from authenticated manifest")
            score = float(row["jensen_shannon"] if model_id == "JSD" else row["probability"])
            if model_id == "JSD":
                if row["pair_type"] != pair.pair_type.value:
                    raise ContractError("JSD pair-type provenance mismatch")
            else:
                logit = float(row["logit"])
                if not np.isfinite(logit) or score != float(np.exp(-np.logaddexp(0, -logit))):
                    raise ContractError("archived score differs from historical float64 sigmoid")
            records.append(CorrespondencePrediction(pair, pair.original_pair_target, score))
        provenance = EvaluationProvenance("train", "primary", train_sha, entries[score_name],
                                          directory + "/" + score_name, _FREEZE_SHA256, _FREEZE_PATH)
        dataset = PredictionSet(model_id, selected.artifact_sha256, selected.score_kind, provenance, tuple(records))
        # Repeat actual byte verification before issuing a handle; no TOCTOU success cache.
        for path, expected in tuple(files.items()):
            read(path, expected)
        handle = object.__new__(AuthenticatedPredictionSet)
        for key, value in (("_dataset", dataset), ("_root", root), ("_fingerprint", dataset.fingerprint),
                           ("_files", tuple(sorted(files.items())))):
            object.__setattr__(handle, key, value)
        return handle
    except (OSError, ValueError, KeyError, TypeError, StopIteration, OverflowError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("incomplete or conflicting authenticated archive evidence") from exc


def resolve_predictions(value):
    """Revalidate trust at every public evaluation/alignment boundary."""
    if type(value) is PredictionSet:
        _validate_exact(value, PredictionSet, "exploratory prediction set")
        if value.model_id.upper() in FROZEN_MODELS:
            # Check even an obviously wrong checkpoint before refusing attribution.
            expected = FROZEN_MODELS[value.model_id.upper()]
            if value.artifact_sha256 != expected.artifact_sha256 or value.score_kind != expected.score_kind:
                raise ContractError("incorrect known-model checkpoint or score interpretation")
            raise ContractError("known model attribution requires authenticated archived TRAIN admission")
        return value, {"level": "unverified_exploratory", "authenticated_model": None,
                       "interpretation": "Caller-supplied exploratory/synthetic scores; no authenticated model attribution."}
    if type(value) is not AuthenticatedPredictionSet:
        raise ContractError("expected exploratory predictions or authenticated archive admission")
    try:
        dataset, original_fingerprint, root, dependencies = (
            value._dataset, value._fingerprint, value._root, value._files)
    except AttributeError as exc:
        raise ContractError("incomplete authenticated archive admission") from exc
    _validate_exact(dataset, PredictionSet, "authenticated prediction set")
    selected = FROZEN_MODELS.get(dataset.model_id)
    if (selected is None or dataset.provenance.split != "train" or dataset.provenance.partition != "primary"
            or dataset.artifact_sha256 != selected.artifact_sha256 or dataset.score_kind != selected.score_kind
            or dataset.fingerprint != original_fingerprint):
        raise ContractError("admitted scores, identities, split or checkpoint changed after admission")
    # Do not trust a caller-recomputed fingerprint or a reconstructed handle's
    # dependency list. Rebuild from the fixed code-owned root and compare all
    # admitted content against that independently verified snapshot.
    fresh = admit_archived_train_predictions(dataset.model_id, repository_root=root)
    if original_fingerprint != fresh._fingerprint or dependencies != fresh._files:
        raise ContractError("admitted content differs from independently authenticated archive")
    return dataset, {"level": "authenticated_archived_train", "authenticated_model": dataset.model_id,
                     "root_evidence_sha256": _FREEZE_SHA256,
                     "interpretation": "Authenticated frozen TRAIN archive continuity; not held-out performance or physical sound-source causality."}

"""Task 2.8 local-only export validation and official-metadata-gated pairing.

No downloads, model imports or training. Source files are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from enum import Enum
from numbers import Integral
from pathlib import Path

from .contracts import (
    OFFICIAL_COUNTS, SPLITS, ContractError, Record, _check_class_names, _check_record,
    align_on_sample_id, annotation_activity, audit_leakage, load_bundle,
)
from .pair_contracts import SamplingPolicy, _digest
from .pair_generator import GenerationConfig, SourceReuse, generate_pairs
from .pair_integrity import audit_generated_manifest

OFFICIAL_FILES = ("Annotations.txt", "trainSet.txt", "valSet.txt", "testSet.txt")
_BASE = ("class_names.json", "README.md", "SOURCE.txt") + tuple(
    f"{split}_{suffix}" for split in SPLITS for suffix in ("emb.npy", "probs.npy", "ids.csv"))
_OPTIONAL = tuple(f"{s}_{suffix}.npy" for s in SPLITS for suffix in ("valid", "timestamps"))


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    return value


def _json(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"


def _read_json(text):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ContractError(f"invalid JSON numeric constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    except (ValueError, TypeError) as exc:
        raise ContractError(f"invalid integration JSON: {exc}") from exc


def _stamp(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_entry(role, path):
    before = _stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if before != _stamp(path):
        raise ContractError(f"input changed while hashing: {role}")
    return {"role": role, "path": str(path.resolve()), "size_bytes": before[2], "sha256": digest.hexdigest()}


def _inventory(paths):
    return [_file_entry(role, path) for role, path in sorted(paths.items())]


def _check_hashes(inventory, expected):
    if expected is None:
        return
    if type(expected) is not dict or not expected:
        raise ContractError("expected hashes must be a nonempty role-to-SHA256 mapping")
    actual = {entry["role"]: entry["sha256"] for entry in inventory}
    for role, digest in expected.items():
        _digest(digest, f"expected hash {role}")
        if role not in actual or actual[role] != digest:
            raise ContractError(f"artifact hash mismatch or unknown role: {role}")


def _export_paths(roots):
    paths, missing = {}, []
    for modality, root in roots.items():
        for name in _BASE:
            path = root / name
            if not path.is_file():
                missing.append(str(path.resolve()))
            else:
                paths[f"{modality}/{name}"] = path
        for name in _OPTIONAL:
            path = root / name
            if path.exists():
                if not path.is_file():
                    raise ContractError(f"expected optional artifact file: {path}")
                paths[f"{modality}/{name}"] = path
    return paths, sorted(missing)


@contextmanager
def _load_exports(roots, counts):
    bundles = {}
    try:
        for modality, root in roots.items():
            bundles[modality] = load_bundle(root, modality, expected_counts=counts)
        for modality, bundle in bundles.items():
            if bundle.missing_inputs:
                raise ContractError(f"incomplete {modality} exports: {bundle.missing_inputs}")
            for split, data in bundle.splits.items():
                n, dim = len(data.records), 256 if modality == "audio" else 512
                if data.embeddings.shape != (n, 10, dim) or data.probabilities is None or data.probabilities.shape != (n, 10, 29):
                    raise ContractError(f"{modality}/{split}: Task 2.8 requires embeddings (N,10,{dim}) and probabilities (N,10,29)")
                if str(data.embeddings.dtype) != "float16" or str(data.probabilities.dtype) != "float16":
                    raise ContractError(f"{modality}/{split}: this final export schema requires float16 arrays")
        yield bundles["audio"], bundles["vision"]
    finally:
        # Close even when an exception traceback or caller retains the bundles.
        for bundle in bundles.values():
            for data in bundle.splits.values():
                for name in ("embeddings", "probabilities", "input_validity", "timestamps"):
                    mapping = getattr(getattr(data, name), "_mmap", None)
                    if mapping is not None:
                        mapping.close()


def _annotation_lines(path):
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = tuple(part.strip() for part in line.split("&"))
        # The independently verified AVE mirror adds exactly this header to the
        # authors' otherwise byte-identical annotation text.
        if not rows and parts == ("Category", "VideoID", "Quality", "StartTime", "EndTime"):
            continue
        if (not rows and len(parts) == 5 and parts[0].lower() in ("category", "class")
                and parts[3].lower() in ("start", "start_time") and parts[4].lower() in ("end", "end_time")):
            continue
        if len(parts) != 5 or not all(parts) or not all(re.fullmatch(r"[0-9]+", p) for p in parts[3:]):
            raise ContractError(f"{path.name}:{number}: expected category&clip_id&quality&integer_start&integer_end")
        rows.append(parts)
    if not rows:
        raise ContractError(f"{path.name}: no annotation rows")
    # Same exact-row deduplication as the inspected teammate metadata loaders.
    return tuple(dict.fromkeys(rows))


def _official_records(root, counts):
    annotations = _annotation_lines(root / "Annotations.txt")
    names = tuple(sorted({row[0] for row in annotations})) + ("background",)
    _check_class_names(names, "official annotation vocabulary")

    def records(rows):
        result = []
        for category, clip, _, start, end in rows:
            if category not in names[:-1]:
                raise ContractError(f"unknown official category: {category}")
            label = names.index(category)
            record = Record(f"{clip}__{label:02d}", clip, category, int(start), int(end), label)
            _check_record(record, "official metadata")
            result.append(record)
        if len({r.sample_id for r in result}) != len(result):
            raise ContractError("official metadata contains duplicate annotation sample IDs")
        return tuple(result)

    all_records = records(annotations)
    splits = {s: records(_annotation_lines(root / name)) for s, name in zip(SPLITS, OFFICIAL_FILES[1:])}
    for split, rows in splits.items():
        if counts is not None and split in counts and len(rows) != counts[split]:
            raise ContractError(f"official {split}: expected {counts[split]} rows, found {len(rows)}")
    if Counter(r for rows in splits.values() for r in rows) != Counter(all_records):
        raise ContractError("official split annotation union differs from Annotations.txt")
    return names, splits


def _verify_official(audio, vision, root, counts):
    names, splits = _official_records(root, counts)
    for bundle in (audio, vision):
        if bundle.class_names != names:
            raise ContractError(f"{bundle.modality}: mapping conflicts with official annotations")
        for split in SPLITS:
            expected = {r.sample_id: r for r in splits[split]}
            actual = {r.sample_id: r for r in bundle.metadata[split]}
            if actual != expected:
                raise ContractError(f"{bundle.modality}/{split}: official annotation mismatch; "
                                    f"missing={sorted(expected.keys()-actual.keys())}; "
                                    f"extra={sorted(actual.keys()-expected.keys())}; "
                                    f"conflicting={sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])}")


def manifest_to_json(manifest, audio, vision, *, artifact_fingerprint=None):
    """Versioned deterministic output, including original rows and derived masks."""
    audit_generated_manifest(manifest, audio, vision)
    if manifest.policy.hard_negative_policy is not None or manifest.config.enable_hard_negatives:
        raise ContractError("Task 2.8 serialization currently requires hard negatives disabled and no hard policy")
    if artifact_fingerprint is not None:
        _digest(artifact_fingerprint, "artifact inventory fingerprint")

    def pair(value):
        return {"audio": _plain(value.audio), "visual": _plain(value.visual),
                "pair_type": value.pair_type.value, "temporal_status": value.temporal_status.value,
                "identity": _plain(value.identity), "original_pair_target": value.original_pair_target,
                "semantic_compatibility": value.semantic_compatibility.value,
                "event_supervision": _plain(value.event_supervision),
                "audio_annotation_activity": annotation_activity([value.audio.record])[0].tolist(),
                "visual_annotation_activity": annotation_activity([value.visual.record])[0].tolist()}

    return _json({"schema": "gc2-pair-manifest-v1", "artifact_fingerprint": artifact_fingerprint,
                  "config": manifest.config, "policy": manifest.policy,
                  "provenance": manifest.provenance, "report": manifest.report,
                  "primary_pairs": [pair(p) for p in manifest.primary_pairs],
                  "challenge_pairs": [pair(p) for p in manifest.challenge_pairs]})


def manifest_from_json(text, audio, vision, *, expected_artifact_fingerprint=None):
    """Rebuild exact typed output by deterministic regeneration and full comparison.

    This deliberately pays the sampling cost again: every persisted field, type,
    source row, ordering and derived mask must match the regenerated manifest.
    It authenticates neither source binaries nor the official origin of bundles.
    """
    data = _read_json(text)
    try:
        if data["schema"] != "gc2-pair-manifest-v1" or data["policy"]["hard_negative_policy"] is not None:
            raise ContractError("unsupported manifest schema or hard policy; Task 2.8 hard negatives are disabled")
        if data["artifact_fingerprint"] != expected_artifact_fingerprint:
            raise ContractError("serialized manifest artifact inventory does not match expected context")
        config_data = dict(data["config"])
        config_data["source_reuse"] = SourceReuse(config_data["source_reuse"])
        config = GenerationConfig(**config_data)
        policy = SamplingPolicy(**data["policy"])
        result = generate_pairs(audio, vision, config=config, policy=policy)
        if manifest_to_json(result, audio, vision, artifact_fingerprint=expected_artifact_fingerprint) != _json(data):
            raise ContractError("serialized manifest differs from deterministic typed regeneration")
        return result
    except (KeyError, TypeError, ValueError) as exc:
        raise ContractError(f"invalid serialized manifest: {exc}") from exc


def run_integration(audio_root, vision_root, official_root, *, output_dir=None,
                    seed=42, expected_counts=OFFICIAL_COUNTS, expected_hashes=None):
    """Validate real files; generate only after all official metadata is available.

    Synthetic file tests must explicitly supply their own expected_counts. The
    CLI always uses official counts. Returned reports never infer official truth
    from predictions or mutually agreeing exporter metadata.
    """
    policy = SamplingPolicy(seed)
    roots = {"audio": Path(audio_root).resolve(), "vision": Path(vision_root).resolve()}
    official_root = Path(official_root).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else None
    if output is not None:
        if any(output == root or output.is_relative_to(root) or root.is_relative_to(output)
               for root in (*roots.values(), official_root)):
            raise ContractError("output directory must be separate from every input root")
        if output.exists():
            raise ContractError("output directory already exists; select a fresh directory to avoid stale results")
    paths, missing_exports = _export_paths(roots)
    missing_official = [str(official_root / n) for n in OFFICIAL_FILES if not (official_root / n).is_file()]
    paths.update({f"official/{name}": official_root / name for name in OFFICIAL_FILES if (official_root / name).is_file()})
    before = _inventory(paths)
    _check_hashes(before, expected_hashes)
    report = {"schema": "gc2-artifact-integration-v1", "status": "BLOCKED", "seed": int(policy.seed),
              "missing_export_files": missing_exports, "missing_official_files": missing_official,
              "binary_validation": "NOT_VALIDATED", "official_metadata": "NOT_VERIFIED",
              "checksums": "COMPUTED_LOCAL_SHA256", "trusted_checksums": "NOT_SUPPLIED" if expected_hashes is None else "SUPPLIED_ENTRIES_MATCH",
              "expected_hash_roles": sorted(expected_hashes) if expected_hashes is not None else [],
              "artifacts": before, "alignment": {}, "leakage": {}, "export_metadata": {},
              "generation_config": {}, "manifests": {}, "outputs": {}}
    documents = {}
    if not missing_exports:
        with _load_exports(roots, expected_counts) as (audio, vision):
            report["binary_validation"] = "VALIDATED_LOCAL_FILES"
            for entry in report["artifacts"]:
                role = entry["role"]
                modality, name = role.split("/", 1)
                if modality in roots and name.endswith(("_emb.npy", "_probs.npy")):
                    split = name.split("_", 1)[0]
                    data = (audio if modality == "audio" else vision).splits[split]
                    array = data.embeddings if name.endswith("_emb.npy") else data.probabilities
                    entry.update(shape=list(array.shape), dtype=str(array.dtype))
            for split in SPLITS:
                alignment = align_on_sample_id(audio, vision, split=split)
                alignment.require_complete()
                report["alignment"][split] = {
                    "matched": len(alignment.sample_ids), "audio_rows": alignment.audio_row_count,
                    "visual_rows": alignment.visual_row_count, "unmatched_audio": alignment.unmatched_audio,
                    "unmatched_visual": alignment.unmatched_visual,
                    "different_row_indices": int((alignment.audio_indices != alignment.visual_indices).sum()),
                    "scope": "EXPORT_CSV_METADATA"}
            report["leakage"] = {b.modality: _plain(audit_leakage(b)) for b in (audio, vision)}
            report["export_metadata"] = {
                b.modality: {s: {"annotation_rows": len(data.records),
                                 "unique_sources": len({r.clip_id for r in data.records}),
                                 "empty_annotation_rows": [
                                     {"sample_id": r.sample_id, "row_index": i,
                                      "annotation_activity": annotation_activity([r])[0].tolist()}
                                     for i, r in enumerate(data.records) if r.end == 0]}
                             for s, data in b.splits.items()} for b in (audio, vision)}
            report["generation_config"] = {
                s: _plain(GenerationConfig(s, easy_negative_budget=len(audio.metadata[s]),
                                           challenge_budget=len(audio.metadata[s]))) for s in SPLITS}
            if not missing_official:
                _verify_official(audio, vision, official_root, expected_counts)
                report["official_metadata"] = "MATCHES_SUPPLIED_OFFICIAL_REFERENCE_FILES"
                context = hashlib.sha256(_json(before).encode("utf-8")).hexdigest()
                report["artifact_inventory_fingerprint"] = context
                for split in SPLITS:
                    n = len(audio.metadata[split])
                    config = GenerationConfig(split, easy_negative_budget=n, challenge_budget=n)
                    manifest = generate_pairs(audio, vision, config=config, policy=policy)
                    text = manifest_to_json(manifest, audio, vision, artifact_fingerprint=context)
                    restored = manifest_from_json(text, audio, vision, expected_artifact_fingerprint=context)
                    if restored != manifest:
                        raise ContractError("typed serialization round trip failed")
                    documents[f"{split}_pairs.json"] = text
                    report["manifests"][split] = _plain(audit_generated_manifest(manifest, audio, vision))
                report["status"] = "COMPLETE"
    # Fresh hashes and file identity stamps guard ordinary concurrent changes.
    # These are local identity checks, not remote authenticity or filesystem locks.
    after = _inventory(paths)
    normalized = [{k: v for k, v in entry.items() if k not in ("shape", "dtype")} for entry in before]
    if after != normalized:
        raise ContractError("input files changed during integration; no outputs published")
    for name, text in documents.items():
        report["outputs"][name] = {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), "size_bytes": len(text.encode("utf-8"))}
    result = _plain(report)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        # Publish the report last. It is the completion marker for this directory.
        for name, text in documents.items():
            (output / name).write_bytes(text.encode("utf-8"))
        summary = (f"Task 2.8: {result['status']}\n\nBinary validation: {result['binary_validation']}.\n"
                   f"Official metadata: {result['official_metadata']}.\n"
                   f"Missing export files: {len(missing_exports)}. Missing official files: {len(missing_official)}.\n"
                   "Hard negatives disabled. No model training or temporal-synchrony claim.\n\n"
                   + "\n".join(f"Missing: {p}" for p in missing_exports + missing_official) + "\n")
        summary += "\nLocal SHA-256 hashes and file sizes: validation.json. Computed hashes are not remote authenticity proof.\n"
        summary += "\nExport CSV alignment (official membership requires the separate reference files):\n"
        for split, data in result["alignment"].items():
            summary += f"- {split}: {data['matched']} matches; {data['different_row_indices']} unequal original row indices.\n"
        for modality, leakage in result["leakage"].items():
            summary += f"\n{modality} source overlaps: " + ", ".join(f"{key}={len(ids)}" for key, ids in leakage["source_overlap"].items()) + ".\n"
            for split, data in result["export_metadata"][modality].items():
                summary += f"- {split}: {data['annotation_rows']} annotations, {data['unique_sources']} sources; empty IDs=" + str([r["sample_id"] for r in data["empty_annotation_rows"]]) + ".\n"
        summary += "\nGenerated pair counts: " + str({s: r["pair_count"] for s, r in result["manifests"].items()}) + ".\n"
        (output / "summary.md").write_bytes(summary.encode("utf-8"))
        (output / "validation.json").write_bytes(_json(result).encode("utf-8"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", default="_local/embeddings/audio_final")
    parser.add_argument("--vision-root", default="_local/embeddings/vision_final/resnet18_v1")
    parser.add_argument("--official-root", default="_local/datasets/AVE")
    parser.add_argument("--output-dir", default="_local/fusion/task28")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-hashes", help="optional local role-to-SHA256 JSON")
    args = parser.parse_args(argv)
    try:
        hashes = _read_json(Path(args.expected_hashes).read_text(encoding="utf-8")) if args.expected_hashes else None
        result = run_integration(args.audio_root, args.vision_root, args.official_root,
                                 output_dir=args.output_dir, seed=args.seed, expected_hashes=hashes)
        print(f"{result['status']}: binaries={result['binary_validation']}; official_metadata={result['official_metadata']}")
        return 0 if result["status"] == "COMPLETE" else 2
    except (ContractError, OSError, UnicodeError) as exc:
        print(f"Task 2.8 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

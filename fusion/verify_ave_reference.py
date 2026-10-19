"""Verify local AVE metadata against pinned authors' references; never download.

Only the HDF5 adapter needs optional h5py. No exporters, predictions, historical
reports, models or pair-generation logic participate in this comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .contracts import OFFICIAL_COUNTS, SPLITS, ContractError, Record, _check_record

AUTHOR_COMMIT = "93f843fed72fb17394df76330df958b0c2dbde76"
AUTHOR_BASE = f"https://raw.githubusercontent.com/YapengTian/AVE-ECCV18/{AUTHOR_COMMIT}/data"
# Checked against the authors' pinned files, independently of the mirror.
REFERENCE_HASHES = {
    "Annotations.txt": "0d24a9ec5cd4ceb779ebd83d74e83ec8e55c5e468cede3f320fb57346c9dd321",
    "train_order.h5": "a216fcdd02782ba9d10c9ea79116120eb3be467740e58290e1b340d42a590456",
    "val_order.h5": "5692aec0efb30e4efa49a867a902c1993b4c88a7d3ec19879349e540e7a5d9ca",
    "test_order.h5": "f9bb79defc1f784ca5dda9cdd3ee52af6d24ccb399a1cfab6bf0d373f5681076",
}
METADATA_FILES = ("Annotations.txt", "trainSet.txt", "valSet.txt", "testSet.txt")
MIRROR_HEADER = b"Category&VideoID&Quality&StartTime&EndTime"
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _rows(data, context, *, allow_header=False):
    if allow_header:
        first, separator, rest = data.partition(b"\n")
        if first in (MIRROR_HEADER, MIRROR_HEADER + b"\r") and separator:
            data = rest
    rows = []
    for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        fields = tuple(line.split("&"))
        if (len(fields) != 5 or any(not f or f != f.strip() for f in fields)
                or any(re.fullmatch(r"[0-9]+", f) is None for f in fields[3:])):
            raise ContractError(f"{context}:{number}: invalid annotation row")
        rows.append(fields)
    if not rows:
        raise ContractError(f"{context}: no annotation rows")
    identities = [(row[0], row[1]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ContractError(f"{context}: duplicate annotation identity")
    return rows, data


def _read_order(data, context):
    try:
        import h5py
    except ImportError as exc:
        raise ContractError("optional h5py is required for official HDF5 references; "
                            "see docs/fusion_artifact_integration.md") from exc
    try:
        with h5py.File(io.BytesIO(data), "r") as handle:
            return np.asarray(handle["order"][:])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ContractError(f"{context}: unreadable HDF5 order dataset: {exc}") from exc


def _compare(author, supplied, orders, counts):
    """Pure comparison over already authenticated reference bytes and indices."""
    rows, _ = _rows(author, "author/Annotations.txt")
    local, without_header = _rows(supplied["Annotations.txt"], "metadata/Annotations.txt", allow_header=True)
    if len(rows) != sum(counts.values()):
        raise ContractError("author annotation count differs from expected split total")
    # Do not normalize whitespace, line endings, ordering, quality or fields.
    if without_header != author or local != rows:
        mismatch_count = sum(a != b for a, b in zip(rows, local)) + abs(len(rows) - len(local))
        raise ContractError(f"annotation text mismatch; mismatched records={mismatch_count}; "
                            "only the exact single mirror header may differ")
    categories = sorted({r[0] for r in rows})
    if len(categories) != 28 or "background" in categories:
        raise ContractError("official vocabulary must contain 28 foreground categories")
    records = []
    for category, clip, quality, start, end in rows:
        label = categories.index(category)
        record = Record(f"{clip}__{label:02d}", clip, category, int(start), int(end), label)
        _check_record(record, "author annotation")
        records.append(record)
    checks, combined, empty = {}, [], []
    for split, filename in zip(SPLITS, METADATA_FILES[1:]):
        order = orders[split]
        if (not isinstance(order, np.ndarray) or order.ndim != 1 or order.dtype.kind not in "iu"
                or len(order) != counts[split]):
            raise ContractError(f"{split}: HDF5 order must be a one-dimensional integer array of length {counts[split]}")
        indices = [int(i) for i in order]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(rows) for i in indices):
            raise ContractError(f"{split}: duplicate or out-of-range official indices")
        combined.extend(indices)
        actual, _ = _rows(supplied[filename], f"metadata/{filename}")
        expected = [rows[i] for i in indices]
        if Counter(actual) != Counter(expected):
            difference = sum((Counter(actual) - Counter(expected)).values()) + sum((Counter(expected) - Counter(actual)).values())
            raise ContractError(f"{split}: split membership/annotation mismatch; differing rows={difference}")
        if actual != expected:
            raise ContractError(f"{split}: split ordering mismatch against zero-based official indices")
        checks[split] = {"records_verified": len(actual), "membership_matches": True,
                         "order_matches": True, "index_min": min(indices), "index_max": max(indices),
                         "order_sha256": _sha(_json(indices).encode("utf-8"))}
        for i in indices:
            record = records[i]
            if record.start == record.end == 0:
                empty.append({"sample_id": record.sample_id, "annotation_index": i, "split": split,
                              "annotation_activity": [False] * 10, "event_compatibility": "UNKNOWN"})
    if len(combined) != len(set(combined)) or set(combined) != set(range(len(rows))):
        raise ContractError("official split indices do not form a disjoint complete annotation partition")
    return {"annotation_rows": len(rows), "class_names": categories + ["background"],
            "annotation_comparison": "EXACT_BYTES_WITH_OPTIONAL_SINGLE_KNOWN_HEADER",
            "split_checks": checks, "empty_annotations": empty,
            "unique_sources": len({r.clip_id for r in records}),
            "multiple_annotation_sources": sum(n > 1 for n in Counter(r.clip_id for r in records).values()),
            "mismatch_count": 0}


def _output_path(output_dir, input_roots):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ContractError("output directory already exists; choose a fresh ignored directory")
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in input_roots):
        raise ContractError("output directory must be separate from input roots")
    if not output.is_relative_to(_REPO_ROOT):
        raise ContractError("output must be inside this repository and Git-ignored")
    ignored = subprocess.run(["git", "-C", str(_REPO_ROOT), "check-ignore", "--quiet", "--no-index", str(output)],
                             capture_output=True, check=False)
    if ignored.returncode != 0:
        raise ContractError("output must be Git-ignored before writing verification evidence")
    return output


def verify_reference(metadata_root, reference_root, *, output_dir=None):
    """Verify all official records and return deterministic, path-independent evidence.

    Missing references or mismatches raise ContractError; no success output is
    published. Output, when requested, must be a fresh Git-ignored directory.
    """
    metadata, reference = Path(metadata_root).resolve(), Path(reference_root).resolve()
    output = _output_path(output_dir, (metadata, reference)) if output_dir is not None else None
    paths = {**{f"metadata/{name}": metadata / name for name in METADATA_FILES},
             **{f"author/{name}": reference / name for name in REFERENCE_HASHES}}
    missing = [role for role, path in paths.items() if not path.is_file()]
    if missing:
        raise ContractError("missing required reference inputs: " + ", ".join(sorted(missing)))
    data = {role: path.read_bytes() for role, path in paths.items()}
    for name, expected in REFERENCE_HASHES.items():
        if _sha(data[f"author/{name}"]) != expected:
            raise ContractError(f"pinned author reference hash mismatch: {name}")
    orders = {split: _read_order(data[f"author/{split}_order.h5"], f"author/{split}_order.h5") for split in SPLITS}
    comparison = _compare(data["author/Annotations.txt"],
                          {name: data[f"metadata/{name}"] for name in METADATA_FILES}, orders, OFFICIAL_COUNTS)
    result = {"schema": "gc2-ave-reference-v1", "status": "PASS", "author_commit": AUTHOR_COMMIT,
              "reference_sources": {name: {"url": f"{AUTHOR_BASE}/{name}", "sha256": digest}
                                    for name, digest in REFERENCE_HASHES.items()},
              "inputs": {role: {"sha256": _sha(content), "size_bytes": len(content)}
                         for role, content in sorted(data.items())}, **comparison}
    if any(path.read_bytes() != data[role] for role, path in paths.items()):
        raise ContractError("reference inputs changed during verification; no evidence published")
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        hashes = {f"official/{name}": _sha(data[f"metadata/{name}"]) for name in METADATA_FILES}
        (output / "verified_metadata_hashes.json").write_bytes(_json(hashes).encode("utf-8"))
        # Completion marker last. Never overwrite earlier verification evidence.
        (output / "verification.json").write_bytes(_json(result).encode("utf-8"))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True, help="four locally supplied AVE text files")
    parser.add_argument("--reference-root", required=True, help="pinned authors' Annotations.txt and three *_order.h5 files")
    parser.add_argument("--output-dir", required=True, help="fresh Git-ignored evidence directory")
    args = parser.parse_args(argv)
    try:
        result = verify_reference(args.metadata_root, args.reference_root, output_dir=args.output_dir)
    except (ContractError, OSError, UnicodeError) as exc:
        print(f"AVE reference verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"PASS: {result['annotation_rows']} annotations; splits="
          f"{[result['split_checks'][s]['records_verified'] for s in SPLITS]}; mismatches=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

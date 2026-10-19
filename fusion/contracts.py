"""Validate external exports and align annotation rows without changing originals.

Only NumPy and the standard library are required. IDs/annotations are metadata,
never predictive features. Index alignment does not establish decoded AV sync.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

SPLITS = ("train", "val", "test")
OFFICIAL_COUNTS = MappingProxyType({"train": 3339, "val": 402, "test": 402})
SECONDS = 10
N_EVENTS = 28
PROBABILITY_ATOL = 1e-3  # float16 softmax storage, summed in float64
_FLOATS = (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))
Modality = Literal["audio", "vision"]


class ContractError(ValueError):
    """An existing artifact violates its declared data contract."""


@dataclass(frozen=True)
class Record:
    sample_id: str
    clip_id: str
    category: str
    start: int
    end: int
    label: int


@dataclass(frozen=True)
class SplitArtifacts:
    records: tuple[Record, ...] | None
    embeddings: NDArray | None
    probabilities: NDArray | None = None
    input_validity: NDArray | None = None
    timestamps: NDArray | None = None

    def __post_init__(self) -> None:
        if self.records is not None:
            object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class Bundle:
    root: Path
    modality: Modality
    requested_splits: tuple[str, ...]
    splits: Mapping[str, SplitArtifacts]
    metadata: Mapping[str, tuple[Record, ...]]
    class_names: tuple[str, ...] | None
    missing_inputs: tuple[str, ...]
    provenance: dict
    expected_counts: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        # Copy before wrapping: mutating a caller's original dictionary/list must
        # not change record order or split membership in an existing bundle.
        object.__setattr__(self, "splits", MappingProxyType(dict(self.splits)))
        object.__setattr__(self, "metadata", MappingProxyType({
            split: tuple(records) for split, records in self.metadata.items()
        }))
        object.__setattr__(self, "requested_splits", tuple(self.requested_splits))
        object.__setattr__(self, "missing_inputs", tuple(self.missing_inputs))
        if self.class_names is not None:
            object.__setattr__(self, "class_names", tuple(self.class_names))
        if self.expected_counts is not None:
            object.__setattr__(self, "expected_counts", MappingProxyType(dict(self.expected_counts)))


@dataclass(frozen=True)
class ValidationReport:
    complete: bool
    missing_inputs: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Alignment:
    split: str
    sample_ids: tuple[str, ...]
    audio_indices: NDArray
    visual_indices: NDArray
    unmatched_audio: tuple[str, ...]
    unmatched_visual: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    notes: tuple[str, ...]
    audio_row_count: int | None = None
    visual_row_count: int | None = None

    def __post_init__(self) -> None:
        # Snapshot caller-owned containers without coercing malformed index
        # dtypes/shapes. Those must fail validation, not be silently repaired.
        for name in ("sample_ids", "unmatched_audio", "unmatched_visual", "missing_inputs", "notes"):
            value = getattr(self, name)
            if isinstance(value, (tuple, list)):
                object.__setattr__(self, name, tuple(value))
        for name in ("audio_indices", "visual_indices"):
            value = getattr(self, name)
            if isinstance(value, np.ndarray):
                snapshot = np.array(value, copy=True, subok=False)
                snapshot.flags.writeable = False
                object.__setattr__(self, name, snapshot)

    def _validation_error(self) -> str | None:
        """One current-state check shared by the boolean and raising guards."""
        if not isinstance(self.split, str) or self.split not in SPLITS:
            return "unknown split"
        for name in ("sample_ids", "unmatched_audio", "unmatched_visual", "missing_inputs", "notes"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(v, str) or not v.strip() for v in values):
                return f"{name} must be a sequence of nonempty strings"
        if not self.sample_ids:
            return "no aligned sample IDs"
        if len(set(self.sample_ids)) != len(self.sample_ids):
            return "duplicate aligned sample IDs"
        for name, count in (("audio_indices", self.audio_row_count),
                            ("visual_indices", self.visual_row_count)):
            indices = getattr(self, name)
            if not isinstance(indices, np.ndarray) or indices.ndim != 1:
                return f"{name} must be a one-dimensional NumPy array"
            if indices.dtype.kind not in ("i", "u"):
                return f"{name} must contain integers, excluding boolean arrays"
            if len(indices) != len(self.sample_ids):
                return f"{name} length must equal sample_ids length ({len(self.sample_ids)})"
            if (not isinstance(count, Integral) or isinstance(count, bool)
                    or not 0 <= count <= np.iinfo(np.intp).max):
                return f"{name}: source row count must be an explicit nonnegative integer within platform bounds"
            if np.any(indices < 0):
                return f"{name} contains negative indices"
            if np.any(indices >= count):
                return f"{name} contains indices outside source row count {count}"
            if len(np.unique(indices)) != len(indices):
                return f"{name} contains duplicate indices"
            # N unique in-bounds indices cover every source row exactly once.
            if len(indices) != count:
                return (f"{name}: full source-row coverage requires {count} matched rows, "
                        f"got {len(indices)}")
        if self.missing_inputs or self.unmatched_audio or self.unmatched_visual:
            return "required inputs or matched rows unavailable"
        return None

    @property
    def complete(self) -> bool:
        return self._validation_error() is None

    def require_complete(self) -> None:
        """Gate future consumers before using the matched row indices."""
        error = self._validation_error()
        if error is not None:
            diagnostics = ""
            if any(isinstance(values, tuple) and len(values) > 0 for values in (
                self.missing_inputs, self.unmatched_audio, self.unmatched_visual
            )):
                diagnostics = (f"; missing inputs={self.missing_inputs}; audio-only IDs={self.unmatched_audio}; "
                               f"vision-only IDs={self.unmatched_visual}")
            raise ContractError(f"{self.split}: alignment pending or invalid: {error}{diagnostics}")


@dataclass(frozen=True)
class LeakageReport:
    duplicate_sample_ids: dict[str, tuple[str, ...]]
    sample_overlap: dict[str, tuple[str, ...]]
    source_overlap: dict[str, tuple[str, ...]]
    multiple_annotations: dict[str, dict[str, tuple[str, ...]]]
    missing_splits: tuple[str, ...]


def _readonly(array: NDArray) -> NDArray:
    array.flags.writeable = False
    return array


def _check_interval(start: int, end: int, context: str) -> None:
    if (not isinstance(start, Integral) or not isinstance(end, Integral)
            or isinstance(start, bool) or isinstance(end, bool)
            or not (0 <= start < end <= SECONDS)):
        raise ContractError(f"{context}: expected integer interval 0 <= start < end <= 10")


def _check_record(record: Record, context: str) -> None:
    if not isinstance(record, Record):
        raise ContractError(f"{context}: expected Record instances")
    for key in ("sample_id", "clip_id", "category"):
        value = getattr(record, key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ContractError(f"{context}: {key} must be nonempty without outer whitespace")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", record.clip_id):
        raise ContractError(f"{context}: invalid clip_id {record.clip_id!r}")
    if record.category == "background":
        raise ContractError(f"{context}: category must be an event, not background")
    if (not isinstance(record.label, Integral) or isinstance(record.label, bool)
            or not 0 <= record.label < N_EVENTS):
        raise ContractError(f"{context}: event label must be an integer from 0 through 27")
    if record.sample_id != f"{record.clip_id}__{record.label:02d}":
        raise ContractError(f"{context}: sample_id conflicts with clip_id and label")
    _check_interval(record.start, record.end, context)


def _read_records(path: Path) -> tuple[Record, ...]:
    try:
        return _read_csv_records(path)
    except UnicodeDecodeError as exc:
        raise ContractError(f"{path}: IDs CSV must be valid UTF-8: {exc}") from exc


def _read_csv_records(path: Path) -> tuple[Record, ...]:
    records: list[Record] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = ("sample_id", "clip_id", "category", "start", "end")
        headers = reader.fieldnames or []
        if not set(required).issubset(headers) or len(set(headers)) != len(headers):
            raise ContractError(f"{path}: unique CSV headers must include {sorted(required)}")
        for line, row in enumerate(reader, start=2):
            context = f"{path}: row {line}"
            if None in row or any(value is None for value in row.values()):
                raise ContractError(f"{context}: malformed CSV row")
            for key in required:
                value = row[key]
                if not value or value != value.strip():
                    raise ContractError(f"{context}: {key} must be nonempty without outer whitespace")
            clip, sample, category = row["clip_id"], row["sample_id"], row["category"]
            match = re.fullmatch(r".+__([0-9]{2})", sample)
            if not match:
                raise ContractError(f"{context}: sample_id must end with a two-digit event label")
            if not all(re.fullmatch(r"[0-9]+", row[key]) for key in ("start", "end")):
                raise ContractError(f"{context}: start/end must be integer seconds")
            start, end = int(row["start"]), int(row["end"])
            record = Record(sample, clip, category, start, end, int(match[1]))
            _check_record(record, context)
            records.append(record)
    if not records:
        raise ContractError(f"{path}: IDs CSV has no samples")
    return tuple(records)


def _class_mapping(path: Path) -> tuple[str, ...]:
    try:
        names = json.loads(path.read_text(encoding="utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise ContractError(f"{path}: invalid class mapping JSON: {exc}") from exc
    if not isinstance(names, list):
        raise ContractError(f"{path}: expected a JSON list of 29 class names")
    _check_class_names(names, str(path))
    return tuple(names)


def _check_class_names(names: Sequence[str], context: str) -> None:
    if len(names) != N_EVENTS + 1:
        raise ContractError(f"{context}: expected 29 class names")
    if any(not isinstance(n, str) or not n.strip() or n != n.strip() for n in names):
        raise ContractError(f"{context}: class names must be nonempty strings without outer whitespace")
    if len(set(names)) != len(names) or names[-1] != "background":
        raise ContractError(f"{context}: expected 29 unique names with background last")
    if list(names[:-1]) != sorted(names[:-1]):
        raise ContractError(f"{context}: event class order must be alphabetical; no automatic remapping")


def _load_array(path: Path) -> NDArray:
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError, EOFError) as exc:
        raise ContractError(f"{path}: cannot load numeric NumPy array: {exc}") from exc
    if not isinstance(array, np.ndarray):
        array.close()
        raise ContractError(f"{path}: expected one .npy array, not an archive")
    return array


def _numeric(array: NDArray, shape: tuple, context: str) -> None:
    if array.shape != shape:
        raise ContractError(f"{context}: expected shape {shape}, got {array.shape}")
    if array.dtype not in _FLOATS:
        raise ContractError(f"{context}: expected float16, float32 or float64, got {array.dtype}")
    # Bound temporary memory even when the source is a large memory-mapped array.
    for offset in range(0, len(array), 128):
        if not np.isfinite(array[offset:offset + 128]).all():
            raise ContractError(f"{context}: NaN or infinity is not allowed")


def _validate_arrays(
    *, embeddings: NDArray | None, probabilities: NDArray | None,
    input_validity: NDArray | None, timestamps: NDArray | None,
    n: int, modality: Modality, split: str,
) -> None:
    for name, array in (("embeddings", embeddings), ("probabilities", probabilities),
                        ("input_validity", input_validity), ("timestamps", timestamps)):
        if array is not None and not isinstance(array, np.ndarray):
            raise ContractError(f"{modality}/{split}: {name} must be a NumPy array")
    if embeddings is not None:
        emb = embeddings
        if emb.ndim != 3 or emb.shape[-1] < 1:
            raise ContractError(f"{modality}/{split}: embeddings must have shape (N,10,D), D > 0")
        dim = 256 if modality == "audio" else emb.shape[-1]
        _numeric(emb, (n, SECONDS, dim), f"{modality}/{split} embeddings")
        if modality == "audio" and emb.dtype != np.dtype("float16"):
            raise ContractError(f"audio/{split}: final audio embeddings must be float16")
    if probabilities is not None:
        probs = probabilities
        _numeric(probs, (n, SECONDS, N_EVENTS + 1), f"{modality}/{split} probabilities")
        if modality == "audio" and probs.dtype != np.dtype("float16"):
            raise ContractError(f"audio/{split}: final audio probabilities must be float16")
        if np.any(probs < 0) or np.any(probs > 1):
            raise ContractError(f"{modality}/{split}: probabilities must be in [0,1]")
        if not np.allclose(probs.sum(-1, dtype=np.float64), 1, atol=PROBABILITY_ATOL, rtol=0):
            raise ContractError(f"{modality}/{split}: probability sums must be 1 within {PROBABILITY_ATOL}")
    if input_validity is not None:
        if input_validity.dtype != np.dtype(bool) or input_validity.shape != (n, SECONDS):
            raise ContractError(f"{modality}/{split}: input validity must be bool (N,10); no coercion")
    if timestamps is not None:
        times = timestamps
        _numeric(times, (n, SECONDS), f"{modality}/{split} timestamps")
        if np.any(times < 0) or np.any(times >= SECONDS) or np.any(np.diff(times, axis=1) < 0):
            raise ContractError(f"{modality}/{split}: timestamps must be nondecreasing and in [0,10)")
        in_bin = (times >= np.arange(SECONDS)) & (times < np.arange(1, SECONDS + 1))
        checked = np.ones((n, SECONDS), dtype=bool) if input_validity is None else input_validity
        if np.any(checked & ~in_bin):
            raise ContractError(f"{modality}/{split}: timestamp outside its declared one-second bin")


def validate_bundle(bundle: Bundle) -> ValidationReport:
    """Revalidate current structure, identities and arrays, including public bundles.

    Stored missing-input diagnostics are retained, but never replace inspecting
    the actual requested split contents. This validates the in-memory snapshot;
    load again to inspect source files that an external writer has changed.
    """
    notes = ["Index alignment does not verify actual decoded audio/video synchronization."]
    if bundle.modality not in ("audio", "vision"):
        raise ContractError(f"Unknown modality {bundle.modality!r}; expected audio or vision")
    requested = bundle.requested_splits
    if not requested or len(set(requested)) != len(requested) or any(s not in SPLITS for s in requested):
        raise ContractError("requested_splits must be unique names from train, val, test")
    if (set(bundle.metadata) | set(bundle.splits)) - set(requested):
        raise ContractError("Bundle contains split data outside requested_splits")
    if bundle.class_names is not None:
        _check_class_names(bundle.class_names, f"{bundle.modality} class mapping")
    missing = list(bundle.missing_inputs)

    def absent(filename: str) -> None:
        missing.append(str(bundle.root / filename))

    if bundle.modality == "audio":
        if bundle.class_names is None:
            absent("class_names.json")
        if bundle.provenance.get("source") is None:
            absent("SOURCE.txt")
        if bundle.provenance.get("readme_available") is not True:
            absent("README.md")
    category_labels: dict[str, int] = {}
    label_categories: dict[int, str] = {}
    dimensions = set()
    for split in requested:
        data = bundle.splits.get(split)
        records = bundle.metadata.get(split)
        if data is not None:
            if not isinstance(data, SplitArtifacts):
                raise ContractError(f"{split}: expected SplitArtifacts")
            if records is not None and records != data.records:
                raise ContractError(f"{split}: metadata and array records differ in membership or order")
            if records is None and data.records is not None:
                raise ContractError(f"{split}: metadata missing for present array records")
        if records is None:
            absent(f"{split}_ids.csv")
        elif not records:
            raise ContractError(f"{split}: IDs records have no samples")
        seen = set()
        for record in records or ():
            _check_record(record, f"{bundle.modality}/{split}")
            if record.sample_id in seen:
                raise ContractError(f"{bundle.modality}/{split}: duplicate sample_id {record.sample_id!r}")
            seen.add(record.sample_id)
            if category_labels.setdefault(record.category, record.label) != record.label:
                raise ContractError(f"{split}: category {record.category!r} has conflicting label indices")
            if label_categories.setdefault(record.label, record.category) != record.category:
                raise ContractError(f"{split}: label {record.label} has conflicting categories")
            if bundle.class_names is not None and bundle.class_names[record.label] != record.category:
                raise ContractError(f"{split}/{record.sample_id}: category conflicts with class mapping")
        if (records is not None and bundle.expected_counts is not None
                and split in bundle.expected_counts and len(records) != bundle.expected_counts[split]):
            raise ContractError(f"{bundle.modality}/{split}: expected {bundle.expected_counts[split]} samples, got {len(records)}")
        if data is None or data.embeddings is None:
            absent(f"{split}_emb.npy")
        if bundle.modality == "audio" and (data is None or data.probabilities is None):
            absent(f"{split}_probs.npy")
        if data is None:
            continue
        if data.probabilities is not None and bundle.class_names is None:
            absent("class_names.json")
            notes.append(f"{bundle.modality}/{split}: probability semantics pending explicit class_names.json")
        available = next((array for array in (
            data.embeddings, data.probabilities, data.input_validity, data.timestamps
        ) if array is not None), None)
        if available is not None and not isinstance(available, np.ndarray):
            raise ContractError(f"{split}: expected NumPy arrays")
        n = len(records) if records is not None else (available.shape[0] if available is not None and available.ndim else 0)
        if available is not None and n < 1:
            raise ContractError(f"{split}: arrays must contain at least one sample")
        _validate_arrays(
            embeddings=data.embeddings, probabilities=data.probabilities,
            input_validity=data.input_validity, timestamps=data.timestamps,
            n=n, modality=bundle.modality, split=split,
        )
        if data.embeddings is not None:
            dimensions.add(data.embeddings.shape[-1])
        for field in ("input_validity", "timestamps"):
            if getattr(data, field) is None:
                notes.append(f"{bundle.modality}/{split}: {field} unavailable (unknown, not inferred)")
        if data.probabilities is None:
            notes.append(f"{bundle.modality}/{split}: probabilities unavailable")
    if len(dimensions) > 1:
        raise ContractError(f"{bundle.modality}: embedding dimension differs across splits")
    if bundle.class_names is None:
        notes.append("Class mapping unavailable; full semantic ID validation is pending cross-modal comparison.")
    missing_inputs = tuple(dict.fromkeys(missing))
    return ValidationReport(not missing_inputs, missing_inputs, tuple(notes))


def load_bundle(
    root: str | Path,
    modality: Modality,
    *,
    splits: Sequence[str] = SPLITS,
    expected_counts: Mapping[str, int] | None = OFFICIAL_COUNTS,
    hash_files: bool = False,
) -> Bundle:
    """Read explicit local paths only. Missing required files produce a partial bundle.

    Audio final files use float16. Vision D is discovered, with float16/32/64
    supported. Official AVE counts are enforced by default. For explicit subsets
    or synthetic fixtures pass expected_counts=None (or a custom count mapping).
    """
    if isinstance(root, str) and not root.strip():
        raise ContractError("An explicit nonempty bundle directory path is required")
    root = Path(root).resolve()
    if not root.is_dir():
        raise ContractError(f"Bundle directory does not exist or is not a directory: {root}")
    if modality not in ("audio", "vision"):
        raise ContractError(f"Unknown modality {modality!r}; expected audio or vision")
    requested = tuple(splits)
    if not requested or len(set(requested)) != len(requested) or any(s not in SPLITS for s in requested):
        raise ContractError("splits must be unique names from train, val, test")
    missing: list[str] = []
    identities: dict[str, dict] = {}

    def inspect(name: str, required: bool = False) -> Path | None:
        path = root / name
        if not path.exists():
            if required:
                missing.append(str(path))
            return None
        if not path.is_file():
            raise ContractError(f"Expected artifact file: {path}")
        stat = path.stat()
        entry = {"path": str(path), "filename": name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if hash_files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            entry["sha256"] = digest.hexdigest()
        identities[name] = entry
        return path

    mapping_path = inspect("class_names.json", modality == "audio")
    names = _class_mapping(mapping_path) if mapping_path else None
    source_path = inspect("SOURCE.txt", modality == "audio")
    readme_path = inspect("README.md", modality == "audio")
    loaded: dict[str, SplitArtifacts] = {}
    metadata: dict[str, tuple[Record, ...]] = {}
    for split in requested:
        ids_path = inspect(f"{split}_ids.csv", True)
        emb_path = inspect(f"{split}_emb.npy", True)
        probs_path = inspect(f"{split}_probs.npy", modality == "audio")
        valid_path = inspect(f"{split}_valid.npy")
        times_path = inspect(f"{split}_timestamps.npy")
        records = _read_records(ids_path) if ids_path else None
        if records is not None:
            metadata[split] = records
            if expected_counts is not None and split in expected_counts and len(records) != expected_counts[split]:
                raise ContractError(f"{modality}/{split}: expected {expected_counts[split]} samples, got {len(records)}")
        arrays: dict[str, NDArray | None] = {}
        for key, path in (("embeddings", emb_path), ("probabilities", probs_path),
                          ("input_validity", valid_path), ("timestamps", times_path)):
            arrays[key] = _load_array(path) if path else None
            if path:
                identities[path.name].update(shape=list(arrays[key].shape), dtype=str(arrays[key].dtype))
        if records is None or emb_path is None:
            # Missing IDs/embeddings must not hide corrupt arrays that do exist.
            first = next((array for array in arrays.values() if array is not None), None)
            n = len(records) if records is not None else (first.shape[0] if first is not None and first.ndim else 0)
            if first is not None:
                if n < 1:
                    raise ContractError(f"{modality}/{split}: arrays must contain at least one sample")
                _validate_arrays(**arrays, n=n, modality=modality, split=split)
        if names is None and probs_path and str(root / "class_names.json") not in missing:
            inspect("class_names.json", True)
        if records is not None and emb_path:
            data = SplitArtifacts(records=records, **arrays)
            loaded[split] = data
    provenance = {
        "root": str(root), "modality": modality, "requested_splits": list(requested),
        "files": identities, "class_names": list(names) if names else None,
        "source": source_path.read_text(encoding="utf-8") if source_path else None,
        "readme_available": readme_path is not None,
        "splits": {s: {"sample_count": len(records), "embedding_dimension":
                       loaded[s].embeddings.shape[-1] if s in loaded and loaded[s].embeddings.ndim else None}
                   for s, records in metadata.items()},
    }
    bundle = Bundle(root, modality, requested, loaded, metadata, names, tuple(missing), provenance, expected_counts)
    validate_bundle(bundle)
    return bundle


def annotation_activity(records: Sequence[Record]) -> NDArray:
    """Ground-truth joint event activity only, never an input-validity mask.

    Task-specific target-supervision masks must be provided by future consumers;
    no correspondence labels or modality-specific presence labels are invented.
    """
    for record in records:
        _check_interval(record.start, record.end, record.sample_id)
    seconds = np.arange(SECONDS)[None, :]
    start = np.array([r.start for r in records], dtype=np.int64)[:, None]
    end = np.array([r.end for r in records], dtype=np.int64)[:, None]
    return _readonly((start <= seconds) & (seconds < end))


def align_on_sample_id(audio: Bundle, vision: Bundle | None = None, *, split: str) -> Alignment:
    """Align in original audio order; retain both source indices and every unmatched ID.

    No arrays are reordered or truncated. Call require_complete() before future
    training uses the matched indices. Missing inputs never count as success.
    """
    if split not in SPLITS:
        raise ContractError(f"Unknown split {split!r}")
    if audio.modality != "audio" or (vision is not None and vision.modality != "vision"):
        raise ContractError("Alignment requires audio then vision bundles")
    reports = [validate_bundle(audio)]
    if vision is not None:
        reports.append(validate_bundle(vision))
        if audio.class_names is not None and vision.class_names is not None and audio.class_names != vision.class_names:
            raise ContractError("Audio and vision semantic class mappings differ; no automatic remapping")
        # A vision export without probabilities may lack a mapping. Validate each
        # observed category/index against the audio mapping rather than guessing it.
        if audio.class_names is not None:
            for records in vision.metadata.values():
                for record in records:
                    if audio.class_names[record.label] != record.category:
                        raise ContractError(f"vision/{record.sample_id}: category conflicts with audio class mapping")
    missing = [item for report in reports for item in report.missing_inputs]
    a = audio.splits.get(split)
    v = vision.splits.get(split) if vision else None
    if a is None:
        missing.append(f"audio/{split}: required split unavailable")
    if v is None:
        missing.append(f"vision/{split}: required split unavailable")
    # For array-bearing splits the array-associated sequence is authoritative.
    # validate_bundle has already checked its equality to the metadata view.
    a_records = (a.records if a else audio.metadata.get(split)) or ()
    v_records = (v.records if v else vision.metadata.get(split) if vision else None) or ()
    lookup = {record.sample_id: (i, record) for i, record in enumerate(v_records)}
    a_ids = {record.sample_id for record in a_records}
    matched, ai, vi, unmatched_a = [], [], [], []
    for i, record in enumerate(a_records):
        other = lookup.get(record.sample_id)
        if other is None:
            unmatched_a.append(record.sample_id)
            continue
        j, visual_record = other
        if record != visual_record:
            raise ContractError(f"{split}/{record.sample_id}: conflicting audio/vision annotation metadata")
        matched.append(record.sample_id)
        ai.append(i)
        vi.append(j)
    return Alignment(
        split, tuple(matched), _readonly(np.array(ai, dtype=np.int64)),
        _readonly(np.array(vi, dtype=np.int64)), tuple(unmatched_a),
        tuple(r.sample_id for r in v_records if r.sample_id not in a_ids),
        tuple(missing), tuple(note for report in reports for note in report.notes),
        audio_row_count=a.embeddings.shape[0] if a is not None and a.embeddings is not None else None,
        visual_row_count=v.embeddings.shape[0] if v is not None and v.embeddings is not None else None,
    )


def audit_leakage(bundle: Bundle) -> LeakageReport:
    """Report annotation/source overlap without changing official partitions.

    Inspect audio and vision separately, including unmatched rows. This checks
    declared source IDs, not file hashes or perceptually duplicated uploads.
    """
    duplicate, multiples, sample_sets, source_sets = {}, {}, {}, {}
    for split in SPLITS:
        if split not in bundle.metadata:
            continue
        records = bundle.metadata[split]
        seen, repeats, clips = set(), set(), {}
        for record in records:
            if record.sample_id in seen:
                repeats.add(record.sample_id)
            seen.add(record.sample_id)
            clips.setdefault(record.clip_id, set()).add(record.sample_id)
        duplicate[split] = tuple(sorted(repeats))
        multiples[split] = {clip: tuple(sorted(ids)) for clip, ids in sorted(clips.items()) if len(ids) > 1}
        sample_sets[split] = seen
        source_sets[split] = set(clips)
    pairs = list(combinations(sample_sets, 2))
    return LeakageReport(
        duplicate,
        {f"{a}&{b}": tuple(sorted(sample_sets[a] & sample_sets[b])) for a, b in pairs},
        {f"{a}&{b}": tuple(sorted(source_sets[a] & source_sets[b])) for a, b in pairs},
        multiples, tuple(s for s in SPLITS if s not in bundle.metadata),
    )

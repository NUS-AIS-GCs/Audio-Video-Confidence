"""Task 2.1 pair metadata and policy validation; no sampling or model inputs.

Original-pair targets describe recording provenance, not semantic compatibility,
temporal synchrony, or physical sound-source causality.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Integral
from typing import Sequence

from .contracts import (
    SPLITS, Bundle, ContractError, Record, _check_class_names, _check_record,
    align_on_sample_id, audit_leakage,
)

POLICY_VERSION = "original-pair-v1"
SAMPLING_METHOD = "sha256-rank-v1"


class PairType(str, Enum):
    ORIGINAL_POSITIVE = "ORIGINAL_POSITIVE"
    CROSS_CLASS_EASY_NEGATIVE = "CROSS_CLASS_EASY_NEGATIVE"
    CROSS_CLASS_HARD_NEGATIVE = "CROSS_CLASS_HARD_NEGATIVE"
    SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE = "SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE"


class SemanticCompatibility(str, Enum):
    """Annotated category compatibility proxy, not physical source truth."""

    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class TemporalStatus(str, Enum):
    NOT_ASSESSED = "NOT_ASSESSED"


class HardPolicyOrigin(str, Enum):
    MANUAL_APPROVED = "MANUAL_APPROVED"
    VALIDATION_CONFUSION = "VALIDATION_CONFUSION"


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ContractError(f"{name}: expected nonempty text without outer whitespace")


def _uint64(value: int, name: str) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or not 0 <= value < 2**64:
        raise ContractError(f"{name}: expected an unsigned 64-bit integer, not a boolean")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{name}: expected a lowercase SHA-256 digest")


def _hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_exact(value, contract_type: type, name: str) -> None:
    """Recheck supplied data with class-owned logic, never an instance hook."""
    if type(value) is not contract_type:
        raise ContractError(f"{name}: expected exact {contract_type.__name__}; subclasses are unsupported")
    try:
        contract_type.__post_init__(value)
    except AttributeError as exc:
        raise ContractError(f"{name}: missing required contract fields") from exc


def _is_enum_member(value, enum_type: type[Enum]) -> bool:
    # isinstance alone also accepts fabricated enum objects not in the schema.
    return type(value) is enum_type and any(value is member for member in enum_type)


@dataclass(frozen=True)
class HardNegativePolicy:
    """Explicit directed (audio category, visual category) neighborhoods.

    Approval/evidence fields are auditable declarations; validation cannot verify
    the truth of approval or that an external evidence file avoided test data.
    """

    policy_id: str
    origin: HardPolicyOrigin
    class_names: tuple[str, ...]
    category_pairs: tuple[tuple[str, str], ...]
    approved_by: str
    evidence_reference: str
    evidence_sha256: str
    evidence_split: str | None = None

    def __post_init__(self) -> None:
        for name in ("policy_id", "approved_by", "evidence_reference"):
            _text(getattr(self, name), name)
        if not _is_enum_member(self.origin, HardPolicyOrigin):
            raise ContractError("origin: expected HardPolicyOrigin enum")
        _digest(self.evidence_sha256, "evidence_sha256")
        if not isinstance(self.class_names, (tuple, list)):
            raise ContractError("class_names: expected an ordered sequence")
        names = tuple(self.class_names)
        _check_class_names(names, "hard policy vocabulary")
        if not isinstance(self.category_pairs, (tuple, list)) or not self.category_pairs:
            raise ContractError("category_pairs: expected nonempty ordered category pairs")
        edges = []
        for edge in self.category_pairs:
            if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                raise ContractError("category_pairs: expected (audio category, visual category)")
            a, v = edge
            if a not in names[:-1] or v not in names[:-1] or a == v:
                raise ContractError("hard categories must be distinct foreground vocabulary entries")
            edges.append((a, v))
        if edges != sorted(set(edges)):
            raise ContractError("category_pairs must be unique and canonically sorted")
        if self.origin == HardPolicyOrigin.VALIDATION_CONFUSION:
            if self.evidence_split != "val":
                raise ContractError("confusion evidence must come from val, never test")
        elif self.evidence_split is not None:
            raise ContractError("manual policy evidence_split must be None")
        object.__setattr__(self, "class_names", names)
        object.__setattr__(self, "category_pairs", tuple(edges))


@dataclass(frozen=True)
class SamplingPolicy:
    seed: int
    hard_negative_policy: HardNegativePolicy | None = None
    version: str = POLICY_VERSION

    def __post_init__(self) -> None:
        _uint64(self.seed, "seed")
        if self.version != POLICY_VERSION:
            raise ContractError("unsupported pair policy version")
        if self.hard_negative_policy is not None:
            _validate_exact(self.hard_negative_policy, HardNegativePolicy, "hard_negative_policy")

    @property
    def fingerprint(self) -> str:
        _validate_exact(self, SamplingPolicy, "policy")
        return _hash({"version": self.version, "seed": int(self.seed),
                      "method": SAMPLING_METHOD, "hard_policy":
                      asdict(self.hard_negative_policy) if self.hard_negative_policy else None})


def split_seed(seed: int, split: str) -> int:
    """Independent deterministic namespaces; no global RNG or Python hash()."""
    _uint64(seed, "seed")
    if split not in SPLITS:
        raise ContractError("unknown source split")
    return int(_hash([POLICY_VERSION, int(seed), split])[:16], 16)


@dataclass(frozen=True)
class SamplingProvenance:
    root_seed: int
    split_seed: int
    policy_fingerprint: str
    audio_metadata_fingerprint: str
    visual_metadata_fingerprint: str
    method: str = SAMPLING_METHOD

    def __post_init__(self) -> None:
        _uint64(self.root_seed, "root_seed")
        _uint64(self.split_seed, "split_seed")
        for name in ("policy_fingerprint", "audio_metadata_fingerprint", "visual_metadata_fingerprint"):
            _digest(getattr(self, name), name)
        if self.method != SAMPLING_METHOD:
            raise ContractError("unsupported or nondeterministic sampling method")


@dataclass(frozen=True)
class PairSource:
    split: str
    record: Record
    row_index: int

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ContractError("unknown source split")
        if type(self.record) is not Record:
            raise ContractError("pair source: expected exact Phase 1 Record; subclasses are unsupported")
        _check_record(self.record, "pair source")
        if not isinstance(self.row_index, Integral) or isinstance(self.row_index, bool) or self.row_index < 0:
            raise ContractError("row_index must be a nonnegative integer")


@dataclass(frozen=True)
class EventSupervision:
    """Annotation eligibility only; these are not input-validity masks.

    Unimodal entries refer to each source's original joint AVE annotation, not
    independent audibility/visibility. Swaps supply no fused presence target.
    """

    audio_annotation_eligible: bool
    visual_annotation_eligible: bool
    joint_event_eligible: bool
    semantic_eligible: bool

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in (
            self.audio_annotation_eligible, self.visual_annotation_eligible,
            self.joint_event_eligible, self.semantic_eligible,
        )):
            raise ContractError("event supervision eligibility must contain booleans")
        both = self.audio_annotation_eligible and self.visual_annotation_eligible
        if self.semantic_eligible != both or (self.joint_event_eligible and not both):
            raise ContractError("event supervision eligibility conflicts with source annotations")


@dataclass(frozen=True)
class PairRecord:
    audio: PairSource
    visual: PairSource
    pair_type: PairType
    policy: SamplingPolicy
    provenance: SamplingProvenance
    temporal_status: TemporalStatus = TemporalStatus.NOT_ASSESSED

    def __post_init__(self) -> None:
        # Recheck shared Phase 1 record guarantees without changing Phase 1.
        _validate_exact(self.audio, PairSource, "audio source")
        _validate_exact(self.visual, PairSource, "visual source")
        if self.audio.split != self.visual.split:
            raise ContractError("cross-split source borrowing is forbidden")
        if not _is_enum_member(self.pair_type, PairType):
            raise ContractError("pair_type: expected supported PairType enum; temporal shifts are unsupported")
        if self.temporal_status is not TemporalStatus.NOT_ASSESSED:
            raise ContractError("temporal synchrony claims are unsupported")
        _validate_exact(self.policy, SamplingPolicy, "policy")
        _validate_exact(self.provenance, SamplingProvenance, "provenance")
        if (self.provenance.root_seed != self.policy.seed
                or self.provenance.split_seed != split_seed(self.policy.seed, self.split)
                or self.provenance.policy_fingerprint != self.policy.fingerprint):
            raise ContractError("sampling provenance conflicts with policy or source split")
        a, v = self.audio.record, self.visual.record
        if (a.category == v.category) != (a.label == v.label):
            raise ContractError("conflicting category/index metadata")
        if self.pair_type == PairType.ORIGINAL_POSITIVE:
            if a != v:
                raise ContractError("original positive requires the same exact sample ID and metadata")
            return
        if a.clip_id == v.clip_id:
            raise ContractError("substituted pairs require different source clip IDs")
        if self.pair_type == PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE:
            if a.category != v.category:
                raise ContractError("same-class challenge requires matching category metadata")
            return
        if a.category == v.category:
            raise ContractError("cross-class negatives require different categories")
        hard = self.policy.hard_negative_policy
        in_hard_policy = hard is not None and (a.category, v.category) in hard.category_pairs
        if self.pair_type == PairType.CROSS_CLASS_HARD_NEGATIVE and not in_hard_policy:
            raise ContractError("hard negatives disabled without an explicit applicable approved policy")
        if self.pair_type == PairType.CROSS_CLASS_EASY_NEGATIVE and in_hard_policy:
            raise ContractError("approved hard-policy edge cannot be labeled easy")

    @property
    def split(self) -> str:
        return self.audio.split

    @property
    def identity(self) -> tuple[str, str, str]:
        # Type, seed, and row order cannot disguise a duplicate source pair.
        return self.split, self.audio.record.sample_id, self.visual.record.sample_id

    @property
    def original_pair_target(self) -> int:
        return int(self.pair_type == PairType.ORIGINAL_POSITIVE)

    @property
    def semantic_compatibility(self) -> SemanticCompatibility:
        """Annotation-category proxy only; empty annotations remain unknown."""
        if self.audio.record.end == 0 or self.visual.record.end == 0:
            return SemanticCompatibility.UNKNOWN
        if self.audio.record.category == self.visual.record.category:
            return SemanticCompatibility.COMPATIBLE
        return SemanticCompatibility.INCOMPATIBLE

    @property
    def event_supervision(self) -> EventSupervision:
        a, v = bool(self.audio.record.end > 0), bool(self.visual.record.end > 0)
        return EventSupervision(a, v, bool(a and v and self.original_pair_target), a and v)

    @property
    def sampling_priority(self) -> str:
        """Candidate rank definition for Task 2.2; does not select candidates."""
        return _hash([SAMPLING_METHOD, int(self.provenance.split_seed),
                      self.audio.record.sample_id, self.visual.record.sample_id, self.pair_type.value])


def pair_order_key(pair: PairRecord) -> tuple[int, str, str]:
    """Canonical output order, separate from candidate selection priority."""
    _validate_exact(pair, PairRecord, "pair")
    return SPLITS.index(pair.split), pair.visual.record.sample_id, pair.audio.record.sample_id


def make_sampling_provenance(
    audio: Bundle, vision: Bundle, *, split: str, policy: SamplingPolicy,
) -> SamplingProvenance:
    """Bind policy to complete Phase 1 rows; metadata hashes are not binary hashes."""
    _validate_exact(policy, SamplingPolicy, "policy")
    align_on_sample_id(audio, vision, split=split).require_complete()
    hard = policy.hard_negative_policy
    if hard is not None and hard.class_names != audio.class_names:
        raise ContractError("hard-policy vocabulary conflicts with bundle class mapping")

    def fingerprint(bundle: Bundle) -> str:
        return _hash({"modality": bundle.modality, "split": split,
                      "classes": bundle.class_names,
                      "rows": [{"sample_id": r.sample_id, "clip_id": r.clip_id,
                                "category": r.category, "label": int(r.label),
                                "start": int(r.start), "end": int(r.end)}
                               for r in bundle.splits[split].records]})

    return SamplingProvenance(int(policy.seed), split_seed(policy.seed, split), policy.fingerprint,
                              fingerprint(audio), fingerprint(vision))


@dataclass(frozen=True)
class SourceOverlap:
    modality: str
    split_pair: str
    clip_ids: tuple[str, ...]


@dataclass(frozen=True)
class PairValidationReport:
    pair_count: int
    source_overlaps: tuple[SourceOverlap, ...]
    missing_audio_metadata_splits: tuple[str, ...]
    missing_visual_metadata_splits: tuple[str, ...]


def validate_pair_manifest(
    pairs: Sequence[PairRecord], audio: Bundle, vision: Bundle, *, policy: SamplingPolicy,
) -> PairValidationReport:
    """Validate supplied records, not generate pairs or certify perceptual labels.

    Every reference must match its original array-associated Phase 1 row. Source
    overlap is reported, never repaired or presented as a source-disjoint split.
    """
    if not isinstance(pairs, (tuple, list)) or not pairs:
        raise ContractError("pairs must be a nonempty ordered list or tuple")
    _validate_exact(policy, SamplingPolicy, "policy")
    pairs = tuple(pairs)
    expected, seen, keys = {}, set(), []
    for pair in pairs:
        _validate_exact(pair, PairRecord, "pair")
        if pair.policy != policy:
            raise ContractError("manifest contains conflicting sampling policies")
        if pair.split not in expected:
            expected[pair.split] = make_sampling_provenance(audio, vision, split=pair.split, policy=policy)
        if pair.provenance != expected[pair.split]:
            raise ContractError("provenance does not match source metadata fingerprints")
        for source, bundle in ((pair.audio, audio), (pair.visual, vision)):
            records = bundle.splits[pair.split].records
            if source.row_index >= len(records) or records[source.row_index] != source.record:
                raise ContractError("source row index or metadata conflicts with Phase 1 bundle")
        if pair.identity in seen:
            raise ContractError("duplicate pair identity")
        seen.add(pair.identity)
        keys.append(pair_order_key(pair))
    if keys != sorted(keys):
        raise ContractError("pair manifest must use canonical deterministic ordering")
    a_report, v_report = audit_leakage(audio), audit_leakage(vision)
    overlaps = tuple(SourceOverlap(modality, key, ids)
                     for modality, report in (("audio", a_report), ("vision", v_report))
                     for key, ids in sorted(report.source_overlap.items()))
    return PairValidationReport(len(pairs), overlaps, a_report.missing_splits, v_report.missing_splits)


__all__ = [
    "PairType", "SemanticCompatibility", "TemporalStatus", "HardPolicyOrigin",
    "HardNegativePolicy", "SamplingPolicy", "SamplingProvenance", "PairSource",
    "EventSupervision", "PairRecord", "SourceOverlap", "PairValidationReport",
    "split_seed", "pair_order_key", "make_sampling_provenance", "validate_pair_manifest",
]

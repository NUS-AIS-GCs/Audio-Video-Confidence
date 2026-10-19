"""Deterministic Tasks 2.2--2.5 metadata generation; no training or file I/O."""
from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Integral

from .contracts import SPLITS, Bundle, ContractError, align_on_sample_id
from .pair_contracts import (
    SAMPLING_METHOD, PairRecord, PairSource, PairType, PairValidationReport,
    SamplingPolicy, SamplingProvenance, _digest, _hash, _is_enum_member,
    _validate_exact, make_sampling_provenance, pair_order_key, validate_pair_manifest,
)

GENERATOR_VERSION = "pair-generator-v1"
_TYPES = tuple(PairType)
_POSITIVE = PairType.ORIGINAL_POSITIVE
_EASY = PairType.CROSS_CLASS_EASY_NEGATIVE
_HARD = PairType.CROSS_CLASS_HARD_NEGATIVE
_CHALLENGE = PairType.SAME_CLASS_DIFFERENT_SOURCE_CHALLENGE


class SourceReuse(str, Enum):
    ALLOW = "ALLOW"
    UNIQUE_PER_TYPE = "UNIQUE_PER_TYPE"


class HardGenerationStatus(str, Enum):
    DISABLED_NO_POLICY = "DISABLED_NO_POLICY"
    DISABLED_BY_CONFIG = "DISABLED_BY_CONFIG"
    ENABLED = "ENABLED"


def _count(value: int, name: str) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ContractError(f"{name}: expected a nonnegative integer")


@dataclass(frozen=True)
class GenerationConfig:
    split: str
    positive_budget: int | None = None  # None retains every aligned annotation.
    easy_negative_budget: int = 0
    hard_negative_budget: int = 0
    challenge_budget: int = 0
    enable_hard_negatives: bool = False
    source_reuse: SourceReuse = SourceReuse.ALLOW
    max_candidate_evaluations: int = 20_000_000

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ContractError("unknown generation split")
        if self.positive_budget is not None:
            _count(self.positive_budget, "positive_budget")
        for name in ("easy_negative_budget", "hard_negative_budget", "challenge_budget",
                     "max_candidate_evaluations"):
            _count(getattr(self, name), name)
        if type(self.enable_hard_negatives) is not bool:
            raise ContractError("enable_hard_negatives must be a boolean")
        if self.hard_negative_budget and not self.enable_hard_negatives:
            raise ContractError("hard-negative budget requires explicit enablement")
        if not _is_enum_member(self.source_reuse, SourceReuse):
            raise ContractError("source_reuse must be a supported SourceReuse enum")
        if self.positive_budget == 0 and not any((self.easy_negative_budget,
                                                self.hard_negative_budget, self.challenge_budget)):
            raise ContractError("all-zero budgets cannot produce a nonempty pair manifest")


@dataclass(frozen=True)
class GenerationProvenance:
    settings_fingerprint: str
    source_provenance: SamplingProvenance
    version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        _digest(self.settings_fingerprint, "settings_fingerprint")
        _validate_exact(self.source_provenance, SamplingProvenance, "source provenance")
        if self.version != GENERATOR_VERSION:
            raise ContractError("unsupported generator version")


@dataclass(frozen=True)
class PairTypeCounts:
    pair_type: PairType
    eligible: int
    requested: int
    selected: int

    def __post_init__(self) -> None:
        if not _is_enum_member(self.pair_type, PairType):
            raise ContractError("invalid pair-count type")
        for name in ("eligible", "requested", "selected"):
            _count(getattr(self, name), name)
        if self.selected != self.requested or self.selected > self.eligible:
            raise ContractError("selected counts must satisfy the exact requested quota")

    @property
    def unselected(self) -> int:
        return self.eligible - self.selected


@dataclass(frozen=True)
class GenerationReport:
    counts: tuple[PairTypeCounts, ...]
    audio_rows: int
    visual_rows: int
    candidate_evaluations: int
    same_source_exclusions: int
    same_sample_exclusions: int  # Subset of same_source_exclusions, not additive.
    unselected_positive_ids: tuple[str, ...]
    hard_status: HardGenerationStatus
    source_validation: PairValidationReport

    def __post_init__(self) -> None:
        if type(self.counts) is not tuple or len(self.counts) != len(_TYPES):
            raise ContractError("counts must contain every pair type in enum order")
        for item, kind in zip(self.counts, _TYPES):
            _validate_exact(item, PairTypeCounts, "pair counts")
            if item.pair_type is not kind:
                raise ContractError("pair-count order conflicts with schema")
        for name in ("audio_rows", "visual_rows", "candidate_evaluations",
                     "same_source_exclusions", "same_sample_exclusions"):
            _count(getattr(self, name), name)
        ids = self.unselected_positive_ids
        if (type(ids) is not tuple or any(not isinstance(s, str) or not s for s in ids)
                or ids != tuple(sorted(set(ids)))):
            raise ContractError("unselected positive IDs must be a sorted unique tuple")
        if not _is_enum_member(self.hard_status, HardGenerationStatus):
            raise ContractError("unsupported hard-generation status")
        if type(self.source_validation) is not PairValidationReport:
            raise ContractError("source validation report must use the Task 2.1 report type")


@dataclass(frozen=True)
class GeneratedManifest:
    config: GenerationConfig
    policy: SamplingPolicy
    primary_pairs: tuple[PairRecord, ...]
    challenge_pairs: tuple[PairRecord, ...]
    provenance: GenerationProvenance
    report: GenerationReport

    def __post_init__(self) -> None:
        _validate_exact(self.config, GenerationConfig, "generation config")
        _validate_exact(self.policy, SamplingPolicy, "sampling policy")
        _validate_exact(self.provenance, GenerationProvenance, "generation provenance")
        _validate_exact(self.report, GenerationReport, "generation report")
        for name in ("primary_pairs", "challenge_pairs"):
            pairs = getattr(self, name)
            if type(pairs) is not tuple:
                raise ContractError(f"{name} must be an immutable tuple")
            for pair in pairs:
                _validate_exact(pair, PairRecord, "generated pair")


def _settings(config: GenerationConfig, policy: SamplingPolicy) -> str:
    fields = asdict(config)
    # Canonicalize NumPy integers accepted by the integer contracts.
    for name, value in fields.items():
        if isinstance(value, Integral) and not isinstance(value, bool):
            fields[name] = int(value)
    return _hash({"version": GENERATOR_VERSION, "config": fields, "policy": policy.fingerprint})


def _quotas(config: GenerationConfig, n: int) -> dict[PairType, int]:
    return dict(zip(_TYPES, (n if config.positive_budget is None else int(config.positive_budget),
                            int(config.easy_negative_budget), int(config.hard_negative_budget),
                            int(config.challenge_budget))))


def _hard_status(config: GenerationConfig, policy: SamplingPolicy) -> HardGenerationStatus:
    if policy.hard_negative_policy is None:
        if config.enable_hard_negatives:
            raise ContractError("hard negatives enabled but no approved hard policy was supplied")
        return HardGenerationStatus.DISABLED_NO_POLICY
    return HardGenerationStatus.ENABLED if config.enable_hard_negatives else HardGenerationStatus.DISABLED_BY_CONFIG


def _census(a_records, v_records, edges):
    """Exact pool/exclusion counts without materializing the Cartesian product."""
    ac, vc = Counter(r.category for r in a_records), Counter(r.category for r in v_records)
    a_clip, v_clip = defaultdict(Counter), defaultdict(Counter)
    for r in a_records:
        a_clip[r.clip_id][r.category] += 1
    for r in v_records:
        v_clip[r.clip_id][r.category] += 1
    common = set(a_clip) & set(v_clip)
    same_source = sum(sum(a_clip[c].values()) * sum(v_clip[c].values()) for c in common)
    same_category_source = sum(n * v_clip[c][cat] for c in common for cat, n in a_clip[c].items())
    challenge = sum(n * vc[cat] for cat, n in ac.items()) - same_category_source
    hard = sum(ac[a] * vc[v] for a, v in edges)
    hard -= sum(a_clip[c][a] * v_clip[c][v] for c in common for a, v in edges)
    easy = len(a_records) * len(v_records) - same_source - challenge - hard
    same_sample = len({r.sample_id for r in a_records} & {r.sample_id for r in v_records})
    return {_POSITIVE: len(a_records), _EASY: easy, _HARD: hard, _CHALLENGE: challenge}, same_source, same_sample


def _kind(a, v, edges):
    if a.clip_id == v.clip_id:
        return None
    if a.category == v.category:
        return _CHALLENGE
    return _HARD if (a.category, v.category) in edges else _EASY


@dataclass(frozen=True)
class _Candidate:
    audio: PairSource
    visual: PairSource
    pair_type: PairType
    priority: str

    @property
    def key(self):
        return self.priority, self.visual.record.sample_id, self.audio.record.sample_id, self.pair_type.value


def _candidate(a, v, kind, provenance):
    # Reuse Task 2.1's canonical hash and exact priority schema. PairRecord later
    # derives every label/mask and revalidates the selected metadata.
    priority = _hash([SAMPLING_METHOD, int(provenance.split_seed),
                      a.record.sample_id, v.record.sample_id, kind.value])
    return _Candidate(a, v, kind, priority)


@dataclass(frozen=True)
class _WorstFirst:
    candidate: _Candidate

    def __lt__(self, other):
        return self.candidate.key > other.candidate.key


class _Pool:
    def __init__(self, quota, reuse):
        self.quota, self.reuse = quota, reuse
        self.heap = []
        self.edges = {}

    def add(self, candidate):
        if self.reuse is SourceReuse.ALLOW:
            item = _WorstFirst(candidate)
            if len(self.heap) < self.quota:
                heapq.heappush(self.heap, item)
            elif candidate.key < self.heap[0].candidate.key:
                heapq.heapreplace(self.heap, item)
        else:
            edge = candidate.visual.record.clip_id, candidate.audio.record.clip_id
            current = self.edges.get(edge)
            if current is None or candidate.key < current.key:
                self.edges[edge] = candidate

    def selected(self):
        if self.reuse is SourceReuse.ALLOW:
            return [item.candidate for item in self.heap]
        return _unique_source_selection(self.edges.values(), self.quota)


def _unique_source_selection(candidates, quota):
    """Deterministic augmenting paths; unlike greedy selection, finds feasible quotas."""
    graph = defaultdict(list)
    for candidate in candidates:
        graph[candidate.visual.record.clip_id].append(candidate)
    for neighbors in graph.values():
        neighbors.sort(key=lambda c: c.key)
    roots = sorted(graph, key=lambda v: (graph[v][0].key, v))
    right_match, selected = {}, {}
    for root in roots:
        if len(selected) == quota:
            break
        queue, seen_left, parent = deque([root]), {root}, {}
        free = None
        while queue and free is None:
            left = queue.popleft()
            for candidate in graph[left]:
                right = candidate.audio.record.clip_id
                if right in parent:
                    continue
                parent[right] = (left, candidate)
                if right not in right_match:
                    free = right
                    break
                other = right_match[right]
                if other not in seen_left:
                    seen_left.add(other)
                    queue.append(other)
        if free is not None:
            right = free
            while True:
                left, candidate = parent[right]
                previous = selected.get(left)
                selected[left], right_match[right] = candidate, left
                if previous is None:
                    break
                right = previous.audio.record.clip_id
    if len(selected) != quota:
        raise ContractError(f"insufficient distinct-source matching: requested {quota}, maximum {len(selected)}")
    return list(selected.values())


def _report(config, policy, a_records, v_records, eligible, same_source, same_sample, pairs, validation):
    quotas = _quotas(config, len(a_records))
    selected = Counter(p.pair_type for p in pairs)
    positive_ids = {p.audio.record.sample_id for p in pairs if p.pair_type is _POSITIVE}
    return GenerationReport(
        tuple(PairTypeCounts(kind, eligible[kind], quotas[kind], selected[kind]) for kind in _TYPES),
        len(a_records), len(v_records),
        len(a_records) * len(v_records) if any(quotas[k] for k in _TYPES[1:]) else 0,
        same_source, same_sample,
        tuple(sorted(r.sample_id for r in a_records if r.sample_id not in positive_ids)),
        _hard_status(config, policy), validation,
    )


def validate_generated_manifest(manifest: GeneratedManifest, audio: Bundle, vision: Bundle) -> PairValidationReport:
    """Revalidate identities, partitions, quotas, settings, source reuse and reports.

    Does not certify which top-ranked candidates were selected by an external
    producer; generate_pairs owns selection. No artifact bytes are hashed here.
    """
    _validate_exact(manifest, GeneratedManifest, "generated manifest")
    config, policy = manifest.config, manifest.policy
    source = make_sampling_provenance(audio, vision, split=config.split, policy=policy)
    expected_provenance = GenerationProvenance(_settings(config, policy), source)
    if manifest.provenance != expected_provenance:
        raise ContractError("generation provenance conflicts with settings or source metadata")
    if any(p.pair_type is _CHALLENGE for p in manifest.primary_pairs):
        raise ContractError("challenge pairs cannot enter the primary distribution")
    if any(p.pair_type is not _CHALLENGE for p in manifest.challenge_pairs):
        raise ContractError("challenge partition contains a primary pair")
    for partition in (manifest.primary_pairs, manifest.challenge_pairs):
        keys = [pair_order_key(p) for p in partition]
        if keys != sorted(keys):
            raise ContractError("generated partitions must have canonical deterministic ordering")
    pairs = tuple(sorted(manifest.primary_pairs + manifest.challenge_pairs, key=pair_order_key))
    if any(p.split != config.split for p in pairs):
        raise ContractError("generated pairs conflict with configured split")
    validation = validate_pair_manifest(pairs, audio, vision, policy=policy)
    a_records, v_records = audio.splits[config.split].records, vision.splits[config.split].records
    edges = frozenset(policy.hard_negative_policy.category_pairs) if policy.hard_negative_policy else frozenset()
    eligible, same_source, same_sample = _census(a_records, v_records, edges)
    expected_report = _report(config, policy, a_records, v_records, eligible, same_source, same_sample, pairs, validation)
    if manifest.report != expected_report:
        raise ContractError("generation counts, exclusions or source report are inconsistent")
    if expected_report.candidate_evaluations > config.max_candidate_evaluations:
        raise ContractError("candidate evaluation budget exceeded")
    if config.source_reuse is SourceReuse.UNIQUE_PER_TYPE:
        for kind in _TYPES[1:]:
            subset = [p for p in pairs if p.pair_type is kind]
            for side in ("audio", "visual"):
                clips = [getattr(p, side).record.clip_id for p in subset]
                if len(set(clips)) != len(clips):
                    raise ContractError("source reuse violates UNIQUE_PER_TYPE")
    return validation


def generate_pairs(audio: Bundle, vision: Bundle, *, config: GenerationConfig,
                   policy: SamplingPolicy) -> GeneratedManifest:
    """Generate one explicit split, with exact global budgets and separate challenges."""
    _validate_exact(config, GenerationConfig, "generation config")
    source_provenance = make_sampling_provenance(audio, vision, split=config.split, policy=policy)
    _hard_status(config, policy)
    alignment = align_on_sample_id(audio, vision, split=config.split)
    alignment.require_complete()
    a_records, v_records = audio.splits[config.split].records, vision.splits[config.split].records
    a_sources = tuple(PairSource(config.split, r, i) for i, r in enumerate(a_records))
    v_sources = tuple(PairSource(config.split, r, i) for i, r in enumerate(v_records))
    quotas = _quotas(config, len(a_records))
    edges = frozenset(policy.hard_negative_policy.category_pairs) if policy.hard_negative_policy else frozenset()
    eligible, same_source, same_sample = _census(a_records, v_records, edges)
    for kind in _TYPES:
        if quotas[kind] > eligible[kind]:
            raise ContractError(f"{kind.value}: requested {quotas[kind]}, only {eligible[kind]} eligible pairs")
    substitutions = any(quotas[k] for k in _TYPES[1:])
    if substitutions and len(a_records) * len(v_records) > config.max_candidate_evaluations:
        raise ContractError("candidate evaluation budget exceeded before enumeration")
    positives = (_candidate(a_sources[int(ai)], v_sources[int(vi)], _POSITIVE, source_provenance)
                 for ai, vi in zip(alignment.audio_indices, alignment.visual_indices))
    selected = heapq.nsmallest(quotas[_POSITIVE], positives, key=lambda c: c.key)
    pools = {kind: _Pool(quotas[kind], config.source_reuse) for kind in _TYPES[1:] if quotas[kind]}
    if substitutions:
        for a in a_sources:
            for v in v_sources:
                kind = _kind(a.record, v.record, edges)
                if kind in pools:
                    pools[kind].add(_candidate(a, v, kind, source_provenance))
        for kind, pool in pools.items():
            try:
                selected.extend(pool.selected())
            except ContractError as exc:
                raise ContractError(f"{kind.value}: {exc}") from exc
    records = [PairRecord(c.audio, c.visual, c.pair_type, policy, source_provenance) for c in selected]
    primary = tuple(sorted((p for p in records if p.pair_type is not _CHALLENGE), key=pair_order_key))
    challenges = tuple(sorted((p for p in records if p.pair_type is _CHALLENGE), key=pair_order_key))
    pairs = tuple(sorted(primary + challenges, key=pair_order_key))
    validation = validate_pair_manifest(pairs, audio, vision, policy=policy)
    result = GeneratedManifest(
        config, policy, primary, challenges, GenerationProvenance(_settings(config, policy), source_provenance),
        _report(config, policy, a_records, v_records, eligible, same_source, same_sample, pairs, validation),
    )
    validate_generated_manifest(result, audio, vision)
    return result


__all__ = [
    "SourceReuse", "HardGenerationStatus", "GenerationConfig", "GenerationProvenance",
    "PairTypeCounts", "GenerationReport", "GeneratedManifest", "generate_pairs", "validate_generated_manifest",
]

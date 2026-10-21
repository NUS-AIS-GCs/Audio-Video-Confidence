"""Task 2.6 diagnostics for validated in-memory manifests, never binary proof.

Reports are immutable output snapshots. They are not validation credentials:
call audit_generated_manifest again against current inputs before consumption.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from .contracts import SPLITS, Bundle, audit_leakage
from .pair_contracts import HardNegativePolicy, PairType, SemanticCompatibility, _hash
from .pair_generator import (
    GeneratedManifest, GenerationProvenance, HardGenerationStatus, PairTypeCounts,
    validate_generated_manifest,
)


@dataclass(frozen=True)
class MetadataLeakage:
    """Both overlap types cover full available metadata, including unselected rows."""

    modality: str
    metadata_fingerprint: str
    split_row_counts: tuple[tuple[str, int], ...]
    duplicate_sample_ids: tuple[tuple[str, tuple[str, ...]], ...]
    sample_overlap: tuple[tuple[str, tuple[str, ...]], ...]
    source_overlap: tuple[tuple[str, tuple[str, ...]], ...]
    multiple_annotations: tuple[tuple[str, str, tuple[str, ...]], ...]
    missing_metadata_splits: tuple[str, ...]


@dataclass(frozen=True)
class PairIntegrityReport:
    split: str
    pair_count: int
    primary_pair_count: int
    challenge_pair_count: int
    pair_type_counts: tuple[PairTypeCounts, ...]
    source_row_references_checked: int
    same_source_substitutions: int
    cross_split_pairs: int
    duplicate_pair_identities: tuple[tuple[str, str, str], ...]
    supplied_split_membership_preserved: bool
    empty_audio_source_pairs: int
    empty_visual_source_pairs: int
    unknown_semantic_pairs: int
    generation_provenance: GenerationProvenance
    hard_generation_status: HardGenerationStatus
    declared_hard_policy: HardNegativePolicy | None
    metadata_leakage: tuple[MetadataLeakage, ...]
    metadata_integrity: Literal["VALIDATED_IN_MEMORY"] = "VALIDATED_IN_MEMORY"
    artifact_binary_integrity: Literal["NOT_VERIFIED"] = "NOT_VERIFIED"
    evidence_contents_verified: Literal[False] = False


def _metadata_leakage(bundle: Bundle) -> MetadataLeakage:
    leakage = audit_leakage(bundle)
    # Unlike a selected split's sampling provenance, this binds the entire
    # available leakage context. Canonicalize accepted NumPy integer metadata.
    metadata = {
        split: [{"sample_id": r.sample_id, "clip_id": r.clip_id,
                 "category": r.category, "label": int(r.label),
                 "start": int(r.start), "end": int(r.end)} for r in bundle.metadata[split]]
        for split in SPLITS if split in bundle.metadata
    }
    fingerprint = _hash({"version": "pair-integrity-v1", "modality": bundle.modality,
                         "classes": bundle.class_names, "metadata": metadata,
                         "requested_splits": [s for s in SPLITS if s in bundle.requested_splits]})
    return MetadataLeakage(
        bundle.modality, fingerprint,
        tuple((s, len(bundle.metadata[s])) for s in SPLITS if s in bundle.metadata),
        tuple((s, leakage.duplicate_sample_ids[s]) for s in SPLITS if s in bundle.metadata),
        tuple(sorted(leakage.sample_overlap.items())),
        tuple(sorted(leakage.source_overlap.items())),
        tuple((s, clip, ids) for s in SPLITS if s in bundle.metadata
              for clip, ids in sorted(leakage.multiple_annotations[s].items())),
        leakage.missing_splits,
    )


def audit_generated_manifest(
    manifest: GeneratedManifest, audio: Bundle, vision: Bundle,
) -> PairIntegrityReport:
    """Fail closed on invalid input, then snapshot leakage and integrity accounting.

    Uses the existing full validator, including source alignment, exact types,
    policy evidence declarations, quotas, partitioning, order and fingerprints.
    Same-source/cross-split substitutions and duplicate pairs therefore raise
    ContractError rather than returning a seemingly successful report. Cross-
    split metadata overlaps remain findings; no official assignments are edited.

    Membership preservation means relative to the supplied Phase 1 bundles.
    Neither their official origin nor embedding/source/evidence bytes are hashed
    or authenticated here. This also does not certify externally selected ranks.
    """
    validate_generated_manifest(manifest, audio, vision)
    pairs = manifest.primary_pairs + manifest.challenge_pairs
    identities = Counter(p.identity for p in pairs)
    preserved = all(
        source.split == manifest.config.split
        and bundle.splits[source.split].records[source.row_index] == source.record
        for pair in pairs for source, bundle in ((pair.audio, audio), (pair.visual, vision))
    )
    return PairIntegrityReport(
        split=manifest.config.split,
        pair_count=len(pairs), primary_pair_count=len(manifest.primary_pairs),
        challenge_pair_count=len(manifest.challenge_pairs),
        pair_type_counts=manifest.report.counts,
        source_row_references_checked=2 * len(pairs),
        same_source_substitutions=sum(
            p.pair_type is not PairType.ORIGINAL_POSITIVE
            and p.audio.record.clip_id == p.visual.record.clip_id for p in pairs),
        cross_split_pairs=sum(p.audio.split != p.visual.split for p in pairs),
        duplicate_pair_identities=tuple(sorted(key for key, count in identities.items() if count > 1)),
        supplied_split_membership_preserved=preserved,
        empty_audio_source_pairs=sum(bool(p.audio.record.end == 0) for p in pairs),
        empty_visual_source_pairs=sum(bool(p.visual.record.end == 0) for p in pairs),
        unknown_semantic_pairs=sum(p.semantic_compatibility is SemanticCompatibility.UNKNOWN for p in pairs),
        generation_provenance=manifest.provenance,
        hard_generation_status=manifest.report.hard_status,
        declared_hard_policy=manifest.policy.hard_negative_policy,
        metadata_leakage=(_metadata_leakage(audio), _metadata_leakage(vision)),
    )


__all__ = ["MetadataLeakage", "PairIntegrityReport", "audit_generated_manifest"]

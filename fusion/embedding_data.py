"""Read-only Phase 2 embedding adaptation; no fitting, scoring or file writes."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import Dataset

from .artifact_integration import _json, _read_json, manifest_from_json, manifest_to_json
from .contracts import Bundle, ContractError, SPLITS, validate_bundle
from .pair_contracts import PairType, SamplingPolicy
from .pair_generator import GeneratedManifest, GenerationConfig, SourceReuse, generate_pairs


def _snapshot(value):
    """Immutable byte backing also prevents re-enabling NumPy write access."""
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def _array_hash(value):
    digest = hashlib.sha256()
    digest.update(str((value.shape, value.dtype.str)).encode())
    digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class EmbeddingFeatures:
    audio: np.ndarray
    visual: np.ndarray


@dataclass(frozen=True)
class EmbeddingProvenance:
    split: str
    manifest_sha256: str
    policy_fingerprint: str
    artifact_fingerprint: str | None
    # Hashes of actual in-memory source arrays, not claims of checkpoint origin.
    source_array_hashes: tuple[tuple[str, str], ...]
    input_export_hashes: tuple[tuple[str, str], ...]


class EmbeddingDataset(Dataset):
    """A validated primary OR challenge partition with lazy pair row gathering.

    Source embeddings are snapshotted once per modality, not duplicated for
    every negative. Items are (feature_dict, float32 target[1]); metadata stays
    in the immutable pair tuple. The boolean mask is the intersection of explicit
    producer validity masks. Missing masks mean all ten exported bins are used,
    NOT that decoded synchronization or physical validity has been verified.
    Annotation intervals never determine this mask.
    """

    def __init__(self, manifest: GeneratedManifest, audio: Bundle, vision: Bundle,
                 *, split: str, partition: str = "primary",
                 limit: int | None = None, artifact_fingerprint: str | None = None):
        if split not in SPLITS or partition not in ("primary", "challenge"):
            raise ContractError("unknown embedding split or partition")
        if limit is not None and (not isinstance(limit, Integral)
                                  or isinstance(limit, bool) or limit < 0):
            raise ContractError("limit must be a nonnegative integer or None")
        if audio.modality != "audio" or vision.modality != "vision":
            raise ContractError("expected audio and vision bundles in that order")
        for bundle in (audio, vision):
            if not validate_bundle(bundle).complete:
                raise ContractError("complete validated source bundles are required")
        # Audited Phase 2 serializer validates artifacts, pair identities, source
        # references, partitions, quotas, ordering, reports and provenance.
        serialized = manifest_to_json(manifest, audio, vision,
                                      artifact_fingerprint=artifact_fingerprint)
        if manifest.config.split != split:
            raise ContractError("requested split conflicts with manifest")
        originals = {p.audio.record.sample_id for p in manifest.primary_pairs
                     if p.pair_type is PairType.ORIGINAL_POSITIVE}
        if originals != {r.sample_id for r in audio.splits[split].records}:
            raise ContractError("full manifest must cover every original annotation")
        allowed = {PairType.ORIGINAL_POSITIVE, PairType.CROSS_CLASS_EASY_NEGATIVE}
        if any(p.pair_type not in allowed for p in manifest.primary_pairs):
            raise ContractError("Phase 4 primary pairs permit originals and easy negatives only")
        arrays, masks, hashes, exports = [], [], [], []
        for bundle, dim in ((audio, 256), (vision, 512)):
            source = bundle.splits[split]
            emb = source.embeddings
            if emb.shape != (len(source.records), 10, dim):
                raise ContractError(f"{bundle.modality} embeddings require (N,10,{dim})")
            value = _snapshot(emb)
            # Phase 1 checks finiteness before the snapshot; recheck the copy.
            if not np.isfinite(value).all():
                raise ContractError("nonfinite embedding snapshot")
            # Vision contracts also permit float64 exports. A finite source
            # value outside float32 range would become Inf in minibatch tensors.
            float32_limit = np.finfo(np.float32).max
            if value.min() < -float32_limit or value.max() > float32_limit:
                raise ContractError("embedding values exceed the finite float32 minibatch range")
            arrays.append(value)
            valid = (np.ones((len(value), 10), dtype=bool) if source.input_validity is None
                     else source.input_validity)
            masks.append(_snapshot(valid))
            for name, array in (("embeddings", value), ("validity", masks[-1])):
                hashes.append((f"{bundle.modality}/{split}/{name}", _array_hash(array)))
            for name, entry in sorted(bundle.provenance.get("files", {}).items()):
                if "sha256" in entry:
                    exports.append((f"{bundle.modality}/{name}", entry["sha256"]))
        pairs = manifest.primary_pairs if partition == "primary" else manifest.challenge_pairs
        self._pairs = pairs if limit is None else pairs[:int(limit)]
        self._split, self._partition = split, partition
        self._complete = limit is None
        self._audio, self._visual = arrays
        self._audio_valid, self._visual_valid = masks
        self._ai = _snapshot(np.array([p.audio.row_index for p in self._pairs], dtype=np.intp))
        self._vi = _snapshot(np.array([p.visual.row_index for p in self._pairs], dtype=np.intp))
        self._targets = _snapshot(np.array([p.original_pair_target for p in self._pairs],
                                          dtype=np.float32).reshape(-1, 1))
        self._provenance = EmbeddingProvenance(
            split, hashlib.sha256(serialized.encode()).hexdigest(), manifest.policy.fingerprint,
            artifact_fingerprint, tuple(hashes), tuple(exports))
        if len(self) and not self.validity.any(axis=1).all():
            raise ContractError("every selected pair needs at least one jointly valid input bin")

    @classmethod
    def from_json(cls, text, audio, vision, *, artifact_fingerprint=None, **kwargs):
        """Use the audited safe manifest reader, including its artifact context."""
        manifest = manifest_from_json(text, audio, vision,
                                     expected_artifact_fingerprint=artifact_fingerprint)
        return cls(manifest, audio, vision, artifact_fingerprint=artifact_fingerprint, **kwargs)

    @classmethod
    def from_scoped_json(cls, text, audio, vision, *, split, expected_manifest_sha256,
                         artifact_fingerprint):
        """TRAIN/val-only reader bound to verified original manifest bytes.

        Regenerate every pair, field, count and source reference using only the
        requested TRAIN/val bundles. Only full-inventory leakage context is opaque:
        it was fixed by the original file hash and is never used as model input.
        This avoids parsing held-out producer labels to reconstruct that context.
        The legacy all-split reader above retains its original exact behavior.
        """
        digest = hashlib.sha256(text.encode()).hexdigest()
        if (split not in ("train", "val") or digest != expected_manifest_sha256
                or any(set(b.requested_splits) != {"train", "val"} for b in (audio, vision))):
            raise ContractError("scoped reader requires verified original bytes and TRAIN/val-only bundles")
        raw = _read_json(text)
        try:
            if (raw["schema"] != "gc2-pair-manifest-v1" or raw["config"]["split"] != split
                    or raw["artifact_fingerprint"] != artifact_fingerprint
                    or raw["policy"]["hard_negative_policy"] is not None):
                raise ContractError("incompatible scoped manifest")
            settings = dict(raw["config"])
            settings["source_reuse"] = SourceReuse(settings["source_reuse"])
            manifest = generate_pairs(audio, vision, config=GenerationConfig(**settings),
                                      policy=SamplingPolicy(**raw["policy"]))
            canonical = _read_json(manifest_to_json(manifest, audio, vision,
                                                    artifact_fingerprint=artifact_fingerprint))
            # These three fields describe unavailable all-split metadata only.
            # Selected-split provenance, pair_count, all pairs and all other
            # report fields must still match exact deterministic regeneration.
            for name in ("source_overlaps", "missing_audio_metadata_splits", "missing_visual_metadata_splits"):
                canonical["report"]["source_validation"][name] = raw["report"]["source_validation"][name]
            if _json(canonical) != _json(raw):
                raise ContractError("scoped manifest differs from deterministic regeneration")
            result = cls(manifest, audio, vision, split=split, artifact_fingerprint=artifact_fingerprint)
            result._provenance = replace(result.provenance, manifest_sha256=digest)
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("invalid scoped manifest") from exc

    @property
    def pairs(self):
        return self._pairs

    @property
    def split(self):
        return self._split

    @property
    def partition(self):
        return self._partition

    @property
    def complete(self):
        return self._complete

    @property
    def provenance(self):
        return self._provenance

    @property
    def targets(self):
        return self._targets

    @property
    def validity(self):
        return _snapshot(self._audio_valid[self._ai] & self._visual_valid[self._vi])

    @property
    def features(self):
        """Explicit materialization for inspection; minibatches use lazy items."""
        return EmbeddingFeatures(_snapshot(self._audio[self._ai]),
                                 _snapshot(self._visual[self._vi]))

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        ai, vi = self._ai[index], self._vi[index]
        return ({"audio": torch.tensor(self._audio[ai], dtype=torch.float32),
                 "visual": torch.tensor(self._visual[vi], dtype=torch.float32),
                 "validity": torch.tensor(self._audio_valid[ai] & self._visual_valid[vi])},
                torch.tensor(self._targets[index], dtype=torch.float32))

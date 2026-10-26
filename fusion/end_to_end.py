"""Strict decoded-media orchestration with an exact TRAIN export compatibility gate.

Compressed-file decoding and unseen-media compatibility are deliberately unsupported
until actual decoder/encoder artifacts and raw-to-export parity are verified.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .contracts import Bundle, ContractError, validate_bundle
from .encoder_interfaces import Artifact, EncoderOutput, EncoderStamp, FrozenEncoder, IntegrationError, digest, fail
from .inference import FrozenFusionInference, load_verified_train_exports
from .inference_contracts import BIN_STARTS, FeatureBatch, RawMediaInput


@dataclass(frozen=True)
class DecodedTensorSource:
    """A trusted decoder's pinned numeric arrays and shared-clock PTS manifest."""
    waveform: Artifact
    frames: Artifact
    manifest: Artifact


@dataclass(frozen=True)
class DecodedMedia:
    raw: RawMediaInput
    audio_sample_ids: tuple[str, ...]
    visual_sample_ids: tuple[str, ...]
    decoder_signature: str
    source_hashes: tuple[tuple[str, str], ...]
    timing_verification: str

    def validate(self, decoder_signature):
        self.raw.validate()
        if self.raw.waveform.device.type != "cpu" or self.raw.frames.device.type != "cpu":
            fail("UNSUPPORTED_DEVICE", "decode", "decoded tensors must be on CPU")
        if self.decoder_signature != decoder_signature or self.timing_verification != "shared_clock_pts_v1":
            fail("UNVERIFIED_TIMING", "decode", "pinned trusted decoder and shared-clock PTS evidence required")
        batch = self.raw.waveform.shape[0]
        for ids in (self.audio_sample_ids, self.visual_sample_ids):
            if (type(ids) is not tuple or len(ids) != batch
                    or any(type(s) is not str or not s for s in ids)):
                fail("INVALID_IDENTITIES", "decode", "one explicit TRAIN sample identity per modality/batch row required")
        if self.raw.audio_validity is None or self.raw.visual_validity is None:
            fail("INVALID_MASK", "decode", "decoded validity must be explicit; no annotation-derived defaults")


class DecodedTensorDecoder:
    """Load canonical, already-decoded inputs. No compressed-media fallback.

    The external trusted decoder must verify source clocks/durations/PTS and do
    explicit FFmpeg mono/resampling and OpenCV RGB/INTER_LINEAR resizing. A manifest
    hash authenticates supplied bytes, not the truth of an untrusted decoder claim.
    """
    is_synthetic = False

    def __init__(self, expected_decoder_signature):
        from .encoder_interfaces import require_digest
        require_digest(expected_decoder_signature)
        self.signature = expected_decoder_signature

    def decode(self, source):
        if type(source) is not DecodedTensorSource:
            fail("UNSUPPORTED_MEDIA", "decode", "provide pinned canonical .npy arrays and a trusted PTS manifest; compressed files unsupported")
        for artifact in (source.waveform, source.frames, source.manifest):
            artifact.verify()
        try:
            meta = json.loads(Path(source.manifest.path).read_text(encoding="utf-8"))
            expected = {"schema", "decoder_signature", "timing_verification", "waveform_sha256", "frames_sha256",
                        "sample_rate", "audio_duration", "video_duration", "audio_start_seconds", "video_start_seconds",
                        "frame_timestamps", "audio_sample_ids", "visual_sample_ids", "audio_validity", "visual_validity"}
            if (type(meta) is not dict or set(meta) != expected or meta["schema"] != "gc2-decoded-ten-bin-v1"
                    or meta["waveform_sha256"] != source.waveform.sha256 or meta["frames_sha256"] != source.frames.sha256):
                fail("INVALID_MEDIA", "decode", "decoded manifest coverage/schema/source hashes differ")
            arrays = []
            for artifact in (source.waveform, source.frames):
                value = np.load(artifact.path, allow_pickle=False)
                if not isinstance(value, np.ndarray):
                    value.close()
                    fail("INVALID_MEDIA", "decode", "plain numeric .npy arrays required")
                arrays.append(value)
            masks = []
            for key in ("audio_validity", "visual_validity"):
                rows = meta[key]
                if (type(rows) is not list or any(type(row) is not list or len(row) != 10
                        or any(type(v) is not bool for v in row) for row in rows)):
                    fail("INVALID_MASK", "decode", "validity JSON values must be explicit booleans")
                masks.append(torch.tensor(rows, dtype=torch.bool))
            raw = RawMediaInput(torch.from_numpy(arrays[0].copy()), torch.from_numpy(arrays[1].copy()),
                meta["sample_rate"], meta["audio_duration"], meta["video_duration"],
                meta["audio_start_seconds"], meta["video_start_seconds"], tuple(meta["frame_timestamps"]), *masks)
            decoded = DecodedMedia(raw, tuple(meta["audio_sample_ids"]), tuple(meta["visual_sample_ids"]),
                meta["decoder_signature"], (("waveform", source.waveform.sha256), ("frames", source.frames.sha256),
                                          ("manifest", source.manifest.sha256)), meta["timing_verification"])
            decoded.validate(self.signature)
            for artifact in (source.waveform, source.frames, source.manifest):
                artifact.verify()
            return decoded
        except IntegrationError:
            raise
        except Exception as exc:
            fail("INVALID_MEDIA", "decode", f"Canonical decode rejected: {exc}")


@dataclass(frozen=True)
class EncodedBundle:
    audio: EncoderOutput
    visual: EncoderOutput
    decoded: DecodedMedia
    synthetic: bool

    def validate(self, expected_audio, expected_visual):
        batch = self.decoded.raw.waveform.shape[0]
        self.audio.validate(batch, expected_audio)
        self.visual.validate(batch, expected_visual)
        # Reuse the foundation's finite/shape/mask/ten-original-bin checks.
        mask = self.audio.validity & self.visual.validity
        if not mask.any(dim=1).all():
            fail("INVALID_MASK", "alignment", "every pair needs a jointly valid bin")
        if self.audio.bin_starts != self.visual.bin_starts:
            fail("TEMPORAL_MISMATCH", "alignment", "audio/video temporal indices differ")
        return mask


def _reference_fingerprint(bundle):
    """Fingerprint live content, not a caller's declared export hashes.

    This is a runtime integrity check anchored only after authenticated loading;
    it is not another archive format or a way to admit caller-supplied evidence.
    """
    if type(bundle) is not Bundle or not validate_bundle(bundle).complete:
        fail("REFERENCE_INTEGRITY", "fusion", "complete original TRAIN reference bundle required")
    if bundle.requested_splits != ("train",) or set(bundle.splits) != {"train"}:
        fail("REFERENCE_INTEGRITY", "fusion", "only the admitted TRAIN reference split is supported")
    source = bundle.splits["train"]
    arrays = {}
    for name in ("embeddings", "probabilities", "input_validity", "timestamps"):
        value = getattr(source, name)
        arrays[name] = None if value is None else (
            value.dtype.str, value.shape, hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest())
    return digest({"root": str(bundle.root), "modality": bundle.modality,
        "requested_splits": bundle.requested_splits, "records": [asdict(r) for r in source.records],
        "metadata": {k: [asdict(r) for r in rows] for k, rows in bundle.metadata.items()},
        "class_names": bundle.class_names, "missing_inputs": bundle.missing_inputs,
        "expected_counts": None if bundle.expected_counts is None else dict(bundle.expected_counts),
        "provenance": bundle.provenance, "arrays": arrays})


@dataclass(frozen=True, slots=True)
class _TrainReference:
    bundle: Bundle
    # Primitive immutable pairs, independent of mutable Bundle.provenance.
    files: tuple[tuple[str, str], ...]
    content_sha256: str


def _bind_reference(bundle, provenance):
    """Internal: called after load_verified_train_exports authenticates both roots."""
    files = tuple((str(bundle.root / name), sha) for name, sha in provenance.export_hashes)
    if not files or bundle.modality != provenance.modality:
        fail("REFERENCE_INTEGRITY", "fusion", "authenticated producer export evidence is required")
    for path, sha in files:
        Artifact(path, sha).verify()
    result = _TrainReference(bundle, files, _reference_fingerprint(bundle))
    for path, sha in files:
        Artifact(path, sha).verify()
    return result


@dataclass(frozen=True, slots=True, init=False, eq=False)
class TrainParityFusion:
    """Narrow real bridge: encoder outputs must reproduce verified TRAIN exports.

    No raw provenance is relabeled as export evidence without a content comparison.
    This permits only source-identified TRAIN reference clips, not unseen raw media.
    Exact float16 export roundtrip is a conservative gate, not an assumed CPU parity.
    """
    is_synthetic: bool = field(default=False, init=False)
    expected_audio: EncoderStamp | None
    expected_visual: EncoderStamp | None
    adapters: dict
    audio: Bundle
    visual: Bundle
    provenance: tuple
    _references: tuple = field(repr=False)
    _provenance_sha256: str = field(repr=False)

    def __init__(self, e3_directory, a2_directory, audio_root, vision_root,
                 expected_audio: EncoderStamp | None = None, expected_visual: EncoderStamp | None = None):
        if ((expected_audio is None) != (expected_visual is None)
                or (expected_audio is not None and (expected_audio.run_identity != "beats_lora_r8_v1 / beats_iter3"
                                                    or expected_visual.run_identity != "resnet18_v1"))):
            fail("PROVENANCE_MISMATCH", "fusion", "only the selected historical producers are supported")
        adapters = {"E3": FrozenFusionInference(e3_directory, candidate="E3"),
                    "A2": FrozenFusionInference(a2_directory, candidate="A2")}
        (audio, visual), provenance = load_verified_train_exports(audio_root, vision_root, adapters["E3"].metadata)
        # Both candidates must be bound to identical source/preprocessing bytes.
        from .inference import expected_provenance
        if expected_provenance(adapters["A2"].metadata) != provenance:
            fail("PROVENANCE_MISMATCH", "fusion", "E3 and A2 producer evidence differs")
        references = tuple(_bind_reference(b, p) for b, p in zip((audio, visual), provenance))
        for name, value in (("is_synthetic", False), ("expected_audio", expected_audio),
                ("expected_visual", expected_visual), ("adapters", adapters), ("audio", audio), ("visual", visual),
                ("provenance", provenance), ("_references", references),
                ("_provenance_sha256", digest([asdict(p) for p in provenance]))):
            object.__setattr__(self, name, value)

    def _validated_references(self):
        """Recheck both original references and pinned files on every raw admission."""
        if not hasattr(self, "_references") or len(self._references) != 2:
            fail("BLOCKED", "fusion", "authenticated TRAIN reference bindings are unavailable")
        try:
            if digest([asdict(p) for p in self.provenance]) != self._provenance_sha256:
                fail("REFERENCE_INTEGRITY", "fusion", "admitted producer evidence changed")
            for live, binding in zip((self.audio, self.visual), self._references):
                if live is not binding.bundle:
                    fail("REFERENCE_INTEGRITY", "fusion", "original TRAIN reference object was replaced")
                for path, sha in binding.files:
                    Artifact(path, sha).verify()
                if _reference_fingerprint(live) != binding.content_sha256:
                    fail("REFERENCE_INTEGRITY", "fusion", "TRAIN reference contents or metadata changed")
        except IntegrationError:
            raise
        except (ContractError, OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            fail("REFERENCE_INTEGRITY", "fusion", f"TRAIN reference integrity could not be established: {exc}")
        return self.audio, self.visual

    def predict(self, bundle):
        if bundle.synthetic:
            fail("SYNTHETIC_INPUT", "fusion", "synthetic encoders cannot prove genuine raw-to-export parity")
        if self.expected_audio is None or self.expected_visual is None:
            fail("BLOCKED", "fusion", "exact raw encoder provenance is not configured; feature replay only")
        references = TrainParityFusion._validated_references(self)
        mask = bundle.validate(self.expected_audio, self.expected_visual)
        if not mask.all():
            fail("PARITY_UNVERIFIED", "fusion", "historical exports contain no decode validity; full-bin raw parity required")
        values = []
        for output, ids, exported in ((bundle.audio, bundle.decoded.audio_sample_ids, references[0]),
                                      (bundle.visual, bundle.decoded.visual_sample_ids, references[1])):
            source = exported.splits["train"]
            index = {r.sample_id: i for i, r in enumerate(source.records)}
            if any(s not in index for s in ids):
                fail("UNSUPPORTED_MEDIA", "fusion", "raw bridge supports verified TRAIN reference identities only")
            # This is the historical producer's export precision, not a new fusion policy.
            produced = output.embeddings.detach().numpy().astype(np.float16)
            if (not np.isfinite(produced).all()
                    or not np.array_equal(produced, source.embeddings[[index[s] for s in ids]])):
                fail("PARITY_UNVERIFIED", "fusion", "actual encoder outputs do not exactly reproduce the TRAIN float16 export")
            values.append(torch.from_numpy(produced.astype(np.float32)))
        features = FeatureBatch(*values, bundle.audio.validity, bundle.visual.validity, self.provenance)
        # No stale admission: also catch reference changes during input validation
        # or comparison, before the first frozen fusion forward can execute.
        TrainParityFusion._validated_references(self)
        # Raw admission must complete this gate, not a caller's replacement method.
        return TrainParityFusion.predict_features(self, features)

    def predict_features(self, features):
        """Existing verified feature replay, explicitly independent of raw encoders."""
        if type(features) is not FeatureBatch:
            fail("INVALID_FEATURES", "fusion", "explicit FeatureBatch required")
        features.validate()
        if features.provenance != self.provenance:
            fail("PROVENANCE_MISMATCH", "fusion", "feature source differs from both selected models")
        return {name: adapter.predict(features) for name, adapter in self.adapters.items()}


@dataclass(frozen=True, slots=True, eq=False)
class MediaFusionPipeline:
    """Decoder -> audio -> visual -> validation/alignment -> frozen fusion.

    Dependency injection is explicit. Synthetic implementations require an opt-in
    and are labeled SYNTHETIC, never evidence of real encoder readiness.
    """
    decoder: object
    audio_encoder: object
    visual_encoder: object
    fusion: object
    allow_synthetic: bool = field(default=False, kw_only=True, repr=False)
    synthetic: bool = field(init=False)
    _modes: tuple = field(init=False, repr=False)
    _producers: tuple = field(init=False, repr=False)
    _decoder_signature: str = field(init=False, repr=False)
    _adapters: tuple = field(init=False, repr=False)

    def __post_init__(self):
        modes = tuple(x.is_synthetic for x in self._components())
        if any(type(mode) is not bool for mode in modes):
            fail("UNVERIFIED_COMPONENT", "setup", "component modes must be explicit booleans")
        object.__setattr__(self, "_modes", modes)
        object.__setattr__(self, "synthetic", any(modes))
        object.__setattr__(self, "_producers", (self.audio_encoder.provenance, self.visual_encoder.provenance))
        object.__setattr__(self, "_decoder_signature", self.decoder.signature)
        adapters = tuple(self.fusion.adapters.items()) if not self.synthetic and type(self.fusion) is TrainParityFusion else ()
        object.__setattr__(self, "_adapters", adapters)
        if self.synthetic and not self.allow_synthetic:
            fail("SYNTHETIC_INPUT", "setup", "synthetic integration requires explicit opt-in")
        self._validate_bindings("setup")

    def _components(self):
        return self.decoder, self.audio_encoder, self.visual_encoder, self.fusion

    def _validate_bindings(self, stage):
        modes = tuple(x.is_synthetic for x in self._components())
        if (any(type(mode) is not bool for mode in modes) or modes != self._modes
                or any(modes) != self.synthetic):
            fail("COMPONENT_MUTATED", stage, "component execution modes changed; construct a new pipeline")
        if (self.decoder.signature != self._decoder_signature
                or (self.audio_encoder.provenance, self.visual_encoder.provenance) != self._producers):
            fail("COMPONENT_MUTATED", stage, "decoder or producer bindings changed")
        if tuple(p.modality for p in self._producers) != ("audio", "vision"):
            fail("PROVENANCE_MISMATCH", "setup", "audio and vision encoder roles must be explicit")
        if not self.synthetic:
            if tuple(type(x) for x in self._components()) != (
                    DecodedTensorDecoder, FrozenEncoder, FrozenEncoder, TrainParityFusion):
                fail("UNVERIFIED_COMPONENT", stage, "production integration requires the verified concrete adapters")
            if (self.fusion.expected_audio, self.fusion.expected_visual) != self._producers:
                fail("PROVENANCE_MISMATCH", stage, "real parity gate must bind both executing producers")
            if (set(self.fusion.adapters) != {"E3", "A2"}
                    or any(self.fusion.adapters.get(k) is not v or type(v) is not FrozenFusionInference
                           for k, v in self._adapters)):
                fail("COMPONENT_MUTATED", stage, "frozen fusion adapter bindings changed")
            # Reject instance-level method overrides in the production path.
            for component, methods in ((self.decoder, ("decode",)),
                    (self.audio_encoder, ("require_ready", "encode")),
                    (self.visual_encoder, ("require_ready", "encode")),
                    (self.fusion, ("predict", "predict_features")),
                    *((v, ("predict",)) for _, v in self._adapters)):
                if any(getattr(getattr(component, name), "__func__", None) is not getattr(type(component), name)
                       for name in methods):
                    fail("COMPONENT_MUTATED", stage, "production component method was replaced")

    def run(self, source):
        stage = "readiness"
        try:
            with torch.no_grad(), torch.autocast("cpu", enabled=False):
                self._validate_bindings(stage)
                # Refuse missing trained assets before either branch executes.
                self.audio_encoder.require_ready()
                self.visual_encoder.require_ready()
                self._validate_bindings(stage)
                stage = "decode"
                decoded = self.decoder.decode(source)
                decoded.validate(self.decoder.signature)
                self._validate_bindings(stage)
                stage = "audio"
                audio = self.audio_encoder.encode(decoded)
                audio.validate(decoded.raw.waveform.shape[0], self.audio_encoder.provenance)
                self._validate_bindings(stage)
                stage = "vision"
                visual = self.visual_encoder.encode(decoded)
                self._validate_bindings(stage)
                bundle = EncodedBundle(audio, visual, decoded, self.synthetic)
                stage = "alignment"
                mask = bundle.validate(self.audio_encoder.provenance, self.visual_encoder.provenance)
                stage = "fusion"
                predictions = (self.fusion.predict(bundle) if self.synthetic
                               else TrainParityFusion.predict(self.fusion, bundle))
                self._validate_bindings(stage)
                if set(predictions) != {"E3", "A2"}:
                    fail("INVALID_FUSION_OUTPUT", "fusion", "both explicit E3/A2 results are required")
                serialized = {k: v.to_dict() for k, v in predictions.items()}
                self._validate_bindings(stage)
                # Matching TRAIN values proves compatibility, not authenticated
                # encoder execution or the truth of external source-clock claims.
                return {"status": "SYNTHETIC" if self.synthetic else "TRAIN_EXPORT_PARITY_MATCHED",
                        "predictions": serialized,
                        "feature_schema": {"audio": list(audio.embeddings.shape), "visual": list(visual.embeddings.shape),
                                           "validity": mask.tolist()},
                        "diagnostics": {"audio_encoder": asdict(audio.provenance), "visual_encoder": asdict(visual.provenance),
                                        "decoder_signature": decoded.decoder_signature, "source_hashes": decoded.source_hashes,
                                        "audio_sample_ids": decoded.audio_sample_ids, "visual_sample_ids": decoded.visual_sample_ids,
                                        "timing": decoded.timing_verification,
                                        "physical_synchronization_independently_verified": False}}
        except IntegrationError:
            raise
        except Exception as exc:
            fail("STAGE_FAILED", stage, f"{stage} rejected input: {exc}")

    def try_run(self, source):
        try:
            return self.run(source)
        except IntegrationError as exc:
            return {"status": "BLOCKED" if exc.code in ("BLOCKED", "MISSING_ARTIFACT") else "REJECTED",
                    "error": exc.to_dict()}

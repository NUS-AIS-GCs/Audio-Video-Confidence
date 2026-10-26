"""Frozen feature inference, pinned to the selected Phase 4/5 artifacts.

No training, calibration, threshold fitting, tracker initialization or fallback.
Only CPU execution is supported in this foundation's verified device contract.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import pickle
from types import MappingProxyType

import numpy as np
import torch

from .contracts import ContractError, load_bundle, validate_bundle
from .inference_contracts import EncoderProvenance, FeatureBatch, PREPROCESSING_CONFIGURATION
from .mlp_training import model_fingerprint
from .model_interface import validate_live_model
from .train_mlp import load_selected_checkpoint


@dataclass(frozen=True)
class FrozenCandidate:
    name: str
    checkpoint_sha256: str
    metadata_sha256: str
    operating_point_sha256: str
    model_sha256: str
    threshold: float


CANDIDATES = MappingProxyType({
    "E3": FrozenCandidate("E3",
        "e7df0de86d873005728c142c3bd31729df803f4cc805110d82e375fb5d695925",
        "fd6abe3f0e62f09e19c096f82b6516195a2c8dba93cd3d534e7f88fcfec48e58",
        "706db41c19e6df2cc39c3c769a17bd4346f745d3f91723bbb58171c81ee8a9ad",
        "d978d0dc9fa9986844936ef74686be7833e69d1291287d77f767307e94fc5025",
        0.009579099009671126),
    "A2": FrozenCandidate("A2",
        "0adf31d64d49e1917bca1ee4835aa2e40c506a41f7b7260a1dd1f68b62ec9c12",
        "aadd9c73e7675249f8192912656ca624aa6d77d50b10cea3f29a1968cce8d460",
        "16994e04c6a8421ec8bc22c1fac33691a4a0228ba544f6e57b0730f98c95b384",
        "1ef5985f9de3e7b9f2994b076ebd16c1b0cd7d4922c0842d12d301aa202c2b8e",
        0.0026640388978268056),
})


def _sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def expected_provenance(metadata):
    """Bind only the TRAIN export evidence; never read validation/test tensors."""
    hashes = dict(metadata.data[0].input_export_hashes)
    identities = dict(metadata.export_identities)
    names = ("SOURCE.txt", "README.md", "class_names.json", "train_emb.npy", "train_ids.csv", "train_probs.npy")
    try:
        return tuple(EncoderProvenance(modality, identities[modality],
                     tuple((name, hashes[f"{modality}/{name}"]) for name in names))
                     for modality in ("audio", "vision"))
    except KeyError as exc:
        raise ContractError("checkpoint lacks required TRAIN producer/preprocessing evidence") from exc


@dataclass(frozen=True)
class PairPrediction:
    logit: float
    sigmoid_score: float
    predicted_target: int
    pairing_decision: str
    valid_bins: tuple[bool, ...]


@dataclass(frozen=True)
class PredictionResult:
    candidate: str
    checkpoint_sha256: str
    model_sha256: str
    threshold: float
    predictions: tuple[PairPrediction, ...]
    provenance: tuple[EncoderProvenance, ...]
    preprocessing_configuration: tuple = PREPROCESSING_CONFIGURATION
    score_semantics: str = "float64 sigmoid of correspondence logit; uncalibrated"
    decision_rule: str = "sigmoid_score >= model-specific frozen threshold"
    timing_basis: str = "export_bin_indices_assumed_synchronous"
    causal_source_identity_claim: bool = False
    diagnostics: object = None

    def to_dict(self):
        return asdict(self)


class FrozenFusionInference:
    def __init__(self, checkpoint_directory, *, candidate="E3", device="cpu"):
        if candidate not in CANDIDATES:
            raise ContractError("unknown frozen candidate; choose E3 or A2 explicitly")
        if str(device) != "cpu":
            raise ContractError("only CPU inference is verified; unsupported device")
        binding = CANDIDATES[candidate]
        directory = Path(checkpoint_directory)
        paths = {"best.pt": binding.checkpoint_sha256,
                 "checkpoint_metadata.json": binding.metadata_sha256,
                 "frozen_operating_point.json": binding.operating_point_sha256}
        try:
            for name, digest in paths.items():
                if _sha(directory / name) != digest:
                    raise ContractError(f"{candidate}: {name} differs from pinned model/threshold evidence")
            model, metadata = load_selected_checkpoint(directory)
            operating_point = _json(directory / "frozen_operating_point.json")
        except (OSError, ValueError, RuntimeError, EOFError, KeyError, TypeError, pickle.UnpicklingError) as exc:
            raise ContractError(f"cannot load frozen {candidate} checkpoint: {exc}") from exc
        if (operating_point["threshold"] != binding.threshold
                or operating_point["model_sha256"] != binding.model_sha256
                or operating_point["selected_epoch"] != metadata.selected_epoch
                or operating_point["validation_manifest_sha256"] != metadata.data[1].manifest_sha256
                or operating_point["decision"] != "probability >= threshold"
                or model_fingerprint(model) != binding.model_sha256):
            raise ContractError("model/threshold/selected-epoch binding mismatch")
        validate_live_model(model, metadata.model)
        if next(model.parameters()).dtype != torch.float32:
            raise ContractError("selected frozen models require float32 weights")
        self._binding, self._metadata = binding, metadata
        self._model = model.to(device="cpu").eval().requires_grad_(False)
        self._provenance = expected_provenance(metadata)

    @property
    def metadata(self):
        return self._metadata

    def predict(self, features: FeatureBatch):
        if type(features) is not FeatureBatch:
            raise ContractError("expected FeatureBatch; raw encoders are not available")
        model_device = next(self._model.parameters()).device
        if model_device.type != "cpu":
            raise ContractError("only CPU inference is verified; unsupported live model device")
        # no_grad does not disable a caller's autocast. Keep validation, forward
        # and conversion in one local float32 boundary; restore caller state on exit.
        with torch.no_grad(), torch.autocast(device_type=model_device.type, enabled=False):
            validity = features.validate()
            if features.provenance != self._provenance:
                raise ContractError("encoder source/preprocessing/export provenance mismatch; raw parity unverified")
            # Reconcile live settings/weights so caller mutations cannot reuse a threshold.
            validate_live_model(self._model, self._metadata.model)
            if model_fingerprint(self._model) != self._binding.model_sha256:
                raise ContractError("frozen model changed; threshold is no longer applicable")
            self._model.eval()
            audio = features.audio.detach().to(device=model_device, dtype=torch.float32)
            visual = features.visual.detach().to(device=model_device, dtype=torch.float32)
            if not torch.isfinite(audio).all() or not torch.isfinite(visual).all():
                raise ContractError("features exceed float32 range")
            output = self._model(audio, visual, validity).logits
            if (type(output) is not torch.Tensor or output.dtype != torch.float32
                    or output.shape != (audio.shape[0], 1)):
                raise ContractError("model must return float32 (B,1) logits")
            logits = output.cpu().numpy()[:, 0]
            if not np.isfinite(logits).all():
                raise ContractError("model must return finite (B,1) logits")
            # Exactly the Phase 4/5 Predictions.probabilities numerical convention.
            scores = np.exp(-np.logaddexp(0, -logits.astype(np.float64)))
            if not np.isfinite(scores).all():
                raise ContractError("nonfinite sigmoid scores")
            predictions = tuple(PairPrediction(float(z), float(s), int(s >= self._binding.threshold),
                "original" if s >= self._binding.threshold else "substituted", tuple(mask.tolist()))
                for z, s, mask in zip(logits, scores, validity))
            return PredictionResult(self._binding.name, self._binding.checkpoint_sha256,
                self._binding.model_sha256, self._binding.threshold, predictions, self._provenance,
                diagnostics=features.diagnostics)


def load_verified_train_exports(audio_root, vision_root, metadata):
    """Read-only, byte-verified exports using the existing Phase 1 contracts.

    Returned bundles contain TRAIN only. All-true export masks mean availability
    in the historical export, not verified decode success or physical synchrony.
    """
    provenance = expected_provenance(metadata)
    bundles = []
    for root, p in zip((Path(audio_root), Path(vision_root)), provenance):
        if any((root / name).exists() for name in ("train_valid.npy", "train_timestamps.npy")):
            raise ContractError("unbound validity/timestamp sidecars cannot change historical export semantics")
        for name, digest in p.export_hashes:
            if _sha(root / name) != digest:
                raise ContractError(f"{p.modality}/{name}: source/preprocessing bytes differ")
        bundle = load_bundle(root, p.modality, splits=("train",))
        if bundle.splits["train"].input_validity is not None or bundle.splits["train"].timestamps is not None:
            raise ContractError("unbound validity/timestamp sidecars appeared during loading")
        if not validate_bundle(bundle).complete:
            raise ContractError("complete validated TRAIN export required")
        # Guard read-time changes against the original pinned bytes.
        for name, digest in p.export_hashes:
            if _sha(root / name) != digest:
                raise ContractError("producer exports changed during loading")
        bundles.append(bundle)
    return tuple(bundles), provenance

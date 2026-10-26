"""Explicit, optional-dependency encoder construction. Never downloads weights.

Factories are unverified on real artifacts until the missing local assets arrive.
Existing producer embedding methods are reused; unsafe historical loaders are not.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import torch

from .contracts import ContractError
from .mlp_training import model_fingerprint


class IntegrationError(ContractError):
    def __init__(self, code, stage, message):
        super().__init__(message)
        self.code, self.stage = code, stage

    def to_dict(self):
        return {"code": self.code, "stage": self.stage, "message": str(self)}


def fail(code, stage, message):
    raise IntegrationError(code, stage, message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()


def require_digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        fail("INVALID_SPEC", "artifact", "expected an explicit lowercase SHA-256; unknown hashes are not allowed")


@dataclass(frozen=True)
class Artifact:
    path: str
    sha256: str

    def verify(self):
        require_digest(self.sha256)
        if type(self.path) is not str or not self.path:
            fail("INVALID_SPEC", "artifact", "explicit artifact path required")
        try:
            with Path(self.path).open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            fail("MISSING_ARTIFACT", "artifact", f"Provide the exact artifact at {self.path}: {exc}")
        if actual != self.sha256:
            fail("HASH_MISMATCH", "artifact", f"Artifact bytes differ: {self.path}")


AUDIO_PREPROCESSING = (
    "mono_float32_16000Hz_160000_samples;no_pad_trim;"
    "BEATs_fbank128_25ms_10ms_scale32768_mean15.41663_std2x6.55582;"
    "pool_to_seconds_time_centres_8_frequency_patches;frozen_768D_mean_std;EmbeddingHead256_GELU"
)
VIDEO_PREPROCESSING = (
    "RGB_uint8_10x224x224;resize_upstream_OpenCV_INTER_LINEAR;"
    "NCHW_float32_div255;mean=.485,.456,.406;std=.229,.224,.225;no_augmentation"
)
DEPENDENCIES = {"audio": ("torch", "numpy", "torchaudio", "pandas", "soundfile", "tqdm"),
                "vision": ("torch", "numpy", "torchvision")}


@dataclass(frozen=True)
class EncoderSpec:
    modality: str
    run_identity: str
    artifacts: tuple[tuple[str, Artifact], ...]
    # Full canonical JSON from actual checkpoint configuration, not guessed defaults.
    configuration_json: str
    preprocessing: str
    runtime_versions: tuple[tuple[str, str], ...]

    def validate(self):
        expected = {"audio": ("base", "trained", "normalization"), "vision": ("trained",)}
        if self.modality not in expected:
            fail("INVALID_SPEC", "encoder", "modality must be audio or vision")
        if type(self.run_identity) is not str or not self.run_identity or type(self.artifacts) is not tuple:
            fail("INVALID_SPEC", "encoder", "explicit run identity and immutable artifact bindings required")
        if self.run_identity != ("beats_lora_r8_v1 / beats_iter3" if self.modality == "audio" else "resnet18_v1"):
            fail("PROVENANCE_MISMATCH", "encoder", "unsupported encoder producer version")
        if any(type(row) is not tuple or len(row) != 2 for row in self.artifacts):
            fail("INVALID_SPEC", "encoder", "artifact roles must be immutable pairs")
        if tuple(k for k, _ in self.artifacts) != expected[self.modality]:
            fail("INVALID_SPEC", "encoder", "required artifact roles/order differ")
        for _, artifact in self.artifacts:
            if type(artifact) is not Artifact:
                fail("INVALID_SPEC", "encoder", "artifact bindings must be Artifact values")
            require_digest(artifact.sha256)
        try:
            cfg = json.loads(self.configuration_json)
            if type(cfg) is not dict or json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False) != self.configuration_json:
                raise ValueError("configuration must be canonical JSON")
        except (ValueError, TypeError) as exc:
            fail("INVALID_SPEC", "encoder", str(exc))
        if self.preprocessing != (AUDIO_PREPROCESSING if self.modality == "audio" else VIDEO_PREPROCESSING):
            fail("PREPROCESSING_MISMATCH", "encoder", "unsupported normalization/preprocessing signature")
        if (type(self.runtime_versions) is not tuple
                or any(type(row) is not tuple or len(row) != 2
                       or any(type(v) is not str or not v for v in row) for row in self.runtime_versions)
                or len(dict(self.runtime_versions)) != len(self.runtime_versions)
                or not set(DEPENDENCIES[self.modality]) <= set(dict(self.runtime_versions))):
            fail("INVALID_SPEC", "encoder", "pin every required encoder runtime version")
        return cfg


def readiness(spec):
    """Inspect all declared assets/dependencies without constructing a model."""
    spec.validate()
    issues = []
    for role, artifact in spec.artifacts:
        try:
            artifact.verify()
        except IntegrationError as exc:
            issues.append({"role": role, **exc.to_dict()})
    for name, version in spec.runtime_versions:
        try:
            available = importlib.util.find_spec(name) is not None
            actual = importlib.metadata.version(name) if available else None
        except (ImportError, ValueError, importlib.metadata.PackageNotFoundError):
            actual = None
        if actual != version:
            issues.append({"code": "DEPENDENCY_MISMATCH", "stage": "runtime",
                           "message": f"Install the verified {name}=={version}; found {actual}"})
    return {"status": "BLOCKED" if issues else "ARTIFACTS_PRESENT_PARITY_UNVERIFIED", "issues": issues}


@dataclass(frozen=True)
class EncoderStamp:
    modality: str
    run_identity: str
    artifacts: tuple[tuple[str, str], ...]
    configuration_sha256: str
    preprocessing_sha256: str
    runtime_versions: tuple[tuple[str, str], ...]


def stamp(spec):
    spec.validate()
    return EncoderStamp(spec.modality, spec.run_identity,
                        tuple((k, a.sha256) for k, a in spec.artifacts),
                        digest(json.loads(spec.configuration_json)), digest(spec.preprocessing), spec.runtime_versions)


@dataclass(frozen=True)
class EncoderOutput:
    embeddings: torch.Tensor
    validity: torch.Tensor
    bin_starts: tuple[float, ...]
    provenance: EncoderStamp

    def validate(self, batch, expected_stamp):
        dim = 256 if expected_stamp.modality == "audio" else 512
        x, mask = self.embeddings, self.validity
        if (type(x) is not torch.Tensor or x.device.type != "cpu" or x.layout != torch.strided or x.dtype != torch.float32
                or x.shape != (batch, 10, dim) or not torch.isfinite(x).all()):
            fail("INVALID_ENCODER_OUTPUT", "encoder", f"expected finite CPU float32 (B,10,{dim})")
        if (type(mask) is not torch.Tensor or mask.device.type != "cpu" or mask.dtype != torch.bool
                or mask.shape != (batch, 10)):
            fail("INVALID_MASK", "encoder", "explicit CPU boolean (B,10) validity required")
        if (type(self.bin_starts) is not tuple or any(type(t) not in (int, float) for t in self.bin_starts)
                or self.bin_starts != tuple(float(t) for t in range(10)) or self.provenance != expected_stamp):
            fail("PROVENANCE_MISMATCH", "encoder", "encoder version, preprocessing or temporal indices differ")


def _safe_payload(artifact):
    artifact.verify()
    try:
        payload = torch.load(artifact.path, map_location="cpu", weights_only=True)
    except Exception as exc:
        fail("CHECKPOINT_LOAD_FAILED", "encoder", f"Safe checkpoint load rejected {artifact.path}: {exc}")
    artifact.verify()
    if type(payload) is not dict:
        fail("CHECKPOINT_LOAD_FAILED", "encoder", "checkpoint payload must be a dictionary")
    return payload


def _strict_state(model, state):
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        fail("ARCHITECTURE_MISMATCH", "encoder", "checkpoint state coverage differs from architecture")
    for name, value in state.items():
        if (type(value) is not torch.Tensor or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            fail("ARCHITECTURE_MISMATCH", "encoder", f"invalid state tensor: {name}")
    model.load_state_dict(state, strict=True)


_LOADED = object()


class FrozenEncoder:
    """Created only by load_encoder after every required weight has loaded."""
    is_synthetic = False

    def __init__(self, spec, model, *, _loaded=None):
        if _loaded is not _LOADED:
            fail("UNTRAINED_ENCODER", "encoder", "use load_encoder; all trained artifacts must be verified first")
        self.spec, self.provenance = spec, stamp(spec)
        self._model = model.cpu().eval().requires_grad_(False)
        self._fingerprint = model_fingerprint(self._model)
        self._structure = repr(self._model)
        self._lora_scales = tuple((n, m.scale) for n, m in self._model.named_modules() if hasattr(m, "scale"))

    def require_ready(self):
        if stamp(self.spec) != self.provenance:
            fail("PROVENANCE_MISMATCH", "encoder", "encoder specification changed")
        for _, artifact in self.spec.artifacts:
            artifact.verify()
        if model_fingerprint(self._model) != self._fingerprint:
            fail("MODEL_MUTATED", "encoder", "frozen encoder state changed")
        if (repr(self._model) != self._structure
                or tuple((n, m.scale) for n, m in self._model.named_modules() if hasattr(m, "scale")) != self._lora_scales):
            fail("MODEL_MUTATED", "encoder", "frozen encoder architecture/settings changed")

    def encode(self, decoded):
        self.require_ready()
        raw = decoded.raw
        raw.validate()
        if raw.waveform.device.type != "cpu" or raw.frames.device.type != "cpu":
            fail("UNSUPPORTED_DEVICE", "encoder", "only CPU decoded inputs are supported")
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            self._model.eval()
            if self.spec.modality == "audio":
                embeddings = self._model.embed(raw.waveform)
            else:
                frames = raw.frames.permute(0, 1, 4, 2, 3).float() / 255.0
                mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(1, 1, 3, 1, 1)
                std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(1, 1, 3, 1, 1)
                embeddings = self._model.embed((frames - mean) / std)
            output = EncoderOutput(embeddings, torch.ones(embeddings.shape[:2], dtype=torch.bool),
                                   tuple(float(t) for t in range(10)), self.provenance)
            output.validate(raw.waveform.shape[0], self.provenance)
        self.require_ready()
        return output


def load_encoder(spec, *, device="cpu"):
    """No executable model is returned before strict trained-state loading."""
    if device != "cpu":
        fail("UNSUPPORTED_DEVICE", "encoder", "only CPU encoder inference is supported")
    cfg = spec.validate()
    report = readiness(spec)
    if report["issues"]:
        fail("BLOCKED", "encoder", json.dumps(report["issues"]))
    assets = dict(spec.artifacts)
    trained = _safe_payload(assets["trained"])
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad(), torch.autocast("cpu", enabled=False):
            if spec.modality == "vision":
                from vision.model import VisionModel
                if (set(cfg) != {"trained_config"} or trained.get("config") != cfg["trained_config"]
                        or cfg["trained_config"].get("backbone") != "resnet18"
                        or cfg["trained_config"].get("emb_dim") != 512):
                    fail("ARCHITECTURE_MISMATCH", "encoder", "exact saved ResNet-18/512D configuration required")
                model = VisionModel(backbone="resnet18", emb_dim=512, pretrained=False)
                _strict_state(model, trained.get("model"))
            else:
                from audio.beats import BEATs, BEATsConfig
                from audio.beats_lora import BeatsLoRA
                from audio.lora import inject_lora
                from audio.model import EmbeddingHead
                base = _safe_payload(assets["base"])
                if (set(cfg) != {"base_config", "trained_config"} or base.get("cfg") != cfg["base_config"]
                        or trained.get("config") != cfg["trained_config"]):
                    fail("ARCHITECTURE_MISMATCH", "encoder", "full actual BEATs and saved LoRA configuration required")
                tc = cfg["trained_config"]
                if (tc != {"kind": "beats_lora", "encoder": "iter3", "r": 8, "alpha": 16.0,
                           "emb_dim": 256, "features": "beats_iter3"}):
                    fail("ARCHITECTURE_MISMATCH", "encoder", "unsupported audio producer configuration")
                bc = BEATsConfig(base["cfg"])
                if (bc.finetuned_model or bc.encoder_embed_dim != 768
                        or bc.encoder_layers != 12 or bc.input_patch_size != 16):
                    fail("ARCHITECTURE_MISMATCH", "encoder", "self-supervised 768D BEATs required")
                backbone = BEATs(bc)
                _strict_state(backbone, base.get("model"))
                assets["normalization"].verify()
                stats = json.loads(Path(assets["normalization"].path).read_text(encoding="utf-8"))
                mean, std = (torch.tensor(stats[k], dtype=torch.float32) for k in ("mean", "std"))
                if (stats.get("dim") != 768 or stats.get("encoder") != "BEATs iter3"
                        or mean.shape != (768,) or std.shape != (768,) or not torch.isfinite(mean).all()
                        or not torch.isfinite(std).all() or not (std > 0).all()):
                    fail("NORMALIZATION_MISMATCH", "encoder", "finite frozen 768D mean and positive stored std required")

                class BoundBeatsLoRA(BeatsLoRA):
                    # Reuse the producer's tokens/embed/train methods. Bypass only
                    # its unpinned loader and cache-dependent normalization constructor.
                    def __init__(self):
                        torch.nn.Module.__init__(self)
                        self.beats = backbone
                        self.n_lora_layers = inject_lora(backbone, ("q_proj", "v_proj"), r=8, alpha=16.)
                        self.register_buffer("mean", mean)
                        self.register_buffer("std", std)
                        self.head = EmbeddingHead(in_dim=768, emb_dim=256)
                        self.emb_dim = 256

                model = BoundBeatsLoRA()
                state = trained.get("model")
                names = {n for n, p in model.named_parameters() if p.requires_grad}
                if not isinstance(state, dict) or set(state) != names:
                    fail("ARCHITECTURE_MISMATCH", "encoder", "every trained LoRA/head tensor must be present")
                # Merge only after exact coverage, shape and dtype validation.
                full = dict(model.state_dict())
                full.update(state)
                _strict_state(model, full)
        result = FrozenEncoder(spec, model, _loaded=_LOADED)
        result.require_ready()
        return result
    except IntegrationError:
        raise
    except Exception as exc:
        fail("ENCODER_BUILD_FAILED", "encoder", f"Exact producer reconstruction failed: {exc}")

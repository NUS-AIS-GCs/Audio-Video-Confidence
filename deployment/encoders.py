"""Independent CPU encoder construction behind the unchanged Phase 6 boundary.

Artifact hashes must come from trusted acquisition evidence. Loading one modality
does not confer raw prediction readiness or authenticate its exported features.
"""
import importlib
import importlib.metadata
import json

from deployment.runtime import DeploymentConfig, DeploymentError, PINNED_VERSIONS
from fusion.encoder_interfaces import (
    AUDIO_PREPROCESSING, VIDEO_PREPROCESSING, DEPENDENCIES,
    Artifact, EncoderSpec, load_encoder,
)


def dependency_issues(modality):
    """Check actual imports as well as versions (find_spec alone misses DLL errors)."""
    if modality not in DEPENDENCIES:
        raise DeploymentError("UNSUPPORTED_ENCODER_MODALITY")
    issues = []
    for name in DEPENDENCIES[modality]:
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            issues.append({"dependency": name, "code": "DEPENDENCY_MISSING"})
            continue
        if actual != PINNED_VERSIONS[name]:
            issues.append({"dependency": name, "code": "DEPENDENCY_VERSION_MISMATCH"})
            continue
        try:
            importlib.import_module(name)
        except Exception:
            issues.append({"dependency": name, "code": "DEPENDENCY_IMPORT_FAILED"})
    return issues


def _bindings(config, modality):
    if type(config) is not DeploymentConfig:
        raise DeploymentError("INVALID_DEPLOYMENT_CONFIGURATION")
    by_name = {setting.name: setting for setting in config.artifacts}
    if len(by_name) != len(config.artifacts):
        raise DeploymentError("DUPLICATE_ARTIFACT_BINDING")
    roles = (("base", "audio_base"), ("trained", "audio_trained"),
             ("normalization", "audio_normalization")) if modality == "audio" else (("trained", "vision_trained"),)
    selected = []
    for role, name in roles:
        if name not in by_name:
            raise DeploymentError("REQUIRED_ARTIFACT_BINDING_MISSING")
        setting = by_name[name]
        # Also validate programmatically constructed configs, not only environment paths.
        try:
            setting.path.resolve().relative_to(config.model_root.resolve())
        except (ValueError, OSError):
            raise DeploymentError("ARTIFACT_PATH_OUTSIDE_MODEL_ROOT") from None
        setting.verify()
        selected.append((role, Artifact(str(setting.path), setting.sha256)))
    return tuple(selected)


def _payload(artifact):
    import torch
    artifact.verify()
    try:
        payload = torch.load(artifact.path, map_location="cpu", weights_only=True)
    except Exception:
        raise DeploymentError("SAFE_CHECKPOINT_LOAD_FAILED") from None
    artifact.verify()
    if type(payload) is not dict or type(payload.get("config")) is not dict:
        raise DeploymentError("INVALID_CHECKPOINT_SCHEMA")
    return payload


def encoder_spec(config, modality):
    """Read actual saved config; never infer it from a filename or model card."""
    if modality not in DEPENDENCIES:
        raise DeploymentError("UNSUPPORTED_ENCODER_MODALITY")
    bindings = _bindings(config, modality)
    assets = dict(bindings)
    trained = _payload(assets["trained"])
    configuration = {"trained_config": trained["config"]}
    if modality == "audio":
        import torch
        assets["base"].verify()
        try:
            base = torch.load(assets["base"].path, map_location="cpu", weights_only=True)
        except Exception:
            raise DeploymentError("SAFE_CHECKPOINT_LOAD_FAILED") from None
        assets["base"].verify()
        if type(base) is not dict or type(base.get("cfg")) is not dict:
            raise DeploymentError("INVALID_AUDIO_CHECKPOINT_CONFIGURATION")
        configuration["base_config"] = base["cfg"]
    try:
        configuration_json = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise DeploymentError("INVALID_CHECKPOINT_CONFIGURATION") from None
    return EncoderSpec(
        modality, "beats_lora_r8_v1 / beats_iter3" if modality == "audio" else "resnet18_v1",
        bindings, configuration_json,
        AUDIO_PREPROCESSING if modality == "audio" else VIDEO_PREPROCESSING,
        tuple((name, PINNED_VERSIONS[name]) for name in DEPENDENCIES[modality]),
    )


def load_modality(config, modality):
    """Return a real FrozenEncoder or a sanitized failure; never a fallback model."""
    issues = dependency_issues(modality)
    if issues:
        raise DeploymentError(issues[0]["code"])
    try:
        spec = encoder_spec(config, modality)
        return load_encoder(spec, device="cpu")
    except DeploymentError:
        raise
    except Exception as exc:
        raise DeploymentError(getattr(exc, "code", "VERIFIED_ENCODER_LOAD_FAILED")) from None


def load_audio_encoder(config):
    return load_modality(config, "audio")


def load_video_encoder(config):
    return load_modality(config, "vision")

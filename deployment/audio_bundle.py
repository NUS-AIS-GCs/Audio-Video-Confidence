"""Safely bind a producer audio bundle to Phase 6 and AudioPredictor numerics.

No downloads, permissive pickle load, raw decoder, tolerance or readiness override.
The model-card digest must come from the approved acquisition, not its filename.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from audio.infer import AudioPredictor
from deployment.encoders import dependency_issues
from deployment.runtime import PINNED_VERSIONS
from fusion.contracts import ContractError
from fusion.encoder_interfaces import (
    AUDIO_PREPROCESSING, DEPENDENCIES, Artifact, EncoderSpec, load_encoder,
)


TRAINED_CONFIG = {"kind": "beats_lora", "encoder": "iter3", "r": 8, "alpha": 16.0,
                  "emb_dim": 256, "features": "beats_iter3"}
HASHED_FILES = frozenset({"encoder.pt", "weights.pt", "norm_stats.json",
                          "class_names.json", "environment.json"})


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_bundle(card_artifact):
    """Check acquisition-bound card and each producer-declared payload digest."""
    if type(card_artifact) is not Artifact or Path(card_artifact.path).name != "model_card.json":
        raise ContractError("explicit acquisition-bound model_card.json Artifact required")
    card_artifact.verify()
    card = _json(card_artifact.path)
    if (card.get("run_id") != "beats_lora_r8_v1" or card.get("config") != TRAINED_CONFIG
            or card.get("encoder") != "iter3" or card.get("features") != "beats_iter3"
            or type(card.get("sample_rate")) is not int or card["sample_rate"] != 16000
            or type(card.get("clip_seconds")) is not int or card["clip_seconds"] != 10
            or type(card.get("sha256")) is not dict or set(card["sha256"]) != HASHED_FILES):
        raise ContractError("unsupported producer audio bundle configuration")
    root = Path(card_artifact.path).resolve().parent
    for name, digest in card["sha256"].items():
        path = (root / name).resolve()
        if path.parent != root:
            raise ContractError("bundle payload escapes its approved directory")
        Artifact(str(path), digest).verify()
    names = _json(root / "class_names.json")
    if (type(names) is not list or len(names) != 29 or names[-1] != "background"
            or any(type(n) is not str or not n for n in names)
            or names[:-1] != sorted(names[:-1]) or len(set(names)) != 29):
        raise ContractError("invalid producer class vocabulary")
    stats = _json(root / "norm_stats.json")
    if type(stats) is not dict or set(stats) != {"mean", "std"}:
        raise ContractError("expected original producer mean/std schema")
    for key in ("mean", "std"):
        value = np.asarray(stats[key])
        if (value.shape != (768,) or value.dtype.kind != "f" or not np.isfinite(value).all()
                or (key == "std" and not (value > 0).all())):
            raise ContractError("invalid frozen producer normalization")
    card_artifact.verify()
    return root, card, stats, names


def bundle_spec(card_artifact, derived_directory):
    """Add audited schema metadata in a separate file, never rewrite producer stats.

    Geometry/identity are checked against actual saved model configurations; mean
    and std arrays are preserved verbatim without recomputation or added epsilon.
    """
    root, card, stats, names = verify_bundle(card_artifact)
    base = torch.load(root / "encoder.pt", map_location="cpu", weights_only=True)
    trained = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
    if (type(base) is not dict or type(base.get("cfg")) is not dict
            or type(trained) is not dict or trained.get("config") != card["config"]):
        raise ContractError("saved checkpoint configurations differ from producer card")
    cfg = base["cfg"]
    if (cfg.get("encoder_layers") != 12 or cfg.get("encoder_embed_dim") != 768
            or cfg.get("input_patch_size") != 16 or cfg.get("finetuned_model", False)):
        raise ContractError("expected self-supervised 12-layer/768D/patch16 BEATs")
    derived = {**stats, "dim": 768, "encoder": "BEATs iter3",
               "source_normalization_sha256": card["sha256"]["norm_stats.json"],
               "source_model_card_sha256": card_artifact.sha256}
    text = json.dumps(derived, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    destination = Path(derived_directory) / "normalization.phase6.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_text(encoding="utf-8") != text:
            raise ContractError("derived normalization exists with different provenance/content")
    else:
        with destination.open("x", encoding="utf-8", newline="\n") as output:
            output.write(text)
    config = json.dumps({"base_config": base["cfg"], "trained_config": trained["config"]},
                        sort_keys=True, separators=(",", ":"), allow_nan=False)
    spec = EncoderSpec(
        "audio", "beats_lora_r8_v1 / beats_iter3",
        (("base", Artifact(str(root / "encoder.pt"), card["sha256"]["encoder.pt"])),
         ("trained", Artifact(str(root / "weights.pt"), card["sha256"]["weights.pt"])),
         ("normalization", Artifact(str(destination), hashlib.sha256(destination.read_bytes()).hexdigest()))),
        config, AUDIO_PREPROCESSING,
        tuple((name, PINNED_VERSIONS[name]) for name in DEPENDENCIES["audio"]),
    )
    verify_bundle(card_artifact)
    return spec, root, card, names


class VerifiedAudioPredictor(AudioPredictor):
    """Reuse AudioPredictor.predict_waveform with a strictly loaded authentic model.

    Bypasses only the producer constructor's unsafe/unpinned loaders. Numerical
    operations remain the producer's inherited waveform path and BeatsLoRA model.
    This is an offline canonical-waveform interface, not a raw-serving endpoint.
    """
    def __init__(self, card_artifact, derived_directory):
        issues = dependency_issues("audio")
        if issues:
            raise ContractError(f"audio candidate runtime unavailable: {issues}")
        self._card_artifact = card_artifact
        spec, self.dir, self.card, self.class_names = bundle_spec(card_artifact, derived_directory)
        self.encoder = load_encoder(spec, device="cpu")
        self.cfg, self.kind, self.device = self.card["config"], "beats_lora", torch.device("cpu")
        self.model = self.encoder._model

    def predict_waveform(self, wav):
        if (type(wav) is not np.ndarray or wav.dtype != np.float32 or wav.shape != (160000,)
                or not np.isfinite(wav).all() or np.any(np.abs(wav) > 1)):
            raise ContractError("exact finite mono float32 160000-sample waveform required; no pad/trim")
        self._card_artifact.verify()
        self.encoder.require_ready()
        if self.model is not self.encoder._model:
            raise ContractError("verified audio model was replaced")
        self.model.eval()
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            result = super().predict_waveform(wav)
        self.encoder.require_ready()
        self._card_artifact.verify()
        if (result.embeddings.shape != (10, 256) or result.embeddings.dtype != np.float32
                or result.probs.shape != (10, 29) or result.probs.dtype != np.float32
                or not np.isfinite(result.embeddings).all() or not np.isfinite(result.probs).all()):
            raise ContractError("invalid genuine AudioPredictor output")
        return result

    def predict_video(self, path):
        raise ContractError("compressed-media serving is not enabled; use audited canonical waveform diagnostics")

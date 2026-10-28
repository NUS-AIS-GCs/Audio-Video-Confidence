"""Original bundle integrity and real CPU architecture checks; no alternate weights."""
import hashlib
import json
import pickle
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from deployment.audio_bundle import (
    HASHED_FILES, TRAINED_CONFIG, VerifiedAudioPredictor, bundle_spec, verify_bundle,
)
from deployment.encoders import dependency_issues
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "_local/audio_handoff/ghanshyam_20261027"
CARD_SHA = "3b98967d720e7fad3de3af37f57487ebd49cf48dc357cef8dbde394a86470bfc"


def fixture_bundle(tmp_path, *, stats=None, extra_hash=None):
    # Verification-only malformed payload fixture, never model construction.
    names = [f"Event{i:02d}" for i in range(28)] + ["background"]
    payloads = {"class_names.json": json.dumps(names).encode(), "environment.json": b"{}",
                "encoder.pt": b"invalid fixture checkpoint", "weights.pt": b"invalid fixture checkpoint",
                "norm_stats.json": json.dumps(stats or {"mean": [0.] * 768, "std": [1.] * 768}).encode()}
    for name, content in payloads.items():
        (tmp_path / name).write_bytes(content)
    card = {"run_id": "beats_lora_r8_v1", "config": dict(TRAINED_CONFIG), "encoder": "iter3",
            "features": "beats_iter3", "sample_rate": 16000, "clip_seconds": 10,
            "sha256": {name: hashlib.sha256(content).hexdigest() for name, content in payloads.items()}}
    if extra_hash:
        card["sha256"].update(extra_hash)
    path = tmp_path / "model_card.json"
    path.write_text(json.dumps(card))
    return Artifact(str(path), hashlib.sha256(path.read_bytes()).hexdigest())


def test_acquisition_card_hash_is_mandatory(tmp_path):
    card = fixture_bundle(tmp_path)
    with pytest.raises(ContractError):
        verify_bundle(Artifact(card.path, "0" * 64))
    with pytest.raises(ContractError):
        verify_bundle(card.path)


def test_payload_change_is_rejected_before_model_load(tmp_path):
    card = fixture_bundle(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"changed")
    with mock.patch("deployment.audio_bundle.torch.load") as loader:
        with pytest.raises(ContractError):
            bundle_spec(card, tmp_path / "derived")
    loader.assert_not_called()


def test_card_cannot_authorize_arbitrary_payload_paths(tmp_path):
    card = fixture_bundle(tmp_path, extra_hash={"../escape.pt": "0" * 64})
    with pytest.raises(ContractError, match="configuration"):
        verify_bundle(card)


@pytest.mark.parametrize("stats", [
    {"mean": [0.] * 767, "std": [1.] * 768},
    {"mean": [0.] * 768, "std": [0.] * 768},
    {"mean": [float('nan')] * 768, "std": [1.] * 768},
    {"mean": [0.] * 768, "std": [1.] * 768, "dim": 768},
])
def test_normalization_schema_and_values_fail_closed(tmp_path, stats):
    card = fixture_bundle(tmp_path, stats=stats)
    with pytest.raises(ContractError, match="normalization|schema"):
        verify_bundle(card)


def test_malformed_checkpoint_uses_only_restricted_loader(tmp_path):
    card = fixture_bundle(tmp_path)
    original_load = torch.load
    calls = []
    def restricted(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)
    with mock.patch("deployment.audio_bundle.torch.load", side_effect=restricted):
        with pytest.raises((pickle.UnpicklingError, RuntimeError, EOFError)):
            bundle_spec(card, tmp_path / "derived")
    assert calls and not (tmp_path / "derived").exists()


@pytest.fixture(scope="module")
def authentic_audio(tmp_path_factory):
    if not (BUNDLE / "encoder.pt").is_file():
        pytest.skip("approved original audio bundle not available; genuine execution unverified")
    if dependency_issues("audio"):
        pytest.skip("candidate CPU audio dependencies unavailable")
    original_load = torch.load
    calls = []
    def restricted(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)
    with mock.patch("torch.load", side_effect=restricted), \
         mock.patch("urllib.request.urlretrieve", side_effect=AssertionError("downloads forbidden")):
        predictor = VerifiedAudioPredictor(Artifact(str(BUNDLE / "model_card.json"), CARD_SHA),
                                           tmp_path_factory.mktemp("derived_audio"))
    assert len(calls) == 4
    return predictor


def test_authentic_base_lora_and_trained_heads(authentic_audio):
    p = authentic_audio
    model = p.model
    assert p.cfg == TRAINED_CONFIG and p.device.type == "cpu"
    assert model.beats.cfg.encoder_layers == 12 and model.beats.cfg.encoder_embed_dim == 768
    assert model.beats.cfg.input_patch_size == 16 and not model.beats.cfg.finetuned_model
    assert model.n_lora_layers == 24
    lora = [m for m in model.modules() if hasattr(m, "lora_A")]
    assert len(lora) == 24 and all(m.lora_A.shape == (8, 768) and m.scale == 2. for m in lora)
    assert model.head.proj[1].weight.shape == (256, 768)
    assert model.head.head[1].weight.shape == (29, 256)
    assert not model.training and not any(parameter.requires_grad for parameter in model.parameters())
    raw = json.loads((BUNDLE / "norm_stats.json").read_text())
    assert torch.equal(model.mean, torch.tensor(raw['mean'], dtype=torch.float32))
    assert torch.equal(model.std, torch.tensor(raw['std'], dtype=torch.float32))
    derived = json.loads(Path(dict(p.encoder.spec.artifacts)['normalization'].path).read_text())
    assert derived['mean'] == raw['mean'] and derived['std'] == raw['std']
    assert derived['source_normalization_sha256'] == p.card['sha256']['norm_stats.json']


@pytest.mark.parametrize("wave", [np.zeros(159999, dtype=np.float32), np.zeros(160000, dtype=np.float64),
                                  np.zeros((1, 160000), dtype=np.float32),
                                  np.full(160000, np.nan, dtype=np.float32),
                                  np.full(160000, 1.1, dtype=np.float32)])
def test_audio_adapter_never_pads_trims_or_repairs(wave, authentic_audio):
    with pytest.raises(ContractError, match="no pad/trim"):
        authentic_audio.predict_waveform(wave)


def test_audio_bundle_does_not_enable_compressed_serving(authentic_audio):
    with pytest.raises(ContractError, match="not enabled"):
        authentic_audio.predict_video("not-decoded.mp4")


def test_actual_audio_predictor_path_shapes_and_autocast_restoration(authentic_audio):
    # Canonical zero waveform is a shape/implementation test, not TRAIN evidence.
    wave = np.zeros(160000, dtype=np.float32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = authentic_audio.predict_waveform(wave)
        assert torch.is_autocast_enabled('cpu')
    assert result.embeddings.shape == (10, 256) and result.probs.shape == (10, 29)
    assert result.embeddings.dtype == result.probs.dtype == np.float32
    assert np.isfinite(result.embeddings).all() and np.isfinite(result.probs).all()
    assert np.allclose(result.probs.sum(1), 1, atol=1e-6)
    authentic_audio.encoder.require_ready()

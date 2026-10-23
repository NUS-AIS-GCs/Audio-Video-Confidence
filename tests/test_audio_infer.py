"""Audio inference bundle: predictor shapes, preprocessing and AudioPrediction semantics.

Uses a randomly initialised CNN bundle, so it needs no dataset, no GPU and no BEATs checkpoint.
"""
import json

import numpy as np
import pytest
import torch

from audio import config as C
from audio.infer import AudioPrediction, AudioPredictor
from audio.model import AudioCNN


@pytest.fixture()
def cnn_bundle(tmp_path):
    torch.manual_seed(0)
    model = AudioCNN(emb_dim=256)
    cfg = {"kind": "cnn", "emb_dim": 256, "in_dim": 100, "features": "audio_logmel_v2"}
    torch.save({"model": model.state_dict(), "config": cfg}, tmp_path / "weights.pt")
    (tmp_path / "norm_stats.json").write_text(json.dumps({"mean": [-8.0] * C.N_MELS, "std": [3.0] * C.N_MELS}))
    names = [f"Event{i:02d}" for i in range(C.N_EVENTS)] + ["background"]
    (tmp_path / "class_names.json").write_text(json.dumps(names))
    (tmp_path / "model_card.json").write_text(json.dumps({"run_id": "unit", "config": cfg, "encoder": None}))
    return tmp_path


def test_predictor_shapes_and_probabilities(cnn_bundle):
    p = AudioPredictor(cnn_bundle, device="cpu")
    out = p.predict_waveform(np.random.default_rng(0).normal(0, 0.1, C.CLIP_SAMPLES).astype(np.float32))
    assert out.probs.shape == (10, C.N_CLASSES)
    assert out.embeddings.shape == (10, 256)
    assert np.allclose(out.probs.sum(1), 1, atol=1e-5)
    assert len(out.second_labels) == 10 and len(out.presence) == 10


def test_short_and_long_audio_are_padded_or_trimmed(cnn_bundle):
    p = AudioPredictor(cnn_bundle, device="cpu")
    wav = np.random.default_rng(1).normal(0, 0.1, C.CLIP_SAMPLES).astype(np.float32)
    full = p.predict_waveform(wav)
    longer = p.predict_waveform(np.concatenate([wav, wav]))           # extra audio after 10 s is ignored
    assert np.allclose(full.probs, longer.probs, atol=1e-6)
    short = p.predict_waveform(wav[: C.SR * 4])                       # 4 s clip -> silence afterwards
    assert short.probs.shape == (10, C.N_CLASSES)


def test_prediction_semantics():
    names = [f"E{i}" for i in range(C.N_EVENTS)] + ["background"]
    probs = np.full((10, C.N_CLASSES), 0.001)
    probs[:, C.BACKGROUND] = 0.9                                      # background everywhere...
    probs[3:6, C.BACKGROUND] = 0.0
    probs[3:6, 5] = 0.9                                               # ...except event 5 in seconds 3-5
    probs /= probs.sum(1, keepdims=True)
    pred = AudioPrediction(probs=probs, embeddings=np.zeros((10, 256)), class_names=names)
    assert pred.presence == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    assert pred.second_labels[4] == "E5" and pred.second_labels[0] == "background"
    assert pred.clip_label == "E5"                                    # clip label ignores background
    assert abs(pred.clip_distribution.sum() - 1) < 1e-9
    assert pred.top_k(3)[0]["class"] == "E5"

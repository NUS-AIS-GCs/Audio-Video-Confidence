"""Vision inference bundle: predictor shapes, preprocessing and VisionPrediction semantics.

Uses a randomly initialised ResNet-18 bundle, so it needs no dataset and no GPU.
"""
import json

import numpy as np
import pytest
import torch

from vision import config as C
from vision.infer import VisionPrediction, VisionPredictor
from vision.model import VisionModel


@pytest.fixture()
def resnet_bundle(tmp_path):
    torch.manual_seed(0)
    model = VisionModel(n_classes=C.N_CLASSES, emb_dim=512, backbone="resnet18", pretrained=False)
    cfg = {"backbone": "resnet18", "emb_dim": 512, "n_classes": C.N_CLASSES}
    torch.save({"model": model.state_dict(), "config": cfg}, tmp_path / "weights.pt")
    names = [f"Event{i:02d}" for i in range(C.N_EVENTS)] + ["background"]
    (tmp_path / "class_names.json").write_text(json.dumps(names, indent=1))
    (tmp_path / "model_card.json").write_text(json.dumps({"run_id": "unit_vision", "config": cfg}))
    return tmp_path


def test_predictor_shapes_and_probabilities(resnet_bundle):
    p = VisionPredictor(resnet_bundle, device="cpu")
    frames = np.random.default_rng(0).integers(0, 256, (10, 224, 224, 3), dtype=np.uint8)
    out = p.predict_frames(frames)
    assert out.probs.shape == (10, C.N_CLASSES)
    assert out.embeddings.shape == (10, 512)
    assert np.allclose(out.probs.sum(1), 1, atol=1e-5)
    assert len(out.second_labels) == 10 and len(out.presence) == 10


def test_prediction_semantics():
    names = [f"E{i}" for i in range(C.N_EVENTS)] + ["background"]
    probs = np.full((10, C.N_CLASSES), 0.001)
    probs[:, C.BACKGROUND] = 0.9                                      # background everywhere...
    probs[2:5, C.BACKGROUND] = 0.0
    probs[2:5, 3] = 0.9                                               # ...except event 3 in seconds 2-4
    probs /= probs.sum(1, keepdims=True)
    pred = VisionPrediction(probs=probs, embeddings=np.zeros((10, 512)), class_names=names)
    assert pred.presence == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    assert pred.second_labels[3] == "E3" and pred.second_labels[0] == "background"
    assert pred.clip_label == "E3"                                    # clip label ignores background
    assert abs(pred.clip_distribution.sum() - 1) < 1e-9
    assert pred.top_k(3)[0]["class"] == "E3"


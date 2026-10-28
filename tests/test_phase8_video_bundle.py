"""Real-bundle verification and rejection paths for the internal Stage 2B seam."""
import hashlib
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from deployment.video_bundle import PREPROCESSING, WEIGHTS_SHA256, VerifiedVideoPredictor, verify_bundle
from deployment.train_video_diagnostics import sample_train_frames
from deployment.paired_diagnostics import InternalTrainPairStudy
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "_local/video_handoff/stage2b_20261028"
CARD_SHA = "12f069feab94d3324f80cf9de08390a5f1a6fdaa7acdb293ada1690262925a1b"


def fixture_card():
    return {"run_id":"resnet18_v1", "backbone":"resnet18", "emb_dim":512,
            "n_classes":29,"preprocessing":dict(PREPROCESSING),
            "sha256":{"weights.pt":WEIGHTS_SHA256,"class_names.json":"0"*64}}


@pytest.fixture(scope="module")
def authentic():
    if not (BUNDLE / "weights.pt").is_file():
        pytest.skip("approved Stage 2B bundle unavailable; actual inference unverified")
    original = torch.load
    def restricted(*args, **kwargs):
        assert kwargs.get("weights_only") is True
        return original(*args, **kwargs)
    with mock.patch("torch.load", side_effect=restricted), \
         mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("download forbidden")):
        return VerifiedVideoPredictor(Artifact(str(BUNDLE / "model_card.json"), CARD_SHA))


def test_authentic_architecture_and_class_order(authentic):
    assert hashlib.sha256((BUNDLE / "weights.pt").read_bytes()).hexdigest() == WEIGHTS_SHA256
    assert isinstance(authentic.model.encoder.fc, torch.nn.Identity)
    assert isinstance(authentic.model.proj, torch.nn.Identity)
    assert authentic.model.head[1].weight.shape == (29, 512)
    assert not authentic.model.training and not any(p.requires_grad for p in authentic.model.parameters())
    frozen_names = json.loads((ROOT / "_local/embeddings/vision_final/resnet18_v1/class_names.json").read_text())
    assert authentic.class_names == frozen_names


@pytest.mark.parametrize("value", [np.zeros((10,224,224,3), dtype=np.float32),
                                  np.zeros((9,224,224,3), dtype=np.uint8), None])
def test_noncanonical_frames_rejected(authentic, value):
    with pytest.raises(ContractError, match="canonical"):
        authentic.predict_frames(value)


def test_compressed_serving_remains_disabled(authentic):
    with pytest.raises(ContractError, match="disabled"):
        authentic.predict_video("anything.mp4")


def test_card_and_payload_are_bound_before_safe_load(tmp_path):
    (tmp_path / "weights.pt").write_bytes(b"incorrect")
    card = fixture_card()
    p = tmp_path / "model_card.json"
    p.write_text(json.dumps(card))
    with mock.patch("torch.load") as loader:
        with pytest.raises(ContractError):
            verify_bundle(Artifact(str(p), hashlib.sha256(p.read_bytes()).hexdigest()))
    loader.assert_not_called()
    with pytest.raises(ContractError):
        verify_bundle(Artifact(str(p), "0" * 64))


@pytest.mark.parametrize("field,value", [("n_classes",28),("emb_dim",256),("backbone","cnn"),
                                        ("preprocessing",{**PREPROCESSING,"fps":2})])
def test_card_cannot_change_training_geometry(tmp_path, field, value):
    card=fixture_card()
    card[field]=value
    p=tmp_path / "model_card.json"
    p.write_text(json.dumps(card))
    with pytest.raises(ContractError, match="configuration"):
        verify_bundle(Artifact(str(p), hashlib.sha256(p.read_bytes()).hexdigest()))


class Capture:
    def __init__(self, *, fps=29.97002997003, count=300, opened=True, fail=None):
        self.fps,self.count,self.opened,self.fail=fps,count,opened,fail
        self.target=0
        self.released=False
    def isOpened(self):return self.opened
    def getBackendName(self):return "diagnostic fixture"
    def set(self, key, value):self.target=value;return True
    def get(self, key):
        import cv2
        return {cv2.CAP_PROP_FPS:self.fps,cv2.CAP_PROP_FRAME_COUNT:self.count,
                cv2.CAP_PROP_POS_MSEC:self.target/self.fps*1000 if self.fps else 0,
                cv2.CAP_PROP_POS_FRAMES:self.target+1}.get(key,0)
    def read(self):
        return (False,None) if self.target==self.fail else (True,np.zeros((224,224,3),np.uint8))
    def release(self):self.released=True


def source_fixture(tmp_path):
    p=tmp_path / "permitted.mp4";p.write_bytes(b"fixture")
    return Artifact(str(p),hashlib.sha256(p.read_bytes()).hexdigest())


@pytest.mark.parametrize("kwargs", [{"opened":False},{"fps":0},{"fps":float('nan')},
                                    {"count":20},{"fail":29},{"fps":.5}])
def test_no_black_repeat_clamp_or_fps_repair(tmp_path, kwargs):
    cap=Capture(**kwargs)
    with mock.patch("deployment.train_video_diagnostics.cv2.VideoCapture",return_value=cap):
        with pytest.raises(ContractError):sample_train_frames(source_fixture(tmp_path))
    assert cap.released


def test_truncation_and_actual_pts_are_preserved(tmp_path):
    cap=Capture()
    with mock.patch("deployment.train_video_diagnostics.cv2.VideoCapture",return_value=cap):
        frames,evidence=sample_train_frames(source_fixture(tmp_path))
    assert frames.shape==(10,224,224,3)
    assert [f['source_frame_index'] for f in evidence['frames']]==[0,29,59,89,119,149,179,209,239,269]
    assert evidence['frames'][1]['timestamp_seconds'] < 1
    assert evidence['repairs_applied'] is False and cap.released


def test_internal_study_rejects_synthetic_predictors():
    with pytest.raises(ContractError, match="authentic"):
        InternalTrainPairStudy(object(),object(),{},None,None,{},None)


def test_unapproved_media_cannot_enter_study():
    study=object.__new__(InternalTrainPairStudy)
    study.sources={}
    with pytest.raises(ContractError,match="unapproved"):
        study.encode("unapproved")


@pytest.mark.parametrize("ids", [[],["unapproved"],["permitted","permitted"]])
def test_compare_rejects_unapproved_or_duplicate_ids(ids):
    study=object.__new__(InternalTrainPairStudy)
    study.sources={"permitted":None}
    with pytest.raises(ContractError,match="unique approved"):
        study.compare(ids,np.zeros((1,10,256),np.float32),np.zeros((1,10,512),np.float32))


@pytest.mark.parametrize("audio,video", [
    (np.zeros((1,10,256),np.float16),np.zeros((1,10,512),np.float32)),
    (np.zeros((1,9,256),np.float32),np.zeros((1,10,512),np.float32)),
    (np.zeros((1,10,256),np.float32),np.full((1,10,512),np.nan,np.float32)),
])
def test_compare_rejects_noncanonical_features(audio,video):
    study=object.__new__(InternalTrainPairStudy)
    study.sources={"permitted":None}
    with pytest.raises(ContractError,match="finite genuine"):
        study.compare(["permitted"],audio,video)

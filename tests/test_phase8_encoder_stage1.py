"""Backend isolation/failure tests; real checkpoint tests never use random weights."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from deployment import encoders
from deployment.encoder_diagnostics import diagnose, embedding_differences
from deployment.runtime import ArtifactSetting, DeploymentConfig, DeploymentError, PINNED_VERSIONS
from fusion.inference_contracts import BIN_STARTS, RawMediaInput


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "_local/drive/artifacts/vision_baseline/resnet18_v1/best.pt"
RECORDED_SHA256 = "6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec"


def config_for(path, digest):
    return DeploymentConfig(path.parent, (ArtifactSetting("vision_trained", path, digest),), ())


def saved_payload(tmp_path, payload):
    path = tmp_path / "invalid.pt"
    torch.save(payload, path)
    return config_for(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_metrics_are_descriptive_and_do_not_introduce_tolerance():
    reference = np.zeros((10, 256), dtype=np.float16)
    actual = np.full((10, 256), 0.5, dtype=np.float32)
    metrics = embedding_differences(actual, reference)
    assert metrics == {"max_absolute_error": 0.5, "mae": 0.5, "rmse": 0.5,
                       "float16_array_equal": False, "float16_differing_values": 2560,
                       "numerical_tolerance_selected": False}
    assert embedding_differences(reference, reference)["float16_array_equal"]


@pytest.mark.parametrize("value", [np.zeros((9, 256)), np.zeros((10, 29)),
                                  np.full((10, 256), np.nan), np.full((10, 256), 1e10),
                                  np.zeros((10, 256), dtype=np.int64)])
def test_metrics_reject_invalid_or_nonexportable_features(value):
    with pytest.raises(ValueError):
        embedding_differences(value, np.zeros((10, 256), dtype=np.float16))


def test_missing_distribution_is_reported():
    with mock.patch.object(encoders.importlib.metadata, "version",
                           side_effect=importlib.metadata.PackageNotFoundError):
        issues = encoders.dependency_issues("vision")
    assert {row["code"] for row in issues} == {"DEPENDENCY_MISSING"}


def test_broken_binary_import_is_reported_without_leaking_path():
    with mock.patch.object(encoders.importlib.metadata, "version", side_effect=PINNED_VERSIONS.__getitem__), \
         mock.patch.object(encoders.importlib, "import_module", side_effect=OSError("secret/path DLL failed")):
        issues = encoders.dependency_issues("vision")
    assert {row["code"] for row in issues} == {"DEPENDENCY_IMPORT_FAILED"}
    assert "secret" not in json.dumps(issues)


def test_version_mismatch_blocks_before_checkpoint_loading(tmp_path):
    with mock.patch.object(encoders.importlib.metadata, "version", return_value="0.0"), \
         mock.patch.object(encoders, "encoder_spec") as build:
        with pytest.raises(DeploymentError, match="DEPENDENCY_VERSION_MISMATCH"):
            encoders.load_video_encoder(config_for(tmp_path / "missing.pt", RECORDED_SHA256))
    build.assert_not_called()


def test_absent_audio_does_not_prevent_independent_vision_spec(tmp_path):
    config = saved_payload(tmp_path, {"config": {"backbone": "resnet18", "emb_dim": 512}})
    spec = encoders.encoder_spec(config, "vision")
    assert spec.validate() == {"trained_config": {"backbone": "resnet18", "emb_dim": 512}}
    assert spec.modality == "vision" and tuple(k for k, _ in spec.artifacts) == ("trained",)
    with pytest.raises(DeploymentError, match="REQUIRED_ARTIFACT_BINDING_MISSING"):
        encoders.encoder_spec(config, "audio")


def test_missing_and_changed_artifacts_fail_closed(tmp_path):
    config = config_for(tmp_path / "absent.pt", RECORDED_SHA256)
    with pytest.raises(DeploymentError, match="MISSING_ARTIFACT"):
        encoders.encoder_spec(config, "vision")
    config.artifacts[0].path.write_bytes(b"not the supplied checkpoint")
    with pytest.raises(DeploymentError, match="HASH_MISMATCH"):
        encoders.encoder_spec(config, "vision")


def test_explicit_hash_is_required_and_unsafe_payload_is_rejected(tmp_path):
    path = tmp_path / "fake.pt"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(DeploymentError, match="EXPECTED_HASH_NOT_CONFIGURED"):
        encoders.encoder_spec(config_for(path, None), "vision")
    with pytest.raises(DeploymentError, match="SAFE_CHECKPOINT_LOAD_FAILED"):
        encoders.encoder_spec(config_for(path, hashlib.sha256(path.read_bytes()).hexdigest()), "vision")


def test_programmatic_paths_and_duplicate_bindings_are_checked(tmp_path):
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    config = DeploymentConfig(tmp_path / "models", (ArtifactSetting("vision_trained", outside, RECORDED_SHA256),), ())
    with pytest.raises(DeploymentError, match="ARTIFACT_PATH_OUTSIDE_MODEL_ROOT"):
        encoders.encoder_spec(config, "vision")
    config = DeploymentConfig(tmp_path, config.artifacts * 2, ())
    with pytest.raises(DeploymentError, match="DUPLICATE_ARTIFACT_BINDING"):
        encoders.encoder_spec(config, "vision")


def test_invalid_checkpoint_configuration_is_rejected(tmp_path):
    config = saved_payload(tmp_path, {"config": []})
    with pytest.raises(DeploymentError, match="INVALID_CHECKPOINT_SCHEMA"):
        encoders.encoder_spec(config, "vision")


def test_diagnostic_readiness_cannot_be_opened_by_a_successful_loader(tmp_path):
    fake = SimpleNamespace(provenance=SimpleNamespace(run_identity="test-only", artifacts=()))
    with mock.patch("deployment.encoder_diagnostics.dependency_issues", return_value=[]), \
         mock.patch("deployment.encoder_diagnostics.load_audio_encoder", return_value=fake), \
         mock.patch("deployment.encoder_diagnostics.load_video_encoder", return_value=fake):
        report = diagnose(DeploymentConfig(tmp_path, (), ()), load=True)
    assert all(row["encoder_loaded"] for row in report["encoders"].values())
    assert report["status"] == "MODEL_NOT_READY" and not report["inference_enabled"]
    assert report["predictions"] is None
    codes = {row["code"] for row in report["deployment"]["issues"]}
    assert {"NOT_IMPLEMENTED", "NOT_VERIFIED"} <= codes


@pytest.fixture(scope="module")
def authentic_video():
    if not CHECKPOINT.is_file():
        pytest.skip("exact supplied ResNet checkpoint unavailable; genuine compatibility unverified")
    issues = encoders.dependency_issues("vision")
    if issues:
        pytest.skip(f"exact candidate CPU dependencies unavailable: {issues}")
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == RECORDED_SHA256
    with mock.patch("torch.hub.download_url_to_file", side_effect=AssertionError("weight download forbidden")):
        encoder = encoders.load_video_encoder(config_for(CHECKPOINT, RECORDED_SHA256))
    return encoder


def test_authentic_checkpoint_architecture_and_trained_head(authentic_video):
    model = authentic_video._model
    assert isinstance(model.encoder.fc, torch.nn.Identity)
    assert isinstance(model.proj, torch.nn.Identity)
    assert model.backbone_name == "resnet18" and model.emb_dim == 512
    assert tuple(model.head[1].weight.shape) == (29, 512)
    assert tuple(model.head[1].bias.shape) == (29,)
    assert model.head[0].p == 0.3
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    authentic_video.require_ready()


def test_authentic_checkpoint_512d_execution_is_shape_smoke_only(authentic_video):
    # Deterministic synthetic pixels test architecture, never TRAIN/raw parity.
    frames = torch.arange(10 * 224 * 224 * 3, dtype=torch.int64).remainder(256).to(torch.uint8)
    raw = RawMediaInput(torch.zeros(1, 160000), frames.reshape(1, 10, 224, 224, 3),
                        16000, 10., 10., 0., 0., BIN_STARTS,
                        torch.ones(1, 10, dtype=torch.bool), torch.ones(1, 10, dtype=torch.bool))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = authentic_video.encode(SimpleNamespace(raw=raw))
        assert torch.is_autocast_enabled("cpu")
    second = authentic_video.encode(SimpleNamespace(raw=raw))
    assert first.embeddings.shape == (1, 10, 512) and first.embeddings.dtype == torch.float32
    assert torch.isfinite(first.embeddings).all() and not first.embeddings.requires_grad
    assert torch.equal(first.embeddings, second.embeddings)
    normalized = raw.frames.permute(0, 1, 4, 2, 3).float() / 255.
    normalized = (normalized - torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)) / \
                 torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    with torch.no_grad():
        logits, embedded = authentic_video._model(normalized)
    assert logits.shape == (1, 10, 29) and torch.isfinite(logits).all()
    assert torch.equal(embedded, first.embeddings)


def test_wrong_architecture_and_incomplete_state_cannot_be_loaded(tmp_path, authentic_video):
    bad = saved_payload(tmp_path, {"config": {"backbone": "cnn", "emb_dim": 512}, "model": {}})
    with pytest.raises(DeploymentError, match="ARCHITECTURE_MISMATCH"):
        encoders.load_video_encoder(bad)
    wrong_head = dict(authentic_video._model.state_dict())
    wrong_head["head.1.weight"] = wrong_head["head.1.weight"][:28]
    bad = saved_payload(tmp_path, {"config": {"backbone": "resnet18", "emb_dim": 512}, "model": wrong_head})
    with pytest.raises(DeploymentError, match="ARCHITECTURE_MISMATCH"):
        encoders.load_video_encoder(bad)
    bad = saved_payload(tmp_path, {"config": {"backbone": "resnet18", "emb_dim": 512}, "model": {}})
    with pytest.raises(DeploymentError, match="ARCHITECTURE_MISMATCH"):
        encoders.load_video_encoder(bad)

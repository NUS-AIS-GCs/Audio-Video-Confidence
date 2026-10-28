"""Authenticated CPU video predictor for internal parity studies only."""
import json
from pathlib import Path

import numpy as np
import torch

from deployment.encoders import dependency_issues
from deployment.runtime import PINNED_VERSIONS
from fusion.contracts import ContractError
from fusion.encoder_interfaces import Artifact, EncoderSpec, VIDEO_PREPROCESSING, DEPENDENCIES, load_encoder
from vision.infer import VisionPredictor


WEIGHTS_SHA256 = "6f1c1fc868efbd397d2ee1ca20c37800bd483911de6ef0727ac0f12871b8b2ec"
PREPROCESSING = {"fps": 1, "clip_seconds": 10, "img_size": [224, 224], "channels": 3,
                 "interpolation": "cv2.INTER_LINEAR", "color_space": "RGB", "scale": "1.0 / 255.0",
                 "mean": [.485, .456, .406], "std": [.229, .224, .225]}


def verify_bundle(card_artifact):
    if type(card_artifact) is not Artifact or Path(card_artifact.path).name != "model_card.json":
        raise ContractError("acquisition-bound model_card.json Artifact required")
    card_artifact.verify()
    root = Path(card_artifact.path).resolve().parent
    card = json.loads(Path(card_artifact.path).read_text(encoding="utf-8"))
    if (card.get("run_id") != "resnet18_v1" or card.get("backbone") != "resnet18"
            or card.get("emb_dim") != 512 or card.get("n_classes") != 29
            or card.get("preprocessing") != PREPROCESSING
            or type(card.get("sha256")) is not dict
            or set(card["sha256"]) != {"weights.pt", "class_names.json"}
            or card["sha256"]["weights.pt"] != WEIGHTS_SHA256):
        raise ContractError("unsupported authenticated video bundle configuration")
    for name, expected in card["sha256"].items():
        payload = (root / name).resolve()
        if payload.parent != root:
            raise ContractError("bundle payload escapes approved directory")
        Artifact(str(payload), expected).verify()
    names = json.loads((root / "class_names.json").read_text(encoding="utf-8"))
    if (type(names) is not list or len(names) != 29 or names[-1] != "background"
            or any(type(n) is not str or not n for n in names)
            or len(set(names)) != 29 or names[:-1] != sorted(names[:-1])):
        raise ContractError("invalid video class vocabulary")
    card_artifact.verify()
    return root, card, names


class VerifiedVideoPredictor(VisionPredictor):
    """Reuse producer frame numerics with the Phase 6 restricted strict loader.

    Compressed-media decoding stays an explicit offline operation. Loading this
    class cannot enable uploads or change the historical TRAIN parity gate.
    """
    def __init__(self, card_artifact):
        if dependency_issues("vision"):
            raise ContractError("candidate CPU vision runtime unavailable")
        self._card_artifact = card_artifact
        self.dir, self.card, self.class_names = verify_bundle(card_artifact)
        self.cfg = {"backbone": "resnet18", "emb_dim": 512}
        spec = EncoderSpec("vision", "resnet18_v1",
            (("trained", Artifact(str(self.dir / "weights.pt"), WEIGHTS_SHA256)),),
            json.dumps({"trained_config": self.cfg}, sort_keys=True, separators=(",", ":")),
            VIDEO_PREPROCESSING, tuple((n, PINNED_VERSIONS[n]) for n in DEPENDENCIES["vision"]))
        self.encoder = load_encoder(spec, device="cpu")
        self.model, self.device = self.encoder._model, torch.device("cpu")

    def predict_frames(self, raw_frames):
        if (type(raw_frames) is not np.ndarray or raw_frames.dtype != np.uint8
                or raw_frames.shape != (10, 224, 224, 3)):
            raise ContractError("canonical RGB uint8 (10,224,224,3) frames required")
        verify_bundle(self._card_artifact)
        self.encoder.require_ready()
        if self.model is not self.encoder._model:
            raise ContractError("verified video model was replaced")
        self.model.eval()
        with torch.no_grad(), torch.autocast("cpu", enabled=False):
            result = super().predict_frames(raw_frames)
        self.encoder.require_ready()
        verify_bundle(self._card_artifact)
        if (result.embeddings.shape != (10, 512) or result.embeddings.dtype != np.float32
                or result.probs.shape != (10, 29) or result.probs.dtype != np.float32
                or not np.isfinite(result.embeddings).all() or not np.isfinite(result.probs).all()):
            raise ContractError("invalid genuine video output")
        return result

    def predict_video(self, path):
        raise ContractError("compressed-media serving disabled; use audited offline frames")

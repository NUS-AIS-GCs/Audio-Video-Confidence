"""Genuine, restricted TRAIN inference. This module grants no public release authority.

Cases are fixed before observing scores. A swapped case uses audio from the next
permitted TRAIN source and video from the selected source, on intended ten-bin
timelines. Achieved source PTS remain diagnostics, not general clock approval.
"""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from threading import Lock
import time
from typing import Literal

import numpy as np
import torch

from deployment.auth import login_owner
from deployment.internal_backend import InternalTrainBackend, response_for_review
from deployment.train_manifest import PERMITTED_IDS
from fusion.contracts import ContractError
from fusion.inference import CANDIDATES


@dataclass(frozen=True)
class TrainCase:
    key: str
    audio_id: str
    video_id: str
    kind: str


# Prespecified diagnostics, not an accuracy benchmark or threshold calibration set.
DEMO_CASES = tuple(
    TrainCase(f"{kind}-{i+1}", PERMITTED_IDS[(i+offset) % len(PERMITTED_IDS)], sid, kind)
    for kind, offset in (("original", 0), ("swapped", 1))
    for i, sid in enumerate(PERMITTED_IDS)
)


@dataclass(frozen=True)
class PairingPrediction:
    case_id: str
    audio_source: str
    video_source: str
    score: float
    logit: float
    processing_seconds: float
    threshold: float = CANDIDATES["E3"].threshold
    model_sha256: str = CANDIDATES["E3"].model_sha256
    checkpoint_sha256: str = CANDIDATES["E3"].checkpoint_sha256
    model_version: str = "E3 / frozen concatenation + MLP"
    status: Literal["EXPERIMENTAL_INTERNAL_TRAIN_ONLY"] = "EXPERIMENTAL_INTERNAL_TRAIN_ONLY"
    schema: str = "aval.train-pairing.v1"
    production_predictions_enabled: Literal[False] = False

    def __post_init__(self):
        case = next((c for c in DEMO_CASES if c.key == self.case_id), None)
        binding = CANDIDATES["E3"]
        if (case is None or (self.audio_source, self.video_source) != (case.audio_id, case.video_id)
                or self.threshold != binding.threshold or self.model_sha256 != binding.model_sha256
                or self.checkpoint_sha256 != binding.checkpoint_sha256
                or self.model_version != "E3 / frozen concatenation + MLP"
                or self.schema != "aval.train-pairing.v1"
                or self.status != "EXPERIMENTAL_INTERNAL_TRAIN_ONLY"
                or self.production_predictions_enabled is not False):
            raise ContractError("unapproved prediction identity or scope")
        if (any(type(v) is not float or not math.isfinite(v)
                for v in (self.score, self.logit, self.processing_seconds))
                or not 0 <= self.score <= 1 or self.processing_seconds < 0
                or self.score != float(np.exp(-np.logaddexp(0., -np.float64(self.logit))))):
            raise ContractError("invalid correspondence score")

    @property
    def classification(self) -> str:
        return ("Original-like pairing" if self.score >= self.threshold
                else "Audio-substituted-like pairing")

    def to_dict(self) -> dict:
        return {**asdict(self), "classification": self.classification,
                "score_semantics": "uncalibrated sigmoid correspondence score",
                "temporal_status": "TRAIN_DIAGNOSTIC_ONLY; general shared-clock approval pending",
                "event_predictions": None, "temporal_predictions": None}


def internal_error(code: str) -> dict:
    return {"schema": "aval.train-pairing.v1", "status": "REJECTED", "predictions": None,
            "error_code": code, "production_predictions_enabled": False}


class InternalPredictionAdapter:
    """One initialized authentic backend per worker; no media upload or client path API."""

    def __init__(self, backend: InternalTrainBackend):
        if type(backend) is not InternalTrainBackend:
            raise ContractError("authentic internal backend required")
        self.backend = backend
        self._lock = Lock()
        self.upload_store = None

    def submit(self, case_id, request) -> dict:
        raw = getattr(request, "request", None)
        owner = login_owner(raw)
        if owner is None:
            return internal_error("INTERNAL_AUTHENTICATION_REQUIRED")
        result = self.predict_case(case_id)
        return result if login_owner(raw) == owner else internal_error("INTERNAL_AUTHENTICATION_REQUIRED")

    def predict_case(self, case_id) -> dict:
        """Offline runner seam; only submit() is registered as a UI callback."""
        case = next((c for c in DEMO_CASES if type(case_id) is str and c.key == case_id), None)
        if case is None:
            return internal_error("UNAPPROVED_TRAIN_CASE")
        if not self._lock.acquire(blocking=False):
            return internal_error("RESOURCE_BUSY")
        started = time.perf_counter()
        try:
            features = {}
            for sid in dict.fromkeys((case.audio_id, case.video_id)):
                source = self.backend.sources[sid]
                source.verify()
                from pathlib import Path
                path = Path(source.path)
                with path.open("rb") as stream:
                    content = stream.read(self.backend.config.max_upload_bytes + 1)
                result = self.backend.predict_internal(sid, content, path.name, include_a2=False)
                visible = response_for_review(result)
                if visible["status"] != "EXPERIMENTAL_INTERNAL_TRAIN_ONLY":
                    return internal_error("TRAIN_INFERENCE_UNAVAILABLE")
                features[sid] = result["_audit_features"]
            a, v = features[case.audio_id][0], features[case.video_id][1]
            if any(type(x) is not np.ndarray or x.dtype != np.float32
                   or x.shape != (1, 10, dim) or not np.isfinite(x).all()
                   for x, dim in ((a, 256), (v, 512))):
                raise ContractError("invalid ten-bin features")
            self.backend._integrity()
            with torch.no_grad(), torch.autocast("cpu", enabled=False):
                logits = self.backend.study.adapters["E3"]._model(
                    torch.from_numpy(a), torch.from_numpy(v), torch.ones(1, 10, dtype=torch.bool)).logits
            if logits.shape != (1, 1) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise ContractError("invalid frozen output")
            z = float(logits[0, 0])
            score = float(np.exp(-np.logaddexp(0., -np.float64(z))))
            self.backend._integrity()
            return PairingPrediction(case.key, case.audio_id, case.video_id, score, z,
                                     float(time.perf_counter() - started)).to_dict()
        except Exception:
            return internal_error("INTERNAL_INFERENCE_FAILED")
        finally:
            # Feature tensors are request-local; never retained in a UI, cache or session.
            self._lock.release()

    def submit_owned_upload(self, handle, request) -> dict:
        """Existing Stage 2H lease -> exact permitted TRAIN bytes -> authentic E3.

        This separate internal path never grants arbitrary upload compatibility.
        The owner and original filename come from server state, not FileData.
        """
        from deployment.uploads import UploadError
        raw = getattr(request, "request", None)
        owner = login_owner(raw)
        if (owner is None or self.upload_store is None
                or raw.scope.get("aval.upload_guard") is not self.upload_store):
            return internal_error("INTERNAL_AUTHENTICATION_REQUIRED")
        if not handle:
            return internal_error("UPLOAD_REQUIRED")
        try:
            with self.upload_store.lease(handle, owner) as (path, filename):
                with path.open("rb") as stream:
                    content = stream.read(self.backend.config.max_upload_bytes + 1)
                result = self.predict_uploaded_train(content, filename)
                return result if login_owner(raw) == owner else internal_error("INTERNAL_AUTHENTICATION_REQUIRED")
        except UploadError:
            return internal_error("UPLOAD_NOT_FOUND")
        except Exception:
            return internal_error("INTERNAL_INFERENCE_FAILED")

    def predict_uploaded_train(self, content, filename):
        """Offline test seam; the UI registers only the authenticated lease method."""
        if type(content) is not bytes or not content:
            return internal_error("UNSUPPORTED_UPLOAD")
        if len(content) > self.backend.config.max_upload_bytes:
            return internal_error("UPLOAD_TOO_LARGE")
        if (type(filename) is not str or not filename or Path(filename).name != filename
                or any(c in filename for c in ('/', '\\', '\x00'))):
            return internal_error("UNSUPPORTED_FILENAME")
        digest = hashlib.sha256(content).hexdigest()
        case = next((c for c in DEMO_CASES if c.kind == "original"
                     and self.backend.sources[c.video_id].sha256 == digest
                     and Path(self.backend.sources[c.video_id].path).name == filename), None)
        if case is None:
            return internal_error("UNAPPROVED_TRAIN_SOURCE")
        if not self._lock.acquire(blocking=False):
            return internal_error("RESOURCE_BUSY")
        started = time.perf_counter()
        try:
            result = response_for_review(self.backend.predict_internal(
                case.video_id, content, filename, include_a2=False))
            if result["status"] != "EXPERIMENTAL_INTERNAL_TRAIN_ONLY":
                return internal_error("TRAIN_INFERENCE_UNAVAILABLE")
            p = result["predictions"]["E3"]
            return PairingPrediction(case.key, case.audio_id, case.video_id, p["score"], p["logit"],
                                     float(time.perf_counter() - started)).to_dict()
        except Exception:
            return internal_error("INTERNAL_INFERENCE_FAILED")
        finally:
            self._lock.release()

"""Separate authenticated TRAIN-only demo entry point. Never the Docker default.

Requires read-only /models and /inputs mounts from the existing permitted manifest.
No arbitrary uploads can reach the experimental backend.
"""
import os
from pathlib import Path
import shutil

from deployment.runtime import DeploymentConfig


def main():
    config = DeploymentConfig.from_environment()
    if not config.authenticated_uploads_enabled:
        raise RuntimeError("AUTHENTICATION_CONFIGURATION_REQUIRED")
    digest = os.environ.get("AVAL_INTERNAL_MANIFEST_SHA256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError("INTERNAL_MANIFEST_BINDING_REQUIRED")
    from deployment.internal_backend import InternalTrainBackend
    from deployment.prediction_adapter import InternalPredictionAdapter
    from deployment.gradio_app import main as serve
    import tempfile
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    try:
        with tempfile.TemporaryDirectory(prefix="aval-internal-derived-") as derived:
            try:
                backend = InternalTrainBackend(config.model_root,
                    Path(os.environ.get("AVAL_INTERNAL_INPUT_ROOT", "/inputs")), digest,
                    shutil.which("ffmpeg"), shutil.which("ffprobe"), Path(derived))
            except Exception:
                raise RuntimeError("INTERNAL_ARTIFACT_VALIDATION_FAILED") from None
            serve(internal_adapter=InternalPredictionAdapter(backend))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

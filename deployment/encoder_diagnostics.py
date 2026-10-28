"""Stage 1 diagnostics only: no decoder, tolerance, fusion scoring or readiness switch."""
import argparse
import json
from pathlib import Path

import numpy as np

from deployment.encoders import dependency_issues, load_audio_encoder, load_video_encoder
from deployment.runtime import DeploymentConfig, DeploymentError, startup_readiness


def embedding_differences(actual, reference):
    """Descriptive metrics for already-authenticated TRAIN rows; not a parity gate.

    This function cannot authenticate identity or split. Callers must do that
    upstream, and retain the existing TrainParityFusion admission checks.
    """
    a, r = np.asarray(actual), np.asarray(reference)
    if (a.shape != r.shape or a.ndim not in (2, 3) or a.shape[-2] != 10
            or a.shape[-1] not in (256, 512) or a.size == 0
            or a.dtype.kind != "f" or r.dtype.kind != "f"
            or not np.isfinite(a).all() or not np.isfinite(r).all()):
        raise ValueError("expected matching finite floating (...,10,256|512) embeddings")
    difference = a.astype(np.float64) - r.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        a16, r16 = a.astype(np.float16), r.astype(np.float16)
    if not np.isfinite(a16).all() or not np.isfinite(r16).all():
        raise ValueError("embeddings exceed the historical float16 export range")
    return {
        "max_absolute_error": float(np.abs(difference).max()),
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "float16_array_equal": bool(np.array_equal(a16, r16)),
        "float16_differing_values": int(np.count_nonzero(a16 != r16)),
        "numerical_tolerance_selected": False,
    }


def diagnose(config, *, load=False):
    """Probe each modality independently; returned status never enables predictions."""
    results = {}
    for modality, loader in (("audio", load_audio_encoder), ("vision", load_video_encoder)):
        issues = dependency_issues(modality)
        result = {"status": "BLOCKED", "dependencies": issues,
                  "encoder_loaded": False, "raw_train_parity": "NOT_VERIFIED"}
        if load and not issues:
            try:
                encoder = loader(config)
                result.update(status="LOADED_PARITY_UNVERIFIED", encoder_loaded=True,
                              provenance={"run_identity": encoder.provenance.run_identity,
                                          "artifacts": list(encoder.provenance.artifacts)})
            except DeploymentError as exc:
                result["error_code"] = exc.code
        results[modality] = result
    return {"schema": "aval.encoder-stage1.v1", "status": "MODEL_NOT_READY",
            "predictions": None, "inference_enabled": False, "encoders": results,
            "deployment": startup_readiness(config).to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load", action="store_true", help="Attempt strict CPU loads; never enables predictions")
    parser.add_argument("--output", type=Path, help="Optional local audit JSON (not historical evidence)")
    args = parser.parse_args()
    try:
        report = diagnose(DeploymentConfig.from_environment(), load=args.load)
    except DeploymentError as exc:
        report = {"status": "MODEL_NOT_READY", "predictions": None,
                  "inference_enabled": False, "error_code": exc.code}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 2  # Stage 1 cannot certify either raw TRAIN parity or production readiness.


if __name__ == "__main__":
    raise SystemExit(main())

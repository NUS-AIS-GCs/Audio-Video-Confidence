"""Offline authentic TRAIN diagnostic runner. Never tunes or opens a release gate."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
from unittest.mock import patch

import torch

from deployment.internal_backend import InternalTrainBackend
from deployment.prediction_adapter import DEMO_CASES, InternalPredictionAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    loads = []
    original_load = torch.load

    def restricted(*a, **kw):
        if kw.get("weights_only") is not True:
            raise RuntimeError("unrestricted checkpoint load")
        loads.append(str(a[0]))
        return original_load(*a, **kw)

    # Verify byte preservation independently of the backend's per-request guards.
    before = {str(p.relative_to(args.models)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in args.models.rglob("*") if p.is_file()}
    with patch("torch.load", side_effect=restricted):
        backend = InternalTrainBackend(args.models, args.inputs, args.manifest_sha256,
            shutil.which("ffmpeg"), shutil.which("ffprobe"), args.output / "derived")
        initialization = time.perf_counter() - start
        loaded = len(loads)
        adapter = InternalPredictionAdapter(backend)
        rows = []
        for case in DEMO_CASES:
            response = adapter.predict_case(case.key)
            if response["status"] != "EXPERIMENTAL_INTERNAL_TRAIN_ONLY":
                raise RuntimeError("genuine case rejected: " + response["error_code"])
            repeat = adapter.predict_case(case.key)
            same = all(response[k] == repeat.get(k) for k in
                       ("score", "logit", "threshold", "classification", "model_sha256"))
            if not same or len(loads) != loaded:
                raise RuntimeError("prediction changed or model reloaded")
            rows.append({"kind": case.kind, "response": response, "repeat_identical": same,
                         "repeat_processing_seconds": repeat["processing_seconds"],
                         "threshold_distance": abs(response["score"] - response["threshold"])})
            print(case.key, response["classification"], response["score"], flush=True)
    backend._integrity()
    after = {str(p.relative_to(args.models)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in args.models.rglob("*") if p.is_file()}
    if before != after:
        raise RuntimeError("model artifacts changed")
    report = {"schema": "aval.prediction-demo-validation.v1", "cases": rows,
        "initialization_seconds": initialization, "checkpoint_load_count": len(loads),
        "load_counts": backend.load_counts, "artifacts_preserved": before == after,
        "artifact_count": len(before), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "smallest_observed_threshold_distance": min(r["threshold_distance"] for r in rows),
        "near_threshold_coverage": "diagnostic distances only; no independent near-threshold release study",
        "numerical_acceptance": "INACTIVE", "historical_parity": "UNRESOLVED",
        "general_shared_clock_approval": False, "production_predictions_enabled": False,
        "public_verdict": "VALIDATION BLOCKED", "readiness": backend.readiness()}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

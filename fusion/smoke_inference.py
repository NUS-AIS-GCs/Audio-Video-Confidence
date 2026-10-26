"""Frozen E3/A2 inference on four verified TRAIN pairs; no metrics or training.

Run: python -B -m fusion.smoke_inference --output _local/phase6_smoke.json
Only TRAIN arrays are loaded. Labels stay in audit diagnostics, never features.
"""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .baselines import score_probabilities
from .contracts import ContractError
from .inference import FrozenFusionInference, _sha, load_verified_train_exports
from .inference_contracts import FeatureBatch

DEFAULTS = {
    "E3": "_local/fusion/phase4_improvement/study_20261024_seed42_01/E3",
    "A2": "_local/fusion/phase5_ablations/study_20261025_seed42_01/A2",
}


def run_smoke(audio_root, vision_root, manifest_path, checkpoint_directories):
    result = {"schema": "gc2-phase6-feature-smoke-v1", "split": "train",
              "raw_encoder_execution": "BLOCKED_MISSING_ARTIFACTS_AND_DEPENDENCIES",
              "raw_export_parity": "UNVERIFIED", "metrics_computed": False, "models": {}}
    for candidate, directory in checkpoint_directories.items():
        adapter = FrozenFusionInference(directory, candidate=candidate)
        metadata = adapter.metadata
        if _sha(manifest_path) != metadata.data[0].manifest_sha256:
            raise ContractError("TRAIN manifest differs from frozen checkpoint input")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if (manifest["config"]["split"] != "train"
                or manifest["artifact_fingerprint"] != metadata.data[0].artifact_fingerprint):
            raise ContractError("TRAIN manifest provenance mismatch")
        (audio, vision), provenance = load_verified_train_exports(audio_root, vision_root, metadata)
        if audio.class_names != vision.class_names:
            raise ContractError("probability vocabularies differ")
        pairs = manifest["primary_pairs"]
        selected = ([p for p in pairs if p["original_pair_target"] == 1][:2]
                    + [p for p in pairs if p["original_pair_target"] == 0][:2])
        if len(selected) != 4:
            raise ContractError("smoke requires two original and two substituted TRAIN pairs")
        ai, vi = [], []
        for pair in selected:
            for side, bundle, indices in (("audio", audio, ai), ("visual", vision, vi)):
                ref = pair[side]
                index = ref["row_index"]
                if (ref["split"] != "train" or type(index) is not int or index < 0
                        or index >= len(bundle.splits["train"].records)
                        or asdict(bundle.splits["train"].records[index]) != ref["record"]):
                    raise ContractError("TRAIN source reference mismatch")
                indices.append(index)
        a, v = audio.splits["train"], vision.splits["train"]
        mask = torch.ones((4, 10), dtype=torch.bool)
        # Historical exports contain no decode validity; use availability only.
        features = FeatureBatch(torch.from_numpy(a.embeddings[ai].astype(np.float32)),
            torch.from_numpy(v.embeddings[vi].astype(np.float32)), mask, mask.clone(), provenance,
            diagnostics={"validity": "export availability only; decode success unknown"})
        first = adapter.predict(features)
        second = adapter.predict(features)
        if first != second:
            raise ContractError("repeated frozen inference is not deterministic")
        # Independent direct forward verifies adapter numerical compatibility.
        with torch.no_grad():
            direct = adapter._model(features.audio, features.visual, mask).logits[:, 0].numpy()
        if not np.array_equal(direct, [p.logit for p in first.predictions]):
            raise ContractError("adapter differs from canonical Phase 4/5 forward")
        # TRAIN-only historical comparison, allowing normal batch/thread GEMM
        # rounding differences. This is an integration check, not evaluation.
        with (Path(directory) / "train_primary_scores.csv").open(newline="", encoding="utf-8") as stream:
            historical = {(r["audio_sample_id"], r["visual_sample_id"]): float(r["logit"])
                          for r in csv.DictReader(stream) if r["split"] == "train" and r["partition"] == "primary"}
        reference = np.array([historical[tuple(p["identity"][1:])] for p in selected])
        if not np.allclose(direct, reference, rtol=1e-5, atol=1e-5):
            raise ContractError("frozen TRAIN logits disagree with historical Phase 4/5 evidence")
        # Same probabilities and all-ten-bin aggregation as Phase 3; no threshold transfer.
        jsd = score_probabilities(a.probabilities[ai], v.probabilities[vi]).pair_scores["jensen_shannon"]
        result["models"][candidate] = {"result": first.to_dict(), "repeated_exactly": True,
            "canonical_forward_exact": True, "phase3_jsd_similarity": jsd.tolist(),
            "historical_train_logit_max_abs_difference": float(np.abs(direct - reference).max()),
            "historical_comparison_tolerance": {"rtol": 1e-5, "atol": 1e-5},
            "pair_audit": [{"identity": p["identity"],
                            "provenance_target": p["original_pair_target"]} for p in selected]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-root", default="_local/embeddings/audio_final")
    parser.add_argument("--vision-root", default="_local/embeddings/vision_final/resnet18_v1")
    parser.add_argument("--manifest", default="_local/fusion/task28_real/train_pairs.json")
    parser.add_argument("--e3", default=DEFAULTS["E3"])
    parser.add_argument("--a2", default=DEFAULTS["A2"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_smoke(args.audio_root, args.vision_root, args.manifest, {"E3": args.e3, "A2": args.a2})
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        # Do not overwrite historical evidence or previously saved smoke results.
        root = Path("_local").resolve()
        if root not in args.output.resolve().parents:
            raise ContractError("smoke report output must be a fresh ignored _local path")
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

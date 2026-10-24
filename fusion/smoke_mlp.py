"""Untrained TRAIN-only integrity smoke check. No metrics, W&B, or file writes."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import default_collate

from .artifact_integration import (
    OFFICIAL_FILES, _check_hashes, _export_paths, _file_entry, _inventory,
    _json, _load_exports, _read_json, _verify_official,
)
from .contracts import ContractError, OFFICIAL_COUNTS
from .embedding_data import EmbeddingDataset
from .mlp_model import LateFusionMLP, MLPConfig


def run_smoke(audio_root, vision_root, official_root, integration_root, *, limit=8,
              expected_counts=OFFICIAL_COUNTS):
    """Check complete original evidence before reading only the TRAIN manifest.

    All-split arrays/metadata and opaque manifest hashes are integrity context
    required by Phase 2. Only TRAIN embeddings enter the untrained model. The
    deterministic subset takes positives AND easy negatives from canonical order.
    """
    if type(limit) is not int or limit < 2:
        raise ContractError("smoke limit must be an integer >= 2")
    roots = {"audio": Path(audio_root).resolve(), "vision": Path(vision_root).resolve()}
    official_root, integration_root = Path(official_root).resolve(), Path(integration_root).resolve()
    report_path = integration_root / "validation.json"
    report_bytes = report_path.read_bytes()
    report = _read_json(report_bytes.decode("utf-8"))
    if (report.get("schema") != "gc2-artifact-integration-v1" or report.get("status") != "COMPLETE"
            or report.get("binary_validation") != "VALIDATED_LOCAL_FILES"
            or report.get("official_metadata") != "MATCHES_SUPPLIED_OFFICIAL_REFERENCE_FILES"):
        raise ContractError("complete Phase 2 evidence is required")
    paths, missing = _export_paths(roots)
    if missing:
        raise ContractError("required producer files are missing")
    paths.update({f"official/{name}": official_root / name for name in OFFICIAL_FILES})
    inventory = _inventory(paths)
    saved = {entry["role"]: entry for entry in report["artifacts"]}
    if len(saved) != len(report["artifacts"]) or set(saved) != {e["role"] for e in inventory}:
        raise ContractError("Phase 2 artifact inventory coverage mismatch")
    _check_hashes(inventory, {role: entry["sha256"] for role, entry in saved.items()})
    if any(e["size_bytes"] != saved[e["role"]]["size_bytes"] for e in inventory):
        raise ContractError("Phase 2 artifact size mismatch")
    context = hashlib.sha256(_json(report["artifacts"]).encode()).hexdigest()
    if context != report["artifact_inventory_fingerprint"]:
        raise ContractError("Phase 2 artifact fingerprint mismatch")
    verified = list(inventory)
    for split in ("train", "val", "test"):
        name = f"{split}_pairs.json"
        entry = _file_entry(f"manifest/{name}", integration_root / name)
        if any(entry[key] != report["outputs"][name][key] for key in ("sha256", "size_bytes")):
            raise ContractError("original manifest hash/size mismatch")
        paths[entry["role"]] = integration_root / name
        verified.append(entry)
    paths["evidence/validation.json"] = report_path
    before = _inventory(paths)
    # Close the gap between initial evidence checks and taking the final
    # before/after inventory: all bytes must still match the verified originals.
    _check_hashes(before, {**{e["role"]: e["sha256"] for e in verified},
                           "evidence/validation.json": hashlib.sha256(report_bytes).hexdigest()})
    with _load_exports(roots, expected_counts) as (audio, vision):
        _verify_official(audio, vision, official_root, expected_counts)
        data = EmbeddingDataset.from_json(
            (integration_root / "train_pairs.json").read_text(encoding="utf-8"), audio, vision,
            split="train", artifact_fingerprint=context)
        n = expected_counts["train"]
        if (len(data) != 2 * n or int(data.targets.sum()) != n
                or len(report["manifests"]["train"]["metadata_leakage"]) != 2):
            raise ContractError("expected full original/easy primary distribution")
        positives = [i for i, target in enumerate(data.targets[:, 0]) if target == 1]
        negatives = [i for i, target in enumerate(data.targets[:, 0]) if target == 0]
        chosen = sorted(positives[:limit // 2] + negatives[:limit - limit // 2])
        batch, targets = default_collate([data[i] for i in chosen])
        mismatches = 0
        for row, index in enumerate(chosen):
            pair = data.pairs[index]
            for side, bundle, key in (("audio", audio, "audio"), ("visual", vision, "visual")):
                source = getattr(pair, side)
                original = bundle.splits["train"]
                mismatches += int(original.records[source.row_index] != source.record)
                mismatches += int(not np.array_equal(batch[key][row].numpy(), original.embeddings[source.row_index]))
        if mismatches:
            raise ContractError("smoke extraction changed original row references or values")
        model = LateFusionMLP(MLPConfig()).eval()
        with torch.inference_mode():
            output = model(**batch, return_probabilities=True)
        result = {"schema": "gc2-fusion-untrained-smoke-v1", "split": "train",
                  "trained": False, "metrics_computed": False, "wandb_run_created": False,
                  "full_context_structural_validation_only": True,
                  "producer_files_verified": len(inventory), "manifest_files_verified": 3,
                  "row_reference_mismatches": mismatches, "subset_size": len(chosen),
                  "positive_count": int(targets.sum()), "negative_count": int((targets == 0).sum()),
                  "audio_shape": list(batch["audio"].shape), "visual_shape": list(batch["visual"].shape),
                  "logit_shape": list(output.logits.shape),
                  "finite_logits": bool(torch.isfinite(output.logits).all()),
                  "finite_probabilities": bool(torch.isfinite(output.probabilities).all()),
                  "parameter_count": sum(p.numel() for p in model.parameters()),
                  "manifest_sha256": data.provenance.manifest_sha256,
                  "artifact_fingerprint": context,
                  "source_array_hashes": data.provenance.source_array_hashes,
                  # Only safe run identifiers; producer notes can contain old
                  # held-out measurements and must never become smoke results.
                  "source_run_ids": {
                      b.modality: [match[1] for line in b.provenance["source"].splitlines()
                                   if (match := re.fullmatch(r"(?:audio|vision) run: ([A-Za-z0-9_-]+)", line))]
                      for b in (audio, vision)},
                  "torch_version": torch.__version__, "numpy_version": np.__version__,
                  "cuda_available": torch.cuda.is_available()}
    if before != _inventory(paths):
        raise ContractError("producer artifacts, evidence or manifests changed during smoke check")
    result["producer_inputs_and_manifests_unchanged"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audio-root", "vision-root", "official-root", "integration-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    print(_json(run_smoke(args.audio_root, args.vision_root, args.official_root,
                          args.integration_root, limit=args.limit)), end="")


if __name__ == "__main__":
    main()

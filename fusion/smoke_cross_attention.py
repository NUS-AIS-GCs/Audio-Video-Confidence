"""Untrained TRAIN-only attention smoke. No metrics, thresholds, W&B or writes."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifact_integration import _json
from .contracts import OFFICIAL_COUNTS
from .cross_attention_model import BidirectionalCrossAttention, CrossAttentionConfig
from .smoke_mlp import _run_smoke


def run_smoke(audio_root, vision_root, official_root, integration_root, *, limit=8,
              expected_counts=OFFICIAL_COUNTS, config=CrossAttentionConfig()):
    return _run_smoke(audio_root, vision_root, official_root, integration_root,
                      limit=limit, expected_counts=expected_counts,
                      model_factory=lambda: BidirectionalCrossAttention(config), inspect_attention=True)


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

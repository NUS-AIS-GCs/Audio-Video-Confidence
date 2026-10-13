"""Export per-second visual embeddings for multimodal fusion (team contract: 10 x D per clip).

Writes to DRIVE_ROOT/embeddings/<run_id>:
  {split}_emb.npy (N, 10, 512) float16
  {split}_ids.csv (same row order)
  SOURCE.txt

Run: python -m vision.export_embeddings --run-id <id>
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from vision import config as C
from vision.dataset import AVEVisual
from vision.evaluate import load_model, pick_device, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=None, help="Custom export directory (default: DRIVE_ROOT/embeddings/<run_id>)")
    a = ap.parse_args()

    device = pick_device()
    model = load_model(C.ARTIFACT_DIR / a.run_id, device)
    out_dir = a.out_dir if a.out_dir is not None else (C.DRIVE_ROOT / "embeddings" / a.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in C.SPLITS:
        loader = DataLoader(AVEVisual(split), batch_size=a.batch, num_workers=a.workers)
        *_, E = predict(model, loader, device)
        np.save(out_dir / f"{split}_emb.npy", E.astype(np.float16))
        shutil.copy(C.FEATURE_DIR / f"{split}_ids.csv", out_dir / f"{split}_ids.csv")
        print(f"{split}: {E.shape}")

    (out_dir / "SOURCE.txt").write_text(f"vision baseline run: {a.run_id}\n")
    print(f"Exported embeddings saved to {out_dir}")


if __name__ == "__main__":
    main()

"""Extract and cache frozen 512-D CLIP ViT-B/16 visual features for AVE dataset.

Caches per-second visual embeddings for 'train', 'val', and 'test' splits to:
  _local/drive/cache/clip_features/{split}_clip_emb.npy

Usage:
  python -m vision.clip_extract [--splits test val train] [--batch 16]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision import config as C
from vision.clip_model import CLIPZeroShotModel
from vision.dataset import AVEVisual
from vision.evaluate import pick_device

CACHE_DIR = C.DRIVE_ROOT / "cache" / "clip_features"


def extract_split(
    model: CLIPZeroShotModel,
    split: str,
    batch_size: int,
    device: torch.device,
    cache_dir: Path,
    overwrite: bool = False
) -> Path:
    out_file = cache_dir / f"{split}_clip_emb.npy"
    if out_file.exists() and not overwrite:
        print(f"[{split}] Already exists at {out_file}, skipping (use --overwrite to re-extract).", flush=True)
        return out_file

    print(f"\n[{split}] Extracting CLIP ViT-B/16 features on {device}...", flush=True)
    ds = AVEVisual(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    E = []
    t0 = time.time()
    with torch.no_grad():
        for step, (x, y, yc) in enumerate(loader):
            # x: (B, 10, 3, 224, 224)
            emb = model.embed(x.to(device))  # (B, 10, 512)
            E.append(emb.cpu().to(torch.float16))

            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                elapsed = time.time() - t0
                clips_done = min((step + 1) * batch_size, len(ds))
                rate = clips_done / elapsed if elapsed > 0 else 0
                print(f"  [{clips_done}/{len(ds)} clips] in {elapsed:.1f}s ({rate:.1f} clips/s)", flush=True)

    E_cat = torch.cat(E, dim=0).numpy()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_file, E_cat)
    print(f"[{split}] Saved {E_cat.shape} {E_cat.dtype} to {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)", flush=True)
    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    print(f"Using compute device: {device}", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing CLIP ViT-B/16...", flush=True)
    model = CLIPZeroShotModel(device=device).to(device)
    model.eval()

    for s in args.splits:
        extract_split(model, s, args.batch, device, CACHE_DIR, overwrite=args.overwrite)

    print("\nAll requested splits successfully extracted and cached!", flush=True)


if __name__ == "__main__":
    main()

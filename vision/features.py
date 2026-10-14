"""Prepare visual features and label arrays for each split, cached in FEATURE_DIR.

Output per split in FEATURE_DIR:
  {split}_y.npy      (N, 10)          int64     per-second label, 28 = background
  {split}_yclip.npy  (N,)             int64     clip category
  {split}_ids.csv    row order: sample_id (= clip_id__label), clip_id, category, start, end
  norm_stats.json    class names and normalization metadata
  {split}_X.npy      (N, 10, 224, 224, 3) uint8 (optional, built if --pack-x is specified)

Run: python -m vision.features [--pack-x]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from vision import config as C
from vision.data_ave import class_names, load_split, second_labels


def build_split_metadata(split: str, pack_x: bool = False) -> None:
    df = load_split(split)
    # Check if frames are available in FRAME_DIR or if videos are ready
    df["has_frames"] = df["clip_id"].map(lambda c: (C.FRAME_DIR / f"{c}.npy").exists())
    missing_count = (~df["has_frames"]).sum()
    if missing_count > 0:
        print(f"{split}: {missing_count} clips don't have extracted frames in {C.FRAME_DIR}.")
        print("Run `python -m vision.extract_frames` to pre-extract frames if needed.")

    # Always generate the deterministic labels and ids
    y = np.stack([second_labels(r.start, r.end, r.label) for r in df.itertuples()])
    yclip = df["label"].to_numpy(np.int64)

    np.save(C.FEATURE_DIR / f"{split}_y.npy", y)
    np.save(C.FEATURE_DIR / f"{split}_yclip.npy", yclip)
    df[["sample_id", "clip_id", "category", "start", "end"]].to_csv(
        C.FEATURE_DIR / f"{split}_ids.csv", index=False
    )
    print(f"{split}: y {y.shape}, yclip {yclip.shape}, ids {len(df)}")

    if pack_x:
        print(f"Packing monolithic {split}_X.npy array...")
        X = np.zeros((len(df), C.CLIP_SECONDS, C.IMG_SIZE, C.IMG_SIZE, C.N_CHANNELS), dtype=np.uint8)
        for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"Pack frames {split}")):
            fpath = C.FRAME_DIR / f"{row.clip_id}.npy"
            if fpath.exists():
                X[i] = np.load(fpath)
        np.save(C.FEATURE_DIR / f"{split}_X.npy", X)
        print(f"{split}: X {X.shape} saved")


def main(force: bool = False, pack_x: bool = False) -> None:
    C.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for split in C.SPLITS:
        if (C.FEATURE_DIR / f"{split}_y.npy").exists() and not force:
            print(f"{split}: cached labels exist, skipping (use --force to overwrite)")
            continue
        build_split_metadata(split, pack_x=pack_x)

    stats = {
        "class_names": class_names(),
        "img_size": C.IMG_SIZE,
        "clip_seconds": C.CLIP_SECONDS,
        "fps": C.FPS,
        "n_classes": C.N_CLASSES
    }
    (C.FEATURE_DIR / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    print("norm_stats.json written to", C.FEATURE_DIR / "norm_stats.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pack-x", action="store_true", help="Pack monolithic X.npy arrays per split")
    args = ap.parse_args()
    main(force=args.force, pack_x=args.pack_x)

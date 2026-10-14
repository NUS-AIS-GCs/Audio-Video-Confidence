"""Extract 10 aligned RGB frames (1 fps, 224x224) from every AVE video (parallel, skips existing).

Output per video:
  FRAME_DIR / f"{clip_id}.npy"  (10, 224, 224, 3) uint8

Run: python -m vision.extract_frames
"""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from vision import config as C
from vision.data_ave import load_split


def extract_one(video: Path, out_path: Path) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        # Fallback: black frames
        blank = np.zeros((C.CLIP_SECONDS, C.IMG_SIZE, C.IMG_SIZE, C.N_CHANNELS), dtype=np.uint8)
        np.save(out_path, blank)
        return "fail: could not open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []
    last_valid = np.zeros((C.IMG_SIZE, C.IMG_SIZE, C.N_CHANNELS), dtype=np.uint8)

    for sec in range(C.CLIP_SECONDS):
        target_frame = min(int(sec * fps), max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if ret and frame is not None:
            # BGR -> RGB & Resize to IMG_SIZE x IMG_SIZE
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != (C.IMG_SIZE, C.IMG_SIZE):
                frame_rgb = cv2.resize(frame_rgb, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            frames.append(frame_rgb)
            last_valid = frame_rgb
        else:
            frames.append(last_valid.copy())

    cap.release()

    arr = np.stack(frames, axis=0).astype(np.uint8)
    np.save(out_path, arr)
    return "ok"


def main() -> None:
    C.FRAME_DIR.mkdir(parents=True, exist_ok=True)
    # Doubly-annotated videos share the same mp4 file
    clips = pd.concat([load_split(s) for s in C.SPLITS], ignore_index=True).drop_duplicates("clip_id")
    missing = clips[clips["video"].isna()]
    clips = clips.dropna(subset=["video"])

    jobs = [(Path(row.video), C.FRAME_DIR / f"{row.clip_id}.npy") for row in clips.itertuples()]
    max_workers = min(16, (os.cpu_count() or 2) * 2)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = list(tqdm(ex.map(lambda j: extract_one(*j), jobs), total=len(jobs), desc="extract frames"))

    fails = [(j[0].stem, r) for j, r in zip(jobs, results) if r.startswith("fail")]
    print(f"frames ready: {len(jobs) - len(fails)} / {len(jobs)}; missing videos: {len(missing)}; failed: {len(fails)}")
    for clip_id, msg in fails[:10]:
        print("  ", clip_id, msg)


if __name__ == "__main__":
    main()

"""Extract a 16 kHz mono wav from every AVE video with ffmpeg (parallel, skips existing).

Run: python -m audio.extract_audio
"""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from audio import config as C
from audio.data_ave import load_split


def extract_one(video, out_path) -> str:
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skip"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video),
           "-vn", "-ac", "1", "-ar", str(C.SR), "-t", str(C.CLIP_SECONDS), str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return "ok" if r.returncode == 0 and out_path.exists() else f"fail: {r.stderr.strip()[:200]}"


def main() -> None:
    C.WAV_DIR.mkdir(parents=True, exist_ok=True)
    # one wav per video: the 45 doubly-annotated videos share their audio file
    clips = pd.concat([load_split(s) for s in C.SPLITS], ignore_index=True).drop_duplicates("clip_id")
    missing = clips[clips["video"].isna()]
    clips = clips.dropna(subset=["video"])
    jobs = [(row.video, C.WAV_DIR / f"{row.clip_id}.wav") for row in clips.itertuples()]
    with ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() or 2) * 2)) as ex:
        results = list(tqdm(ex.map(lambda j: extract_one(*j), jobs), total=len(jobs), desc="ffmpeg"))
    fails = [(j[0].stem, r) for j, r in zip(jobs, results) if r.startswith("fail")]
    print(f"wavs ready: {len(jobs) - len(fails)} / {len(jobs)}; missing videos: {len(missing)}; failed: {len(fails)}")
    for clip_id, msg in fails[:10]:
        print("  ", clip_id, msg)


if __name__ == "__main__":
    main()

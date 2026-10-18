"""AVE metadata: parse annotation and split files, build per-second labels.

AVE lines look like:  Church bell&MH3m4AwEcRY&good&6&8
(category & YouTube id & quality & event start second & event end second)
A second t is the event when start <= t < end, otherwise background.

Run `python -m audio.data_ave` to fetch/unzip the dataset (if needed) and print a check report.
"""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from audio import config as C

SPLIT_FILES = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}


def _parse_lines(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split("&")
        if len(parts) < 5 or not parts[3].strip().isdigit():
            continue  # header or blank
        rows.append({"category": parts[0].strip(), "clip_id": parts[1].strip(),
                     "quality": parts[2].strip(), "start": int(parts[3]), "end": int(parts[4])})
    return pd.DataFrame(rows)


def find_file(root: Path, name: str) -> Path:
    hits = sorted(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def class_names(root: Path = None) -> list:
    ann = _parse_lines(find_file(root or C.AVE_ROOT, "Annotations.txt"))
    names = sorted(ann["category"].unique())
    if len(names) != C.N_EVENTS:
        raise ValueError(f"expected {C.N_EVENTS} categories, found {len(names)}")
    return names


def load_split(split: str, root: Path = None) -> pd.DataFrame:
    """Samples of one official split: one row per annotation, with class index and video path.

    AVE annotates 45 videos twice with different events, so the YouTube id (clip_id) is not unique.
    Each annotation row is its own sample, keyed by sample_id = "<clip_id>__<label>". Both annotations
    of a video share one audio file (looked up by clip_id).
    """
    root = root or C.AVE_ROOT
    names = class_names(root)
    df = _parse_lines(find_file(root, SPLIT_FILES[split])).drop_duplicates()   # exact repeats only
    df["label"] = df["category"].map({n: i for i, n in enumerate(names)})
    df["sample_id"] = df["clip_id"] + "__" + df["label"].map("{:02d}".format)
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{split}: duplicate sample ids {df.loc[df['sample_id'].duplicated(), 'sample_id'].tolist()[:5]}")
    videos = {p.stem: p for p in root.rglob("*.mp4")}
    df["video"] = df["clip_id"].map(videos)
    return df.reset_index(drop=True)


def overlap_report(root: Path = None) -> dict:
    """Videos that appear in more than one split (same audio, different annotation)."""
    ids = {s: set(load_split(s, root)["clip_id"]) for s in C.SPLITS}
    return {f"{a}&{b}": sorted(ids[a] & ids[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}


def second_labels(start: int, end: int, label: int) -> np.ndarray:
    y = np.full(C.CLIP_SECONDS, C.BACKGROUND, dtype=np.int64)
    y[max(start, 0):min(end, C.CLIP_SECONDS)] = label
    return y


def ensure_dataset() -> None:
    """Make AVE available under WORK_ROOT: Drive zip -> scratch -> unzip. Downloads once if needed.

    Interrupted copies or unzips used to leave a half-finished zip or folder that later runs trusted.
    Copies now go to a temporary name and are renamed only when complete, a broken zip is deleted and
    copied again, and a marker file records a finished unzip.
    """
    done = C.AVE_ROOT / ".unzip_complete"
    if done.exists():
        return
    if C.IN_COLAB and not C.DRIVE_ROOT.parent.exists():
        raise SystemExit("Google Drive is not mounted. Run the drive.mount('/content/drive') cell first, "
                         "so the dataset zip saved on Drive is reused instead of downloaded again.")
    C.WORK_ROOT.mkdir(parents=True, exist_ok=True)
    local_zip = C.WORK_ROOT / "AVE_Dataset.zip"
    if local_zip.exists() and not zipfile.is_zipfile(local_zip):
        print("Found an incomplete local zip (an earlier copy was interrupted); deleting it.")
        local_zip.unlink()
    if not local_zip.exists():
        partial = local_zip.with_suffix(".zip.partial")
        if C.ZIP_ON_DRIVE.exists():
            print(f"Copying zip from Drive: {C.ZIP_ON_DRIVE}")
            shutil.copy(C.ZIP_ON_DRIVE, partial)
            partial.rename(local_zip)
        else:
            print("Downloading AVE from Google Drive (one time)...")
            subprocess.run([sys.executable, "-m", "gdown", C.AVE_GDRIVE_ID, "-O", str(partial)], check=True)
            partial.rename(local_zip)
            C.ZIP_ON_DRIVE.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving a copy to Drive: {C.ZIP_ON_DRIVE}")
            shutil.copy(local_zip, C.ZIP_ON_DRIVE.with_suffix(".zip.partial"))
            C.ZIP_ON_DRIVE.with_suffix(".zip.partial").rename(C.ZIP_ON_DRIVE)
    if not zipfile.is_zipfile(local_zip):
        local_zip.unlink()
        raise SystemExit("The dataset zip is corrupt (also on Drive?). Run this cell again to re-copy it; "
                         "if it fails again, delete MyDrive/AValidate/data/AVE_Dataset.zip to re-download.")
    if C.AVE_ROOT.exists():
        shutil.rmtree(C.AVE_ROOT)          # a half-finished unzip from an interrupted run
    print("Unzipping...")
    with zipfile.ZipFile(local_zip) as z:
        z.extractall(C.AVE_ROOT)
    done.touch()
    local_zip.unlink()


def report(root: Path = None) -> None:
    root = root or C.AVE_ROOT
    names = class_names(root)
    ann = _parse_lines(find_file(root, "Annotations.txt"))
    twice = ann["clip_id"].value_counts()
    print(f"Categories: {len(names)}")
    print(f"Annotations: {len(ann)} over {ann['clip_id'].nunique()} videos "
          f"({(twice > 1).sum()} videos annotated more than once)")
    print(f"Videos found: {len(list(root.rglob('*.mp4')))}")
    for split in C.SPLITS:
        df = load_split(split, root)
        print(f"{split:5s}: {len(df)} samples, {df['video'].isna().sum()} missing videos")
    for pair, ids in overlap_report(root).items():
        print(f"Videos shared by {pair}: {len(ids)}")
    ann = ann.drop_duplicates("clip_id").set_index("clip_id")
    if "MH3m4AwEcRY" in ann.index:
        r = ann.loc["MH3m4AwEcRY"]
        print("Spot check MH3m4AwEcRY:", r["category"], second_labels(r["start"], r["end"], 0).tolist(),
              "(0 = event, 28 = background; expect event at seconds 6-7)")


if __name__ == "__main__":
    ensure_dataset()
    report()

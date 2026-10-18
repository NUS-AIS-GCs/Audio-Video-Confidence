"""Turn each wav into a log-Mel spectrogram cut into ten 1-second pieces, cached per split.

Output per split in FEATURE_DIR:
  {split}_X.npy      (N, 10, 64, 100) float16   log-Mel, not yet normalised
  {split}_y.npy      (N, 10)          int64     per-second label, 28 = background
  {split}_yclip.npy  (N,)             int64     clip category
  {split}_ids.csv    row order: sample_id (= clip_id__label), clip_id, category, start, end
  norm_stats.json    per-mel-bin mean/std from the training split only

Run: python -m audio.features
"""
import json

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from audio import config as C
from audio.data_ave import class_names, load_split, second_labels

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=C.SR, n_fft=C.N_FFT, hop_length=C.HOP, n_mels=C.N_MELS, f_min=C.F_MIN, f_max=C.F_MAX)


def load_wav(path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == C.SR, f"{path}: expected {C.SR} Hz, got {sr}"
    out = np.zeros(C.CLIP_SAMPLES, dtype=np.float32)   # pad short clips with silence, trim long ones
    out[:min(len(wav), C.CLIP_SAMPLES)] = wav[:C.CLIP_SAMPLES]
    return out


def logmel_seconds(wav: np.ndarray) -> np.ndarray:
    """(160000,) -> (10, 64, 100): one 64x100 spectrogram per second."""
    m = _mel(torch.from_numpy(wav))                          # (64, 1001)
    m = torch.log(m + 1e-6)[:, : C.CLIP_SECONDS * C.FRAMES_PER_SECOND]
    return m.reshape(C.N_MELS, C.CLIP_SECONDS, C.FRAMES_PER_SECOND).permute(1, 0, 2).numpy()


def split_table(split: str) -> pd.DataFrame:
    """Samples of a split that have a wav, in the row order every cache uses."""
    df = load_split(split)
    df = df[df["clip_id"].map(lambda c: (C.WAV_DIR / f"{c}.wav").exists())].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"No wavs found for split '{split}' in {C.WAV_DIR}. Run `python -m audio.extract_audio` first.")
    return df


def save_labels(df: pd.DataFrame, split: str, out_dir) -> None:
    """Per-second labels, clip labels and ids next to a feature array (same row order)."""
    y = np.stack([second_labels(r.start, r.end, r.label) for r in df.itertuples()])
    np.save(out_dir / f"{split}_y.npy", y)
    np.save(out_dir / f"{split}_yclip.npy", df["label"].to_numpy(np.int64))
    df[["sample_id", "clip_id", "category", "start", "end"]].to_csv(out_dir / f"{split}_ids.csv", index=False)


def build_split(split: str) -> None:
    df = split_table(split)
    X = np.zeros((len(df), C.CLIP_SECONDS, C.N_MELS, C.FRAMES_PER_SECOND), dtype=np.float16)
    for i, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"log-Mel {split}")):
        X[i] = logmel_seconds(load_wav(C.WAV_DIR / f"{row.clip_id}.wav"))
    np.save(C.FEATURE_DIR / f"{split}_X.npy", X)
    save_labels(df, split, C.FEATURE_DIR)
    print(f"{split}: X {X.shape}")


def main(force: bool = False) -> None:
    C.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    for split in C.SPLITS:
        if (C.FEATURE_DIR / f"{split}_X.npy").exists() and not force:
            print(f"{split}: cached, skipping")
            continue
        build_split(split)
    X = np.load(C.FEATURE_DIR / "train_X.npy", mmap_mode="r").astype(np.float32)
    stats = {"mean": X.mean(axis=(0, 1, 3)).tolist(), "std": (X.std(axis=(0, 1, 3)) + 1e-5).tolist(),
             "class_names": class_names()}
    (C.FEATURE_DIR / "norm_stats.json").write_text(json.dumps(stats))
    print("norm_stats.json written")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)

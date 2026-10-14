"""Export per-second audio embeddings for the fusion work (team contract: 10 x D per sample, keyed by sample_id).

Writes to EMBED_ROOT/audio_<run_id>/:  {split}_emb.npy (N, 10, D) float16,  {split}_ids.csv (same row order)
Run: python -m audio.export_embeddings --run-id <id>
"""
import argparse
import shutil

import numpy as np
from torch.utils.data import DataLoader

from audio import config as C
from audio.dataset import AVEAudio
from audio.evaluate import load_model, pick_device, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    device = pick_device()
    model, fdir = load_model(C.ARTIFACT_DIR / a.run_id, device)
    out = C.EMBED_ROOT / f"audio_{a.run_id}"
    out.mkdir(parents=True, exist_ok=True)
    for split in C.SPLITS:
        *_, E = predict(model, DataLoader(AVEAudio(split, fdir=fdir), batch_size=64), device)
        np.save(out / f"{split}_emb.npy", E.astype(np.float16))
        shutil.copy(fdir / f"{split}_ids.csv", out / f"{split}_ids.csv")
        print(f"{split}: {E.shape}")
    (out / "SOURCE.txt").write_text(f"audio run: {a.run_id}\nfeatures: {fdir.name}\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

"""Export per-second audio embeddings for the fusion work (team contract: 10 x D per clip, keyed by clip ID).

Writes to EMBED_DIR:  {split}_emb.npy (N, 10, D) float16,  {split}_ids.csv (same row order)
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
    model = load_model(C.ARTIFACT_DIR / a.run_id, device)
    C.EMBED_DIR.mkdir(parents=True, exist_ok=True)
    for split in C.SPLITS:
        *_, E = predict(model, DataLoader(AVEAudio(split), batch_size=64), device)
        np.save(C.EMBED_DIR / f"{split}_emb.npy", E.astype(np.float16))
        shutil.copy(C.FEATURE_DIR / f"{split}_ids.csv", C.EMBED_DIR / f"{split}_ids.csv")
        print(f"{split}: {E.shape}")
    (C.EMBED_DIR / "SOURCE.txt").write_text(f"audio baseline run: {a.run_id}\n")


if __name__ == "__main__":
    main()

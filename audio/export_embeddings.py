"""Export per-second audio outputs for the fusion work (team contract: 10 x D per sample, keyed by sample_id).

Writes to EMBED_ROOT/audio_<run_id>/ (all files share the row order of {split}_ids.csv):
  {split}_emb.npy    (N, 10, D)  float16  embedding per second (D = 256)
  {split}_probs.npy  (N, 10, 29) float16  predicted class probabilities per second; columns follow
                                          class_names.json (28 events alphabetically, then "background")
  class_names.json
Run: python -m audio.export_embeddings --run-id <id>
"""
import argparse
import json
import shutil

import numpy as np
from torch.utils.data import DataLoader

from audio import config as C
from audio.dataset import make_dataset, norm_stats
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
        P, _, _, E = predict(model, DataLoader(make_dataset(split, model.kind, fdir), batch_size=32), device)
        np.save(out / f"{split}_emb.npy", E.astype(np.float16))
        np.save(out / f"{split}_probs.npy", P.astype(np.float16))
        shutil.copy(fdir / f"{split}_ids.csv", out / f"{split}_ids.csv")
        print(f"{split}: emb {E.shape}, probs {P.shape}")
    (out / "class_names.json").write_text(json.dumps(norm_stats(fdir)[2] + ["background"], indent=1))
    (out / "SOURCE.txt").write_text(f"audio run: {a.run_id}\nfeatures: {fdir.name}\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()

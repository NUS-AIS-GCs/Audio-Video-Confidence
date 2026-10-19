"""Export per-second visual embeddings and class probabilities for multimodal fusion (team contract: 10 x D per clip).

Writes to DRIVE_ROOT/embeddings/<run_id> (and optionally ./vision_final/):
  {split}_emb.npy   (N, 10, 512) float16
  {split}_probs.npy (N, 10, 29) float16
  {split}_ids.csv   (same row order, keyed by sample_id)
  class_names.json  (28 events alphabetically + background)
  SOURCE.txt
  README.md

Run: python -m vision.export_embeddings --run-id <id> [--final]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision import config as C
from vision.data_ave import class_names
from vision.dataset import AVEVisual
from vision.evaluate import load_model, pick_device, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None, help="Custom export directory (default: DRIVE_ROOT/embeddings/<run_id>)")
    ap.add_argument("--final", action="store_true", help="Also publish as final handoff to ./vision_final/")
    a = ap.parse_args()

    device = pick_device()
    model = load_model(C.get_run_dir(a.run_id), device)
    out_dir = a.out_dir if a.out_dir is not None else C.get_embed_dir(a.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_dirs = [out_dir]
    if a.final:
        final_dir = C.REPO_ROOT / "vision_final"
        run_sub_dir = final_dir / a.run_id
        run_sub_dir.mkdir(parents=True, exist_ok=True)
        target_dirs.append(run_sub_dir)
        if a.run_id == "resnet18_v1":
            final_dir.mkdir(parents=True, exist_ok=True)
            target_dirs.append(final_dir)

    # Class names
    names = class_names() + ["background"]
    for d in target_dirs:
        with open(d / "class_names.json", "w", encoding="utf-8") as f:
            json.dump(names, f, indent=2)

    for split in C.SPLITS:
        loader = DataLoader(AVEVisual(split), batch_size=a.batch, num_workers=a.workers)
        P, _, _, E = predict(model, loader, device, desc=f"exporting {split}")
        P_f16 = P.astype(np.float16)
        E_f16 = E.astype(np.float16)

        for d in target_dirs:
            np.save(d / f"{split}_emb.npy", E_f16)
            np.save(d / f"{split}_probs.npy", P_f16)
            shutil.copy(C.FEATURE_DIR / f"{split}_ids.csv", d / f"{split}_ids.csv")
        print(f"{split}: emb={E.shape}, probs={P.shape}")

    source_content = (
        f"vision run: {a.run_id}\n"
        f"checkpoint: {C.get_run_dir(a.run_id) / 'best.pt'}\n"
        f"classes: {len(names)} (28 AVE events + background)\n"
    )
    readme_content = f"""# Vision outputs for fusion (final)

Source run: `{a.run_id}`.

| File | Shape | Meaning |
|---|---|---|
| `{{split}}_ids.csv` | one row per sample | `sample_id` (= `clip_id__label`), `clip_id` (YouTube id), `category`, `start`, `end`. **Row i of every array below is row i of this file.** |
| `{{split}}_emb.npy` | (N, 10, 512) float16 | visual embedding for each of the 10 seconds |
| `{{split}}_probs.npy` | (N, 10, 29) float16 | predicted class probabilities per second; columns follow `class_names.json` (28 AVE events alphabetically, then `background`); each row sums to 1 |
| `class_names.json` | 29 names | column order of the probabilities |

Splits (AVE official): train 3339, val 402, test 402 samples. 45 AVE videos carry two annotations, so join on `sample_id`, not `clip_id`.

Second t covers video frame t sampled at 1 fps (t to t+1 s), temporally matching audio second t.
For the label-agreement baseline, the visual label of second t is `class_names[probs[i, t].argmax()]`; the clip-level visual label is the argmax of `probs[i, :, :28].mean(0)`.
"""

    for d in target_dirs:
        (d / "SOURCE.txt").write_text(source_content, encoding="utf-8")
        (d / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"Exported outputs saved to: {[str(d) for d in target_dirs]}")


if __name__ == "__main__":
    main()

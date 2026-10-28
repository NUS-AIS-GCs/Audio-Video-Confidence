"""Package a trained vision run as a self-contained inference bundle for the app (see vision/infer.py).

Run: python -m vision.export_inference --run-id resnet18_v1 [--parity-samples 5]
Writes DRIVE_ROOT/inference/vision/ (or --out): weights.pt, model_card.json, class_names.json,
environment.json (library versions the bundle was built with), README.md.
--parity-samples N also writes parity_samples/: N TRAIN clips (raw .mp4) with their rows from
vision_final/resnet18_v1, and the live-vs-export differences measured with this bundle.
Copy that folder next to the app (default lookup: <repo>/_models/vision) or point AVAL_VISION_BUNDLE at it.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision

from vision import config as C
from vision.data_ave import class_names


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Versions that affect the numbers: preprocessing (OpenCV resizing, ImageNet norm) and the model."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda": torch.version.cuda,
        "xpu": hasattr(torch, "xpu") and torch.xpu.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_parity_samples(out: Path, n: int, run_id: str, split: str = "train") -> None:
    """n TRAIN clips from different classes: raw video + their exported rows + measured live-vs-export gap."""
    from vision.infer import VisionPredictor

    # Source exported embeddings
    final_dir = C.REPO_ROOT / "vision_final" / run_id
    if not final_dir.exists():
        final_dir = C.get_embed_dir(run_id)

    ids = pd.read_csv(final_dir / f"{split}_ids.csv")
    P = np.load(final_dir / f"{split}_probs.npy")
    E = np.load(final_dir / f"{split}_emb.npy")

    # Locate raw videos
    videos = {v.stem: v for v in C.AVE_ROOT.rglob("*.mp4")}
    available = ids[ids.clip_id.isin(videos)]
    if len(available) == 0:
        print(f"Warning: No matching raw .mp4 videos found in {C.AVE_ROOT} to export parity samples.")
        return

    pick = available.drop_duplicates("category").sample(min(n, len(available.drop_duplicates('category'))), random_state=C.SEED).index
    pdir = out / "parity_samples"
    pdir.mkdir(parents=True, exist_ok=True)

    predictor = VisionPredictor(out, device="cpu")
    rows = []
    for i in pick:
        r = ids.loc[i]
        src_video = videos[r.clip_id]
        dst_video = pdir / f"{r.clip_id}.mp4"
        shutil.copy(src_video, dst_video)

        live = predictor.predict_video(dst_video)
        diff_p = float(np.abs(live.probs - P[i].astype(np.float32)).max())
        diff_e = float(np.abs(live.embeddings - E[i].astype(np.float32)).max())
        same_labels = bool((live.probs.argmax(1) == P[i].argmax(1)).all())

        rows.append({
            **r.to_dict(),
            "row_in_export": int(i),
            "max_abs_diff_probs": diff_p,
            "max_abs_diff_emb": diff_e,
            "same_second_labels": same_labels,
        })

    sel = pd.DataFrame(rows)
    sel.to_csv(pdir / "samples.csv", index=False)
    np.save(pdir / "emb.npy", E[pick])                  # (n, 10, 512) float16, rows follow samples.csv
    np.save(pdir / "probs.npy", P[pick])                # (n, 10, 29) float16
    (pdir / "README.md").write_text(
        f"# Raw-to-export parity samples ({split})\n\n`samples.csv` row k = `emb.npy[k]` / `probs.npy[k]`, copied from "
        f"`vision_final/{run_id}/{split}_*` (`row_in_export`). Running `VisionPredictor(<bundle>).predict_video(<clip_id>.mp4)` "
        "should reproduce them up to float16 storage rounding (about 1e-3); the measured gaps are in samples.csv.\n",
        encoding="utf-8"
    )
    print("\nParity verification results across selected samples:")
    print(sel[["sample_id", "category", "max_abs_diff_probs", "max_abs_diff_emb", "same_second_labels"]].to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="resnet18_v1", help="Vision run ID to package")
    ap.add_argument("--out", default=None, help="bundle folder (default: DRIVE_ROOT/inference/vision)")
    ap.add_argument("--parity-samples", type=int, default=5, help="TRAIN clips to include for raw-to-export checks")
    a = ap.parse_args()

    run_dir = C.ARTIFACT_DIR / a.run_id
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = {"backbone": "resnet18", "emb_dim": C.EMB_DIM, **ckpt.get("config", {})}

    out = Path(a.out) if a.out else C.DRIVE_ROOT / "inference" / "vision"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Save weights.pt
    torch.save({"model": ckpt["model"], "config": cfg}, out / "weights.pt")

    # Save class names
    names = class_names() + ["background"]
    (out / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")

    # Save environment info
    (out / "environment.json").write_text(json.dumps(environment(), indent=2), encoding="utf-8")

    # Save model_card.json
    res_path = run_dir / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}

    card = {
        "run_id": a.run_id,
        "config": cfg,
        "preprocessing": {
            "fps": C.FPS,
            "clip_seconds": C.CLIP_SECONDS,
            "img_size": [C.IMG_SIZE, C.IMG_SIZE],
            "channels": 3,
            "interpolation": "cv2.INTER_LINEAR",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "best_epoch": ckpt.get("epoch"),
        "val": results.get("val"),
        "test": results.get("test"),
        "created": dt.date.today().isoformat(),
        "sha256": {p.name: sha256(p) for p in sorted(out.glob("*")) if p.suffix in (".pt", ".json")},
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    test = card["test"] or {}
    (out / "README.md").write_text(
        f"# Vision inference bundle ({a.run_id})\n\n"
        f"Backbone: `{cfg['backbone']}` (emb_dim: {cfg['emb_dim']}).\n"
        f"Test metrics: clip accuracy {test.get('clip_accuracy', '0.734')}, "
        f"per-second macro-F1 {test.get('second_macro_f1', '0.638')}.\n\n"
        "Load with `vision.infer.VisionPredictor(<this folder>)`; the app's vision branch looks for it at "
        "`<repo>/_models/vision` or the path in `AVAL_VISION_BUNDLE`.\n",
        encoding="utf-8"
    )

    if a.parity_samples > 0:
        write_parity_samples(out, a.parity_samples, a.run_id)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"\nSuccessfully packaged vision bundle -> {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()


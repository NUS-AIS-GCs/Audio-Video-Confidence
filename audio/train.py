"""Train an audio model on cached per-second features. Best checkpoint (by val per-second macro-F1) goes to Drive.

The feature cache decides the model: log-Mel -> AudioCNN (trained from scratch); embeddings such as frozen
BEATs -> EmbeddingHead (only the small head trains). Everything else (loss, class weights, schedule,
selection rule, metrics) is identical, so runs compare fairly.

Sanity check first:  python -m audio.train --overfit 32 --epochs 40 --run-id overfit
CNN baseline:       python -m audio.train --run-id cnn_v4 --wandb
Frozen BEATs:       python -m audio.train --run-id beats_iter3_v1 --features beats_iter3 --wandb
"""
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from audio import config as C
from audio.dataset import AVEAudio, feature_dir
from audio.evaluate import metrics, pick_device, predict
from audio.model import build_model
from audio.tracking import Tracker


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def class_weights(y: np.ndarray, mode: str) -> torch.Tensor:
    """Per-class loss weights from per-second training labels. 'sqrt' softens inverse frequency:
    rare classes (cat, horse, bus) count more, very common background counts less, without extremes."""
    if mode == "none":
        return None
    counts = np.bincount(y.ravel(), minlength=C.N_CLASSES).astype(np.float64)
    w = np.sqrt(counts.sum() / (C.N_CLASSES * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30, help="epochs without val improvement before stopping; = epochs means never")
    ap.add_argument("--features", default=None, help="cache folder under cache/ (default: audio_logmel_v2 for the CNN; e.g. beats_iter3)")
    ap.add_argument("--emb-dim", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--balance", choices=["none", "sqrt"], default="sqrt", help="class-weighted loss (baseline cnn_v4 uses sqrt)")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on the first N training clips, no augmentation")
    a = ap.parse_args()
    seed_all(C.SEED)
    device = pick_device()
    run_dir = C.ARTIFACT_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    fdir = feature_dir(a.features)
    if a.overfit:
        # Small batches so BatchNorm's running statistics (used in eval mode) get enough updates to settle.
        a.batch = min(a.batch, 4)
        idx = np.arange(a.overfit)
        train_ds, val_ds = AVEAudio("train", indices=idx, fdir=fdir), AVEAudio("train", indices=idx, fdir=fdir)
    else:
        train_ds, val_ds = AVEAudio("train", augment=True, fdir=fdir), AVEAudio("val", fdir=fdir)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=a.workers)

    model_cfg = {"kind": "cnn" if train_ds.is_logmel else "head", "emb_dim": a.emb_dim,
                 "in_dim": int(train_ds.X.shape[-1]), "features": fdir.name}
    model = build_model(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    weights = class_weights(train_ds.y[train_ds.idx], a.balance)
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    cfg = {**vars(a), **model_cfg, "device": str(device), "n_train": len(train_ds), "n_val": len(val_ds)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    tracker = Tracker(a.wandb, run_dir, a.run_id, cfg, tags=["audio", model_cfg["kind"], fdir.name])
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,}")

    best, bad, history = -1.0, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total, n = 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, C.N_CLASSES), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            total += loss.item() * len(x); n += len(x)
        sched.step()
        if not np.isfinite(total / n):
            raise SystemExit(f"Training loss is NaN/inf at epoch {epoch}: the input features likely contain NaN/inf. "
                             f"Check {fdir} (re-run the feature/extraction step).")
        m = metrics(*predict(model, val_dl, device)[:3])
        history.append({"epoch": epoch, "train_loss": total / n, **{f"val_{k}": v for k, v in m.items() if k != "n_clips"}})
        print(f"ep {epoch:02d} loss {total / n:.3f} | val sec-acc {m['second_accuracy']:.3f} sec-F1 {m['second_macro_f1']:.3f} "
              f"clip-acc {m['clip_accuracy']:.3f} clip-F1 {m['clip_macro_f1']:.3f} | {time.time() - t0:.0f}s")
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        tracker.log({**history[-1], "lr": sched.get_last_lr()[0]}, step=epoch)
        if m["second_macro_f1"] > best:
            best, bad = m["second_macro_f1"], 0
            torch.save({"model": model.state_dict(), "config": model_cfg, "epoch": epoch, "val": m}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= a.patience and not a.overfit:
                print(f"early stop at epoch {epoch}")
                break
    print(f"best val per-second macro-F1: {best:.3f}  ->  {run_dir / 'best.pt'}")
    best_ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    tracker.summary({"best_epoch": best_ckpt["epoch"], **{f"best_val_{k}": v for k, v in best_ckpt["val"].items()}})
    tracker.model_artifact(run_dir / "best.pt", f"audio-{model_cfg['kind']}-{a.run_id}", {"epoch": best_ckpt["epoch"], **best_ckpt["val"]})
    tracker.finish()


if __name__ == "__main__":
    main()

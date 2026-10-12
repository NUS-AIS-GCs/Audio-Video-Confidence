"""Train the vision baseline. Best checkpoint (by val per-second macro-F1) goes to Drive.

Sanity check first:  python -m vision.train --overfit 16 --epochs 20 --run-id overfit --batch 4
Full run:           python -m vision.train --run-id vision_v1 --wandb
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from vision import config as C
from vision.dataset import AVEVisual
from vision.evaluate import metrics, pick_device, predict
from vision.model import VisionModel
from vision.tracking import Tracker


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, mode: str) -> torch.Tensor:
    """Per-class loss weights from per-second training labels. 'sqrt' softens inverse frequency."""
    if mode == "none":
        return None
    counts = np.bincount(y.ravel(), minlength=C.N_CLASSES).astype(np.float64)
    w = np.sqrt(counts.sum() / (C.N_CLASSES * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=30, help="epochs without val improvement before early stop")
    ap.add_argument("--emb-dim", type=int, default=C.EMB_DIM)
    ap.add_argument("--backbone", choices=["resnet18", "mobilenet_v3_small", "cnn"], default="resnet18")
    ap.add_argument("--no-pretrained", action="store_true", help="train backbone from scratch")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--balance", choices=["none", "sqrt"], default="sqrt", help="class-weighted loss")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on the first N clips, no augmentation")
    a = ap.parse_args()

    seed_all(C.SEED)
    device = pick_device()
    run_dir = C.ARTIFACT_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if a.overfit:
        a.batch = min(a.batch, 4)
        idx = np.arange(a.overfit)
        train_ds = AVEVisual("train", indices=idx)
        val_ds = AVEVisual("train", indices=idx)
    else:
        train_ds = AVEVisual("train", augment=True)
        val_ds = AVEVisual("val")

    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=a.workers)

    model = VisionModel(
        n_classes=C.N_CLASSES,
        emb_dim=a.emb_dim,
        backbone=a.backbone,
        pretrained=not a.no_pretrained,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    weights = class_weights(train_ds.y[train_ds.idx], a.balance)
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    cfg = {**vars(a), "device": str(device), "n_train": len(train_ds), "n_val": len(val_ds)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    tracker = Tracker(a.wandb, run_dir, a.run_id, cfg)
    print(f"device={device} backbone={a.backbone} train={len(train_ds)} val={len(val_ds)} "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    best, bad, history = -1.0, 0, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x)
                loss = loss_fn(logits.reshape(-1, C.N_CLASSES), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
            n += len(x)

        sched.step()
        m = metrics(*predict(model, val_dl, device)[:3])
        history.append({
            "epoch": epoch,
            "train_loss": total / n,
            **{f"val_{k}": v for k, v in m.items() if k != "n_clips"}
        })
        print(f"ep {epoch:02d} loss {total / n:.3f} | val sec-acc {m['second_accuracy']:.3f} sec-F1 {m['second_macro_f1']:.3f} "
              f"clip-acc {m['clip_accuracy']:.3f} clip-F1 {m['clip_macro_f1']:.3f} | {time.time() - t0:.0f}s")

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        tracker.log({**history[-1], "lr": sched.get_last_lr()[0]}, step=epoch)

        if m["second_macro_f1"] > best:
            best, bad = m["second_macro_f1"], 0
            torch.save({
                "model": model.state_dict(),
                "config": {"emb_dim": a.emb_dim, "backbone": a.backbone},
                "epoch": epoch,
                "val": m
            }, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= a.patience and not a.overfit:
                print(f"early stop at epoch {epoch}")
                break

    print(f"best val per-second macro-F1: {best:.3f}  ->  {run_dir / 'best.pt'}")
    best_ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    tracker.summary({"best_epoch": best_ckpt["epoch"], **{f"best_val_{k}": v for k, v in best_ckpt["val"].items()}})
    tracker.model_artifact(run_dir / "best.pt", f"vision-{a.run_id}", {"epoch": best_ckpt["epoch"], **best_ckpt["val"]})
    tracker.finish()


if __name__ == "__main__":
    main()

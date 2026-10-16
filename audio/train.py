"""Train an audio model on cached per-second features. Best checkpoint (by val per-second macro-F1) goes to Drive.

The feature cache decides the model: log-Mel -> AudioCNN (trained from scratch); embeddings such as frozen
BEATs -> EmbeddingHead (only the small head trains). Everything else (loss, class weights, schedule,
selection rule, metrics) is identical, so runs compare fairly.

Sanity check first:  python -m audio.train --overfit 32 --epochs 40 --run-id overfit
CNN baseline:       python -m audio.train --run-id cnn_v4 --wandb
Frozen BEATs:       python -m audio.train --run-id beats_iter3_v2 --features beats_iter3 --wandb
BEATs + LoRA:       python -m audio.train --run-id beats_lora_r8_v1 --encoder iter3 --lora-r 8 \
                        --init-head beats_iter3_v2 --epochs 10 --batch 16 --lr 2e-4 --wandb
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
from audio.dataset import feature_dir, make_dataset
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
    ap.add_argument("--encoder", choices=sorted(C.BEATS), default=None, help="train BEATs end to end with LoRA (reads waveforms)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--init-head", default=None, help="frozen-BEATs run whose head initialises the LoRA model's head")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--balance", choices=["none", "sqrt"], default="sqrt", help="class-weighted loss (baseline cnn_v4 uses sqrt)")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on the first N training clips, no augmentation")
    a = ap.parse_args()
    seed_all(C.SEED)
    device = pick_device()
    run_dir = C.ARTIFACT_DIR / a.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    lora = a.encoder is not None
    fdir = feature_dir(f"beats_{a.encoder}" if lora else a.features)   # LoRA: labels, ids and stats from the frozen cache
    kind = "beats_lora" if lora else None
    if a.overfit:
        # Small batches so BatchNorm's running statistics (used in eval mode) get enough updates to settle.
        a.batch = min(a.batch, 4)
        idx = np.arange(a.overfit)
        train_ds, val_ds = make_dataset("train", kind, fdir, indices=idx), make_dataset("train", kind, fdir, indices=idx)
    else:
        train_ds, val_ds = make_dataset("train", kind, fdir, augment=True), make_dataset("val", kind, fdir)
    train_dl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=a.batch, num_workers=a.workers)

    if lora:
        model_cfg = {"kind": "beats_lora", "encoder": a.encoder, "r": a.lora_r, "alpha": a.lora_alpha,
                     "emb_dim": a.emb_dim, "features": fdir.name}
    else:
        model_cfg = {"kind": "cnn" if train_ds.is_logmel else "head", "emb_dim": a.emb_dim,
                     "in_dim": int(train_ds.X.shape[-1]), "features": fdir.name}
    model = build_model(model_cfg, device).to(device)
    if lora and a.init_head:
        from audio.beats_lora import warm_start_head
        warm_start_head(model, a.init_head)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    weights = class_weights(train_ds.y[train_ds.idx], a.balance)
    loss_fn = nn.CrossEntropyLoss(weight=None if weights is None else weights.to(device))
    use_amp = device.type == "cuda"
    # bfloat16 has float32's range (no overflow) where the GPU supports it (L4, A100); T4 falls back to float16.
    amp_dtype = torch.bfloat16 if (lora and use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    cfg = {**vars(a), **model_cfg, "device": str(device), "n_train": len(train_ds), "n_val": len(val_ds)}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    tracker = Tracker(a.wandb, run_dir, a.run_id, cfg, tags=["audio", model_cfg["kind"], fdir.name])
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} params={sum(p.numel() for p in model.parameters()):,} "
          f"trainable={sum(p.numel() for p in trainable):,}" + (f" amp={str(amp_dtype).split('.')[-1]}" if use_amp else ""))
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    def trainable_state():   # LoRA runs save only what trained (LoRA + head); the frozen BEATs comes from its checkpoint
        return {k: v for k, v in model.state_dict().items() if not lora or k in trainable_names}

    best, bad, history = -1.0, 0, []
    if lora:   # before any training the model equals the warm-start run: a wiring check
        m0 = metrics(*predict(model, val_dl, device)[:3])
        print(f"ep 00 (before training) | val sec-acc {m0['second_accuracy']:.3f} sec-F1 {m0['second_macro_f1']:.3f} "
              f"clip-acc {m0['clip_accuracy']:.3f} clip-F1 {m0['clip_macro_f1']:.3f}")
        tracker.log({"epoch": 0, **{f"val_{k}": v for k, v in m0.items() if k != "n_clips"}}, step=0)
        best = m0["second_macro_f1"]   # the warm start is a candidate: LoRA must beat it to replace it
        torch.save({"model": trainable_state(), "config": model_cfg, "epoch": 0, "val": m0}, run_dir / "best.pt")
    for epoch in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total, n = 0.0, 0
        for x, y, _ in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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
            torch.save({"model": trainable_state(), "config": model_cfg, "epoch": epoch, "val": m}, run_dir / "best.pt")
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

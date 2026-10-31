"""Train a linear probe (or MLP probe) on top of frozen CLIP ViT-B/16 visual features.

Reads pre-extracted features from:
  _local/drive/cache/clip_features/{split}_clip_emb.npy

Complies with the AVE fusion contract:
  Per-second visual embeddings: (N, 10, 512)
  Per-second probabilities:     (N, 10, 29)

Usage:
  python -m vision.clip_train_probe --run-id clip_linear_v1 --epochs 25 --lr 1e-3 --export
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vision import config as C
from vision.clip_extract import CACHE_DIR
from vision.clip_model import CLIPLinearProbe
from vision.data_ave import class_names
from vision.evaluate import metrics, pick_device, save_confusion


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(y: np.ndarray, mode: str) -> torch.Tensor:
    if mode == "none":
        return None
    counts = np.bincount(y.ravel(), minlength=C.N_CLASSES).astype(np.float64)
    w = np.sqrt(counts.sum() / (C.N_CLASSES * np.maximum(counts, 1)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class MLPProbe(nn.Module):
    """2-layer non-linear probe for CLIP embeddings."""
    def __init__(self, emb_dim: int = 512, hidden_dim: int = 512, n_classes: int = 29, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, emb: torch.Tensor):
        logits = self.net(emb)
        return logits, emb


def load_cached_split(split: str):
    emb_path = CACHE_DIR / f"{split}_clip_emb.npy"
    y_path = C.FEATURE_DIR / f"{split}_y.npy"
    yclip_path = C.FEATURE_DIR / f"{split}_yclip.npy"
    ids_path = C.FEATURE_DIR / f"{split}_ids.csv"

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Cached CLIP embeddings not found at {emb_path}. "
            f"Run `python -m vision.clip_extract --splits {split}` first."
        )

    emb = np.load(emb_path).astype(np.float32)  # (N, 10, 512)
    y = np.load(y_path)                        # (N, 10)
    yclip = np.load(yclip_path)                # (N,)
    df_ids = pd.read_csv(ids_path)
    return emb, y, yclip, df_ids


def evaluate_probe(model: nn.Module, emb: np.ndarray, y: np.ndarray, yclip: np.ndarray, device: torch.device):
    model.eval()
    t_emb = torch.from_numpy(emb).to(device)
    with torch.no_grad():
        logits, _ = model(t_emb)
        probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    res = metrics(probs, y, yclip)
    return res, probs


def export_probe_artifacts(
    run_id: str,
    model: nn.Module,
    splits_data: dict,
    test_metrics: dict,
    device: torch.device
):
    print(f"\n--- Exporting final multimodal assets for {run_id} ---", flush=True)
    vision_final_dir = C.REPO_ROOT / "vision_final" / run_id
    embed_dir = C.get_embed_dir(run_id)
    vision_final_dir.mkdir(parents=True, exist_ok=True)
    embed_dir.mkdir(parents=True, exist_ok=True)

    names = class_names()
    with open(vision_final_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)
    with open(embed_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(names, f, indent=2)

    for split, (emb, y, yclip, df_ids) in splits_data.items():
        _, probs = evaluate_probe(model, emb, y, yclip, device)

        # Save to vision_final
        np.save(vision_final_dir / f"{split}_emb.npy", emb.astype(np.float16))
        np.save(vision_final_dir / f"{split}_probs.npy", probs.astype(np.float16))
        df_ids.to_csv(vision_final_dir / f"{split}_ids.csv", index=False)

        # Mirror to drive embeddings
        np.save(embed_dir / f"{split}_emb.npy", emb.astype(np.float16))
        np.save(embed_dir / f"{split}_probs.npy", probs.astype(np.float16))
        df_ids.to_csv(embed_dir / f"{split}_ids.csv", index=False)
        print(f"  [{split}] Exported {emb.shape} to {vision_final_dir}", flush=True)

    source_content = (
        f"model: openai/clip-vit-base-patch16\n"
        f"probe: linear\n"
        f"run_id: {run_id}\n"
        f"emb_dim: 512\n"
        f"n_classes: 29\n"
        f"sampling: 1 fps (10 frames/video), 224x224, CLIP normalized\n"
        f"test_clip_acc: {test_metrics['clip_accuracy']:.4f}\n"
        f"test_clip_f1: {test_metrics['clip_macro_f1']:.4f}\n"
        f"test_sec_acc: {test_metrics['second_accuracy']:.4f}\n"
        f"test_sec_f1: {test_metrics['second_macro_f1']:.4f}\n"
    )
    (vision_final_dir / "SOURCE.txt").write_text(source_content, encoding="utf-8")
    (embed_dir / "SOURCE.txt").write_text(source_content, encoding="utf-8")

    readme_content = f"""# Vision Embeddings Handoff ({run_id})

- **Backbone**: `openai/clip-vit-base-patch16` (Frozen, 512-D projection)
- **Classifier**: Linear Probe (512 -> 29) trained on AVE training split
- **Test Performance**:
  - Clip Accuracy: **{test_metrics['clip_accuracy'] * 100:.2f}%**
  - Clip Macro-F1: **{test_metrics['clip_macro_f1']:.4f}**
  - Second Accuracy: **{test_metrics['second_accuracy'] * 100:.2f}%**
  - Second Macro-F1: **{test_metrics['second_macro_f1']:.4f}**

## File Formats
- `{{split}}_emb.npy`: (N, 10, 512) float16 visual embeddings
- `{{split}}_probs.npy`: (N, 10, 29) float16 per-second class probabilities
- `{{split}}_ids.csv`: sample_id, clip_id, order
- `class_names.json`: 29 class ordering (28 events + background at index 28)
"""
    (vision_final_dir / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"Assets and documentation written to {vision_final_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="clip_linear_v1")
    ap.add_argument("--probe-type", choices=["linear", "mlp"], default="linear")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--balance", choices=["none", "sqrt"], default="sqrt")
    ap.add_argument("--export", action="store_true", default=True)
    ap.add_argument("--device", default="auto", choices=["auto", "xpu", "cpu"])
    args = ap.parse_args()

    seed_all(C.SEED)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "xpu":
        device = torch.device("xpu")
    else:
        device = pick_device()
    print(f"Using compute device: {device}", flush=True)

    run_dir = C.get_run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cached CLIP embeddings...", flush=True)
    train_emb, train_y, train_yclip, train_ids = load_cached_split("train")
    val_emb, val_y, val_yclip, val_ids = load_cached_split("val")
    test_emb, test_y, test_yclip, test_ids = load_cached_split("test")

    print(f"Loaded: train {train_emb.shape}, val {val_emb.shape}, test {test_emb.shape}", flush=True)

    # Tensor datasets and data loaders
    train_ds = TensorDataset(torch.from_numpy(train_emb), torch.from_numpy(train_y))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)

    # Instantiate model
    if args.probe_type == "linear":
        model = CLIPLinearProbe(emb_dim=512, n_classes=C.N_CLASSES, dropout=args.dropout).to(device)
    else:
        model = MLPProbe(emb_dim=512, hidden_dim=512, n_classes=C.N_CLASSES, dropout=args.dropout).to(device)

    # Loss and optimizer
    weights = class_weights(train_y, args.balance)
    if weights is not None:
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, foreach=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_f1 = -1.0
    history = []
    t0 = time.time()

    print(f"\n--- Training {args.probe_type.upper()} Probe for {args.epochs} epochs ---", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_emb, batch_y in train_loader:
            batch_emb = batch_emb.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_emb)
            loss = criterion(logits.view(-1, C.N_CLASSES), batch_y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Evaluate on val
        val_m, _ = evaluate_probe(model, val_emb, val_y, val_yclip, device)

        rec = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_clip_accuracy": val_m["clip_accuracy"],
            "val_clip_macro_f1": val_m["clip_macro_f1"],
            "val_second_accuracy": val_m["second_accuracy"],
            "val_second_macro_f1": val_m["second_macro_f1"],
            "lr": scheduler.get_last_lr()[0]
        }
        history.append(rec)

        print(
            f"Ep {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f} | "
            f"Val Sec Acc: {val_m['second_accuracy']*100:.2f}% | "
            f"Val Sec F1: {val_m['second_macro_f1']:.4f} | "
            f"Val Clip Acc: {val_m['clip_accuracy']*100:.2f}% | "
            f"Val Clip F1: {val_m['clip_macro_f1']:.4f}",
            flush=True
        )

        if val_m["second_macro_f1"] > best_val_f1:
            best_val_f1 = val_m["second_macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_m,
                "probe_type": args.probe_type,
                "emb_dim": 512,
            }, run_dir / "best.pt")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    print(f"\nTraining completed in {time.time() - t0:.1f}s. Best Val Sec F1: {best_val_f1:.4f}", flush=True)

    # Load best model for test evaluation
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']}", flush=True)

    test_m, test_probs = evaluate_probe(model, test_emb, test_y, test_yclip, device)
    print("\n" + "="*50, flush=True)
    print(f"TEST BENCHMARK RESULTS ({args.run_id}):", flush=True)
    print(f"  Clip Accuracy:    {test_m['clip_accuracy'] * 100:.2f}%", flush=True)
    print(f"  Clip Macro-F1:    {test_m['clip_macro_f1']:.4f}", flush=True)
    print(f"  Second Accuracy:  {test_m['second_accuracy'] * 100:.2f}%", flush=True)
    print(f"  Second Macro-F1:  {test_m['second_macro_f1']:.4f}", flush=True)
    print("="*50 + "\n", flush=True)

    # Save test confusion matrix
    clip_pred = test_probs[:, :, :C.N_EVENTS].mean(1).argmax(-1)
    save_confusion(test_yclip, clip_pred, class_names(), run_dir / "confusion_test.png")

    results_data = {
        "val": ckpt["val_metrics"],
        "test": test_m,
        "config": vars(args),
        "best_epoch": ckpt["epoch"]
    }
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    if args.export:
        splits_data = {
            "train": (train_emb, train_y, train_yclip, train_ids),
            "val": (val_emb, val_y, val_yclip, val_ids),
            "test": (test_emb, test_y, test_yclip, test_ids),
        }
        export_probe_artifacts(args.run_id, model, splits_data, test_m, device)


if __name__ == "__main__":
    main()

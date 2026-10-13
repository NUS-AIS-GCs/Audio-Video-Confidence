"""Metrics for the vision baseline, and the test-set evaluation.

Per second (29 classes, AVE's labelling):  accuracy, macro-F1
Per clip   (28 classes):                   accuracy, macro-F1 from mean softmax over seconds, background excluded

Run: python -m vision.evaluate --run-id <id>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def f1_score(y_true, y_pred, labels=None, average="macro", zero_division=0):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        f1s = []
        for c in labels:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            denom = 2 * tp + fp + fn
            f1s.append(2.0 * tp / denom if denom > 0 else float(zero_division))
        return float(np.mean(f1s)) if f1s else 0.0

    def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
        n = len(labels)
        idx_map = {lbl: i for i, lbl in enumerate(labels)}
        cm = np.zeros((n, n), dtype=np.float64)
        for t, p in zip(y_true, y_pred):
            if t in idx_map and p in idx_map:
                cm[idx_map[t], idx_map[p]] += 1
        if normalize == "true":
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        return cm

from vision import config as C
from vision.data_ave import class_names
from vision.dataset import AVEVisual
from vision.model import VisionModel
from vision.tracking import Tracker


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device, desc: str = None):
    """Returns per-second probabilities (N,10,29), labels (N,10), clip labels (N,), embeddings (N,10,D)."""
    model.eval()
    P, Y, YC, E = [], [], [], []
    iterator = loader
    if desc and len(loader) > 1:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
        except ImportError:
            pass
    for x, y, yc in iterator:
        logits, emb = model(x.to(device))
        P.append(torch.softmax(logits.float(), -1).cpu())
        E.append(emb.float().cpu())
        Y.append(y)
        YC.append(yc)
    return (
        torch.cat(P).numpy(),
        torch.cat(Y).numpy(),
        torch.cat(YC).numpy(),
        torch.cat(E).numpy(),
    )


def metrics(P: np.ndarray, Y: np.ndarray, YC: np.ndarray) -> dict:
    sec_pred, sec_true = P.argmax(-1).ravel(), Y.ravel()
    clip_pred = P[:, :, : C.N_EVENTS].mean(1).argmax(-1)
    return {
        "second_accuracy": float(accuracy_score(sec_true, sec_pred)),
        "second_macro_f1": float(f1_score(sec_true, sec_pred, labels=list(range(C.N_CLASSES)), average="macro", zero_division=0)),
        "clip_accuracy": float(accuracy_score(YC, clip_pred)),
        "clip_macro_f1": float(f1_score(YC, clip_pred, labels=list(range(C.N_EVENTS)), average="macro", zero_division=0)),
        "n_clips": int(len(YC)),
    }


def save_confusion(YC: np.ndarray, clip_pred: np.ndarray, names: list, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(YC, clip_pred, labels=list(range(C.N_EVENTS)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.N_EVENTS))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(C.N_EVENTS))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Vision baseline: clip-level confusion (test, row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load_model(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    cfg = ckpt["config"]
    model = VisionModel(
        n_classes=C.N_CLASSES,
        emb_dim=cfg.get("emb_dim", C.EMB_DIM),
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--wandb", action="store_true", help="add test results to the training run in W&B")
    a = ap.parse_args()

    run_dir = C.ARTIFACT_DIR / a.run_id
    device = pick_device()
    model = load_model(run_dir, device)
    names = class_names()

    results = {}
    for split in ("val", "test"):
        loader = DataLoader(AVEVisual(split), batch_size=a.batch, num_workers=a.workers)
        P, Y, YC, _ = predict(model, loader, device)
        results[split] = metrics(P, Y, YC)
        if split == "test":
            save_confusion(YC, P[:, :, : C.N_EVENTS].mean(1).argmax(-1), names, run_dir / "confusion_test.png")

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    try:
        tracker = Tracker(a.wandb, run_dir, a.run_id, resume=True)
        tracker.summary({f"{s}_{k}": v for s, m in results.items() for k, v in m.items()})
        tracker.image("confusion_test", run_dir / "confusion_test.png")
    finally:
        tracker.finish()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

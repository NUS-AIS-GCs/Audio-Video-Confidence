"""Metrics for the audio baseline, and the test-set evaluation.

Per second (29 classes, AVE's labelling):  accuracy, macro-F1
Per clip   (28 classes):                   accuracy, macro-F1 from mean softmax over seconds, background excluded

Run: python -m audio.evaluate --run-id <id>
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from audio import config as C
from audio.dataset import AVEAudio, norm_stats
from audio.model import AudioCNN
from audio.tracking import Tracker


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    """Returns per-second probabilities (N,10,29), labels (N,10), clip labels (N,), embeddings (N,10,D)."""
    model.eval()
    P, Y, YC, E = [], [], [], []
    for x, y, yc in loader:
        logits, emb = model(x.to(device))
        P.append(torch.softmax(logits.float(), -1).cpu()); E.append(emb.float().cpu())
        Y.append(y); YC.append(yc)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(YC).numpy(), torch.cat(E).numpy()


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


def save_confusion(YC, clip_pred, names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(YC, clip_pred, labels=list(range(C.N_EVENTS)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(C.N_EVENTS)); ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticks(range(C.N_EVENTS)); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Audio baseline: clip-level confusion (test, row-normalised)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def load_model(run_dir, device):
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    model = AudioCNN(emb_dim=ckpt["config"]["emb_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--wandb", action="store_true", help="add test results to the training run in W&B")
    a = ap.parse_args()
    run_dir = C.ARTIFACT_DIR / a.run_id
    device = pick_device()
    model = load_model(run_dir, device)
    names = norm_stats()[2]
    results = {}
    for split in ("val", "test"):
        P, Y, YC, _ = predict(model, DataLoader(AVEAudio(split), batch_size=a.batch), device)
        results[split] = metrics(P, Y, YC)
        if split == "test":
            save_confusion(YC, P[:, :, : C.N_EVENTS].mean(1).argmax(-1), names, run_dir / "confusion_test.png")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    tracker = Tracker(a.wandb, run_dir, a.run_id, resume=True)
    tracker.summary({f"{s}_{k}": v for s, m in results.items() for k, v in m.items()})
    tracker.image("confusion_test", run_dir / "confusion_test.png")
    tracker.finish()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

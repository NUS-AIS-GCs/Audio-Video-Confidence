"""Zero-shot evaluation of CLIP ViT-B/16 on AVE dataset.

Evaluates test (or val/train) splits without any fine-tuning.
Computes:
  - Clip accuracy (28 events)
  - Clip macro-F1 (28 events)
  - Per-second accuracy (29 classes)
  - Per-second macro-F1 (29 classes)

Usage:
  python -m vision.clip_zero_shot [--split test] [--batch 8] [--run-id clip_zeroshot_v1]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vision import config as C
from vision.clip_model import CLIPZeroShotModel
from vision.data_ave import class_names
from vision.dataset import AVEVisual
from vision.evaluate import metrics, pick_device, save_confusion


def evaluate_split(
    model: CLIPZeroShotModel,
    split: str,
    batch_size: int,
    device: torch.device
):
    print(f"\n--- Evaluating zero-shot CLIP ViT-B/16 on {split} split ---")
    ds = AVEVisual(split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    P, Y, YC, E = [], [], [], []
    t0 = time.time()
    
    with torch.no_grad():
        for step, (x, y, yc) in enumerate(loader):
            # x: (B, 10, 3, 224, 224)
            logits, emb = model(x.to(device))
            probs = torch.softmax(logits.float(), dim=-1).cpu()
            
            P.append(probs)
            E.append(emb.cpu())
            Y.append(y)
            YC.append(yc)

            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                elapsed = time.time() - t0
                clips_done = min((step + 1) * batch_size, len(ds))
                print(f"  [{clips_done}/{len(ds)} clips] in {elapsed:.1f}s ({clips_done / elapsed:.1f} clips/s)")

    P = torch.cat(P).numpy()
    E = torch.cat(E).numpy()
    Y = torch.cat(Y).numpy()
    YC = torch.cat(YC).numpy()

    eval_results = metrics(P, Y, YC)
    return eval_results, P, E, Y, YC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="clip_zeroshot_v1")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--export", action="store_true", help="Export embeddings and probs to vision_final/ and drive")
    args = ap.parse_args()

    device = pick_device()
    print(f"Using device: {device}")

    run_dir = C.get_run_dir(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP ViT-B/16 ZeroShot Model...")
    model = CLIPZeroShotModel(device=device).to(device)

    splits_to_eval = C.SPLITS if args.split == "all" else [args.split]
    all_metrics = {}

    for s in splits_to_eval:
        res, P, E, Y, YC = evaluate_split(model, s, args.batch, device)
        all_metrics[s] = res
        print(f"\n{s.upper()} RESULTS:")
        print(f"  Clip Accuracy:      {res['clip_accuracy'] * 100:.2f}%")
        print(f"  Clip Macro-F1:      {res['clip_macro_f1']:.4f}")
        print(f"  Second Accuracy:    {res['second_accuracy'] * 100:.2f}%")
        print(f"  Second Macro-F1:    {res['second_macro_f1']:.4f}")

        # Save confusion matrix for test split
        if s == "test":
            clip_pred = P[:, :, : C.N_EVENTS].mean(1).argmax(-1)
            save_confusion(YC, clip_pred, class_names(), run_dir / "confusion_test.png")

        # Save intermediate arrays for the run
        if args.export:
            embed_dir = C.get_embed_dir(args.run_id)
            embed_dir.mkdir(parents=True, exist_ok=True)
            np.save(embed_dir / f"{s}_emb.npy", E.astype(np.float16))
            np.save(embed_dir / f"{s}_probs.npy", P.astype(np.float16))

    # Save results.json
    results_path = run_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

"""Package a trained audio run as a self-contained inference bundle for the app (see audio/infer.py).

Run: python -m audio.export_inference --run-id beats_lora_r8_v1 [--parity-samples 5]
Writes DRIVE_ROOT/inference/audio/ (or --out): weights.pt, model_card.json, norm_stats.json,
class_names.json, encoder.pt (BEATs checkpoint copy, BEATs-based models only), environment.json
(library versions the bundle was built with), README.md.
--parity-samples N also writes parity_samples/: N TRAIN clips (raw .mp4) with their rows from
embeddings/audio_final, and the live-vs-export differences measured with this bundle.
Copy that folder next to the app (default lookup: <repo>/_models/audio) or point AVAL_AUDIO_BUNDLE at it.
"""
import argparse
import datetime as dt
import json
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audio import config as C
from audio.beats_extract import find_checkpoint, sha256
from audio.dataset import feature_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default=None, help="bundle folder (default: DRIVE_ROOT/inference/audio)")
    ap.add_argument("--parity-samples", type=int, default=0, help="TRAIN clips to include for raw-to-export checks")
    a = ap.parse_args()
    run_dir = C.ARTIFACT_DIR / a.run_id
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu")
    cfg = {"kind": "cnn", **ckpt["config"]}
    fdir = feature_dir(cfg.get("features"))
    out = Path(a.out) if a.out else C.DRIVE_ROOT / "inference" / "audio"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    torch.save({"model": ckpt["model"], "config": cfg}, out / "weights.pt")
    stats = json.loads((fdir / "norm_stats.json").read_text())
    (out / "norm_stats.json").write_text(json.dumps({"mean": stats["mean"], "std": stats["std"]}))
    (out / "class_names.json").write_text(json.dumps(stats["class_names"] + ["background"], indent=1))
    encoder = None
    if cfg["kind"] in ("head", "beats_lora"):
        encoder = cfg.get("encoder") or fdir.name.removeprefix("beats_")
        shutil.copy(find_checkpoint(encoder), out / "encoder.pt")

    (out / "environment.json").write_text(json.dumps(environment(), indent=2))

    res = run_dir / "results.json"
    results = json.loads(res.read_text()) if res.exists() else {}
    card = {"run_id": a.run_id, "config": cfg, "encoder": encoder, "features": fdir.name,
            "best_epoch": ckpt.get("epoch"), "val": results.get("val"), "test": results.get("test"),
            "sample_rate": C.SR, "clip_seconds": C.CLIP_SECONDS, "created": dt.date.today().isoformat(),
            "sha256": {p.name: sha256(p) for p in sorted(out.glob("*")) if p.suffix in (".pt", ".json")}}
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    test = card["test"] or {}
    (out / "README.md").write_text(
        f"# Audio inference bundle\n\nRun `{a.run_id}` ({cfg['kind']}). Test: clip accuracy {test.get('clip_accuracy', 'n/a')}, "
        f"per-second macro-F1 {test.get('second_macro_f1', 'n/a')}.\n\n"
        "Load with `audio.infer.AudioPredictor(<this folder>)`; the app's audio branch looks for it at "
        "`<repo>/_models/audio` or the path in `AVAL_AUDIO_BUNDLE`.\n")
    if a.parity_samples:
        write_parity_samples(out, a.parity_samples)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"bundle -> {out} ({size:.0f} MB)")


def environment() -> dict:
    """Versions that affect the numbers: preprocessing (torchaudio fbank, ffmpeg resampling) and the model."""
    import torchaudio
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    return {"python": platform.python_version(), "torch": torch.__version__, "torchaudio": torchaudio.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "ffmpeg": ff,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_parity_samples(out: Path, n: int, split: str = "train") -> None:
    """n TRAIN clips from different classes: raw video + their exported rows + measured live-vs-export gap."""
    from audio.infer import AudioPredictor
    final = C.EMBED_ROOT / "audio_final"
    ids = pd.read_csv(final / f"{split}_ids.csv")
    P, E = np.load(final / f"{split}_probs.npy"), np.load(final / f"{split}_emb.npy")
    videos = {v.stem: v for v in C.AVE_ROOT.rglob("*.mp4")}
    pick = ids[ids.clip_id.isin(videos)].drop_duplicates("category").sample(n, random_state=C.SEED).index
    pdir = out / "parity_samples"
    pdir.mkdir()
    predictor = AudioPredictor(out)
    rows = []
    for i in pick:
        r = ids.loc[i]
        shutil.copy(videos[r.clip_id], pdir / f"{r.clip_id}.mp4")
        live = predictor.predict_video(pdir / f"{r.clip_id}.mp4")
        rows.append({**r.to_dict(), "row_in_export": int(i),
                     "max_abs_diff_probs": float(np.abs(live.probs - P[i].astype(np.float32)).max()),
                     "max_abs_diff_emb": float(np.abs(live.embeddings - E[i].astype(np.float32)).max()),
                     "same_second_labels": bool((live.probs.argmax(1) == P[i].argmax(1)).all())})
    sel = pd.DataFrame(rows)
    sel.to_csv(pdir / "samples.csv", index=False)
    np.save(pdir / "emb.npy", E[pick])                  # (n, 10, 256) float16, rows follow samples.csv
    np.save(pdir / "probs.npy", P[pick])                # (n, 10, 29) float16
    (pdir / "README.md").write_text(
        f"# Raw-to-export parity samples ({split})\n\n`samples.csv` row k = `emb.npy[k]` / `probs.npy[k]`, copied from "
        f"`embeddings/audio_final/{split}_*` (`row_in_export`). Running `AudioPredictor(<bundle>).predict_video(<clip_id>.mp4)` "
        "should reproduce them up to float16 storage rounding (about 1e-3); the measured gaps are in samples.csv.\n")
    print(sel[["sample_id", "category", "max_abs_diff_probs", "max_abs_diff_emb", "same_second_labels"]].to_string(index=False))


if __name__ == "__main__":
    main()

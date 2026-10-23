"""Package a trained audio run as a self-contained inference bundle for the app (see audio/infer.py).

Run: python -m audio.export_inference --run-id beats_lora_r8_v1
Writes DRIVE_ROOT/inference/audio/ (or --out): weights.pt, model_card.json, norm_stats.json,
class_names.json, encoder.pt (BEATs checkpoint copy, BEATs-based models only), README.md.
Copy that folder next to the app (default lookup: <repo>/_models/audio) or point AVAL_AUDIO_BUNDLE at it.
"""
import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import torch

from audio import config as C
from audio.beats_extract import find_checkpoint, sha256
from audio.dataset import feature_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default=None, help="bundle folder (default: DRIVE_ROOT/inference/audio)")
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
    size = sum(p.stat().st_size for p in out.glob("*")) / 1e6
    print(f"bundle -> {out} ({size:.0f} MB)")


if __name__ == "__main__":
    main()

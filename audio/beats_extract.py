"""Frozen BEATs: one 768-D embedding per second for every AVE sample, cached per split.

BEATs turns 10 s of 16 kHz audio into 62 time steps x 8 frequency patches (each patch = 16 frames of
10 ms = 0.16 s). We average the 8 frequency patches, then average the time steps whose centre falls in
each second, giving (10, 768) per sample: the same per-second layout as the log-Mel features.

Output in CACHE_ROOT/beats_<ckpt>/:
  {split}_X.npy (N, 10, 768) float16, {split}_y.npy, {split}_yclip.npy, {split}_ids.csv, norm_stats.json

Run: python -m audio.beats_extract --ckpt iter3        (main result, checkpoint downloaded manually)
     python -m audio.beats_extract --ckpt iter3_as2m   (comparison, downloaded automatically)
"""
import argparse
import hashlib
import json
import urllib.request

import numpy as np
import torch
from tqdm import tqdm

from audio import config as C
from audio.beats import BEATs, BEATsConfig
from audio.data_ave import class_names
from audio.features import load_wav, save_labels, split_table

FREQ_PATCHES = 8          # 128 mel bins / 16
PATCH_FRAMES = 16         # frames per time patch
FRAME_SECONDS = 0.01      # BEATs fbank hop


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_checkpoint(name: str):
    spec = C.BEATS[name]
    hits = [p for p in sorted(C.PRETRAINED_DIR.glob(spec["glob"]))
            if not any(x in p.name.lower() for x in spec["exclude"])]
    if hits:
        return hits[0]
    if not spec["url"]:
        raise SystemExit(
            f"BEATs '{name}' checkpoint not found in {C.PRETRAINED_DIR}.\n"
            f"Download it in your browser from the link in audio/config.py (BEATS comment) and upload it there, "
            f"keeping a name that starts with 'BEATs_iter3'.")
    C.PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)
    dest = C.PRETRAINED_DIR / spec["glob"]
    print(f"Downloading {name} from {spec['url']} ...")
    urllib.request.urlretrieve(spec["url"], dest)
    return dest


def load_beats(name: str, device) -> BEATs:
    path = find_checkpoint(name)
    spec = C.BEATS[name]
    if spec["sha256"]:
        got = sha256(path)
        if got != spec["sha256"]:
            raise SystemExit(f"{path.name}: sha256 {got[:12]}... does not match the expected {spec['sha256'][:12]}...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(ckpt["cfg"])
    if cfg.finetuned_model:
        raise SystemExit(f"{path.name} is a fine-tuned classifier, not a pre-trained encoder. Use the pre-trained checkpoint.")
    model = BEATs(cfg)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded BEATs '{name}' from {path.name}: {cfg.encoder_layers} layers, {cfg.encoder_embed_dim}-D")
    return model.to(device).eval()


def pool_to_seconds(x: torch.Tensor) -> torch.Tensor:
    """(B, T*8, D) BEATs tokens -> (B, 10, D): mean over frequency, then over the time steps in each second."""
    b, n, d = x.shape
    t = n // FREQ_PATCHES
    x = x[:, : t * FREQ_PATCHES].reshape(b, t, FREQ_PATCHES, d).mean(2)          # tokens are time-major
    centres = (torch.arange(t, device=x.device) * PATCH_FRAMES + PATCH_FRAMES / 2) * FRAME_SECONDS
    sec = centres.floor().long().clamp(max=C.CLIP_SECONDS - 1)
    out = torch.zeros(b, C.CLIP_SECONDS, d, device=x.device, dtype=x.dtype).index_add_(1, sec, x)
    counts = torch.bincount(sec, minlength=C.CLIP_SECONDS).clamp(min=1).to(x.dtype)
    return out / counts[None, :, None]


@torch.no_grad()
def embed_split(model, split: str, out_dir, device, batch: int) -> None:
    df = split_table(split)
    X = np.zeros((len(df), C.CLIP_SECONDS, model.cfg.encoder_embed_dim), dtype=np.float16)
    ids = df["clip_id"].tolist()
    for i in tqdm(range(0, len(ids), batch), desc=f"BEATs {split}"):
        wav = torch.from_numpy(np.stack([load_wav(C.WAV_DIR / f"{c}.wav") for c in ids[i:i + batch]])).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            tokens, _ = model.extract_features(wav, padding_mask=torch.zeros_like(wav, dtype=torch.bool))
        X[i:i + len(wav)] = pool_to_seconds(tokens.float()).cpu().numpy()
    np.save(out_dir / f"{split}_X.npy", X)
    save_labels(df, split, out_dir)
    print(f"{split}: X {X.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", choices=sorted(C.BEATS), required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from audio.evaluate import pick_device
    device = pick_device()
    out_dir = C.CACHE_ROOT / f"beats_{a.ckpt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in C.SPLITS if a.force or not (out_dir / f"{s}_X.npy").exists()]
    if todo:
        model = load_beats(a.ckpt, device)
        for split in todo:
            embed_split(model, split, out_dir, device, a.batch)
    else:
        print("all splits cached, skipping")
    X = np.load(out_dir / "train_X.npy").astype(np.float32)
    X = X.reshape(-1, X.shape[-1])                      # every training second, per embedding dimension
    stats = {"mean": X.mean(0).tolist(), "std": (X.std(0) + 1e-5).tolist(), "class_names": class_names(),
             "encoder": f"BEATs {a.ckpt}", "dim": int(X.shape[-1])}
    (out_dir / "norm_stats.json").write_text(json.dumps(stats))
    print(f"norm_stats.json written -> {out_dir}")


if __name__ == "__main__":
    main()

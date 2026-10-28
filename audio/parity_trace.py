"""Intermediate tensors of the audio pipeline for chosen clips, to tell decoder differences from device/precision ones.

Run: python -m audio.parity_trace --bundle <inference/audio> --videos a.mp4 b.mp4 ... --out trace/ [--device cpu|cuda]
Per clip it writes <clip_id>.npz (all float32 unless noted) and, once, trace_manifest.json:
  pcm_int16         (N,)        decoded wav samples exactly as ffmpeg wrote them (pcm_s16le), N = decoded length
  wav               (160000,)   float32 after soundfile read + pad/trim to 10 s (audio.features.load_wav)
  fbank             (T, 128)    BEATs filterbank (x32768 scaling and normalisation applied inside BEATs)
  tokens_pooled     (10, 768)   BEATs output averaged over frequency patches and pooled into 10 one-second bins
  tokens_normed     (10, 768)   after the frozen 768-D train mean/std
  emb               (10, 256)   final per-second embedding (what audio_final stores, there as float16)
  probs             (10, 29)
The manifest records ffmpeg and library versions, device, dtypes and the sha256 of every decoded wav.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio import config as C
from audio.beats_extract import pool_to_seconds
from audio.features import load_wav
from audio.infer import AudioPredictor


@torch.no_grad()
def trace_clip(predictor: AudioPredictor, video: Path, out_dir: Path) -> dict:
    m = predictor.model
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", str(C.SR), "-t", str(C.CLIP_SECONDS), str(wav_path)], check=True)
        pcm, sr = sf.read(wav_path, dtype="int16")
        wav = load_wav(wav_path)
    dev = predictor.device
    x = torch.from_numpy(wav)[None].to(dev)
    fbank = m.beats.preprocess(x.float())[0]
    pooled = pool_to_seconds(m.tokens(x).float())[0]
    normed = (pooled - m.mean) / m.std
    emb = m.head.embed(normed[None])[0]
    probs = torch.softmax(m.head.head(emb[None]).float(), -1)[0]
    arrays = dict(pcm_int16=pcm, wav=wav, fbank=fbank, tokens_pooled=pooled, tokens_normed=normed, emb=emb, probs=probs)
    arrays = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in arrays.items()}
    np.savez(out_dir / f"{video.stem}.npz", **arrays)
    return {"clip_id": video.stem, "decoded_samples": int(len(pcm)), "decoded_sample_rate": int(sr),
            "padded_samples": int(max(0, C.CLIP_SAMPLES - len(pcm))), "trimmed_samples": int(max(0, len(pcm) - C.CLIP_SAMPLES)),
            "wav_sha256": hashlib.sha256(wav.tobytes()).hexdigest(),
            "dtypes": {k: str(v.dtype) for k, v in arrays.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    predictor = AudioPredictor(a.bundle, device=a.device)
    if predictor.kind != "beats_lora":
        raise SystemExit("parity_trace supports BEATs + LoRA bundles")
    clips = [trace_clip(predictor, Path(v), out) for v in a.videos]
    ff = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    import torchaudio
    manifest = {"bundle_run": predictor.card["run_id"], "device": str(predictor.device), "ffmpeg": ff,
                "torch": torch.__version__, "torchaudio": torchaudio.__version__, "numpy": np.__version__,
                "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
                "autocast": "disabled (fp32 end to end)", "clips": clips}
    (out / "trace_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

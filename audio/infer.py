"""Live audio inference for one clip, from a self-contained inference bundle.

The predictor applies exactly the training-time preprocessing, so for an AVE clip its output equals that clip's
row in the offline export (`embeddings/audio_*`, up to float16 storage rounding). A fusion model trained on
the exports can therefore take live audio without changes.

Bundle (written by `python -m audio.export_inference`):
  model_card.json   run id, model config, scores, checksums
  weights.pt        trained weights (LoRA + head for BEATs + LoRA)
  norm_stats.json   input normalisation from training
  class_names.json  29 names: 28 AVE events alphabetically, then "background"
  encoder.pt        BEATs checkpoint (BEATs-based models only)

Usage:
  p = AudioPredictor("/path/to/inference/audio")
  out = p.predict_video("clip.mp4")      # or p.predict_waveform(wav_16k_mono)
  out.probs (10, 29), out.embeddings (10, 256), out.clip_label, out.second_labels
"""
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from audio import config as C


@dataclass
class AudioPrediction:
    probs: np.ndarray            # (10, 29) per-second class probabilities, columns = class_names
    embeddings: np.ndarray       # (10, 256) per-second embedding (fusion input)
    class_names: list            # 29 names, last = "background"

    @property
    def second_labels(self) -> list:
        """Most likely label for each second (may be 'background')."""
        return [self.class_names[i] for i in self.probs.argmax(1)]

    @property
    def presence(self) -> list:
        """1 where the most likely label of the second is an event (not background)."""
        return [int(i != C.BACKGROUND) for i in self.probs.argmax(1)]

    @property
    def clip_distribution(self) -> np.ndarray:
        """Clip-level event distribution over the 28 events: mean per-second probabilities, background
        excluded, renormalised. Its argmax is the clip label used in evaluation."""
        p = self.probs[:, : C.N_EVENTS].mean(0)
        return p / p.sum()

    @property
    def clip_label(self) -> str:
        return self.class_names[int(self.clip_distribution.argmax())]

    @property
    def clip_confidence(self) -> float:
        return float(self.clip_distribution.max())

    def top_k(self, k: int = 5) -> list:
        d = self.clip_distribution
        return [{"class": self.class_names[i], "prob": float(d[i])} for i in np.argsort(-d)[:k]]


def waveform_from_video(path, seconds: int = C.CLIP_SECONDS) -> np.ndarray:
    """First `seconds` of a video's audio as 16 kHz mono float32, padded with silence if shorter.
    Same ffmpeg settings as audio/extract_audio.py."""
    from audio.features import load_wav
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
               "-vn", "-ac", "1", "-ar", str(C.SR), "-t", str(seconds), str(wav)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not wav.exists():
            raise RuntimeError(f"Could not extract audio from {path}: {r.stderr.strip()[:300] or 'no audio stream'}")
        return load_wav(wav)


class AudioPredictor:
    def __init__(self, bundle_dir, device=None):
        from audio.evaluate import pick_device
        self.dir = Path(bundle_dir)
        self.card = json.loads((self.dir / "model_card.json").read_text())
        self.class_names = json.loads((self.dir / "class_names.json").read_text())
        stats = json.loads((self.dir / "norm_stats.json").read_text())
        self.device = torch.device(device) if device else pick_device()
        self.cfg = self.card["config"]
        self.kind = self.cfg["kind"]
        weights = torch.load(self.dir / "weights.pt", map_location="cpu")["model"]
        mean, std = torch.tensor(stats["mean"]), torch.tensor(stats["std"])

        if self.kind == "beats_lora":
            from audio.beats_lora import BeatsLoRA
            self.model = BeatsLoRA(self.cfg["encoder"], self.cfg["r"], self.cfg["alpha"], self.cfg["emb_dim"],
                                   device=self.device, beats_path=self.dir / "encoder.pt", stats=(mean, std))
            missing, unexpected = self.model.load_state_dict(weights, strict=False)
            bad = [k for k in missing if k not in ("mean", "std") and not (k.startswith("beats.") and "lora_" not in k)]
            if unexpected or bad:
                raise RuntimeError(f"weights.pt does not fit the model: unexpected={unexpected[:3]} missing={bad[:3]}")
        elif self.kind == "head":
            from audio.beats_extract import load_beats
            from audio.model import EmbeddingHead
            self.beats = load_beats(self.card["encoder"], self.device, path=self.dir / "encoder.pt")
            self.model = EmbeddingHead(in_dim=self.cfg["in_dim"], emb_dim=self.cfg["emb_dim"])
            self.model.load_state_dict(weights)
            self.mean, self.std = mean.to(self.device), std.to(self.device)
        elif self.kind == "cnn":
            from audio.model import AudioCNN
            self.model = AudioCNN(emb_dim=self.cfg["emb_dim"])
            self.model.load_state_dict(weights)
            self.mean, self.std = mean.view(1, -1, 1).to(self.device), std.view(1, -1, 1).to(self.device)
        else:
            raise ValueError(f"unknown model kind {self.kind!r}")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict_waveform(self, wav: np.ndarray) -> AudioPrediction:
        """wav: 16 kHz mono float32. Trimmed or padded to 10 s exactly as in training."""
        clip = np.zeros(C.CLIP_SAMPLES, dtype=np.float32)
        clip[: min(len(wav), C.CLIP_SAMPLES)] = wav[: C.CLIP_SAMPLES]
        x = torch.from_numpy(clip)[None].to(self.device)
        if self.kind == "beats_lora":
            logits, emb = self.model(x)
        elif self.kind == "head":
            from audio.beats_extract import pool_to_seconds
            tokens, _ = self.beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))
            sec = pool_to_seconds(tokens.float())
            # training saw the cached embeddings in float16, so round the same way
            sec = sec.half().float()
            logits, emb = self.model((sec - self.mean) / self.std)
        else:
            from audio.features import logmel_seconds
            m = torch.from_numpy(logmel_seconds(clip)).half().float().to(self.device)   # cache was float16
            logits, emb = self.model(((m - self.mean) / self.std).unsqueeze(1)[None])
        probs = torch.softmax(logits.float(), -1)[0].cpu().numpy()
        return AudioPrediction(probs=probs, embeddings=emb[0].float().cpu().numpy(), class_names=self.class_names)

    def predict_video(self, path) -> AudioPrediction:
        return self.predict_waveform(waveform_from_video(path))

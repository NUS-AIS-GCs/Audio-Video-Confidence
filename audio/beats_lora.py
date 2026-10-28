"""BEATs + LoRA: the pretrained encoder adapted with small trainable side paths, plus the per-second head.

waveform (B, 160000) -> BEATs filterbank (float32) -> BEATs transformer with LoRA on q/v projections
-> tokens pooled to (B, 10, 768) exactly as in beats_extract -> normalised with the frozen cache's stats
-> EmbeddingHead -> logits (B, 10, 29), embedding (B, 10, 256)

With the head warm-started from a frozen-BEATs run and LoRA's B matrices at zero, the model starts out
identical to that frozen run. Its "epoch 0" scores should match, which checks the wiring.
"""
import torch
from torch import nn

from audio import config as C
from audio.beats_extract import load_beats, pool_to_seconds
from audio.dataset import norm_stats
from audio.lora import LoRALinear, inject_lora
from audio.model import EmbeddingHead


class BeatsLoRA(nn.Module):
    def __init__(self, encoder: str, r: int, alpha: float, emb_dim: int, fdir=None, device="cpu",
                 beats_path=None, stats=None):
        """fdir: frozen embedding cache (training). For inference bundles pass beats_path and stats=(mean, std)."""
        super().__init__()
        self.beats = load_beats(encoder, device, path=beats_path)
        self.n_lora_layers = inject_lora(self.beats, ("q_proj", "v_proj"), r=r, alpha=alpha)
        mean, std = stats if stats is not None else norm_stats(fdir)[:2]   # 768-D stats of the frozen embedding cache
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.head = EmbeddingHead(in_dim=self.beats.cfg.encoder_embed_dim, emb_dim=emb_dim)
        self.emb_dim = emb_dim

    def train(self, mode: bool = True):
        """Frozen BEATs stays in eval mode (no dropout, no layer drop), so it produces the same features the
        frozen run's head was trained on; only LoRA's own dropout and the head's dropout act in training."""
        super().train(mode)
        self.beats.eval()
        for m in self.beats.modules():
            if isinstance(m, LoRALinear):
                m.drop.train(mode)
        return self

    def tokens(self, wav: torch.Tensor) -> torch.Tensor:
        """Same steps as BEATs.extract_features (no padding), with the filterbank kept in float32:
        BEATs scales audio to +-32768 first, and the squared values overflow float16."""
        b = self.beats
        with torch.autocast(device_type=wav.device.type, enabled=False):
            fbank = b.preprocess(wav.float())
        x = b.patch_embedding(fbank.unsqueeze(1))
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = b.layer_norm(x)
        if b.post_extract_proj is not None:
            x = b.post_extract_proj(x)
        x, _ = b.encoder(b.dropout_input(x), padding_mask=None)
        return x

    def embed(self, wav: torch.Tensor) -> torch.Tensor:
        sec = pool_to_seconds(self.tokens(wav).float())
        return self.head.embed((sec - self.mean) / self.std)

    def forward(self, wav: torch.Tensor):
        emb = self.embed(wav)
        return self.head.head(emb), emb


def build_beats_lora(cfg: dict, device="cpu") -> BeatsLoRA:
    from audio.dataset import feature_dir
    return BeatsLoRA(cfg["encoder"], cfg["r"], cfg["alpha"], cfg["emb_dim"], feature_dir(cfg["features"]), device)


def warm_start_head(model: BeatsLoRA, run_id: str) -> None:
    """Copy the head weights of a frozen-BEATs run (same cache, same head shape)."""
    ckpt = torch.load(C.ARTIFACT_DIR / run_id / "best.pt", map_location="cpu")
    if ckpt["config"].get("kind") != "head":
        raise SystemExit(f"--init-head {run_id} is not a frozen-embedding (head) run")
    model.head.load_state_dict(ckpt["model"])
    print(f"head warm-started from {run_id}")

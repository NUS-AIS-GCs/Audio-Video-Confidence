"""CLIP ViT-B/16 visual encoder and zero-shot / linear probe models for AVE.

Provides:
  - CLIPVisualEncoder: frozen or fine-tunable OpenAI CLIP ViT-B/16 (512-D projection)
  - CLIPZeroShotModel: zero-shot text-prompt classifier (no training required)
  - CLIPLinearProbe: lightweight trainable head on top of frozen CLIP embeddings
"""
from typing import List, Optional, Tuple
import torch
from torch import nn
from transformers import CLIPModel, CLIPTokenizer

from vision import config as C
from vision.data_ave import class_names

CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
EMB_DIM = 512

# Standard CLIP visual normalization
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)

# Hand-crafted event prompt templates for AVE classes
DEFAULT_PROMPTS = [
    "a photo or video frame of an accordion being played",
    "a photo or video frame of an acoustic guitar being played",
    "a photo or video frame of a baby crying or an infant crying",
    "a photo or video frame of a banjo being played",
    "a photo or video frame of a dog barking",
    "a photo or video frame of a bus on the street or road",
    "a photo or video frame of a cat meowing",
    "a photo or video frame of a chainsaw cutting wood or running",
    "a photo or video frame of a church bell ringing in a belfry or tower",
    "a photo or video frame of a clock or clock tower ticking",
    "a photo or video frame of a woman speaking or talking",
    "a photo or video frame of an airplane or fixed-wing aircraft",
    "a photo or video frame of a flute being played",
    "a photo or video frame of food frying or sizzling in a pan",
    "a photo or video frame of a goat bleating",
    "a photo or video frame of a helicopter flying or hovering",
    "a photo or video frame of a horse running or neighing",
    "a photo or video frame of a man speaking or talking",
    "a photo or video frame of a mandolin being played",
    "a photo or video frame of a motorcycle riding or revving",
    "a photo or video frame of a race car or auto racing on a track",
    "a photo or video frame of rodents, a rat or a mouse",
    "a photo or video frame of a person blowing a shofar horn",
    "a photo or video frame of a toilet flushing with water",
    "a photo or video frame of a train or train horn on tracks",
    "a photo or video frame of a large truck or semi-truck on the road",
    "a photo or video frame of a ukulele being played",
    "a photo or video frame of a violin or fiddle being played",
    "a background video scene without any event or sound source visible",
]


def build_text_embeddings(
    model: CLIPModel,
    tokenizer: CLIPTokenizer,
    device: Optional[torch.device] = None,
    prompts: Optional[List[str]] = None
) -> torch.Tensor:
    """Encode 29 class prompts into L2-normalized text vectors (29, 512)."""
    if prompts is None:
        prompts = DEFAULT_PROMPTS
    
    m_device = next(model.parameters()).device
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(m_device)
    with torch.no_grad():
        out = model.get_text_features(**inputs)
        text_feats = out.pooler_output if hasattr(out, "pooler_output") else out
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
    if device is not None:
        text_feats = text_feats.to(device)
    return text_feats


class CLIPZeroShotModel(nn.Module):
    """Zero-shot vision classifier using CLIP ViT-B/16 text-image cosine similarity.
    
    Complies with team contract:
      embed(x)   -> (B, 10, 512) float tensor
      forward(x) -> (logits (B, 10, 29), emb (B, 10, 512))
    """
    def __init__(
        self,
        model_name: str = CLIP_MODEL_NAME,
        prompts: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip.eval()

        # Freeze all CLIP parameters for zero-shot
        for p in self.clip.parameters():
            p.requires_grad = False

        self.prompts = prompts or DEFAULT_PROMPTS
        # Register text embeddings as persistent buffer
        text_feats = build_text_embeddings(self.clip, self.tokenizer, prompts=self.prompts)
        self.register_buffer("text_features", text_feats)  # (29, 512)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Extract L2-normalized 512-D visual features per frame.
        
        Args:
            x: (B, 10, 3, 224, 224) float tensor normalized or in [0..1]
        Returns:
            emb: (B, 10, 512) visual embeddings
        """
        b, s, c, h, w = x.shape
        flat_x = x.view(b * s, c, h, w)
        
        # Convert ImageNet-normalized or raw [0..1] input to exact CLIP normalization
        if flat_x.min() < -0.1:
            # Unnormalize from ImageNet stats back to [0..1]
            in_mean = torch.tensor([0.485, 0.456, 0.406], device=flat_x.device).view(1, 3, 1, 1)
            in_std = torch.tensor([0.229, 0.224, 0.225], device=flat_x.device).view(1, 3, 1, 1)
            flat_x = flat_x * in_std + in_mean
            flat_x = torch.clamp(flat_x, 0.0, 1.0)

        # Apply standard CLIP normalization
        clip_mean = CLIP_MEAN.to(flat_x.device).squeeze(0)
        clip_std = CLIP_STD.to(flat_x.device).squeeze(0)
        flat_x = (flat_x - clip_mean) / clip_std

        out = self.clip.get_image_features(pixel_values=flat_x)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.view(b, s, -1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute zero-shot cosine similarity logits and embeddings.
        
        Returns:
            logits: (B, 10, 29)
            emb:    (B, 10, 512)
        """
        emb = self.embed(x)  # (B, 10, 512)
        # Cosine similarity scaled by CLIP temperature
        logit_scale = self.clip.logit_scale.exp()
        logits = logit_scale * torch.matmul(emb, self.text_features.t())  # (B, 10, 29)
        return logits, emb


class CLIPLinearProbe(nn.Module):
    """Linear probe classification head on top of frozen 512-D CLIP visual embeddings."""
    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        n_classes: int = C.N_CLASSES,
        dropout: float = 0.1
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(emb_dim, n_classes)
        )

    def forward(self, emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """emb: (B, 10, 512) -> logits (B, 10, 29), emb (B, 10, 512)"""
        logits = self.head(emb)
        return logits, emb

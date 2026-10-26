"""Strict AVE inference boundaries; diagnostics are never predictive features."""
from dataclasses import dataclass, field
from typing import Mapping

import torch

from .contracts import ContractError

BIN_STARTS = tuple(float(t) for t in range(10))
PREPROCESSING_PROFILE = "audited-producer-code-v1;raw-export-parity-unverified"
PREPROCESSING_CONFIGURATION = (
    ("audio", "16kHz mono; historical ffmpeg -ac 1 -ar 16000 -t 10; "
     "BEATs iter3 fbank128/25ms/10ms; (fbank-15.41663)/(2*6.55582); "
     "q/v LoRA r8 alpha16; 8 frequency patches averaged then time-centre bins; "
     "frozen TRAIN 768D mean/std then 768->256 GELU head; stats/checkpoint digests unavailable"),
    ("vision", "ResNet18 fc=Identity; 512D projection Identity; 29-class head excluded from embedding; "
     "historical OpenCV int(t*fps), RGB, 224x224 INTER_LINEAR; /255; "
     "ImageNet mean .485,.456,.406 std .229,.224,.225; export augmentation disabled"),
    ("verification", "code and export declarations audited; actual decoder versions, checkpoint "
     "configuration, normalization statistics and raw-to-export parity unverified"),
)


@dataclass(frozen=True)
class EncoderProvenance:
    """Export evidence is not evidence of executable encoder checkpoint parity.

    Hash declarations at a caller boundary are assertions. The local export
    reader verifies bytes; arbitrary external tensors require a trusted producer.
    None explicitly represents the unavailable encoder checkpoint digest.
    """
    modality: str
    run_identity: str
    export_hashes: tuple[tuple[str, str], ...]
    preprocessing_profile: str = PREPROCESSING_PROFILE
    encoder_checkpoint_sha256: str | None = None
    origin: str = "validated_export"


@dataclass(frozen=True)
class FeatureBatch:
    audio: torch.Tensor
    visual: torch.Tensor
    audio_validity: torch.Tensor
    visual_validity: torch.Tensor
    provenance: tuple[EncoderProvenance, EncoderProvenance]
    audio_bin_starts: tuple[float, ...] = BIN_STARTS
    visual_bin_starts: tuple[float, ...] = BIN_STARTS
    duration_seconds: float = 10.0
    timing_basis: str = "export_bin_indices_assumed_synchronous"
    diagnostics: Mapping = field(default_factory=dict)

    def validate(self):
        for value, dim in ((self.audio, 256), (self.visual, 512)):
            if (type(value) is not torch.Tensor or value.ndim != 3 or value.shape[0] < 1
                    or tuple(value.shape[1:]) != (10, dim) or not value.is_floating_point()
                    or value.layout != torch.strided or value.device.type == "meta"
                    or not torch.isfinite(value).all()):
                raise ContractError(f"expected finite floating (B,10,{dim}) encoder output")
        if self.audio.shape[0] != self.visual.shape[0]:
            raise ContractError("modality batch sizes differ")
        for mask, value in ((self.audio_validity, self.audio), (self.visual_validity, self.visual)):
            if (type(mask) is not torch.Tensor or mask.dtype != torch.bool
                    or mask.layout != torch.strided
                    or mask.shape != value.shape[:2] or mask.device != value.device):
                raise ContractError("producer validity must be boolean (B,10) on feature device")
        if (type(self.audio_bin_starts) is not tuple or type(self.visual_bin_starts) is not tuple
                or any(type(t) not in (float, int) for t in self.audio_bin_starts + self.visual_bin_starts)
                or self.audio_bin_starts != BIN_STARTS or self.visual_bin_starts != BIN_STARTS
                or type(self.duration_seconds) not in (float, int) or self.duration_seconds != 10
                or self.timing_basis != "export_bin_indices_assumed_synchronous"):
            raise ContractError("AVE requires aligned original bins [t,t+1), t=0..9, duration 10 seconds")
        if (type(self.provenance) is not tuple or len(self.provenance) != 2
                or any(type(p) is not EncoderProvenance for p in self.provenance)
                or tuple(p.modality for p in self.provenance) != ("audio", "vision")):
            raise ContractError("explicit audio and vision encoder provenance required")
        mask = self.audio_validity.to("cpu") & self.visual_validity.to("cpu")
        if not mask.any(dim=1).all():
            raise ContractError("every pair needs a jointly valid bin; no fabrication or all-masked inference")
        return mask


@dataclass(frozen=True)
class RawMediaInput:
    """Proposed canonical raw boundary, not a decoder or runnable encoder.

    Upstream conversion must be explicit and audited. Full valid ten-second
    content is required: BEATs mixes context across bins before fusion masking.
    """
    waveform: torch.Tensor
    frames: torch.Tensor
    sample_rate: int
    audio_duration: float
    video_duration: float
    audio_start_seconds: float
    video_start_seconds: float
    frame_timestamps: tuple[float, ...]
    audio_validity: torch.Tensor | None = None
    visual_validity: torch.Tensor | None = None

    def validate(self):
        w, f = self.waveform, self.frames
        if (type(w) is not torch.Tensor or w.ndim != 2 or w.shape[0] < 1
                or w.shape[1] != 160000 or w.dtype != torch.float32
                or w.layout != torch.strided or w.device.type == "meta"
                or not torch.isfinite(w).all() or (w.abs() > 1).any()
                or type(self.sample_rate) is not int or self.sample_rate != 16000):
            raise ContractError("canonical audio requires finite mono float32 (B,160000), [-1,1], 16000 Hz")
        if (type(f) is not torch.Tensor or f.dtype != torch.uint8
                or f.layout != torch.strided or f.device.type == "meta"
                or tuple(f.shape) != (w.shape[0], 10, 224, 224, 3)):
            raise ContractError("canonical video requires RGB uint8 (B,10,224,224,3)")
        if (any(type(x) not in (int, float) for x in (self.audio_duration, self.video_duration,
                                                    self.audio_start_seconds, self.video_start_seconds))
                or self.audio_duration != 10 or self.video_duration != 10
                or self.audio_start_seconds != 0 or self.video_start_seconds != 0
                or type(self.frame_timestamps) is not tuple or self.frame_timestamps != BIN_STARTS
                or any(type(t) not in (float, int) for t in self.frame_timestamps)):
            raise ContractError("only synchronized ten-second clips with explicit local zero origin are supported")
        for mask in (self.audio_validity, self.visual_validity):
            if mask is not None and (type(mask) is not torch.Tensor or mask.dtype != torch.bool
                                    or mask.layout != torch.strided or mask.device.type == "meta"
                                    or tuple(mask.shape) != (w.shape[0], 10) or not mask.all()):
                raise ContractError("raw missing/invalid segments cannot be repaired by fusion masking")


def encode_raw_media(media: RawMediaInput):
    if type(media) is not RawMediaInput:
        raise ContractError("raw file decoding is unsupported; supply canonical RawMediaInput")
    media.validate()
    raise ContractError("raw encoder execution BLOCKED: BEATs iter3, trained LoRA/head, normalization "
                        "stats and trained resnet18_v1 checkpoint unavailable; parity unverified")

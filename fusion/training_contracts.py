"""Phase 4 training interfaces only; no epoch loop, experiment or checkpoint I/O."""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .contracts import ContractError
from .embedding_data import EmbeddingDataset, EmbeddingProvenance
from .cross_attention_model import CrossAttentionConfig, UnidirectionalAttentionConfig
from .mlp_model import MLPConfig
from .model_interface import model_config_from_dict
from .pair_contracts import PairType

FORGE_BASE_URL = "https://api.forge.coreweave.com"
FORGE_ENTITY = "GC2-AValidate"
FORGE_PROJECT = "avalidate"


def _positive_int(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ContractError(f"{name} must be an integer >= {0 if zero else 1}")


def _finite(value, name, *, positive=False):
    if type(value) not in (float, int) or not math.isfinite(value) or (positive and value <= 0):
        raise ContractError(f"{name} must be finite" + (" and positive" if positive else ""))


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContractError(f"{name} requires a SHA-256 digest")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8

    def __post_init__(self):
        if self.name not in ("adam", "adamw"):
            raise ContractError("optimizer must be adam or adamw")
        _finite(self.learning_rate, "learning_rate", positive=True)
        _finite(self.weight_decay, "weight_decay")
        _finite(self.epsilon, "epsilon", positive=True)
        if self.weight_decay < 0:
            raise ContractError("weight_decay must be nonnegative")
        if (type(self.betas) is not tuple or len(self.betas) != 2
                or any(type(b) not in (float, int) or not math.isfinite(b) or not 0 <= b < 1
                       for b in self.betas)):
            raise ContractError("betas must be a tuple of two finite values in [0,1)")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 20
    device: str = "cpu"
    num_workers: int = 0
    deterministic: bool = True
    max_gradient_norm: float = 5.0
    # Default is an unweighted pair average, not an assumption of equal counts.
    class_balance: str = "unweighted"
    positive_weight: float | None = None
    optimizer: OptimizerConfig = OptimizerConfig()
    selection_metric: str = "loss"
    training_split: str = "train"
    selection_split: str = "val"
    selection_partition: str = "primary"
    source_overlap_policy: str = "official_annotation_split"
    wandb_base_url: str = FORGE_BASE_URL
    wandb_entity: str = FORGE_ENTITY
    wandb_project: str = FORGE_PROJECT

    def __post_init__(self):
        _positive_int(self.seed, "seed", zero=True)
        if self.seed >= 2**32:
            raise ContractError("seed must fit NumPy unsigned 32-bit seed handling")
        for name in ("batch_size", "epochs"):
            _positive_int(getattr(self, name), name)
        _positive_int(self.num_workers, "num_workers", zero=True)
        if type(self.deterministic) is not bool:
            raise ContractError("deterministic must be boolean")
        if not isinstance(self.device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device) is None:
            raise ContractError("device must be cpu, cuda, or cuda:index")
        _finite(self.max_gradient_norm, "max_gradient_norm", positive=True)
        if self.class_balance == "unweighted":
            if self.positive_weight is not None:
                raise ContractError("unweighted loss cannot specify positive_weight")
        elif self.class_balance == "explicit_positive_weight":
            _finite(self.positive_weight, "positive_weight", positive=True)
        else:
            raise ContractError("unknown class-balance contract")
        if type(self.optimizer) is not OptimizerConfig:
            raise ContractError("expected OptimizerConfig")
        self.optimizer.__post_init__()
        if self.selection_metric not in ("loss", "roc_auc", "average_precision", "accuracy", "f1"):
            raise ContractError("unsupported validation selection metric")
        if (self.training_split, self.selection_split, self.selection_partition) != ("train", "val", "primary"):
            raise ContractError("optimization uses train; selection uses val primary only")
        if self.source_overlap_policy != "official_annotation_split":
            raise ContractError("only the documented official annotation split is supported")
        if (self.wandb_base_url, self.wandb_entity, self.wandb_project) != (
                FORGE_BASE_URL, FORGE_ENTITY, FORGE_PROJECT):
            raise ContractError("future experiments must use the required CoreWeave Forge destination")

    @property
    def fingerprint(self):
        self.__post_init__()
        return _fingerprint(asdict(self))


def seed_everything(config: TrainingConfig):
    config.__post_init__()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = config.deterministic


def resolve_device(config: TrainingConfig):
    config.__post_init__()
    device = torch.device(config.device)
    if device.type == "cuda" and (not torch.cuda.is_available()
                                or (device.index is not None and device.index >= torch.cuda.device_count())):
        raise ContractError("requested CUDA device is unavailable")
    return device


def build_optimizer(model, config: OptimizerConfig):
    if type(config) is not OptimizerConfig:
        raise ContractError("expected OptimizerConfig")
    config.__post_init__()
    cls = torch.optim.AdamW if config.name == "adamw" else torch.optim.Adam
    return cls(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
               betas=config.betas, eps=config.epsilon)


def validate_targets(targets, *, shape=None, require_both_classes=False):
    if (not isinstance(targets, torch.Tensor) or not targets.is_floating_point()
            or targets.ndim != 2 or targets.shape[1] != 1 or targets.shape[0] == 0
            or (shape is not None and targets.shape != shape)
            or not torch.isfinite(targets).all()
            or not ((targets == 0) | (targets == 1)).all() or targets.requires_grad):
        raise ContractError("targets must be separate finite binary floating (B,1) values without gradients")
    if require_both_classes and not ((targets == 0).any() and (targets == 1).any()):
        raise ContractError("the full primary optimization/selection dataset must contain both classes")


def correspondence_loss(logits, targets, config: TrainingConfig):
    config.__post_init__()
    if (not isinstance(logits, torch.Tensor) or not logits.is_floating_point()
            or logits.ndim != 2 or logits.shape[1] != 1 or not torch.isfinite(logits).all()):
        raise ContractError("logits must be finite floating (B,1)")
    validate_targets(targets, shape=logits.shape)
    if targets.device != logits.device or targets.dtype != logits.dtype:
        raise ContractError("targets and logits must share device and dtype")
    weight = None if config.positive_weight is None else logits.new_tensor([config.positive_weight])
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)
    if not torch.isfinite(loss):
        raise ContractError("nonfinite correspondence loss")
    return loss


def require_primary_dataset(dataset, role):
    if role not in ("train", "val"):
        raise ContractError("training and selection loaders reject final-test roles")
    if (type(dataset) is not EmbeddingDataset or dataset.split != role
            or dataset.partition != "primary" or not dataset.complete):
        raise ContractError("require a complete matching train/val primary dataset")
    if any(p.pair_type not in (PairType.ORIGINAL_POSITIVE, PairType.CROSS_CLASS_EASY_NEGATIVE)
           for p in dataset.pairs):
        raise ContractError("unauthorized primary training pair type")
    validate_targets(torch.tensor(dataset.targets), require_both_classes=True)


def validate_split_isolation(train, validation):
    """Reject annotation leakage; report official source-video overlap explicitly.

    Official AVE is not source-disjoint. Different annotations from the same
    source remain a known limitation, not permission to borrow rows across splits.
    """
    require_primary_dataset(train, "train")
    require_primary_dataset(validation, "val")
    def references(dataset, field):
        return {getattr(getattr(p, side).record, field)
                for p in dataset.pairs for side in ("audio", "visual")}
    if references(train, "sample_id") & references(validation, "sample_id"):
        raise ContractError("train/validation annotation identities overlap")
    return tuple(sorted(references(train, "clip_id") & references(validation, "clip_id")))


def _seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, config: TrainingConfig, *, role):
    config.__post_init__()
    require_primary_dataset(dataset, role)
    device = resolve_device(config)
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=role == "train",
                      drop_last=False, num_workers=config.num_workers,
                      pin_memory=device.type == "cuda", generator=generator, worker_init_fn=_seed_worker)


def move_batch(batch, device):
    """Transfer only the explicit feature schema and separate targets."""
    features, targets = batch
    if type(features) is not dict or set(features) != {"audio", "visual", "validity"}:
        raise ContractError("batch features must contain only audio, visual and validity")
    validate_targets(targets)
    return ({name: value.to(device=device, non_blocking=True) for name, value in features.items()},
            targets.to(device=device, non_blocking=True))


def backward_checked(loss, model, *, max_gradient_norm):
    """Backpropagate and clip; callers own zero_grad and the optimizer step.

    No optimizer state or parameters are updated here. A failed guard clears
    gradients so a caller cannot accidentally step using corrupt values.
    """
    _finite(max_gradient_norm, "max_gradient_norm", positive=True)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    try:
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss):
            raise ContractError("backward requires a finite scalar loss")
        loss.backward()
        if not parameters or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise ContractError("missing or nonfinite model gradients")
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm, error_if_nonfinite=True)
        return float(norm)
    except (ContractError, RuntimeError) as exc:
        model.zero_grad(set_to_none=True)
        raise ContractError("finite-loss/gradient guard failed; gradients cleared") from exc


@dataclass(frozen=True)
class CheckpointMetadata:
    """JSON metadata beside a FUTURE checkpoint; never serializes model weights.

    optimizer_state_present describes the companion checkpoint, not state stored
    in this JSON. Epoch zero / no selected epoch is a valid pre-training schema.
    """
    model: MLPConfig | CrossAttentionConfig | UnidirectionalAttentionConfig
    training: TrainingConfig
    git_commit: str
    code_sha256: str
    data: tuple[EmbeddingProvenance, EmbeddingProvenance]
    export_identities: tuple[tuple[str, str], ...]
    software_versions: tuple[tuple[str, str], ...]
    source_overlap: tuple[str, ...]
    epoch: int = 0
    optimizer_steps: int = 0
    optimizer_state_present: bool = False
    selected_epoch: int | None = None
    validation_metrics: tuple[tuple[str, float], ...] = ()
    schema: str = "gc2-fusion-checkpoint-v1"

    def __post_init__(self):
        if (type(self.model) not in (MLPConfig, CrossAttentionConfig, UnidirectionalAttentionConfig)
                or type(self.training) is not TrainingConfig):
            raise ContractError("checkpoint requires strict model and training configs")
        self.model.__post_init__()
        self.training.__post_init__()
        expected_schema = ("gc2-fusion-checkpoint-v1" if type(self.model) is MLPConfig
                           else "gc2-fusion-cross-attention-checkpoint-v1")
        if self.schema != expected_schema:
            raise ContractError("checkpoint schema must match the architecture")
        if not isinstance(self.git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", self.git_commit) is None:
            raise ContractError("git_commit must be a full 40-character revision")
        _digest(self.code_sha256, "code_sha256")
        if (type(self.data) is not tuple or len(self.data) != 2
                or any(type(d) is not EmbeddingProvenance for d in self.data)
                or tuple(d.split for d in self.data) != ("train", "val")):
            raise ContractError("checkpoint must bind train and val provenance, never test")
        for data in self.data:
            for name in ("manifest_sha256", "policy_fingerprint"):
                _digest(getattr(data, name), name)
            if data.artifact_fingerprint is not None:
                _digest(data.artifact_fingerprint, "artifact_fingerprint")
            for inventory in (data.source_array_hashes, data.input_export_hashes):
                if (type(inventory) is not tuple
                        or any(type(row) is not tuple or len(row) != 2 for row in inventory)):
                    raise ContractError("hash inventories must be immutable tuples")
                roles = []
                for role, digest in inventory:
                    if not isinstance(role, str) or not role.strip():
                        raise ContractError("invalid input role")
                    _digest(digest, "input hash")
                    roles.append(role)
                if len(set(roles)) != len(roles):
                    raise ContractError("duplicate input hash role")
            expected = {f"{m}/{data.split}/{a}" for m in ("audio", "vision")
                        for a in ("embeddings", "validity")}
            if {r for r, _ in data.source_array_hashes} != expected:
                raise ContractError("source array fingerprint coverage is incomplete")
        for inventory in (self.export_identities, self.software_versions):
            if (type(inventory) is not tuple or not inventory
                    or any(type(row) is not tuple or len(row) != 2
                           or any(not isinstance(s, str) or not s.strip() for s in row) for row in inventory)
                    or len({k for k, _ in inventory}) != len(inventory)):
                raise ContractError("identities and versions require unique nonempty string pairs")
        if {k for k, _ in self.export_identities} != {"audio", "vision"}:
            raise ContractError("both modality export identities must be recorded")
        if not {"python", "numpy", "torch"} <= {k for k, _ in self.software_versions}:
            raise ContractError("Python, NumPy and PyTorch versions must be recorded")
        if (type(self.source_overlap) is not tuple or any(not isinstance(s, str) or not s.strip()
                                                       for s in self.source_overlap)
                or self.source_overlap != tuple(sorted(set(self.source_overlap)))):
            raise ContractError("source overlap must be a sorted unique tuple")
        _positive_int(self.epoch, "epoch", zero=True)
        _positive_int(self.optimizer_steps, "optimizer_steps", zero=True)
        if self.epoch > self.training.epochs or type(self.optimizer_state_present) is not bool:
            raise ContractError("invalid checkpoint epoch or optimizer state declaration")
        if self.optimizer_steps and not self.optimizer_state_present:
            raise ContractError("optimizer steps require a companion optimizer state")
        if self.selected_epoch is not None:
            _positive_int(self.selected_epoch, "selected_epoch")
            if self.selected_epoch > self.epoch:
                raise ContractError("selected epoch cannot exceed checkpoint epoch")
        if (type(self.validation_metrics) is not tuple
                or any(type(row) is not tuple or len(row) != 2 for row in self.validation_metrics)):
            raise ContractError("validation_metrics must be immutable")
        names = []
        for name, value in self.validation_metrics:
            if name not in ("loss", "roc_auc", "average_precision", "accuracy", "f1"):
                raise ContractError("unknown validation metric")
            _finite(value, name)
            if value < 0 or (name != "loss" and value > 1):
                raise ContractError("validation metric out of range")
            names.append(name)
        if len(set(names)) != len(names):
            raise ContractError("duplicate validation metrics")
        if self.selected_epoch is not None and self.training.selection_metric not in names:
            raise ContractError("selected epoch requires its validation selection metric")
        if self.selected_epoch is None and names:
            raise ContractError("selection metrics require an explicit selected epoch")

    @property
    def experiment_fingerprint(self):
        self.__post_init__()
        fields = asdict(self)
        for name in ("epoch", "optimizer_steps", "optimizer_state_present", "selected_epoch", "validation_metrics"):
            fields.pop(name)
        return _fingerprint(fields)

    def to_dict(self):
        self.__post_init__()
        return {**asdict(self), "experiment_fingerprint": self.experiment_fingerprint}

    @classmethod
    def from_dict(cls, value):
        """Strict reconstruction after JSON decoding; unknown fields fail closed."""
        try:
            if type(value) is not dict:
                raise ContractError("checkpoint metadata must be an object")
            fields = dict(value)
            fingerprint = fields.pop("experiment_fingerprint")
            model = model_config_from_dict(fields.pop("model"), fields["schema"])
            training = dict(fields.pop("training"))
            optimizer = dict(training.pop("optimizer"))
            optimizer["betas"] = tuple(optimizer["betas"])
            data = []
            for entry in fields.pop("data"):
                entry = dict(entry)
                for name in ("source_array_hashes", "input_export_hashes"):
                    entry[name] = tuple(tuple(row) for row in entry[name])
                data.append(EmbeddingProvenance(**entry))
            for name in ("export_identities", "software_versions", "validation_metrics"):
                fields[name] = tuple(tuple(row) for row in fields[name])
            if not isinstance(fields["source_overlap"], (tuple, list)):
                raise ContractError("source overlap must be a sequence")
            fields["source_overlap"] = tuple(fields["source_overlap"])
            result = cls(model=model,
                         training=TrainingConfig(optimizer=OptimizerConfig(**optimizer), **training),
                         data=tuple(data), **fields)
            if result.experiment_fingerprint != fingerprint:
                raise ContractError("checkpoint experiment fingerprint mismatch")
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("malformed checkpoint metadata") from exc

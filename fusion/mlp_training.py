"""Learned-fusion optimization and frozen evaluation; no network or experiment I/O."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
from numbers import Integral

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate

from .baseline_metrics import ranking_metrics
from .baseline_thresholds import classification_metrics
from .contracts import ContractError
from .embedding_data import EmbeddingDataset, _snapshot
from .training_contracts import (
    TrainingConfig, _fingerprint, backward_checked, correspondence_loss,
    move_batch, require_primary_dataset, resolve_device,
)


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@dataclass
class EarlyStopping:
    """Strictly greater AUC wins; exact ties retain the earliest epoch."""
    patience: int = 5
    best_auc: float = -1.0
    best_epoch: int = 0
    stale_epochs: int = 0
    last_epoch: int = 0

    def __post_init__(self):
        if type(self.patience) is not int or self.patience < 1:
            raise ContractError("patience must be a positive integer")

    def observe(self, epoch, auc):
        if (type(epoch) is not int or epoch != self.last_epoch + 1
                or type(auc) not in (int, float) or not math.isfinite(auc) or not 0 <= auc <= 1):
            raise ContractError("selection requires consecutive epochs and finite primary validation AUC")
        if self.should_stop:
            raise ContractError("early stopping has already fired")
        self.last_epoch = epoch
        improved = auc > self.best_auc
        if improved:
            self.best_auc, self.best_epoch, self.stale_epochs = float(auc), epoch, 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.stale_epochs >= self.patience


class _IndexedTrainingDataset(Dataset):
    """Worker-safe original indices, separate from predictive features/targets."""
    def __init__(self, source):
        self.source = source

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self):
            raise ContractError("training sampler requires in-range original integer indices")
        features, targets = self.source[index]
        return features, targets, int(index)


def _indexed_training_loader(loader):
    # Reuse the actual batch sampler and generator, without iterating either in
    # advance. Indices travel with fetched items, including worker prefetches.
    # Arbitrary collators could drop/reorder data independently of the indices.
    if (not isinstance(loader, DataLoader) or loader.batch_sampler is None
            or loader.collate_fn is not default_collate):
        raise ContractError("training requires a batched DataLoader with default collation")
    require_primary_dataset(loader.dataset, "train")
    if loader.drop_last or getattr(loader.batch_sampler, "drop_last", False):
        raise ContractError("training must retain the final partial minibatch")
    ordering = {"in_order": loader.in_order} if hasattr(loader, "in_order") else {}
    return DataLoader(
        _IndexedTrainingDataset(loader.dataset), batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers, collate_fn=default_collate,
        pin_memory=loader.pin_memory, timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn, multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator, prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers, **ordering,
    )


def train_epoch(model, loader, optimizer, config: TrainingConfig):
    """Only complete TRAIN primary loaders may update parameters."""
    indexed = _indexed_training_loader(loader)
    visited = np.zeros(len(loader.dataset), dtype=bool)
    model.train()
    loss_sum, count, norms = 0.0, 0, []
    device = resolve_device(config)
    for features, targets, indices in indexed:
        indices = indices.numpy()
        if len(np.unique(indices)) != len(indices) or visited[indices].any():
            raise ContractError("optimization epoch contains duplicate original dataset indices")
        visited[indices] = True
        features, targets = move_batch((features, targets), device)
        optimizer.zero_grad(set_to_none=True)
        loss = correspondence_loss(model(**features).logits, targets, config)
        norm = backward_checked(loss, model, max_gradient_norm=config.max_gradient_norm)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ContractError("optimizer produced nonfinite parameters")
        loss_sum += float(loss.detach()) * len(targets)
        count += len(targets)
        norms.append(norm)
    if not visited.all() or not count or count != len(loader.dataset):
        raise ContractError("optimization epoch did not cover the complete TRAIN primary dataset")
    return {"loss": loss_sum / count, "count": count, "optimizer_steps": len(norms),
            "gradient_norm_mean": float(np.mean(norms)), "gradient_norm_max": float(max(norms))}


@dataclass(frozen=True)
class Predictions:
    split: str
    partition: str
    manifest_sha256: str
    model_sha256: str
    identities: tuple
    targets: np.ndarray
    logits: np.ndarray
    loss: float

    def __post_init__(self):
        if self.split not in ("train", "val") or self.partition not in ("primary", "challenge"):
            raise ContractError("test or unknown predictions are forbidden")
        y, z = np.asarray(self.targets), np.asarray(self.logits)
        if (y.ndim != 1 or z.shape != y.shape or y.dtype.kind not in "iu" or not len(y)
                or not np.isin(y, [0, 1]).all() or z.dtype.kind != "f" or not np.isfinite(z).all()
                or type(self.identities) is not tuple or len(self.identities) != len(y)
                or len(set(self.identities)) != len(y) or not math.isfinite(self.loss)):
            raise ContractError("malformed or incomplete prediction records")
        if self.partition == "challenge" and np.any(y != 0):
            raise ContractError("challenge provenance targets must all be negative")
        for name in ("manifest_sha256", "model_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("predictions require model and manifest SHA-256")
        object.__setattr__(self, "targets", _snapshot(y))
        object.__setattr__(self, "logits", _snapshot(z))

    @property
    def probabilities(self):
        # Float64 sigmoid avoids unnecessary float32 saturation/ties. It is
        # still a classifier output, without any independent calibration claim.
        return _snapshot(np.exp(-np.logaddexp(0, -self.logits.astype(np.float64))))

    @property
    def fingerprint(self):
        return _fingerprint({"split": self.split, "partition": self.partition,
                             "manifest": self.manifest_sha256, "model": self.model_sha256,
                             "identities": self.identities, "targets": self.targets.tolist(),
                             "logits": self.logits.tolist()})


def predict(model, dataset, config: TrainingConfig, *, frozen_policy=None):
    """Canonical train/val inference; challenges require an already frozen model/threshold."""
    if type(dataset) is not EmbeddingDataset or dataset.split not in ("train", "val") or not dataset.complete:
        raise ContractError("inference requires complete train/val embedding data; never test")
    before = model_fingerprint(model)
    if dataset.partition == "primary":
        require_primary_dataset(dataset, dataset.split)
    elif (dataset.split != "val" or type(frozen_policy) is not FrozenOperatingPoint
          or before != frozen_policy.model_sha256
          or dataset.provenance.manifest_sha256 != frozen_policy.validation_manifest_sha256):
        raise ContractError("only separate validation challenges with frozen model/threshold may be scored")
    model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=resolve_device(config).type == "cuda")
    logits, targets, total = [], [], 0.0
    with torch.inference_mode():
        for batch in loader:
            features, y = move_batch(batch, resolve_device(config))
            z = model(**features).logits
            total += float(correspondence_loss(z, y, config)) * len(y)
            logits.append(z.cpu().numpy().reshape(-1).copy())
            targets.append(y.cpu().numpy().reshape(-1).astype(np.int64))
    if before != model_fingerprint(model):
        raise ContractError("inference changed model parameters")
    return Predictions(dataset.split, dataset.partition, dataset.provenance.manifest_sha256, before,
                       tuple(p.identity for p in dataset.pairs), np.concatenate(targets),
                       np.concatenate(logits), total / len(dataset))


def primary_metrics(predictions, *, threshold=0.5):
    if type(predictions) is not Predictions or predictions.partition != "primary":
        raise ContractError("ranking/classification metrics require primary train/val predictions")
    auc, ap = ranking_metrics(predictions.targets, predictions.probabilities)
    if auc.value is None or ap.value is None:
        raise ContractError("primary metrics require both target classes")
    from dataclasses import asdict
    classification = classification_metrics(predictions.targets, predictions.probabilities, threshold=threshold)
    return {"loss": predictions.loss, "roc_auc": auc.value, "average_precision": ap.value,
            "threshold": float(threshold), **asdict(classification),
            "brier_score_descriptive": float(np.mean((predictions.probabilities - predictions.targets)**2))}


@dataclass(frozen=True)
class FrozenOperatingPoint:
    threshold: float
    model_sha256: str
    selected_epoch: int
    validation_manifest_sha256: str
    validation_prediction_fingerprint: str
    objective: str = "maximum_validation_primary_binary_f1"
    tie_break: str = "largest_threshold"
    decision: str = "probability >= threshold"

    def __post_init__(self):
        if (type(self.threshold) is not float or not math.isfinite(self.threshold)
                or not 0 <= self.threshold <= np.nextafter(1., np.inf)
                or type(self.selected_epoch) is not int or self.selected_epoch < 1
                or self.objective != "maximum_validation_primary_binary_f1"
                or self.tie_break != "largest_threshold" or self.decision != "probability >= threshold"):
            raise ContractError("invalid frozen operating-point contract")
        for value in (self.model_sha256, self.validation_manifest_sha256, self.validation_prediction_fingerprint):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("frozen policy requires SHA-256 provenance")


def fit_operating_point(predictions, *, selected_epoch):
    if type(predictions) is not Predictions or (predictions.split, predictions.partition) != ("val", "primary"):
        raise ContractError("threshold fitting requires selected-model validation primary predictions")
    primary_metrics(predictions)  # Require both classes, finite complete inputs.
    scores, y = predictions.probabilities, predictions.targets
    candidates = [*np.unique(scores), np.nextafter(float(scores.max()), np.inf)]
    def objective(threshold):
        counts = classification_metrics(y, scores, threshold=threshold).counts
        denominator = 2*counts.tp + counts.fp + counts.fn
        return Fraction(2*counts.tp, denominator) if denominator else Fraction(0)
    threshold = float(max(candidates, key=lambda t: (objective(t), t)))
    return FrozenOperatingPoint(threshold, predictions.model_sha256, selected_epoch,
                                predictions.manifest_sha256, predictions.fingerprint)


def evaluate_frozen(predictions, policy):
    if type(policy) is not FrozenOperatingPoint or predictions.model_sha256 != policy.model_sha256:
        raise ContractError("evaluation model differs from frozen selected model")
    if predictions.split == "val" and predictions.manifest_sha256 != policy.validation_manifest_sha256:
        raise ContractError("frozen validation manifest differs")
    if predictions.partition == "primary":
        return primary_metrics(predictions, threshold=policy.threshold)
    if predictions.split != "val":
        raise ContractError("only validation challenge diagnostics are authorized")
    accepted = int(np.sum(predictions.probabilities >= policy.threshold))
    n = len(predictions.targets)
    return {"count": n, "false_acceptance_count": accepted, "rejection_count": n-accepted,
            "false_acceptance_rate": accepted/n, "rejection_rate": (n-accepted)/n,
            "threshold": policy.threshold, "ranking_metrics": "undefined: only negative provenance targets",
            "interpretation": "same-category substitution diagnostic; not evidence of physical causality"}

"""Interchangeable loss strategies for one experiment."""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float = float("nan")


class LossStrategy(abc.ABC):
    name = ""
    uses_classifier = False
    two_views = False
    balanced_train = False
    balanced_eval = False
    checkpoint_adapter = "supcon_checkpoint"

    @abc.abstractmethod
    def run_epoch(self, model, loader, device: str, optimizer=None) -> EpochMetrics:
        raise NotImplementedError


class CrossEntropyStrategy(LossStrategy):
    name = "ce"
    uses_classifier = True
    checkpoint_adapter = "classifier_checkpoint"

    def __init__(self, *, label_smoothing: float = 0.0, **_) -> None:
        import torch

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def run_epoch(self, model, loader, device: str, optimizer=None) -> EpochMetrics:
        import torch

        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total = 0
        correct = 0
        for images, targets, _, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            known = targets >= 0
            if not known.any():
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = model(images[known])
                loss = self.criterion(logits, targets[known])
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            count = int(known.sum())
            total_loss += float(loss.detach()) * count
            total += count
            correct += int(logits.argmax(dim=1).eq(targets[known]).sum())
        return EpochMetrics(total_loss / max(total, 1), correct / max(total, 1))


class SupConLoss:
    def __init__(self, temperature: float = 0.07) -> None:
        self.temperature = temperature

    def __call__(self, features, labels):
        import torch

        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError("SupCon features must have shape [batch, 2, dim]")
        batch = features.shape[0]
        flattened = torch.cat(torch.unbind(features, dim=1), dim=0)
        repeated_labels = labels.repeat(2)
        logits = flattened @ flattened.T / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        self_mask = torch.eye(2 * batch, dtype=torch.bool, device=features.device)
        positive_mask = (
            repeated_labels[:, None].eq(repeated_labels[None, :]) & ~self_mask
        )
        exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
        log_prob = logits - torch.log(
            exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        )
        positive_count = positive_mask.sum(dim=1)
        if (positive_count == 0).any():
            raise ValueError(
                "a SupCon anchor has no positive; use two views and class-aware batches"
            )
        return -((log_prob * positive_mask).sum(dim=1) / positive_count).mean()


class SupConStrategy(LossStrategy):
    name = "supcon"
    two_views = True
    balanced_train = True

    def __init__(self, *, temperature: float = 0.07, **_) -> None:
        self.criterion = SupConLoss(temperature)

    def run_epoch(self, model, loader, device: str, optimizer=None) -> EpochMetrics:
        import torch

        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total = 0
        for views, targets, _, _ in loader:
            first, second = views
            targets = targets.to(device)
            images = torch.cat([first, second], dim=0).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                projections = model(images, return_projection=True)
                batch = targets.shape[0]
                paired = torch.stack([projections[:batch], projections[batch:]], dim=1)
                loss = self.criterion(paired, targets)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total += len(targets)
        return EpochMetrics(total_loss / max(total, 1))


class BatchHardTripletLoss:
    def __init__(self, margin: float = 0.2) -> None:
        self.margin = margin

    def __call__(self, embeddings, labels):
        import torch

        distances = torch.cdist(embeddings, embeddings)
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        positives = same & ~eye
        negatives = ~same
        if not positives.any(dim=1).all() or not negatives.any(dim=1).all():
            raise ValueError(
                "triplet batches need a positive and a negative for every sample"
            )
        hardest_positive = (
            distances.masked_fill(~positives, float("-inf")).max(dim=1).values
        )
        hardest_negative = (
            distances.masked_fill(~negatives, float("inf")).min(dim=1).values
        )
        return torch.relu(hardest_positive - hardest_negative + self.margin).mean()


class TripletStrategy(LossStrategy):
    name = "triplet"
    balanced_train = True
    balanced_eval = True

    def __init__(self, *, margin: float = 0.2, **_) -> None:
        self.criterion = BatchHardTripletLoss(margin)

    def run_epoch(self, model, loader, device: str, optimizer=None) -> EpochMetrics:
        import torch

        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total = 0
        for images, targets, _, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                embeddings = model(images, return_projection=True)
                loss = self.criterion(embeddings, targets)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
            total += len(targets)
        return EpochMetrics(total_loss / max(total, 1))


LOSS_STRATEGIES: dict[str, type[LossStrategy]] = {
    strategy.name: strategy
    for strategy in (CrossEntropyStrategy, SupConStrategy, TripletStrategy)
}


def create_loss_strategy(name: str, **kwargs) -> LossStrategy:
    try:
        return LOSS_STRATEGIES[name](**kwargs)
    except KeyError as exc:
        raise KeyError(
            f"unknown loss {name!r}; available={sorted(LOSS_STRATEGIES)}"
        ) from exc

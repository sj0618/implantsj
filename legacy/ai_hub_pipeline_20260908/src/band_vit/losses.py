"""Cross-entropy classification with optional band-attention regularization."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float = float("nan")
    classification_loss: float = float("nan")
    band_loss: float = float("nan")
    macro_precision: float = float("nan")
    macro_recall: float = float("nan")
    macro_f1: float = float("nan")
    weighted_precision: float = float("nan")
    weighted_recall: float = float("nan")
    weighted_f1: float = float("nan")


class CrossEntropyStrategy:
    """Standard classifier objective with optional band-attention regularization."""

    checkpoint_adapter = "classifier_checkpoint"

    def __init__(
        self,
        *,
        label_smoothing: float = 0.0,
        band_loss_weight: float = 0.0,
        band_alpha: float = 4.0,
        **_,
    ) -> None:
        import torch

        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.band_loss_weight = band_loss_weight
        self.band_alpha = band_alpha

    def run_epoch(self, model, loader, device: str, optimizer=None) -> EpochMetrics:
        import torch
        import torch.nn.functional as functional
        from sklearn.metrics import precision_recall_fscore_support

        from .models import get_cls_patch_attention

        training = optimizer is not None
        model.train(training)
        total_loss = 0.0
        total_classification_loss = 0.0
        total_band_loss = 0.0
        total = 0
        correct = 0
        used_band = False
        all_targets: list[int] = []
        all_predictions: list[int] = []

        for batch in loader:
            if len(batch) == 5:
                images, targets, _, _, band_masks = batch
            else:
                images, targets, _, _ = batch
                band_masks = None
            images = images.to(device)
            targets = targets.to(device)
            known = targets >= 0
            if not known.any():
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                logits = model(images[known])
                known_targets = targets[known]
                classification_loss = self.criterion(logits, known_targets)
                band_loss = torch.zeros((), device=device)
                if training and self.band_loss_weight > 0:
                    if band_masks is None:
                        raise ValueError(
                            "band_loss_weight > 0 requires band masks in the training loader"
                        )
                    band_masks = band_masks.to(device).float()[known]
                    attention = get_cls_patch_attention(model).float()
                    if attention.shape != band_masks.shape:
                        raise ValueError(
                            "attention/mask patch count mismatch: "
                            f"attention={tuple(attention.shape)}, "
                            f"mask={tuple(band_masks.shape)}"
                        )
                    prior = 1.0 + self.band_alpha * band_masks
                    prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    band_loss = functional.kl_div(
                        attention.clamp_min(1e-8).log(),
                        prior,
                        reduction="batchmean",
                    )
                    used_band = True
                loss = classification_loss + self.band_loss_weight * band_loss
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            predictions = logits.argmax(dim=1)
            count = int(known.sum().item())
            total_loss += float(loss.detach()) * count
            total_classification_loss += float(classification_loss.detach()) * count
            total_band_loss += float(band_loss.detach()) * count
            total += count
            correct += int(predictions.eq(known_targets).sum())
            all_targets.extend(known_targets.detach().cpu().tolist())
            all_predictions.extend(predictions.detach().cpu().tolist())

        if all_targets:
            macro = precision_recall_fscore_support(
                all_targets,
                all_predictions,
                average="macro",
                zero_division=0,
            )
            weighted = precision_recall_fscore_support(
                all_targets,
                all_predictions,
                average="weighted",
                zero_division=0,
            )
        else:
            macro = weighted = (float("nan"),) * 4

        return EpochMetrics(
            loss=total_loss / max(total, 1),
            accuracy=correct / max(total, 1),
            classification_loss=total_classification_loss / max(total, 1),
            band_loss=(
                total_band_loss / max(total, 1) if used_band else float("nan")
            ),
            macro_precision=float(macro[0]),
            macro_recall=float(macro[1]),
            macro_f1=float(macro[2]),
            weighted_precision=float(weighted[0]),
            weighted_recall=float(weighted[1]),
            weighted_f1=float(weighted[2]),
        )

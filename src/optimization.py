"""Optimizer and scheduler builders selected directly by CLI names."""

from __future__ import annotations


OPTIMIZERS = ("adamw", "sgd")
SCHEDULERS = ("cosine", "step", "none")


def create_optimizer(
    name: str,
    parameters,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
):
    import torch

    if name == "adamw":
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=momentum,
        )
    raise KeyError(f"unknown optimizer {name!r}; available={list(OPTIMIZERS)}")


def create_scheduler(
    name: str,
    optimizer,
    *,
    epochs: int,
    step_size: int,
    gamma: float,
):
    import torch

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    if name == "none":
        return None
    raise KeyError(f"unknown scheduler {name!r}; available={list(SCHEDULERS)}")

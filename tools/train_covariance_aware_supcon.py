#!/usr/bin/env python3
"""Train SupCon with an episodic covariance-aware prototype loss."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset

from src.data import ImageClassificationDataset, build_transforms, label_mapping, load_manifest
from src.metric_learning import SupConLoss, extract_embeddings, prototype_predict
from src.metrics import compute_metrics
from src.models import MetricModel, save_checkpoint
from src.utils import count_parameters, log, make_run_dir, seed_everything, write_json


def covariance_scores(
    support: torch.Tensor,
    support_y: torch.Tensor,
    query: torch.Tensor,
    classes: int,
    covariance_weight: float,
    beta: float,
    logdet_weight: float,
    covariance_mode: str = "legacy",
    covariance_shrinkage: float = 0.1,
) -> torch.Tensor:
    centers = torch.stack([support[support_y == index].mean(0) for index in range(classes)])
    residuals = support - centers[support_y]
    pooled_var = residuals.pow(2).mean(0)
    if covariance_mode == "pooled_full":
        if not 0.0 <= covariance_shrinkage < 1.0:
            raise ValueError("covariance_shrinkage must be in [0, 1)")
        covariance = residuals.T @ residuals / max(1, residuals.shape[0])
        mean_variance = covariance.diag().mean().clamp_min(beta)
        eye = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
        covariance = (1.0 - covariance_shrinkage) * covariance
        covariance = covariance + covariance_shrinkage * mean_variance * eye
        covariance = covariance + beta * eye
        precision = torch.linalg.inv(covariance)
        delta = query[:, None, :] - centers[None, :, :]
        return torch.einsum("nkd,de,nke->nk", delta, precision, delta)

    class_var = torch.stack(
        [(support[support_y == index] - centers[index]).pow(2).mean(0) for index in range(classes)]
    )
    if covariance_mode == "pooled_diagonal":
        variance = pooled_var[None, :].expand(classes, -1)
        logdet_weight = 0.0
    elif covariance_mode == "legacy":
        variance = (1.0 - covariance_weight) * pooled_var[None, :] + covariance_weight * class_var
    else:
        raise ValueError(f"unknown covariance_mode: {covariance_mode}")
    variance = variance.clamp_min(beta)
    delta = query[:, None, :] - centers[None, :, :]
    scores = (delta.pow(2) / variance[None, :, :]).sum(-1)
    if logdet_weight:
        scores = scores + logdet_weight * variance.log().sum(1)[None, :]
    return scores


def covariance_loss(
    support: torch.Tensor,
    support_y: torch.Tensor,
    query: torch.Tensor,
    query_y: torch.Tensor,
    classes: int,
    covariance_weight: float,
    beta: float,
    logdet_weight: float,
    covariance_mode: str,
    covariance_shrinkage: float,
) -> torch.Tensor:
    return F.cross_entropy(
        -covariance_scores(
            support,
            support_y,
            query,
            classes,
            covariance_weight,
            beta,
            logdet_weight,
            covariance_mode,
            covariance_shrinkage,
        ),
        query_y,
    )


class EpisodeViews(Dataset):
    def __init__(self, manifest: str | Path, image_size: int):
        self.first = ImageClassificationDataset(
            manifest,
            split="train",
            transform=build_transforms(image_size, train=True, strong_aug=True),
            return_path=False,
        )
        self.second = ImageClassificationDataset(
            manifest,
            split="train",
            transform=build_transforms(image_size, train=True, strong_aug=True),
            return_path=False,
        )

    def __len__(self) -> int:
        return len(self.first)

    def __getitem__(self, index: int):
        first, label = self.first[index]
        second, _ = self.second[index]
        return first, second, label


def sample_episode(
    class_indices: dict[int, np.ndarray],
    support_shots: int,
    query_shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support, support_y, query, query_y = [], [], [], []
    for label in sorted(class_indices):
        selected = rng.choice(class_indices[label], support_shots + query_shots, replace=False)
        support.extend(selected[:support_shots])
        query.extend(selected[support_shots:])
        support_y.extend([label] * support_shots)
        query_y.extend([label] * query_shots)
    return (
        np.asarray(support),
        np.asarray(support_y),
        np.asarray(query),
        np.asarray(query_y),
    )


def extract_embedding_variant(model, loader, device, use_projection: bool):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            x, y, _ = batch
            features = model.forward_features(x.to(device, non_blocking=True))
            z = model.projector(features) if use_projection else features
            zs.append(F.normalize(z, dim=1).cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


def pooled_mahalanobis_predict(train_z: np.ndarray, train_y: np.ndarray, query_z: np.ndarray) -> np.ndarray:
    classes = np.unique(train_y)
    centers = np.stack([train_z[train_y == label].mean(axis=0) for label in classes])
    residuals = train_z - centers[np.searchsorted(classes, train_y)]
    precision = LedoitWolf().fit(residuals).precision_.astype(np.float64)
    delta = query_z[:, None, :] - centers[None, :, :]
    scores = np.einsum("nkd,de,nke->nk", delta, precision, delta)
    return classes[scores.argmin(axis=1)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--experiment-name", default="covariance_aware_supcon")
    parser.add_argument("--model-name", default="vit_small_patch16_224.augreg_in21k_ft_in1k")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--episodes-per-epoch", type=int, default=20)
    parser.add_argument("--support-shots", type=int, default=5)
    parser.add_argument("--query-shots", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--covariance-weight", type=float, default=0.75)
    parser.add_argument("--covariance-beta", type=float, default=1e-3)
    parser.add_argument("--logdet-weight", type=float, default=1.0)
    parser.add_argument("--covariance-loss-weight", type=float, default=1.0)
    parser.add_argument("--covariance-mode", choices=("legacy", "pooled_diagonal", "pooled_full"), default="legacy")
    parser.add_argument("--covariance-shrinkage", type=float, default=0.1)
    parser.add_argument("--loss-embedding", choices=("backbone", "projection"), default="backbone")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    manifest = Path(args.manifest).resolve()
    output = Path(args.output_root).resolve()
    df = load_manifest(manifest, split=None)
    class_to_idx, _ = label_mapping(df)
    labels = sorted(class_to_idx, key=class_to_idx.get)
    if len(labels) < 2:
        raise ValueError("at least two source classes are required")
    train_df = df[df["split"].astype(str) == "train"].reset_index(drop=True)
    class_indices = {
        index: np.flatnonzero(train_df["label"].map(class_to_idx.__getitem__).to_numpy() == index)
        for index in range(len(labels))
    }
    required = args.support_shots + args.query_shots
    if min(map(len, class_indices.values())) < required:
        raise ValueError(f"each class needs {required} train images per episode")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = MetricModel(
        args.model_name,
        embedding_dim=args.embedding_dim,
        pretrained=True,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    dataset = EpisodeViews(manifest, args.image_size)
    supcon = SupConLoss(args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = GradScaler(enabled=device.type == "cuda")
    run_dir = make_run_dir(output, args.experiment_name)
    write_json(run_dir / "args.json", {**vars(args), "manifest": str(manifest), "class_to_idx": class_to_idx})
    log(f"Run dir: {run_dir}")
    log(f"Trainable parameters: {count_parameters(model):,}")

    valid_loader = torch.utils.data.DataLoader(
        ImageClassificationDataset(
            manifest,
            split="valid",
            class_to_idx=class_to_idx,
            transform=build_transforms(args.image_size, train=False),
            return_path=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        ImageClassificationDataset(
            manifest,
            split="train",
            class_to_idx=class_to_idx,
            transform=build_transforms(args.image_size, train=False),
            return_path=True,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    best_score = -1.0
    history = []
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_supcon = total_cov = 0.0
        for _ in range(args.episodes_per_epoch):
            support_idx, support_y_np, query_idx, query_y_np = sample_episode(
                class_indices, args.support_shots, args.query_shots, rng
            )
            first = torch.stack([dataset[int(index)][0] for index in np.concatenate([support_idx, query_idx])]).to(device)
            second = torch.stack([dataset[int(index)][1] for index in np.concatenate([support_idx, query_idx])]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                h1 = model.forward_features(first)
                h2 = model.forward_features(second)
                projected = F.normalize(model.projector(torch.cat([h1, h2], dim=0)), dim=1)
                n = len(support_idx) + len(query_idx)
                supcon_labels = torch.tensor(
                    np.concatenate([support_y_np, query_y_np, support_y_np, query_y_np]),
                    device=device,
                    dtype=torch.long,
                )
                supcon_value = supcon(projected, supcon_labels)
                backbone_support = F.normalize(h1[: len(support_idx)], dim=1)
                backbone_query = F.normalize(h1[len(support_idx) : n], dim=1)
                support_y = torch.tensor(support_y_np, device=device, dtype=torch.long)
                query_y = torch.tensor(query_y_np, device=device, dtype=torch.long)
                covariance_value = covariance_loss(
                    (backbone_support if args.loss_embedding == "backbone" else projected[: len(support_idx)]),
                    support_y,
                    (backbone_query if args.loss_embedding == "backbone" else projected[len(support_idx) : n]),
                    query_y,
                    len(labels),
                    args.covariance_weight,
                    args.covariance_beta,
                    args.logdet_weight,
                    args.covariance_mode,
                    args.covariance_shrinkage,
                )
                loss = supcon_value + args.covariance_loss_weight * covariance_value
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_supcon += float(supcon_value.item())
            total_cov += float(covariance_value.item())
        scheduler.step()

        model.eval()
        if args.covariance_mode == "legacy":
            train_z, train_y, _ = extract_embeddings(model, train_eval_loader, device)
            valid_z, valid_y, _ = extract_embeddings(model, valid_loader, device)
            valid_pred, _ = prototype_predict(train_z, train_y, valid_z)
        else:
            train_z, train_y = extract_embedding_variant(model, train_eval_loader, device, args.loss_embedding == "projection")
            valid_z, valid_y = extract_embedding_variant(model, valid_loader, device, args.loss_embedding == "projection")
            valid_pred = pooled_mahalanobis_predict(train_z, train_y, valid_z)
        valid_metrics = compute_metrics(valid_y, valid_pred, labels)
        row = {
            "epoch": epoch,
            "supcon_loss": total_supcon / args.episodes_per_epoch,
            "covariance_loss": total_cov / args.episodes_per_epoch,
            "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"],
            "seconds": time.time() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(
            f"epoch={epoch:03d} supcon={row['supcon_loss']:.4f} covariance={row['covariance_loss']:.4f} "
            f"valid_acc={row['valid_accuracy']:.4f} valid_macro_f1={row['valid_macro_f1']:.4f}"
        )
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, valid_metrics, vars(args), class_to_idx)
        if row["valid_macro_f1"] > best_score:
            best_score = row["valid_macro_f1"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, valid_metrics, vars(args), class_to_idx)
            log(f"New best valid macro_f1={best_score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

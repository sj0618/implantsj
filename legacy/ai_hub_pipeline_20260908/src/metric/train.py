"""Metric 임베딩 학습 — embedding_loss 축의 모든 선택지.

  supcon          two-view P×K 배치 + SupCon
  triplet         P×K 배치 + batch-hard Triplet
  arcface         margin softmax CE
  covaware_supcon SupCon + episodic covariance-aware prototype loss
                  (Mahalanobis 거리에서 잘 분리되는 임베딩을 직접 학습)
  none            학습 생략 — 드라이버가 백본 임베딩으로 바로 평가

매 epoch valid를 mean-prototype + cosine으로 평가해 macro_f1 기준 best 저장.
출처: implantsj src/metric_learning.py, src/arcface.py,
      tools/train_covariance_aware_supcon.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .core import (
    ImageClassificationDataset,
    PKBatchSampler,
    build_transforms,
    choose_device,
    compute_metrics,
    create_loader,
    label_mapping,
    load_manifest,
    log,
    resolve_pk_batch_size,
    seed_everything,
    write_json,
)
from .models import (
    ArcFaceHead,
    BatchHardTripletLoss,
    MetricModel,
    SupConLoss,
    build_metric_model,
    covariance_loss,
    load_checkpoint,
    save_checkpoint,
    unwrap_state_dict,
)
from .prototypes import build_prototypes
from .scoring import Scorer

TRAIN_LOSSES = ("supcon", "triplet", "arcface", "covaware_supcon")


class TwoViewDataset(Dataset):
    """같은 이미지의 서로 다른 augmentation 2장 (SupCon / covaware용)."""

    def __init__(self, manifest: str, split: str, class_to_idx: Dict[str, int], image_size: int):
        self.base1 = ImageClassificationDataset(
            manifest, split=split, class_to_idx=class_to_idx,
            transform=build_transforms(image_size, train=True, strong_aug=True),
        )
        self.base2 = ImageClassificationDataset(
            manifest, split=split, class_to_idx=class_to_idx,
            transform=build_transforms(image_size, train=True, strong_aug=True),
        )

    def __len__(self):
        return len(self.base1)

    def __getitem__(self, idx):
        x1, y = self.base1[idx]
        x2, _ = self.base2[idx]
        return x1, x2, y


def extract_embeddings(model, loader, device, source: str = "projection"):
    """(features, int_labels, paths). source=backbone이면 projection 미통과."""
    model.eval()
    zs, ys, paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, p = batch
            else:
                x, y = batch
                p = [""] * len(y)
            x = x.to(device, non_blocking=True)
            z = model.forward_features(x) if source == "backbone" else model(x)
            zs.append(z.float().cpu().numpy())
            ys.extend(y.numpy().tolist())
            paths.extend(list(p))
    return np.concatenate(zs, axis=0), np.asarray(ys), paths


def _valid_prototype_score(model, manifest, class_to_idx, image_size, batch_size, num_workers, device):
    """valid를 mean-prototype + cosine으로 평가 (epoch별 best 선택 기준)."""
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    classes = [idx_to_class[i] for i in range(len(idx_to_class))]
    train_loader = create_loader(manifest, "train", class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)
    valid_loader = create_loader(manifest, "valid", class_to_idx, image_size, batch_size, num_workers, train=False, return_path=True)
    train_z, train_y, _ = extract_embeddings(model, train_loader, device)
    valid_z, valid_y, _ = extract_embeddings(model, valid_loader, device)
    train_labels = np.asarray(classes, dtype=object)[train_y]
    protos = build_prototypes(train_z, train_labels, classes, "mean")
    scores = Scorer(distance="cosine").fit(train_z, train_labels, classes).score(valid_z, protos)
    return compute_metrics(valid_y, scores.argmin(axis=1), classes)


def _sample_episode(class_indices, support_shots: int, query_shots: int, rng):
    """클래스마다 support/query를 뽑아 하나의 episode 구성."""
    support, support_y, query, query_y = [], [], [], []
    for label in sorted(class_indices):
        selected = rng.choice(class_indices[label], support_shots + query_shots, replace=False)
        support.extend(selected[:support_shots])
        query.extend(selected[support_shots:])
        support_y.extend([label] * support_shots)
        query_y.extend([label] * query_shots)
    return np.asarray(support), np.asarray(support_y), np.asarray(query), np.asarray(query_y)


def train_metric(
    *,
    manifest: str,
    output_dir: str | Path,
    backbone_name: str,
    backbone_registry: Dict[str, dict],
    loss_name: str,
    epochs: int = 80,
    batch_size: int = 32,
    samples_per_class: int = 2,
    lr: float = 3e-5,
    weight_decay: float = 0.05,
    embedding_dim: int = 128,
    temperature: float = 0.07,
    triplet_margin: float = 0.2,
    arcface_s: float = 30.0,
    arcface_m: float = 0.5,
    covaware: dict | None = None,
    dropout: float = 0.1,
    drop_path_rate: float = 0.1,
    grad_checkpointing: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    patience: int = 12,
    device: str = "auto",
    amp: bool = True,
) -> Path:
    if loss_name not in TRAIN_LOSSES:
        raise ValueError(f"loss_name must be one of {TRAIN_LOSSES} (none은 학습을 건너뜀)")

    seed_everything(seed)
    resolved = choose_device(device)
    torch_device = torch.device(resolved)
    covaware = dict(covaware or {})

    df = load_manifest(manifest, split=None)
    class_to_idx, _ = label_mapping(df)
    run_dir = Path(output_dir).expanduser().resolve()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    model, image_size = build_metric_model(
        backbone_name, backbone_registry, embedding_dim,
        dropout=dropout, drop_path_rate=drop_path_rate,
        grad_checkpointing=grad_checkpointing,
    )
    model = model.to(torch_device)

    args_payload = {
        "manifest": str(manifest), "backbone": backbone_name, "loss": loss_name,
        "epochs": epochs, "batch_size": batch_size, "samples_per_class": samples_per_class,
        "lr": lr, "weight_decay": weight_decay, "embedding_dim": embedding_dim,
        "temperature": temperature, "triplet_margin": triplet_margin,
        "arcface_s": arcface_s, "arcface_m": arcface_m, "covaware": covaware,
        "image_size": image_size, "seed": seed, "device": resolved,
    }
    write_json(run_dir / "args.json", args_payload)
    write_json(run_dir / "class_to_idx.json", class_to_idx)
    log(f"metric train: backbone={backbone_name} loss={loss_name} → {run_dir}")

    # ---- loss별 학습 구성 ----
    arc_head = None
    episode_ctx = None
    if loss_name == "arcface":
        arc_head = ArcFaceHead(embedding_dim, len(class_to_idx), s=arcface_s, m=arcface_m).to(torch_device)
        criterion = torch.nn.CrossEntropyLoss()
        params = list(model.parameters()) + list(arc_head.parameters())
        loader = create_loader(manifest, "train", class_to_idx, image_size, batch_size, num_workers, train=True)
    elif loss_name == "covaware_supcon":
        criterion = SupConLoss(temperature)
        params = model.parameters()
        dataset = TwoViewDataset(manifest, "train", class_to_idx, image_size)
        train_df = dataset.base1.df
        targets = train_df["label"].map(class_to_idx.__getitem__).to_numpy()
        class_indices = {i: np.flatnonzero(targets == i) for i in range(len(class_to_idx))}
        support_shots = int(covaware.get("support_shots", 5))
        query_shots = int(covaware.get("query_shots", 5))
        required = support_shots + query_shots
        too_small = {i: len(v) for i, v in class_indices.items() if len(v) < required}
        if too_small:
            raise ValueError(
                f"covaware_supcon: 클래스마다 support+query={required}장이 필요합니다. 부족: {too_small}"
            )
        episode_ctx = {
            "dataset": dataset, "class_indices": class_indices,
            "support_shots": support_shots, "query_shots": query_shots,
            "episodes_per_epoch": int(covaware.get("episodes_per_epoch", 20)),
            "rng": np.random.default_rng(seed),
        }
        loader = None
    else:
        criterion = SupConLoss(temperature) if loss_name == "supcon" else BatchHardTripletLoss(triplet_margin)
        params = model.parameters()
        ds = TwoViewDataset(manifest, "train", class_to_idx, image_size)
        train_labels = ds.base1.df["label"].astype(str).tolist()
        sampler_bs = resolve_pk_batch_size(train_labels, batch_size, samples_per_class)
        if sampler_bs != batch_size:
            log(f"P×K 배치 조정: {batch_size} → {sampler_bs}")
        sampler = PKBatchSampler(train_labels, sampler_bs, samples_per_class, seed)
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = amp and torch_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    patience_count = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        if arc_head is not None:
            arc_head.train()
        total_loss, n, nonfinite = 0.0, 0, 0
        t0 = time.time()

        if loss_name == "covaware_supcon":
            total_loss, n, nonfinite = _run_covaware_epoch(
                model, episode_ctx, criterion, optimizer, scaler, use_amp,
                torch_device, len(class_to_idx), covaware,
            )
        else:
            if loss_name != "arcface":
                loader.batch_sampler.set_epoch(epoch)
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if loss_name == "arcface":
                        x, y = batch
                        x, y = x.to(torch_device, non_blocking=True), y.to(torch_device, non_blocking=True)
                        loss = criterion(arc_head(model(x), y), y)
                    elif loss_name == "supcon":
                        x1, x2, y = batch
                        x = torch.cat([x1, x2], dim=0).to(torch_device, non_blocking=True)
                        y = torch.cat([y, y], dim=0).to(torch_device, non_blocking=True)
                        loss = criterion(model(x), y)
                    else:  # triplet
                        x1, _, y = batch
                        x1, y = x1.to(torch_device, non_blocking=True), y.to(torch_device, non_blocking=True)
                        loss = criterion(model(x1), y)
                # positive pair가 없는 SupCon 배치는 non-finite → weight 오염 방지
                if not torch.isfinite(loss):
                    nonfinite += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                bs = int(y.size(0))
                total_loss += float(loss.item()) * bs
                n += bs

        scheduler.step()
        if n == 0:
            raise RuntimeError(
                f"epoch={epoch}: 모든 배치가 non-finite loss ({nonfinite}개). "
                f"batch_size를 올리거나 클래스 수를 줄여야 함."
            )
        if nonfinite:
            log(f"WARNING non-finite loss 배치 {nonfinite}개 skip")

        valid = _valid_prototype_score(model, manifest, class_to_idx, image_size, batch_size, num_workers, torch_device)
        score = valid["macro_f1"]
        history.append({
            "epoch": epoch, "train_loss": total_loss / max(1, n),
            "valid_accuracy": valid["accuracy"], "valid_macro_f1": score,
            "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0,
        })
        pd.DataFrame(history).to_csv(run_dir / "metrics" / "history.csv", index=False)
        log(f"epoch={epoch:03d} loss={total_loss / max(1, n):.4f} "
            f"valid_acc={valid['accuracy']:.4f} valid_macro_f1={score:.4f}")

        # NaN weight 체크포인트 저장 방지
        bad = [k for k, t in model.state_dict().items()
               if torch.is_tensor(t) and not torch.isfinite(t.float()).all()]
        if bad:
            raise RuntimeError(f"epoch={epoch}: NaN/Inf 텐서 {len(bad)}개. 체크포인트 저장 거부.")

        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, optimizer, epoch, valid, args_payload, class_to_idx)
        if score > best_score:
            best_score = score
            patience_count = 0
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, optimizer, epoch, valid, args_payload, class_to_idx)
            log(f"new best valid macro_f1={best_score:.4f}")
        else:
            patience_count += 1
            if patience_count >= patience:
                log(f"early stopping at epoch={epoch}")
                break

    return run_dir


def _run_covaware_epoch(model, ctx, supcon, optimizer, scaler, use_amp, device, num_classes, covaware):
    """SupCon + episodic covariance-aware prototype loss 한 epoch."""
    import torch.nn.functional as F

    dataset = ctx["dataset"]
    total_loss, n, nonfinite = 0.0, 0, 0
    loss_embedding = covaware.get("loss_embedding", "backbone")
    cov_loss_weight = float(covaware.get("loss_weight", 1.0))

    for _ in range(ctx["episodes_per_epoch"]):
        s_idx, s_y_np, q_idx, q_y_np = _sample_episode(
            ctx["class_indices"], ctx["support_shots"], ctx["query_shots"], ctx["rng"]
        )
        order = np.concatenate([s_idx, q_idx])
        first = torch.stack([dataset[int(i)][0] for i in order]).to(device)
        second = torch.stack([dataset[int(i)][1] for i in order]).to(device)
        n_support, n_total = len(s_idx), len(s_idx) + len(q_idx)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            h1 = model.forward_features(first)
            h2 = model.forward_features(second)
            projected = F.normalize(model.projector(torch.cat([h1, h2], dim=0)), dim=1)
            supcon_labels = torch.tensor(
                np.concatenate([s_y_np, q_y_np, s_y_np, q_y_np]), device=device, dtype=torch.long
            )
            supcon_value = supcon(projected, supcon_labels)

            if loss_embedding == "backbone":
                cov_support = F.normalize(h1[:n_support], dim=1)
                cov_query = F.normalize(h1[n_support:n_total], dim=1)
            else:
                cov_support = projected[:n_support]
                cov_query = projected[n_support:n_total]
            cov_value = covariance_loss(
                cov_support,
                torch.tensor(s_y_np, device=device, dtype=torch.long),
                cov_query,
                torch.tensor(q_y_np, device=device, dtype=torch.long),
                num_classes,
                float(covaware.get("covariance_weight", 0.75)),
                float(covaware.get("beta", 1e-3)),
                float(covaware.get("logdet_weight", 1.0)),
                covaware.get("covariance_mode", "legacy"),
                float(covaware.get("covariance_shrinkage", 0.1)),
            )
            loss = supcon_value + cov_loss_weight * cov_value

        if not torch.isfinite(loss):
            nonfinite += 1
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * n_total
        n += n_total
    return total_loss, n, nonfinite


def load_trained_model(
    checkpoint: str | Path,
    backbone_name: str,
    backbone_registry: Dict[str, dict],
    embedding_dim: int,
    device,
) -> tuple[MetricModel, int]:
    """학습된 metric 체크포인트 로드 (백본 구조는 레지스트리에서 재구성)."""
    model, image_size = build_metric_model(backbone_name, backbone_registry, embedding_dim)
    state = unwrap_state_dict(load_checkpoint(checkpoint, map_location="cpu"))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), image_size

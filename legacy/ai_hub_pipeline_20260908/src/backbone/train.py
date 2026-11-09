"""백본 학습 — 4종 × finetune 범위 × 프로토콜.

백본 4종:
  vit          일반 ViT
  band-vit     ViT + band-attention prior loss (band 영역에 집중하도록 유도)
  dinov2       일반 DINOv2
  band-dinov2  DINOv2 + band-attention prior loss (band-vit와 동일 방식)

finetune 범위 (finetune_mode):
  full     전체 파라미터 학습
  partial  마지막 n개 block + norm/head만 학습, 나머지 frozen

프로토콜 (protocol):
  closed27_open15  27클래스 manifest로만 학습 (open 15는 백본에 노출 금지 — leakage 방지)
  full42           42클래스 전부 학습

산출물: 분류 head를 제외한 백본 weight → Stage 2/3이 레지스트리 경로로 끌어다 쓴다.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

from .core import (
    HEAD_PREFIXES,
    BackboneDataset,
    BackboneSpec,
    apply_finetune_mode,
    attach_attention_capture,
    band_attention_loss,
    choose_device,
    class_weights,
    compute_metrics,
    create_model,
    describe_model,
    log,
    make_loader,
    patch_grid_size,
    read_manifest,
    save_backbone,
    seed_everything,
    write_json,
)


def _run_epoch(model, loader, device, criterion, optimizer, *, use_band: bool,
               band_alpha: float, band_weight: float, scaler, use_amp: bool) -> dict:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "ce": 0.0, "band": 0.0}
    seen, correct = 0, 0
    targets_all, preds_all = [], []

    for batch in loader:
        if use_band:
            images, targets, band_masks = batch
            band_masks = band_masks.to(device, non_blocking=True)
        else:
            images, targets = batch
            band_masks = None
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            ce = criterion(logits, targets)
            band = torch.zeros((), device=device)
            # band loss는 학습할 때만 (평가에서는 attention을 강제할 이유가 없음)
            if training and use_band and band_weight > 0:
                band = band_attention_loss(model, band_masks, band_alpha)
            loss = ce + band_weight * band

        if training:
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0
            )
            scaler.step(optimizer)
            scaler.update()

        predictions = logits.argmax(dim=1)
        count = int(targets.size(0))
        totals["loss"] += float(loss.detach()) * count
        totals["ce"] += float(ce.detach()) * count
        totals["band"] += float(band.detach()) * count
        seen += count
        correct += int(predictions.eq(targets).sum())
        targets_all.extend(targets.detach().cpu().tolist())
        preds_all.extend(predictions.detach().cpu().tolist())

    seen = max(seen, 1)
    return {
        "loss": totals["loss"] / seen,
        "ce_loss": totals["ce"] / seen,
        "band_loss": totals["band"] / seen,
        "accuracy": correct / seen,
        "_targets": targets_all,
        "_preds": preds_all,
    }


def train_backbone(
    *,
    spec: BackboneSpec,
    manifest: str,
    protocol: str,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    finetune_mode: str = "full",
    trainable_blocks: int = 4,
    band_alpha: float = 4.0,
    band_loss_weight: float = 0.5,
    attention_blocks: int = 6,
    mask_column: str = "mask_path",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    use_class_weights: bool = False,
    dropout: float = 0.1,
    drop_path_rate: float = 0.1,
    grad_checkpointing: bool = False,
    num_workers: int = 4,
    seed: int = 42,
    patience: int = 12,
    device: str = "auto",
    amp: bool = True,
    embed_plan=None,
) -> Path:
    seed_everything(seed)
    resolved = choose_device(device)
    torch_device = torch.device(resolved)
    run_dir = Path(output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # band-* 백본만 mask 필요
    frame = read_manifest(manifest, require_mask=spec.use_band, mask_column=mask_column)
    classes = sorted(frame["label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    log(f"[{spec.name}/{finetune_mode}] protocol={protocol} classes={len(classes)} "
        f"samples={len(frame)} band={spec.use_band}")

    model = create_model(spec, len(classes), dropout=dropout, drop_path_rate=drop_path_rate,
                         grad_checkpointing=grad_checkpointing)
    model_info = describe_model(model, spec)
    log(f"  model: {model_info['parameters']:,} params, depth={model_info['depth']}, "
        f"embed_dim={model_info['embed_dim']}")
    finetune_info = apply_finetune_mode(model, finetune_mode, trainable_blocks)
    log(f"  finetune: {finetune_info}")

    patch_grid = None
    if spec.use_band:
        attach_attention_capture(model, attention_blocks)
        patch_grid = patch_grid_size(model)
        log(f"  band-attention: patch_grid={patch_grid} blocks={attention_blocks} "
            f"alpha={band_alpha} weight={band_loss_weight}")
    model = model.to(torch_device)

    splits = {s: frame[frame["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test")}
    if splits["train"].empty or splits["valid"].empty:
        raise ValueError(f"manifest에 train/valid split이 필요합니다: {manifest}")
    loaders = {
        "train": make_loader(
            BackboneDataset(splits["train"], class_to_idx, spec.image_size, train=True,
                            patch_grid=patch_grid, mask_column=mask_column),
            batch_size, num_workers, shuffle=True),
        "valid": make_loader(
            BackboneDataset(splits["valid"], class_to_idx, spec.image_size, train=False,
                            patch_grid=patch_grid, mask_column=mask_column),
            batch_size, num_workers, shuffle=False),
    }
    if not splits["test"].empty:
        loaders["test"] = make_loader(
            BackboneDataset(splits["test"], class_to_idx, spec.image_size, train=False,
                            patch_grid=patch_grid, mask_column=mask_column),
            batch_size, num_workers, shuffle=False)

    weights = None
    if use_class_weights:
        weights = class_weights(splits["train"], classes).to(torch_device)
        log(f"  class weights 적용: 최소 {float(weights.min()):.3f} / "
            f"최대 {float(weights.max()):.3f} (빈도 역수)")
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weights)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    use_amp = amp and torch_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config_payload = {
        **model_info, "use_band": spec.use_band,
        "grad_checkpointing": grad_checkpointing,
        "use_class_weights": use_class_weights,
        "protocol": protocol, "manifest": str(manifest),
        "finetune": finetune_info, "num_classes": len(classes), "labels": classes,
        "band": {"alpha": band_alpha, "loss_weight": band_loss_weight,
                 "attention_blocks": attention_blocks} if spec.use_band else None,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "device": resolved,
    }
    write_json(run_dir / "config.json", config_payload)

    best_f1, patience_count, history = -1.0, 0, []
    # best 가중치 복사본은 나중에 쓸 때만 (test 평가·임베딩) 들고 있는다 — giant 백본이면 수 GB 다.
    keep_best = ("test" in loaders) or (embed_plan is not None and embed_plan.enabled)
    best_state = None
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_stats = _run_epoch(model, loaders["train"], torch_device, criterion, optimizer,
                                 use_band=spec.use_band, band_alpha=band_alpha,
                                 band_weight=band_loss_weight, scaler=scaler, use_amp=use_amp)
        valid_stats = _run_epoch(model, loaders["valid"], torch_device, criterion, None,
                                 use_band=spec.use_band, band_alpha=band_alpha,
                                 band_weight=band_loss_weight, scaler=scaler, use_amp=use_amp)
        scheduler.step()
        valid_metrics = compute_metrics(valid_stats["_targets"], valid_stats["_preds"], classes)

        history.append({
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_stats["loss"], "train_ce": train_stats["ce_loss"],
            "train_band_loss": train_stats["band_loss"], "train_accuracy": train_stats["accuracy"],
            "valid_loss": valid_stats["loss"], "valid_accuracy": valid_metrics["accuracy"],
            "valid_macro_f1": valid_metrics["macro_f1"], "seconds": time.time() - t0,
        })
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(f"  epoch={epoch:03d} train_loss={train_stats['loss']:.4f} "
            f"(ce={train_stats['ce_loss']:.4f} band={train_stats['band_loss']:.4f}) "
            f"valid_acc={valid_metrics['accuracy']:.4f} valid_macro_f1={valid_metrics['macro_f1']:.4f}")

        if valid_metrics["macro_f1"] > best_f1:
            best_f1 = valid_metrics["macro_f1"]
            patience_count = 0
            save_backbone(checkpoint_path, model, spec, {
                "epoch": epoch, "protocol": protocol, "finetune": finetune_info,
                "valid_metrics": valid_metrics, "labels": classes,
                "band": config_payload["band"],
            })
            if keep_best:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"  best backbone 저장: {checkpoint_path} (valid_macro_f1={best_f1:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                log(f"  early stopping at epoch={epoch}")
                break

    # best epoch 가중치로 되돌린다 — test 평가와 임베딩은 저장된 checkpoint 와 같은 모델이어야 한다.
    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {"best_valid_macro_f1": best_f1, "checkpoint": str(checkpoint_path),
               **config_payload}
    if "test" in loaders:
        test_stats = _run_epoch(model, loaders["test"], torch_device, criterion, None,
                                use_band=spec.use_band, band_alpha=band_alpha,
                                band_weight=band_loss_weight, scaler=scaler, use_amp=use_amp)
        summary["test_metrics"] = compute_metrics(test_stats["_targets"], test_stats["_preds"], classes)
        log(f"  test: {summary['test_metrics']}")
    write_json(run_dir / "summary.json", summary)

    # ---- 학습 종료 hook: 전체 이미지 backbone/logits 임베딩 (best 가중치 기준) ----
    if embed_plan is not None and embed_plan.enabled and best_state is not None:
        from embeddings import square_pad_transform, timm_classifier_forward, try_dump_embeddings

        # band-* 의 attention 캡처 hook 은 학습용이다. 붙여둔 채 임베딩을 뽑으면 block 마다
        # [B, heads, tokens, tokens] 를 쌓아 학습보다 메모리를 더 쓴다 → 떼고 뽑는다.
        for handle in getattr(model, "_band_attention_handles", []):
            handle.remove()
        model._band_attention_handles, model._band_attention_stores = [], []
        # 백본 checkpoint 는 분류 head 를 뺀다(save_backbone). head logits 를 재현할 수 있도록
        # head 가중치를 따로 남기고 npz 가 그 경로를 가리키게 한다.
        head_state = {k: v for k, v in best_state.items() if k.startswith(HEAD_PREFIXES)}
        head_checkpoint = run_dir / "head_best.pt"
        if head_state:
            torch.save(head_state, head_checkpoint)
        else:   # head 이름이 HEAD_PREFIXES 밖이면 save_backbone 도 못 벗겼으니 백본 checkpoint 안에 있다
            log(f"  head 파라미터를 {HEAD_PREFIXES} 로 찾지 못함 → 백본 checkpoint 에 포함된 채 저장됨")
        try_dump_embeddings(
            output_dir=run_dir / "embeddings", plan=embed_plan,
            forward=timm_classifier_forward(model),
            transform=square_pad_transform(spec.image_size),
            stage="backbone", model_name=spec.timm_model, checkpoint=checkpoint_path,
            device=torch_device, head_kind="logits",
            extra={"backbone": spec.name, "finetune_mode": finetune_mode,
                   "protocol": protocol, "labels": classes,
                   "head_in_backbone_checkpoint": not head_state,
                   "head_checkpoint": str(head_checkpoint) if head_state else ""},
        )
    return Path(checkpoint_path)

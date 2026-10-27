"""feature 모델 학습 + 예측 확률 추출 — train.mode 두 가지.

  separate  feature마다 백본+head 를 통째로 하나씩 (독립 모델, 간섭 없음, feature 수만큼 학습)
            → checkpoints/<feature>.pt
  shared    백본 1개 + feature마다 head (multi-head, 학습 1회, 표현 공유)
            → checkpoints/shared.pt

실행 단위는 yaml의 run.features로 정한다:
  all           feature 전부
  [System, ...] 지정한 feature만 → separate 는 나눠서 여러 번 돌릴 수 있다
run.skip_existing: true 면 이미 학습된 체크포인트는 건너뛰고 재사용한다
(폐쇄망에서 feature 를 며칠에 걸쳐 나눠 돌릴 때 필요). shared 는 저장된 head 목록이
이번 run.features 와 같을 때만 재사용한다.

체크포인트마다 학습이 끝나면 전체 이미지의 backbone/head 임베딩을 npz 로 남긴다
(src/embeddings.py, base.yaml embeddings.*): separate 는 embeddings/<feature>/, shared 는 embeddings/.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from .core import (
    AttributeDataset,
    attribute_class_weights,
    build_feature_model,
    build_shared_model,
    build_transforms,
    log,
    make_loader,
    masked_cross_entropy,
    write_json,
)
from .spec_table import SpecTable

MODES = ("separate", "shared")


def _attribute_metrics(probabilities: Dict[str, np.ndarray], class_index: np.ndarray,
                       table: SpecTable, attributes: Sequence[str]) -> dict:
    """feature별 정확도/macro-F1 (unknown 정답은 분모에서 제외)."""
    from sklearn.metrics import f1_score

    per_attribute = {}
    accuracies, f1s = [], []
    for a in attributes:
        if a not in probabilities:
            continue
        ai = table.attributes.index(a)
        truth = table.target[class_index, ai]
        keep = truth >= 0
        if not keep.any():
            per_attribute[a] = {"n": 0, "note": "정답이 전부 unknown이라 평가 불가"}
            continue
        predicted = probabilities[a].argmax(axis=1)
        acc = float((predicted[keep] == truth[keep]).mean())
        f1 = float(f1_score(truth[keep], predicted[keep], average="macro", zero_division=0))
        per_attribute[a] = {"n": int(keep.sum()), "accuracy": acc, "macro_f1": f1,
                            "values": table.vocab[a]}
        accuracies.append(acc)
        f1s.append(f1)
    return {
        "per_attribute": per_attribute,
        "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "mean_macro_f1": float(np.mean(f1s)) if f1s else 0.0,
    }


@torch.no_grad()
def predict_probabilities(model, loader, device, attributes: Sequence[str]):
    """({feature: [N, values] 확률}, class_index [N], paths [N]) — 모델의 head 전부를 한 번에."""
    model.eval()
    chunks: Dict[str, list] = {a: [] for a in attributes}
    class_index, paths = [], []
    for images, targets, batch_paths in loader:
        logits = model(images.to(device, non_blocking=True))
        for a in attributes:
            chunks[a].append(torch.softmax(logits[a].float(), dim=1).cpu().numpy())
        class_index.extend(targets.numpy().tolist())
        paths.extend(list(batch_paths))
    probabilities = {a: np.concatenate(v, axis=0) for a, v in chunks.items()}
    return probabilities, np.asarray(class_index), paths


def _train_model(
    model, attributes: Sequence[str], table: SpecTable, loaders: dict, device, checkpoint: Path,
    *, mode: str, epochs: int, lr: float, weight_decay: float, label_smoothing: float,
    patience: int, amp: bool, run_dir: Path, history_name: str, class_weights: dict | None = None,
) -> float:
    """모델 하나(separate: head 1개, shared: head n개)를 학습한다.

    선택 기준은 학습하는 feature 들의 valid macro-F1 평균. best 체크포인트를 저장하고
    최고 점수를 돌려준다. history_<history_name>.csv 에 feature 별 F1 도 같이 남긴다.
    """
    attributes = list(attributes)
    target_lut = torch.from_numpy(table.target).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best, best_epoch, history = -1.0, -1, []
    for epoch in range(1, epochs + 1):
        model.train()
        running, batches = 0.0, 0
        t0 = time.time()
        for images, targets, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            target = target_lut[targets.to(device)]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # logits 에 있는 head 만 loss 에 들어간다 → separate/shared 가 같은 코드로 학습된다
                loss = masked_cross_entropy(model(images), target, table.attributes,
                                            label_smoothing, class_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
            batches += 1
        scheduler.step()

        probabilities, class_index, _ = predict_probabilities(model, loaders["valid"], device, attributes)
        metrics = _attribute_metrics(probabilities, class_index, table, attributes)
        score = metrics["mean_macro_f1"]
        per_attribute = {a: float(metrics["per_attribute"].get(a, {}).get("macro_f1", float("nan")))
                         for a in attributes}
        row = {"epoch": epoch, "train_loss": running / max(batches, 1),
               "valid_macro_f1": score, "valid_accuracy": metrics["mean_accuracy"],
               "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - t0}
        if len(attributes) > 1:
            row.update({f"valid_macro_f1__{a}": f1 for a, f1 in per_attribute.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / f"history_{history_name}.csv", index=False)
        log(f"    epoch={epoch:03d} loss={running / max(batches, 1):.4f} "
            f"valid_macro_f1={score:.4f}")

        if score > best:
            best, best_epoch = score, epoch
            payload = {
                "model_state_dict": model.state_dict(), "epoch": epoch, "mode": mode,
                "attributes": attributes, "classes": table.classes,
                "valid_macro_f1": score, "valid_macro_f1_per_attribute": per_attribute,
            }
            if mode == "separate":
                # 예전 체크포인트와 같은 키 (attribute: str, values: list)
                payload["attribute"] = attributes[0]
                payload["values"] = table.vocab[attributes[0]]
            else:
                payload["values"] = {a: table.vocab[a] for a in attributes}
            torch.save(payload, checkpoint)
        if epoch - best_epoch >= patience:
            log(f"    {patience} epoch 개선 없음 → 조기종료")
            break

    log(f"    best valid_macro_f1={best:.4f} (epoch {best_epoch}) → {checkpoint.name}")
    return best


def _dump_embeddings(model, *, embed_plan, output_dir: Path, image_size: int, checkpoint: Path,
                     backbone_entry: dict, device, extra: dict) -> None:
    """학습 종료 hook — 전체 이미지 backbone feature + head logits 를 npz 로."""
    if embed_plan is None or not embed_plan.enabled:
        return
    from embeddings import try_dump_embeddings

    model.eval()

    def forward(images):
        features = model.forward_features(images)
        return features, model.forward_heads(features)

    try_dump_embeddings(
        output_dir=output_dir, plan=embed_plan, forward=forward,
        transform=build_transforms(image_size, False), stage="attributes",
        model_name=backbone_entry["timm"], checkpoint=checkpoint, device=device,
        head_kind="logits", extra=extra,
    )


def train_attributes(
    *,
    backbone_entry: dict,
    table: SpecTable,
    run_features: Sequence[str],
    frames: Dict[str, pd.DataFrame],
    image_size: int,
    run_dir: str | Path,
    mode: str = "separate",
    skip_existing: bool = True,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 5e-5,
    weight_decay: float = 0.05,
    label_smoothing: float = 0.1,
    use_class_weights: bool = False,
    dropout: float = 0.1,
    drop_path_rate: float = 0.1,
    num_workers: int = 4,
    patience: int = 12,
    device: str = "cuda",
    amp: bool = True,
    pretrained: bool = True,
    embed_plan=None,
) -> dict:
    """mode 에 따라 feature 모델을 학습하고 각 split 의 예측 확률을 돌려준다.

    separate: feature마다 독립 모델. 이번 실행에서 학습하지 않은 feature 라도 체크포인트가
              있으면 불러와 추론만 한다 (나눠서 돌린 결과를 모아 fuse 할 수 있도록).
    shared  : run_features 전부를 head 로 갖는 모델 하나.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}: {mode!r}")
    run_dir = Path(run_dir).expanduser().resolve()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)

    class_to_idx = {c: i for i, c in enumerate(table.classes)}
    loaders = {
        split: make_loader(
            AttributeDataset(frame, class_to_idx, image_size, train=(split == "train")),
            batch_size, num_workers, shuffle=(split == "train"))
        for split, frame in frames.items()
    }
    for split, loader in loaders.items():
        log(f"  {split}: {len(loader.dataset)}장")

    weights = None
    if use_class_weights:
        train_index = np.asarray([class_to_idx[str(l)] for l in frames["train"]["label"]
                                  if str(l) in class_to_idx])
        weights = attribute_class_weights(table.target, table.attributes, table.vocab,
                                          train_index, torch_device)
        log("  class weights 적용 (feature별 값 빈도 역수)")

    to_run = [a for a in run_features if a in table.attributes]
    unknown = [a for a in run_features if a not in table.attributes]
    if unknown:
        log(f"  표에서 제외된 feature라 건너뜀: {unknown}")
    if not to_run:
        raise ValueError("학습할 feature가 없습니다 (run.features / spec table 확인)")

    outputs: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in loaders}
    class_index_by_split: Dict[str, np.ndarray] = {}
    paths_by_split: Dict[str, list] = {}
    checkpoints: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    train_kwargs = dict(epochs=epochs, lr=lr, weight_decay=weight_decay,
                        label_smoothing=label_smoothing, patience=patience, amp=amp,
                        run_dir=run_dir, class_weights=weights)

    def collect(model, attributes: Sequence[str]) -> None:
        for split, loader in loaders.items():
            probabilities, class_index, paths = predict_probabilities(
                model, loader, torch_device, attributes)
            outputs[split].update(probabilities)
            class_index_by_split[split] = class_index
            paths_by_split[split] = paths

    if mode == "shared":
        checkpoint = run_dir / "checkpoints" / "shared.pt"
        head_sizes = {a: len(table.vocab[a]) for a in to_run}
        log(f"  shared multi-head 모델: 백본 1개 + head {len(head_sizes)}개 {head_sizes}")
        model = build_shared_model(backbone_entry, head_sizes, dropout=dropout,
                                   drop_path_rate=drop_path_rate, pretrained=pretrained).to(torch_device)
        reused = False
        if skip_existing and checkpoint.is_file():
            payload = torch.load(checkpoint, map_location=torch_device)
            saved_sizes = {a: len(v) for a, v in (payload.get("values") or {}).items()} \
                if isinstance(payload.get("values"), dict) else {}
            if list(payload.get("attributes", [])) == to_run and saved_sizes == head_sizes:
                model.load_state_dict(payload["model_state_dict"])
                scores = {a: float(v) for a, v in payload.get("valid_macro_f1_per_attribute", {}).items()}
                reused = True
                log(f"    기존 shared 체크포인트 재사용 "
                    f"(valid_macro_f1={float(payload.get('valid_macro_f1', float('nan'))):.4f})")
            else:
                log(f"    기존 shared.pt 의 head {payload.get('attributes')}/{saved_sizes} ≠ "
                    f"이번 {to_run}/{head_sizes} → 다시 학습")
        if not reused:
            _train_model(model, to_run, table, loaders, torch_device, checkpoint,
                         mode="shared", history_name="shared", **train_kwargs)
            payload = torch.load(checkpoint, map_location=torch_device)
            model.load_state_dict(payload["model_state_dict"])
            scores = {a: float(v) for a, v in payload["valid_macro_f1_per_attribute"].items()}
        for a in to_run:
            checkpoints[a] = str(checkpoint)
        collect(model, to_run)
        _dump_embeddings(model, embed_plan=embed_plan, output_dir=run_dir / "embeddings",
                         image_size=image_size, checkpoint=checkpoint, backbone_entry=backbone_entry,
                         device=torch_device, extra={"mode": "shared", "attributes": to_run})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        for position, attribute in enumerate(to_run, 1):
            checkpoint = run_dir / "checkpoints" / f"{attribute}.pt"
            num_values = len(table.vocab[attribute])
            log(f"  [{position}/{len(to_run)}] feature={attribute} ({num_values}종) 독립 모델")

            model = build_feature_model(backbone_entry, attribute, num_values, dropout=dropout,
                                        drop_path_rate=drop_path_rate, pretrained=pretrained).to(torch_device)
            if skip_existing and checkpoint.is_file():
                payload = torch.load(checkpoint, map_location=torch_device)
                model.load_state_dict(payload["model_state_dict"])
                scores[attribute] = float(payload.get("valid_macro_f1", float("nan")))
                log(f"    기존 체크포인트 재사용 (valid_macro_f1={scores[attribute]:.4f})")
            else:
                scores[attribute] = _train_model(
                    model, [attribute], table, loaders, torch_device, checkpoint,
                    mode="separate", history_name=attribute, **train_kwargs)
                model.load_state_dict(torch.load(checkpoint, map_location=torch_device)["model_state_dict"])
            checkpoints[attribute] = str(checkpoint)
            collect(model, [attribute])
            _dump_embeddings(model, embed_plan=embed_plan, output_dir=run_dir / "embeddings" / attribute,
                             image_size=image_size, checkpoint=checkpoint, backbone_entry=backbone_entry,
                             device=torch_device, extra={"mode": "separate", "attribute": attribute})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    trained = list(outputs[next(iter(outputs))].keys())
    metrics = {split: _attribute_metrics(outputs[split], class_index_by_split[split], table, trained)
               for split in outputs}
    write_json(run_dir / "attribute_metrics.json", metrics)
    write_json(run_dir / "checkpoints.json",
               {"mode": mode, "checkpoints": checkpoints,
                # NaN 은 엄격한 JSON 파서가 거부한다 → null
                "valid_macro_f1": {a: (None if v != v else v) for a, v in scores.items()}})

    return {
        "mode": mode,
        "probabilities": outputs,
        "class_index": class_index_by_split,
        "paths": paths_by_split,
        "attributes": trained,
        "metrics": metrics,
        "checkpoints": checkpoints,
    }

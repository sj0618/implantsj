"""얼굴인식(ArcFace) 모델 위에 13 feature 추출을 얹은 실험. VLM 은 쓰지 않는다.

목적은 클래스 이름을 맞히는 게 아니라 **처음 보는 임플란트 시스템의 이미지를 보고
그 시스템이 13 feature 각각에 대해 어떤 값을 갖는지 표를 채우는 것**이다.
정답의 출처는 `configs/class_features.csv` 한 장이고, feature 를 뽑는 일은
VLM 이 아니라 여기서 학습하는 attribute head 가 한다.

기존 것을 그대로 재사용한다:
  - models.MetricModel      : 백본 + projector, forward_features 로 raw 도 꺼냄
  - arcface.ArcFaceHead     : margin softmax
  - class_features.*        : 클래스 -> 13 feature 정답표, 값 vocab, target 행렬
  - data.*, utils.*         : manifest / loader / run dir

새로 얹는 것은 두 조각이다.

1) 멀티태스크 학습
   같은 임베딩 위에서 ArcFace margin loss(클래스 분리)와 attribute CE(feature 예측)를
   동시에 건다. margin 만 걸면 임베딩이 클래스 정체성만 담고, attribute 만 걸면
   갤러리가 쓸모없어진다. 둘 다 걸어야 갤러리와 head 가 같은 표현을 공유한다.

2) open-set 분기 추론
   갤러리 최근접 유사도가 threshold 이상이면 **이미 아는 시스템**이므로 그 클래스의
   표 행을 그대로 답한다(추측할 이유가 없다). threshold 미만이면 **처음 보는
   시스템**이므로 attribute head 의 예측으로 표를 채운다.

   이 구조 덕분에 새 임플란트 시스템이 생겨도 재학습이 필요 없다. 레퍼런스 몇 장을
   갤러리에 등록하고 표에 행 하나를 추가하면 그때부터 "아는 시스템"이 된다.

폐쇄망 실행 예:
  python -m src.arcface_attributes train     --manifest m.csv --out outputs/afa
  python -m src.arcface_attributes gallery   --manifest m.csv --ckpt <run>/model.pt --out <run>/gallery.npz
  python -m src.arcface_attributes recognize --manifest m.csv --ckpt <run>/model.pt \
                                             --gallery <run>/gallery.npz --split test --threshold 0.3
  python -m src.arcface_attributes selftest   # 데이터 없이 학습·분기 동작만 확인
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .arcface import ArcFaceHead, _resolve_device
from .class_features import (
    DEFAULT_TABLE,
    DEFAULT_TRAIN_ATTRIBUTES,
    FeatureTable,
    build_targets,
    load_aliases,
    load_table,
    match_labels,
)
from .data import create_loader, load_manifest
from .models import MetricModel
from .utils import log, make_run_dir, seed_everything, write_json

# attribute head 를 어느 표현에서 뽑을지. 어느 쪽이 나은지는 측정으로 정한다.
HEAD_SOURCES = ("embedding", "backbone")


# ---------------------------------------------------------------------------
# 1) 모델 — ArcFace 임베딩 + attribute head
# ---------------------------------------------------------------------------
class ArcFaceAttributeModel(nn.Module):
    """ArcFace 인식과 13 feature 예측을 한 백본에서 동시에 한다.

    head_source
      "embedding" : 128d L2 정규화 임베딩에서 뽑는다. 갤러리와 head 가 정확히 같은
                    공간을 본다. 다만 압축된 차원이라 12개 attribute 에 좁을 수 있다.
      "backbone"  : projector 이전 raw feature(백본 num_features)에서 뽑는다.
                    정보 손실이 없는 대신 갤러리와 다른 공간을 본다.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        head_sizes: Dict[str, int],
        embedding_dim: int = 128,
        pretrained: bool = True,
        s: float = 30.0,
        m: float = 0.5,
        head_source: str = "embedding",
        dropout: float = 0.0,
    ):
        super().__init__()
        if head_source not in HEAD_SOURCES:
            raise ValueError(f"head_source must be one of {HEAD_SOURCES}, got {head_source!r}")
        self.encoder = MetricModel(model_name, embedding_dim=embedding_dim, pretrained=pretrained)
        self.head = ArcFaceHead(embedding_dim, num_classes, s=s, m=m)
        self.head_source = head_source
        raw_dim = self.encoder.backbone.num_features
        dim = embedding_dim if head_source == "embedding" else raw_dim
        self.attributes = list(head_sizes.keys())
        self.attr_heads = nn.ModuleDict(
            {a: nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, k)) for a, k in head_sizes.items()}
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(L2 정규화 임베딩, raw 백본 feature). 백본을 두 번 돌리지 않는다."""
        raw = self.encoder.forward_features(x)
        emb = F.normalize(self.encoder.projector(raw), dim=1)
        return emb, raw

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)[0]

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        emb, raw = self.encode(x)
        src = emb if self.head_source == "embedding" else raw
        attr_logits = {a: h(src) for a, h in self.attr_heads.items()}
        return self.head(emb, labels), attr_logits, emb


# ---------------------------------------------------------------------------
# 2) 정답표 준비 — manifest 의 모든 클래스로 vocab 을 만든다
# ---------------------------------------------------------------------------
class AttributeSpec:
    """표·vocab·클래스별 정답행을 한 곳에 모은다.

    vocab 은 train 클래스가 아니라 **manifest 전체 클래스**로 만든다. unseen 클래스만
    가진 값도 평가에서 보이게 하기 위해서다. 그런 값은 학습에서 한 번도 정답이 된 적이
    없으므로 head 가 원리상 예측할 수 없다. 숨기지 않고 도달 불가로 세어 보고한다.
    """

    def __init__(
        self,
        manifest_labels: Sequence[str],
        train_labels: Sequence[str],
        feature_table: str | Path = DEFAULT_TABLE,
        label_aliases: Optional[str] = None,
        attributes: Optional[Sequence[str]] = None,
    ):
        all_labels = sorted({str(x) for x in manifest_labels})
        tdf = load_table(feature_table)
        mapping, unmatched = match_labels(all_labels, tdf, load_aliases(label_aliases))
        table, target = build_targets(all_labels, mapping, tdf, attributes or DEFAULT_TRAIN_ATTRIBUTES)

        self.table: FeatureTable = table
        self.attributes: List[str] = table.attributes
        self.dropped_constant: List[str] = table.dropped_constant
        self.unmatched: List[str] = list(unmatched)
        # label -> target 행 인덱스. build_targets 는 매칭된 라벨만 입력 순서대로 담는다.
        matched = [l for l in all_labels if l in mapping]
        self.label_to_row: Dict[str, int] = {l: i for i, l in enumerate(matched)}
        self.target = target  # [matched, attributes] int64, unknown 은 -1

        train_matched = [l for l in sorted({str(x) for x in train_labels}) if l in self.label_to_row]
        self.train_labels: List[str] = train_matched
        self.unseen_labels: List[str] = [l for l in matched if l not in set(train_matched)]

        # 학습에서 한 번도 정답이 아니었던 값 = head 가 예측할 수 없는 값
        self.unreachable: Dict[str, List[str]] = {}
        train_rows = [self.label_to_row[l] for l in train_matched]
        for ai, a in enumerate(self.attributes):
            seen_idx = {int(v) for v in self.target[train_rows, ai] if int(v) >= 0} if train_rows else set()
            missing = [v for i, v in enumerate(self.table.vocab[a]) if i not in seen_idx]
            if missing:
                self.unreachable[a] = missing

    @property
    def head_sizes(self) -> Dict[str, int]:
        return {a: len(self.table.vocab[a]) for a in self.attributes}

    def row_for(self, label: str) -> Optional[np.ndarray]:
        i = self.label_to_row.get(str(label))
        return None if i is None else self.target[i]

    def majority_baseline(self) -> Dict[str, int]:
        """train 클래스에서 가장 흔한 값. '아무것도 안 배운 모델'의 점수 기준선."""
        rows = [self.label_to_row[l] for l in self.train_labels]
        out: Dict[str, int] = {}
        for ai, a in enumerate(self.attributes):
            vals = [int(v) for v in self.target[rows, ai] if int(v) >= 0] if rows else []
            out[a] = int(np.bincount(vals).argmax()) if vals else -1
        return out

    def describe(self) -> dict:
        return {
            "attributes": self.attributes,
            "dropped_constant": self.dropped_constant,
            "unmatched_labels": self.unmatched,
            "train_classes": len(self.train_labels),
            "unseen_classes": len(self.unseen_labels),
            "unseen_label_names": self.unseen_labels,
            "vocab_sizes": self.head_sizes,
            "unreachable_values": self.unreachable,
        }


# ---------------------------------------------------------------------------
# 3) 학습
# ---------------------------------------------------------------------------
def train(
    manifest: str,
    out: str,
    model_name: str = "vit_base_patch16_224",
    feature_table: str = str(DEFAULT_TABLE),
    label_aliases: Optional[str] = None,
    head_source: str = "embedding",
    attr_weight: float = 1.0,
    embedding_dim: int = 128,
    epochs: int = 40,
    batch_size: int = 32,
    image_size: int = 224,
    lr: float = 5e-5,
    s: float = 30.0,
    m: float = 0.5,
    num_workers: int = 8,
    seed: int = 42,
    device: str = "auto",
    pretrained: bool = True,
    holdout_classes: int = 0,
) -> Path:
    device = _resolve_device(device)
    seed_everything(seed)
    run_dir = make_run_dir(out, f"arcface_attr_{head_source}")

    df_all = load_manifest(manifest, split=None)
    train_labels = _train_label_set(df_all, holdout_classes)
    spec = AttributeSpec(df_all["label"].tolist(), train_labels, feature_table, label_aliases)
    if not spec.attributes:
        raise SystemExit("학습할 attribute 가 없다. 표와 데이터 클래스가 매칭되는지 확인한다.")
    if not spec.train_labels:
        raise SystemExit("표에 매칭된 train 클래스가 없다. configs/class_features.csv 를 확인한다.")

    class_to_idx = {l: i for i, l in enumerate(spec.train_labels)}
    num_classes = len(class_to_idx)
    log(f"[afa] head_source={head_source} classes={num_classes} "
        f"attributes={len(spec.attributes)} device={device} backbone={model_name}")
    log(f"[afa] attribute={spec.attributes}")
    if spec.dropped_constant:
        log(f"[afa] 값이 1종뿐이라 제외: {spec.dropped_constant}")
    if spec.unmatched:
        log(f"[afa] 표에 없어 제외된 클래스: {spec.unmatched}")
    if spec.unseen_labels:
        log(f"[afa] unseen(갤러리에 안 넣음) {len(spec.unseen_labels)}개: {spec.unseen_labels}")
    if spec.unreachable:
        log(f"[afa] head 가 예측할 수 없는 값(학습에 없음): {spec.unreachable}")
    if not spec.unseen_labels:
        log("[afa] 경고: unseen 클래스가 0개다. 갤러리가 모든 클래스를 알고 있어 "
            "head 분기가 한 번도 돌지 않는다. 이 실험의 목적인 '처음 보는 시스템의 "
            "표 채우기'는 측정되지 않는다. --holdout-classes N 을 주거나 unseen 이 "
            "분리된 데이터셋을 쓴다.")

    fit_manifest = str(_write_filtered_manifest(manifest, class_to_idx,
                                                run_dir / "fit_manifest.csv"))
    train_loader = create_loader(fit_manifest, split="train", class_to_idx=class_to_idx,
                                 image_size=image_size, batch_size=batch_size,
                                 num_workers=num_workers, train=True)
    val_loader = _maybe_loader(fit_manifest, "valid", class_to_idx, image_size, batch_size, num_workers) \
        or _maybe_loader(fit_manifest, "val", class_to_idx, image_size, batch_size, num_workers)

    lut = _target_lut(spec, class_to_idx).to(device)
    model = ArcFaceAttributeModel(model_name, num_classes, spec.head_sizes, embedding_dim,
                                  pretrained, s, m, head_source).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, device, lut, spec.attributes, attr_weight, optimizer)
        va = _run_epoch(model, val_loader, device, lut, spec.attributes, attr_weight, None) if val_loader else None
        scheduler.step()
        history.append({"epoch": epoch, "train": tr, "val": va, "lr": scheduler.get_last_lr()[0]})
        msg = (f"[afa] epoch {epoch:03d} loss={tr['loss']:.4f} "
               f"cls_acc={tr['cls_acc']:.3f} attr_acc={tr['attr_acc']:.3f}")
        if va:
            msg += f" | val cls={va['cls_acc']:.3f} attr={va['attr_acc']:.3f}"
        log(msg)
        cur = (va or tr)["attr_acc"]
        if cur >= best:
            best = cur
            _save(run_dir / "model.pt", model, class_to_idx, spec,
                  dict(model_name=model_name, embedding_dim=embedding_dim, s=s, m=m,
                       head_source=head_source, fit_manifest=fit_manifest))

    write_json(run_dir / "history.json",
               {"best_attr_acc": best, "spec": spec.describe(), "history": history})
    log(f"[afa] done. best attr_acc={best:.3f}  ->  {run_dir/'model.pt'}")
    return run_dir


def _run_epoch(model, loader, device, lut, attributes, attr_weight, optimizer):
    training = optimizer is not None
    model.train(training)
    tot_loss = cls_ok = n = 0
    attr_ok = attr_n = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        attr_t = lut[targets]                      # [B, A]  unknown 은 -1
        with torch.set_grad_enabled(training):
            cls_logits, attr_logits, _ = model(images, targets if training else None)
            loss = F.cross_entropy(cls_logits, targets)
            a_loss = images.new_zeros(())
            for ai, a in enumerate(attributes):
                a_loss = a_loss + F.cross_entropy(attr_logits[a], attr_t[:, ai], ignore_index=-1)
            loss = loss + attr_weight * a_loss / max(1, len(attributes))
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        tot_loss += float(loss) * images.size(0)
        cls_ok += int((cls_logits.argmax(1) == targets).sum())
        n += images.size(0)
        for ai, a in enumerate(attributes):
            valid = attr_t[:, ai] >= 0
            if bool(valid.any()):
                attr_ok += int((attr_logits[a][valid].argmax(1) == attr_t[valid, ai]).sum())
                attr_n += int(valid.sum())
    return {"loss": tot_loss / max(1, n),
            "cls_acc": cls_ok / max(1, n),
            "attr_acc": attr_ok / max(1, attr_n)}


# ---------------------------------------------------------------------------
# 4) 갤러리 — 클래스 centroid 와 그 클래스의 정답 표 행을 함께 등록한다
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_gallery(manifest: str, ckpt: str, out: str, split: str = "train",
                  image_size: int = 224, batch_size: int = 32, num_workers: int = 8,
                  device: str = "auto") -> Path:
    device = _resolve_device(device)
    model, class_to_idx, spec_meta, _ = load_model(ckpt, device)
    model.eval()
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    loader = create_loader(manifest, split=split, class_to_idx=class_to_idx,
                           image_size=image_size, batch_size=batch_size,
                           num_workers=num_workers, train=False)
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for images, targets in loader:
        emb = model.embed(images.to(device)).cpu()
        for e, y in zip(emb, targets.tolist()):
            if y < 0:           # 갤러리에 넣을 클래스가 아니면 건너뛴다
                continue
            sums[y] = sums.get(y, torch.zeros_like(e)) + e
            counts[y] = counts.get(y, 0) + 1

    order = sorted(sums)
    names = [idx_to_class[y] for y in order]
    centroids = np.stack([F.normalize(sums[y] / counts[y], dim=0).numpy() for y in order])
    rows = np.stack([np.asarray(spec_meta["rows"][idx_to_class[y]], dtype="int64") for y in order])

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, names=np.array(names), centroids=centroids, attr_rows=rows,
             attributes=np.array(spec_meta["attributes"]),
             counts=np.array([counts[y] for y in order]))
    log(f"[afa] gallery: {len(names)} systems x {rows.shape[1]} attributes -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 5) 인식 + 표 채우기 — open-set 분기
# ---------------------------------------------------------------------------
@torch.no_grad()
def recognize(manifest: str, ckpt: str, gallery: str, split: str = "test",
              threshold: float = 0.30, image_size: int = 224, batch_size: int = 32,
              num_workers: int = 8, device: str = "auto",
              feature_table: str = str(DEFAULT_TABLE),
              label_aliases: Optional[str] = None,
              report: Optional[str] = None) -> dict:
    device = _resolve_device(device)
    model, class_to_idx, spec_meta, _ = load_model(ckpt, device)
    model.eval()
    attributes: List[str] = list(spec_meta["attributes"])

    g = np.load(gallery, allow_pickle=True)
    gal_names = [str(x) for x in g["names"]]
    gal = F.normalize(torch.from_numpy(g["centroids"]).float(), dim=1).to(device)
    gal_rows = g["attr_rows"]

    # 정답은 manifest 전체 클래스 기준으로 다시 만든다. unseen 클래스의 정답도 필요하다.
    df_all = load_manifest(manifest, split=None)
    spec = AttributeSpec(df_all["label"].tolist(), list(class_to_idx.keys()),
                         feature_table, label_aliases, attributes)
    baseline = spec.majority_baseline()

    loader = create_loader(manifest, split=split, class_to_idx=class_to_idx,
                           image_size=image_size, batch_size=batch_size,
                           num_workers=num_workers, train=False, return_path=True)
    known = set(gal_names)

    # split 의 라벨을 순서대로 얻기 위해 loader 대신 manifest 를 같이 쓴다.
    df_split = load_manifest(manifest, split=split)
    path_to_label = dict(zip(df_split["path"].astype(str), df_split["label"].astype(str)))

    A = len(attributes)
    stat = {k: np.zeros(A, dtype="int64") for k in ("ok", "n", "base_ok")}
    branch = {"gallery": 0, "head": 0}
    seen_stat = {k: np.zeros(A, dtype="int64") for k in ("ok", "n")}
    unseen_stat = {k: np.zeros(A, dtype="int64") for k in ("ok", "n")}
    n_probe = 0

    for images, _t, paths in loader:
        emb, raw = model.encode(images.to(device))
        src = emb if model.head_source == "embedding" else raw
        head_pred = np.stack(
            [model.attr_heads[a](src).argmax(1).cpu().numpy() for a in attributes], axis=1
        )
        sims = emb @ gal.T
        best_sim, best_idx = sims.max(dim=1)
        for b, p in enumerate(paths):
            label = path_to_label.get(str(p), "")
            truth = spec.row_for(label)
            if truth is None:
                continue
            n_probe += 1
            hit = float(best_sim[b]) >= threshold
            pred = gal_rows[int(best_idx[b])] if hit else head_pred[b]
            branch["gallery" if hit else "head"] += 1
            bucket = seen_stat if label in known else unseen_stat
            for ai, a in enumerate(attributes):
                t = int(truth[ai])
                if t < 0:            # 표가 unknown 인 칸은 맞다 틀리다 판정하지 않는다
                    continue
                stat["n"][ai] += 1
                stat["ok"][ai] += int(int(pred[ai]) == t)
                stat["base_ok"][ai] += int(baseline[a] == t)
                bucket["n"][ai] += 1
                bucket["ok"][ai] += int(int(pred[ai]) == t)

    def acc(ok, n):
        return [float(o) / float(x) if x else float("nan") for o, x in zip(ok, n)]

    per_attr = {
        a: {"acc": acc(stat["ok"], stat["n"])[i],
            "majority_baseline": acc(stat["base_ok"], stat["n"])[i],
            "n": int(stat["n"][i])}
        for i, a in enumerate(attributes)
    }
    macro = float(np.nanmean(acc(stat["ok"], stat["n"]))) if A else float("nan")
    macro_base = float(np.nanmean(acc(stat["base_ok"], stat["n"]))) if A else float("nan")
    result = {
        "split": split,
        "threshold": threshold,
        "head_source": model.head_source,
        "n_probes": n_probe,
        "branch": branch,
        "gallery_size": len(gal_names),
        "macro_attr_acc": macro,
        "macro_majority_baseline": macro_base,
        "macro_gain_over_baseline": macro - macro_base,
        "per_attribute": per_attr,
        "seen_macro_acc": float(np.nanmean(acc(seen_stat["ok"], seen_stat["n"]))) if A else float("nan"),
        "unseen_macro_acc": float(np.nanmean(acc(unseen_stat["ok"], unseen_stat["n"]))) if A else float("nan"),
        "unseen_probe_count": int(unseen_stat["n"].max()) if A else 0,
        "spec": spec.describe(),
    }
    log(f"[afa] macro attr acc={macro:.3f} (최빈값 baseline {macro_base:.3f}, "
        f"차이 {macro - macro_base:+.3f}) | 갤러리 {branch['gallery']} / head {branch['head']}")
    log(f"[afa] seen={result['seen_macro_acc']:.3f}  unseen={result['unseen_macro_acc']:.3f} "
        f"(unseen 판정 칸 {result['unseen_probe_count']}개)")
    if report:
        write_json(report, result)
    return result


# ---------------------------------------------------------------------------
# 저장 / 로딩 / 유틸
# ---------------------------------------------------------------------------
def _save(path: Path, model: ArcFaceAttributeModel, class_to_idx: Dict[str, int],
          spec: AttributeSpec, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {l: [int(v) for v in spec.target[i]] for l, i in spec.label_to_row.items()}
    torch.save({
        "state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "spec": {"attributes": spec.attributes, "vocab": spec.table.vocab, "rows": rows},
        "cfg": cfg,
    }, path)


def load_model(ckpt: str, device: str = "cpu"):
    obj = torch.load(ckpt, map_location=device, weights_only=False)
    cfg, class_to_idx, spec_meta = obj["cfg"], obj["class_to_idx"], obj["spec"]
    head_sizes = {a: len(spec_meta["vocab"][a]) for a in spec_meta["attributes"]}
    model = ArcFaceAttributeModel(
        cfg["model_name"], num_classes=len(class_to_idx), head_sizes=head_sizes,
        embedding_dim=cfg["embedding_dim"], pretrained=False, s=cfg["s"], m=cfg["m"],
        head_source=cfg.get("head_source", "embedding"),
    ).to(device)
    model.load_state_dict(obj["state_dict"])
    return model, class_to_idx, spec_meta, cfg


def _write_filtered_manifest(manifest: str, class_to_idx: Dict[str, int], out: Path) -> Path:
    """train/valid 에서 학습 대상이 아닌 클래스를 뺀 manifest 를 만든다.

    holdout 으로 unseen 을 만들면 그 클래스가 train split 에 그대로 남아 있다. 그대로
    학습하면 정답 인덱스가 없어 터진다. test split 은 건드리지 않는다 — 거기 있는
    unseen 이 이 실험의 평가 대상이다.
    """
    df = load_manifest(manifest, split=None)
    split = df["split"].astype(str)
    label = df["label"].astype(str)
    is_fit = split.isin(["train", "valid", "val"])
    drop = is_fit & ~label.isin(set(class_to_idx))
    kept = df[~drop].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(out, index=False)
    if int(drop.sum()):
        log(f"[afa] holdout: train/valid 에서 {int(drop.sum())}장 제외 "
            f"({sorted(set(label[drop]))})")
    return out


def _train_label_set(df: pd.DataFrame, holdout_classes: int) -> List[str]:
    """train split 의 클래스. holdout_classes>0 이면 희귀한 것부터 그만큼 뺀다.

    unseen 이 하나도 없으면 head 분기가 한 번도 안 돌아 실험이 성립하지 않는다.
    데이터셋에 unseen 이 이미 분리돼 있으면 0 으로 두면 된다.
    """
    tr = df[df["split"].astype(str) == "train"]
    labels = tr["label"].astype(str)
    if holdout_classes <= 0:
        return sorted(labels.unique())
    counts = labels.value_counts()
    keep = list(counts.index[: max(0, len(counts) - holdout_classes)])
    return sorted(str(x) for x in keep)


def _target_lut(spec: AttributeSpec, class_to_idx: Dict[str, int]) -> torch.Tensor:
    lut = torch.full((len(class_to_idx), len(spec.attributes)), -1, dtype=torch.long)
    for label, i in class_to_idx.items():
        row = spec.row_for(label)
        if row is not None:
            lut[i] = torch.from_numpy(np.asarray(row, dtype="int64"))
    return lut


def _maybe_loader(manifest, split, class_to_idx, image_size, batch_size, num_workers):
    try:
        df = load_manifest(manifest, split=split)
    except Exception:
        return None
    if len(df) == 0:
        return None
    return create_loader(manifest, split=split, class_to_idx=class_to_idx,
                         image_size=image_size, batch_size=batch_size,
                         num_workers=num_workers, train=False)


# ---------------------------------------------------------------------------
# self-test — 데이터 없이 두 가지를 확인한다.
#   (a) 멀티태스크가 실제로 학습되는가 (ArcFace 와 attribute head 동시에)
#   (b) open-set 분기가 threshold 대로 갈라지는가
# ---------------------------------------------------------------------------
def selftest() -> None:
    torch.manual_seed(0)
    D, C, A, K, B = 16, 4, 3, 3, 64

    # (a) 멀티태스크 학습. 클래스 centroid 를 만들고 attribute 는 클래스의 함수로 둔다.
    head = ArcFaceHead(D, C, s=30.0, m=0.5)
    attr = nn.ModuleDict({f"a{i}": nn.Linear(D, K) for i in range(A)})
    table = torch.randint(0, K, (C, A))          # 클래스 -> attribute 정답
    table[0, 0] = -1                             # unknown 칸도 하나 섞는다
    centers = F.normalize(torch.randn(C, D), dim=1)
    enc = nn.Linear(D, D)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters())
                           + list(attr.parameters()), lr=0.05)

    def batch():
        y = torch.randint(0, C, (B,))
        x = F.normalize(centers[y] + 0.15 * torch.randn(B, D), dim=1)
        return x, y

    for _ in range(300):
        x, y = batch()
        z = F.normalize(enc(x), dim=1)
        loss = F.cross_entropy(head(z, y), y)
        for ai, a in enumerate(attr):
            loss = loss + F.cross_entropy(attr[a](z), table[y, ai], ignore_index=-1) / A
        opt.zero_grad(); loss.backward(); opt.step()

    x, y = batch()
    z = F.normalize(enc(x), dim=1)
    cls_acc = (head(z, None).argmax(1) == y).float().mean().item()
    ok = n = 0
    for ai, a in enumerate(attr):
        t = table[y, ai]
        v = t >= 0
        ok += int((attr[a](z)[v].argmax(1) == t[v]).sum()); n += int(v.sum())
    attr_acc = ok / n
    assert cls_acc > 0.9, f"클래스 학습 실패: {cls_acc:.3f}"
    assert attr_acc > 0.9, f"attribute 학습 실패: {attr_acc:.3f}"

    # (b) open-set 분기. 갤러리에 등록된 것과 아주 다른 임베딩은 head 로 가야 한다.
    gal = F.normalize(centers.clone(), dim=1)
    near = F.normalize(centers[1] + 0.01 * torch.randn(D), dim=0)
    far = F.normalize(-centers[1] + 0.01 * torch.randn(D), dim=0)
    tau = 0.5
    assert float((near @ gal.T).max()) >= tau, "가까운 probe 가 갤러리에 안 붙었다"
    assert float((far @ gal.T).max()) < tau, "먼 probe 가 갤러리에 붙었다"

    print(f"[selftest] OK  cls_acc={cls_acc:.3f}  attr_acc={attr_acc:.3f}  분기 판정 정상")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser("arcface_attributes",
                                description="ArcFace 기반 13 feature 표 채우기 (VLM 미사용)")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--manifest", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="vit_base_patch16_224")
    t.add_argument("--feature-table", default=str(DEFAULT_TABLE))
    t.add_argument("--label-aliases", default=None)
    t.add_argument("--head-source", default="embedding", choices=HEAD_SOURCES)
    t.add_argument("--attr-weight", type=float, default=1.0)
    t.add_argument("--embedding-dim", type=int, default=128)
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--image-size", type=int, default=224)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--s", type=float, default=30.0)
    t.add_argument("--m", type=float, default=0.5)
    t.add_argument("--num-workers", type=int, default=8)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", default="auto")
    t.add_argument("--holdout-classes", type=int, default=0,
                   help="train 에서 희귀 클래스 N 개를 빼 unseen 으로 만든다")
    t.add_argument("--no-pretrained", action="store_true")

    g = sub.add_parser("gallery")
    g.add_argument("--manifest", required=True)
    g.add_argument("--ckpt", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--split", default="train")
    g.add_argument("--image-size", type=int, default=224)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--device", default="auto")

    r = sub.add_parser("recognize")
    r.add_argument("--manifest", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--gallery", required=True)
    r.add_argument("--split", default="test")
    r.add_argument("--threshold", type=float, default=0.30)
    r.add_argument("--image-size", type=int, default=224)
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--num-workers", type=int, default=8)
    r.add_argument("--device", default="auto")
    r.add_argument("--feature-table", default=str(DEFAULT_TABLE))
    r.add_argument("--label-aliases", default=None)
    r.add_argument("--report", default=None)

    sub.add_parser("selftest")

    a = p.parse_args()
    if a.cmd == "train":
        train(a.manifest, a.out, a.model_name, a.feature_table, a.label_aliases,
              a.head_source, a.attr_weight, a.embedding_dim, a.epochs, a.batch_size,
              a.image_size, a.lr, a.s, a.m, a.num_workers, a.seed, a.device,
              pretrained=not a.no_pretrained, holdout_classes=a.holdout_classes)
    elif a.cmd == "gallery":
        build_gallery(a.manifest, a.ckpt, a.out, a.split, a.image_size,
                      a.batch_size, a.num_workers, a.device)
    elif a.cmd == "recognize":
        recognize(a.manifest, a.ckpt, a.gallery, a.split, a.threshold, a.image_size,
                  a.batch_size, a.num_workers, a.device, a.feature_table,
                  a.label_aliases, a.report)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()

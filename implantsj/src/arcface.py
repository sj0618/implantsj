"""얼굴인식(face recognition) 방식의 임플란트 시스템 인식.

기존 코드는 건드리지 않고 이 파일만 새로 얹는다. 재사용하는 것들:
  - models.MetricModel      : 백본 + L2 정규화 임베딩 (이미 있음)
  - data.*                  : manifest 로딩 / transform / DataLoader (이미 있음)
  - utils.*                 : seed / run dir / json (이미 있음)

새로 추가하는 것(= 얼굴인식에서 그대로 가져온 두 조각):
  1) ArcFaceHead   : margin softmax(ArcFace). triplet/SupCon 대신 임베딩을 또렷하게.
  2) gallery/probe : 학습에 없던 임플란트 시스템(unseen)도 레퍼런스 몇 장만 등록하면
                     재학습 없이 최근접 갤러리로 인식. threshold 미만이면 "모름"(open-set).

폐쇄망 실행 예:
  python -m src.arcface train    --manifest outputs/manifest.csv --out outputs/arcface
  python -m src.arcface gallery  --manifest outputs/manifest.csv --ckpt outputs/arcface/model.pt \
                                 --split train --out outputs/arcface/gallery.npz
  python -m src.arcface recognize --manifest outputs/manifest.csv --ckpt outputs/arcface/model.pt \
                                 --gallery outputs/arcface/gallery.npz --split test --threshold 0.3
  python -m src.arcface selftest   # 데이터 없이 학습 가능성만 확인
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import create_loader, label_mapping, load_manifest
from .models import MetricModel
from .utils import log, make_run_dir, seed_everything, write_json


# ---------------------------------------------------------------------------
# 1) ArcFace head — 얼굴인식의 핵심 loss. 임베딩은 이미 L2 정규화되어 들어온다.
#    cos(θ+m) = cosθ·cos m − sinθ·sin m 을 target 클래스에만 적용하고 s배 스케일.
# ---------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        # 수치 안정용 상수 (margin 경계 밖에서 단조성 보장)
        self._cos_m = math.cos(m)
        self._sin_m = math.sin(m)
        self._th = math.cos(math.pi - m)
        self._mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        # embeddings: [B, D] (L2 정규화 가정), weight도 정규화 → 순수 코사인
        cos = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        if labels is None:
            return cos * self.s  # 추론: 그냥 스케일된 코사인 로짓
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self._cos_m - sin * self._sin_m           # cos(θ+m)
        phi = torch.where(cos > self._th, phi, cos - self._mm)  # 경계 밖은 단조 대체
        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = onehot * phi + (1.0 - onehot) * cos          # target만 margin 적용
        return logits * self.s


class ArcFaceModel(nn.Module):
    """백본(MetricModel) + ArcFaceHead 를 하나로 묶어 저장/로딩을 단순화."""

    def __init__(self, model_name: str, num_classes: int, embedding_dim: int = 128,
                 pretrained: bool = True, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.encoder = MetricModel(model_name, embedding_dim=embedding_dim, pretrained=pretrained)
        self.head = ArcFaceHead(embedding_dim, num_classes, s=s, m=m)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)  # 이미 L2 정규화됨

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.embed(x), labels)


# ---------------------------------------------------------------------------
# 2) 학습
# ---------------------------------------------------------------------------
def train_arcface(
    manifest: str,
    out: str,
    model_name: str = "vit_base_patch16_224",
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
) -> Path:
    device = _resolve_device(device)
    seed_everything(seed)
    run_dir = make_run_dir(out, "arcface")

    df = load_manifest(manifest, split=None)
    class_to_idx, idx_to_class = label_mapping(df)
    num_classes = len(class_to_idx)
    log(f"[arcface] classes={num_classes} device={device} backbone={model_name}")

    train_loader = create_loader(
        manifest, split="train", class_to_idx=class_to_idx, image_size=image_size,
        batch_size=batch_size, num_workers=num_workers, train=True,
    )
    val_loader = _maybe_loader(manifest, "valid", class_to_idx, image_size, batch_size, num_workers) \
        or _maybe_loader(manifest, "val", class_to_idx, image_size, batch_size, num_workers)

    model = ArcFaceModel(model_name, num_classes, embedding_dim, pretrained, s, m).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss()

    best_acc, history = -1.0, []
    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, device, criterion, optimizer)
        va = _run_epoch(model, val_loader, device, criterion, None) if val_loader else None
        scheduler.step()
        row = {"epoch": epoch, "train": tr, "val": va, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        msg = f"[arcface] epoch {epoch:03d} train_loss={tr['loss']:.4f} acc={tr['acc']:.3f}"
        if va:
            msg += f" | val_loss={va['loss']:.4f} acc={va['acc']:.3f}"
        log(msg)
        cur = va["acc"] if va else tr["acc"]
        if cur >= best_acc:
            best_acc = cur
            _save(run_dir / "model.pt", model, class_to_idx,
                  dict(model_name=model_name, embedding_dim=embedding_dim, s=s, m=m))

    write_json(run_dir / "history.json", {"best_acc": best_acc, "history": history})
    log(f"[arcface] done. best_acc={best_acc:.3f}  ->  {run_dir/'model.pt'}")
    return run_dir


def _run_epoch(model, loader, device, criterion, optimizer):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images, targets if training else None)
            loss = criterion(logits, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss) * images.size(0)
        correct += int((logits.argmax(1) == targets).sum())
        total += images.size(0)
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}


# ---------------------------------------------------------------------------
# 3) Gallery 구축 (얼굴 "등록") — 클래스별 임베딩 centroid.
#    새 임플란트 시스템은 여기에 레퍼런스만 추가하면 됨(재학습 X).
# ---------------------------------------------------------------------------
@torch.no_grad()
def build_gallery(manifest: str, ckpt: str, out: str, split: str = "train",
                  image_size: int = 224, batch_size: int = 32, num_workers: int = 8,
                  device: str = "auto") -> Path:
    device = _resolve_device(device)
    model, class_to_idx, _ = load_model(ckpt, device)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    model.eval()

    loader = create_loader(manifest, split=split, class_to_idx=class_to_idx,
                           image_size=image_size, batch_size=batch_size,
                           num_workers=num_workers, train=False)
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for images, targets in loader:
        emb = model.embed(images.to(device)).cpu()
        for e, y in zip(emb, targets.tolist()):
            sums[y] = sums.get(y, torch.zeros_like(e)) + e
            counts[y] = counts.get(y, 0) + 1

    names, centroids = [], []
    for y in sorted(sums):
        c = F.normalize(sums[y] / counts[y], dim=0)  # 평균 후 재정규화
        names.append(idx_to_class[y])
        centroids.append(c.numpy())
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, names=np.array(names), centroids=np.stack(centroids),
             counts=np.array([counts[y] for y in sorted(sums)]))
    log(f"[arcface] gallery: {len(names)} systems -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 4) 인식 (probe) — 코사인 최근접 + threshold 미만이면 "unknown"(open-set).
# ---------------------------------------------------------------------------
@torch.no_grad()
def recognize(manifest: str, ckpt: str, gallery: str, split: str = "test",
              threshold: float = 0.30, image_size: int = 224, batch_size: int = 32,
              num_workers: int = 8, device: str = "auto",
              report: Optional[str] = None) -> dict:
    device = _resolve_device(device)
    model, class_to_idx, _ = load_model(ckpt, device)
    model.eval()
    g = np.load(gallery, allow_pickle=True)
    gal_names = list(g["names"])
    gal = F.normalize(torch.from_numpy(g["centroids"]).float(), dim=1).to(device)

    # split 라벨은 open-set 평가에서 갤러리에 없을 수도 있음(unseen).
    df = load_manifest(manifest, split=split)
    known = set(gal_names)
    loader = create_loader(manifest, split=split, class_to_idx=class_to_idx,
                           image_size=image_size, batch_size=batch_size,
                           num_workers=num_workers, train=False, return_path=True)

    # 정답 이름은 targets 이 아니라 manifest 에서 경로로 찾는다. unseen 클래스는
    # class_to_idx 에 없어 targets 이 -1 로 오기 때문이다(idx_to_class 로는 못 찾는다).
    path_to_label = dict(zip(df["path"].astype(str), df["label"].astype(str)))

    n = correct = seen_total = 0
    unknown_pred = 0
    for images, targets, paths in loader:
        emb = model.embed(images.to(device))
        sims = emb @ gal.T                      # [B, G] 코사인
        best_sim, best_idx = sims.max(dim=1)
        for bs, bi, pth in zip(best_sim.tolist(), best_idx.tolist(), paths):
            true_name = path_to_label.get(str(pth), "")
            if not true_name:
                continue
            pred = gal_names[bi] if bs >= threshold else "__unknown__"
            if pred == "__unknown__":
                unknown_pred += 1
            if true_name in known:              # seen 클래스만 정확도 집계
                seen_total += 1
                correct += int(pred == true_name)
            n += 1

    acc = correct / seen_total if seen_total else float("nan")
    result = {"split": split, "threshold": threshold, "n_probes": n,
              "seen_probes": seen_total, "seen_acc": acc,
              "predicted_unknown": unknown_pred, "gallery_size": len(gal_names)}
    log(f"[arcface] recognize: seen_acc={acc:.3f} on {seen_total} seen probes, "
        f"{unknown_pred}/{n} predicted unknown (τ={threshold})")
    if report:
        write_json(report, result)
    return result


# ---------------------------------------------------------------------------
# 저장/로딩 & 유틸
# ---------------------------------------------------------------------------
def _save(path: Path, model: ArcFaceModel, class_to_idx: Dict[str, int], cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "class_to_idx": class_to_idx, "cfg": cfg}, path)


def load_model(ckpt: str, device: str = "cpu"):
    obj = torch.load(ckpt, map_location=device)
    cfg, class_to_idx = obj["cfg"], obj["class_to_idx"]
    model = ArcFaceModel(cfg["model_name"], num_classes=len(class_to_idx),
                         embedding_dim=cfg["embedding_dim"], pretrained=False,
                         s=cfg["s"], m=cfg["m"]).to(device)
    model.load_state_dict(obj["state_dict"])
    return model, class_to_idx, cfg


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
# self-test: 데이터 없이 ArcFace가 실제로 학습되는지(=margin softmax가 임베딩을
# 클래스별로 분리하는지) 합성 데이터로 확인. ponytail: 최소 실행 검증 1개.
# ---------------------------------------------------------------------------
def selftest() -> None:
    torch.manual_seed(0)
    D, C, B = 16, 4, 64
    head = ArcFaceHead(D, C, s=30.0, m=0.5)

    # (a) 추론 로짓은 [B, C] 이고 margin=0.5 를 target에 실제로 적용하는지
    z = F.normalize(torch.randn(B, D), dim=1)
    y = torch.randint(0, C, (B,))
    logits = head(z, y)
    assert logits.shape == (B, C), logits.shape
    infer = head(z, None)
    # target 로짓은 margin 때문에 추론(margin 없는) 로짓보다 작아야 정상
    tgt_train = logits.gather(1, y.view(-1, 1))
    tgt_infer = infer.gather(1, y.view(-1, 1))
    assert (tgt_train <= tgt_infer + 1e-4).all(), "margin이 target을 낮추지 않음"

    # (b) 선형 임베딩 + ArcFace 로 분리 가능한 합성 문제를 실제로 학습하는지
    centers = F.normalize(torch.randn(C, D), dim=1)
    def batch():
        yb = torch.randint(0, C, (B,))
        xb = centers[yb] + 0.15 * torch.randn(B, D)
        return F.normalize(xb, dim=1), yb
    enc = nn.Linear(D, D)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=0.05)
    crit = nn.CrossEntropyLoss()
    for _ in range(300):
        xb, yb = batch()
        out = head(F.normalize(enc(xb), dim=1), yb)
        loss = crit(out, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    xb, yb = batch()
    acc = (head(F.normalize(enc(xb), dim=1), None).argmax(1) == yb).float().mean().item()
    assert acc > 0.9, f"학습 실패: acc={acc:.3f}"
    print(f"[selftest] OK  final_loss={loss.item():.4f}  acc={acc:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser("arcface", description="얼굴인식 방식 임플란트 인식")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--manifest", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--model-name", default="vit_base_patch16_224")
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
    r.add_argument("--report", default=None)

    sub.add_parser("selftest")

    a = p.parse_args()
    if a.cmd == "train":
        train_arcface(a.manifest, a.out, a.model_name, a.embedding_dim, a.epochs,
                      a.batch_size, a.image_size, a.lr, a.s, a.m, a.num_workers,
                      a.seed, a.device, pretrained=not a.no_pretrained)
    elif a.cmd == "gallery":
        build_gallery(a.manifest, a.ckpt, a.out, a.split, a.image_size,
                      a.batch_size, a.num_workers, a.device)
    elif a.cmd == "recognize":
        recognize(a.manifest, a.ckpt, a.gallery, a.split, a.threshold, a.image_size,
                  a.batch_size, a.num_workers, a.device, a.report)
    elif a.cmd == "selftest":
        selftest()


if __name__ == "__main__":
    main()

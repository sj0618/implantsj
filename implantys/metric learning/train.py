#!/usr/bin/env python3
"""
학습 — M2~M7 을 한 스크립트로 커버한다.

  python train.py --cond M2    # frozen backbone + linear head, 13개 학습
  python train.py --cond M3    #                                 9개 학습
  python train.py --cond M4    # full FT, 13개 학습
  python train.py --cond M5    # full FT,  9개 학습
  python train.py --cond M6    # full FT + SupCon, 13개 학습
  python train.py --cond M7    # full FT + SupCon,  9개 학습

산출물
  ckpt/{cond}.pt            backbone + head + 메타 (extract.py 가 읽는다)
  logs/{cond}.jsonl         epoch 별 loss/acc (plots.py --curves 가 읽는다)
  softmax_reference.csv     softmax head 직접 분류 성능 (report.py Q5)

모델 선택은 **val macro-F1** 기준. val loss 가 아니다 —
불균형이 53:1 이라 loss 는 큰 클래스에 끌려간다.

구조: backbone → projection head → 분류기. 프로토타입 분류에 쓰이는 임베딩은
projection 출력이고, SupCon 도 여기에 건다(SupCon 표준 구성). frozen 조건에서도
projection 과 분류기는 학습되므로 임베딩이 M1 Base 와 달라진다. 이유는 model.py 참조.

SupCon 주의: 배치 안에 같은 클래스가 2장 이상 있어야 양성쌍이 생긴다.
소수 클래스(nobel train 3장)는 무작위 배치에 거의 안 잡히므로
--balanced-sampler 로 클래스 균형 샘플링을 쓰는 것을 기본값으로 둔다.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ImplantDataset, load_split
from metrics import macro_f1
from model import ImplantModel, pick_device

_ROOT = Path(__file__).parent
CKPT_DIR = _ROOT / "ckpt"
LOG_DIR = _ROOT / "logs"
SOFTMAX_CSV = _ROOT / "softmax_reference.csv"

# 조건 → (학습 범위, backbone 동결 여부, SupCon, 평가 시나리오 코드)
CONDITIONS = {
    "M2": ("all", True, False, "E1"),
    "M3": ("open", True, False, "E2"),
    "M4": ("all", False, False, "E1"),
    "M5": ("open", False, False, "E2"),
    "M6": ("all", False, True, "E1"),
    "M7": ("open", False, True, "E2"),
}


# ---------------------------------------------------------------- SupCon


class SupConLoss(nn.Module):
    """Khosla et al. 2020. 배치 내 같은 라벨을 양성쌍으로 당긴다.

    양성쌍이 없는 샘플(배치에 그 클래스가 혼자)은 손실에서 제외한다.
    포함시키면 0 을 더해 평균을 희석시켜 손실이 조용히 작아진다.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, emb, labels):
        emb = F.normalize(emb, dim=1)
        sim = emb @ emb.T / self.t

        # 수치 안정화: 행별 최대값을 빼도 softmax 결과는 같다
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        n = emb.size(0)
        eye = torch.eye(n, dtype=torch.bool, device=emb.device)
        pos = (labels[:, None] == labels[None, :]) & ~eye

        exp = torch.exp(sim) * (~eye)
        log_prob = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)

        n_pos = pos.sum(1)
        valid = n_pos > 0
        if not valid.any():
            return emb.sum() * 0.0  # 그래프 유지용 0
        mean_log_prob = (pos * log_prob).sum(1)[valid] / n_pos[valid]
        return -mean_log_prob.mean()


# ---------------------------------------------------------------- 데이터


def build_loaders(scope, batch_size, num_workers, train_aug, balanced, meta):
    tr = ImplantDataset("train", scope=scope, meta=meta, train_aug=train_aug)
    va = ImplantDataset("val", scope=scope, meta=meta, train_aug=False)

    if balanced:
        counts = Counter(lbl for _, lbl in tr.samples)
        w = [1.0 / counts[lbl] for _, lbl in tr.samples]
        sampler = WeightedRandomSampler(w, num_samples=len(tr), replacement=True)
        tr_loader = DataLoader(tr, batch_size=batch_size, sampler=sampler,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)

    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return tr, tr_loader, va, va_loader


# ---------------------------------------------------------------- 평가


@torch.no_grad()
def evaluate_head(model, loader, device, n_classes):
    """softmax head 직접 분류 성능. 프로토타입 분류와는 별개의 참조값."""
    model.eval()
    ys, ps, losses = [], [], []
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        _, logits = model(x)
        losses.append(F.cross_entropy(logits, y).item())
        ys.append(y.cpu().numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return {
        "loss": float(np.mean(losses)),
        "acc": float((y == p).mean()),
        "macro_f1": macro_f1(y, p, n_classes),
    }


# ---------------------------------------------------------------- 학습


def train(args):
    scope, freeze, use_supcon, scenario = CONDITIONS[args.cond]
    tag = args.tag or args.cond
    device = pick_device(args.device)
    meta = load_split()

    # frozen 조건은 augmentation 을 끈다 — backbone 이 안 변하니
    # 매 epoch 다른 입력을 주면 head 만 흔들려 수렴이 나빠진다.
    train_aug = args.aug and not freeze
    balanced = args.balanced_sampler and use_supcon

    tr, tr_loader, va, va_loader = build_loaders(
        scope, args.batch_size, args.num_workers, train_aug, balanced, meta)
    n_classes = len(tr.classes)

    print(f"[{tag}] scope={scope}({n_classes}클래스)  frozen={freeze}  "
          f"supcon={use_supcon}  device={device}")
    print(f"  train {len(tr)}장 / val {len(va)}장  "
          f"aug={train_aug}  balanced_sampler={balanced}")
    if use_supcon and not balanced:
        print("  [주의] SupCon 인데 balanced sampler 가 꺼져 있다. "
              "소수 클래스가 배치에 안 잡히면 양성쌍이 안 생긴다.")

    model = ImplantModel(n_classes, pretrained=True, freeze_backbone=freeze).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    supcon = SupConLoss(args.temperature) if use_supcon else None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    LOG_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"{tag}.jsonl"
    log_path.write_text("")

    best = {"macro_f1": -1.0, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, ces, scs = [], [], []
        for x, y, _ in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                emb, logits = model(x)
                ce = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                sc = supcon(emb.float(), y) if supcon is not None else None
                loss = ce if sc is None else ce + args.supcon_weight * sc
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            ces.append(ce.item())
            scs.append(0.0 if sc is None else sc.item())
        sched.step()

        val = evaluate_head(model, va_loader, device, n_classes)
        # train_ce 를 따로 남긴다. val_loss 는 CE 만이라, SupCon 조건에서
        # train_loss(CE+SupCon)와 빼면 과적합 격차가 무의미해진다.
        rec = {"epoch": epoch,
               "train_loss": float(np.mean(losses)),   # 실제 최적화 대상
               "train_ce": float(np.mean(ces)),        # val_loss 와 비교 가능한 값
               "train_supcon": float(np.mean(scs)),
               "val_loss": val["loss"], "val_acc": val["acc"],
               "val_macro_f1": val["macro_f1"], "lr": sched.get_last_lr()[0]}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        msg = (f"  ep{epoch:>3}  train_ce={rec['train_ce']:.4f}  "
               f"val_loss={val['loss']:.4f}  val_acc={val['acc']:.4f}  "
               f"val_macroF1={val['macro_f1']:.4f}")
        if supcon is not None:
            msg += f"  supcon={rec['train_supcon']:.4f}"
        print(msg)

        # 모델 선택 기준은 val macro-F1 (불균형 53:1 이라 loss 는 큰 클래스에 끌려간다)
        if val["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val["macro_f1"], "epoch": epoch}
            torch.save({
                "backbone": model.backbone.state_dict(),
                "proj": model.proj.state_dict(),   # 임베딩을 만드는 층 — 없으면 extract 실패
                "head": model.head.state_dict(),
                "classes": tr.classes,
                "train_scope": scope,
                "cond": tag,
                "epoch": epoch,
                "val_macro_f1": val["macro_f1"],
            }, CKPT_DIR / f"{tag}.pt")

    print(f"  best: epoch {best['epoch']}  val_macro_f1={best['macro_f1']:.4f}")

    # --- softmax 참조값 (test split, best 체크포인트로)
    ckpt = torch.load(CKPT_DIR / f"{tag}.pt", map_location="cpu")
    model.backbone.load_state_dict(ckpt["backbone"])
    model.proj.load_state_dict(ckpt["proj"])
    model.head.load_state_dict(ckpt["head"])
    model.to(device)
    te = ImplantDataset("test", scope=scope, meta=meta)
    te_loader = DataLoader(te, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers)
    test = evaluate_head(model, te_loader, device, n_classes)
    print(f"  [softmax 참조값 / test] acc={test['acc']:.4f}  "
          f"macro_f1={test['macro_f1']:.4f}")
    _append_softmax(tag, scenario, n_classes, len(te), test)

    print(f"\n다음: python extract.py --cond {tag} --ckpt ckpt/{tag}.pt")


def _append_softmax(cond, scenario, n_classes, n_test, res):
    fields = ["cond", "scenario", "n_classes", "n_test", "acc", "macro_f1"]
    rows = []
    if SOFTMAX_CSV.is_file():
        with open(SOFTMAX_CSV, newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["cond"] != cond]
    rows.append({"cond": cond, "scenario": scenario, "n_classes": n_classes,
                 "n_test": n_test, "acc": round(res["acc"], 6),
                 "macro_f1": round(res["macro_f1"], 6)})
    rows.sort(key=lambda r: r["cond"])
    with open(SOFTMAX_CSV, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--tag", default=None,
                    help="산출물 이름(ckpt/logs/softmax 행). 기본은 --cond 값. "
                         "설정만 바꿔 변형 조건을 돌릴 때 쓴다 "
                         "(예: --cond M6 --no-balanced-sampler --tag M6-nobal).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None,
                    help="기본: frozen 1e-3 / full FT 1e-5")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--supcon-weight", type=float, default=1.0)
    ap.add_argument("--aug", action="store_true", default=True)
    ap.add_argument("--no-aug", dest="aug", action="store_false")
    ap.add_argument("--balanced-sampler", action="store_true", default=True)
    ap.add_argument("--no-balanced-sampler", dest="balanced_sampler",
                    action="store_false")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.lr is None:
        # frozen head 는 크게, backbone 을 푸는 쪽은 작게.
        # 사전학습 표현을 소량 데이터로 뭉개지 않으려면 1e-5 수준이 필요하다.
        args.lr = 1e-3 if CONDITIONS[args.cond][1] else 1e-5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()

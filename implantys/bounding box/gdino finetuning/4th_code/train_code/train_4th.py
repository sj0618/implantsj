#!/usr/bin/env python3
"""
Grounding DINO LoRA 파인튜닝. repo의 train.py를 참고해 새로 썼다.

repo train.py 대비 고친 것:

  1. steps_per_epoch = len(dataset) // batch_size  →  len(train_loader)
     74//4=18 인데 실제 배치는 ceil(74/4)=19 이라 OneCycleLR에 total_steps가
     540으로 들어가고 실제로는 570번 step()이 불려 epoch 28.4에서 죽었다.
     1차 학습이 epoch 25까지만 남은 원인.
  2. validate()를 실제로 부른다. repo는 함수만 있고 루프에서 안 불러 val loss가
     기록되지 않았다. (RULES 4)
  3. val mAP@.5:.95 / AP50을 epoch마다 잰다. repo에는 아예 없었다.
     best epoch 선택 지표라 이게 없으면 규칙 5를 지킬 수 없다.
  4. seed 고정. repo는 seed를 안 박아 재현이 안 됐다. (RULES 4)
  5. 매 epoch 체크포인트 저장 + 사후 선택. early stopping 없음. (RULES 5)
  6. 규칙 4의 산출물을 강제로 찍는다: config.json / metrics.csv / steps.csv /
     best_epoch.json

──────────────────────────────────────────────────────────────────────────────
4차 전용본. 2nd/train_code/train.py를 복사해 **데이터 지정 방식만** 바꿨다.
학습 로직(모델·loss·optimizer·scheduler·seed·best epoch 선택)은 원본과 동일하다.

  원본: --data 01 02   → 라운드 폴더를 병합 (val도 라운드별 val을 합침)
  4차 : --set 206      → 4th/dataset/train_206 + 4th/dataset/val (고정 val 299장)

라운드 병합은 make_dataset.py에서 이미 끝났으므로 여기서는 폴더를 그대로 읽는다.

  python train_code/train_4th.py --set 206 --lr 1e-4 --run-id FT-206v2
  python train_code/train_4th.py --set 506 --lr 1e-4 --run-id FT-506v2
"""

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

REPO = Path("/workspace/Grounding-Dino-FineTuning")
BASE = REPO / "4th"
DATASET = BASE / "dataset"          # make_dataset.py 산출물 (train_206/306/506, val, test200)

SEED = 42                    # resplit_73.py / sample_test37.py와 동일. 분할과 맞춘다.
TEXT_PROMPT = "implant"
VAL_BOX_THR = 0.05           # val mAP용. AP는 낮은 threshold의 전체 예측으로 잰다.
VAL_TEXT_THR = 0.20


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def split_hash(csv_path: Path) -> dict:
    """어떤 데이터로 돌렸는지 나중에 확정할 수 있게. (RULES 4)"""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    names = sorted({r["image_name"] for r in rows})
    return {
        "n_images": len(names),
        "n_boxes": len(rows),
        "split_hash": sha("\n".join(names)),
        "box_hash": sha("\n".join(sorted(
            f"{r['image_name']}:{r['bbox_x']},{r['bbox_y']},"
            f"{r['bbox_width']},{r['bbox_height']}" for r in rows))),
    }


def code_version() -> dict:
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        rev = "unknown"
    return {"repo_git_rev": rev,
            "train_py_sha": sha(Path(__file__).read_text(encoding="utf-8"))}


def build_val_gt(csv_path: Path, img_dir: Path):
    """val CSV -> COCO GT. 학습에 쓴 것과 같은 파일에서 만들어야 어긋나지 않는다."""
    from pycocotools.coco import COCO

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    names = sorted({r["image_name"] for r in rows})
    name2id = {n: i + 1 for i, n in enumerate(names)}

    images, id2wh = [], {}
    for n, i in name2id.items():
        with Image.open(img_dir / n) as im:
            w, h = im.size
        images.append({"id": i, "file_name": n, "width": w, "height": h})
        id2wh[i] = (w, h)

    anns = []
    for k, r in enumerate(rows, 1):
        x, y = float(r["bbox_x"]), float(r["bbox_y"])
        bw, bh = float(r["bbox_width"]), float(r["bbox_height"])
        anns.append({"id": k, "image_id": name2id[r["image_name"]], "category_id": 1,
                     "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": 1, "name": TEXT_PROMPT}]}
    gt.createIndex()
    return gt, name2id, id2wh


@torch.no_grad()
def val_ap(model, gt, name2id, id2wh, img_dir, device="cuda"):
    """epoch마다 val에서 AP50 / mAP@.5:.95. best epoch 선택 지표. (RULES 5)

    predict()를 안 쓰고 모델 출력을 직접 읽는다 — phrase가 필요 없고,
    PeftModel에서 .tokenizer 접근이 걸릴 여지를 없앤다.
    """
    import contextlib, io, copy
    from torchvision.ops import box_convert
    from pycocotools.cocoeval import COCOeval
    from groundingdino.util.inference import load_image, preprocess_caption

    model.eval()
    caption = preprocess_caption(TEXT_PROMPT)
    dets = []
    for name, img_id in name2id.items():
        _, image = load_image(str(img_dir / name))
        out = model(image[None].to(device), captions=[caption])
        logits = out["pred_logits"].sigmoid()[0]        # (nq, 256)
        boxes = out["pred_boxes"][0]                    # (nq, 4) cxcywh 정규화
        scores = logits.max(dim=1)[0]
        keep = scores > VAL_BOX_THR
        if keep.sum() == 0:
            continue
        b = boxes[keep].cpu()
        s = scores[keep].cpu().numpy()
        W, H = id2wh[img_id]
        xyxy = box_convert(b * torch.tensor([W, H, W, H], dtype=b.dtype),
                           in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for (x1, y1, x2, y2), sc in zip(xyxy, s):
            dets.append({"image_id": img_id, "category_id": 1,
                         "bbox": [float(x1), float(y1),
                                  float(x2 - x1), float(y2 - y1)],
                         "score": float(sc)})
    if not dets:
        return 0.0, 0.0
    ev = COCOeval(copy.deepcopy(gt), gt.loadRes(copy.deepcopy(dets)), iouType="bbox")
    ev.evaluate(); ev.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        ev.summarize()
    return float(ev.stats[1]), float(ev.stats[0])       # ap50, map


class Trainer:
    def __init__(self, model, lr, steps_per_epoch, num_epochs, device="cuda",
                 class_coef=1.0, bbox_coef=5.0, giou_coef=1.0, eos_coef=0.1,
                 max_txt_len=256, max_grad_norm=5.0):
        from groundingdino.util.matchers import build_matcher
        from groundingdino.util.losses import SetCriterion
        from groundingdino.util.lora import get_lora_optimizer_params

        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(get_lora_optimizer_params(model), lr=lr)

        total_steps = steps_per_epoch * num_epochs
        self.scheduler = OneCycleLR(self.optimizer, max_lr=lr, total_steps=total_steps,
                                    pct_start=0.1, div_factor=25, final_div_factor=1e4,
                                    anneal_strategy="cos")
        self.total_steps = total_steps

        matcher = build_matcher(set_cost_class=class_coef * 2,
                                set_cost_bbox=bbox_coef, set_cost_giou=giou_coef)
        # repo와 동일: matcher보다 loss에서 bbox에 가중을 더 준다
        self.w_loss = {"loss_ce": class_coef, "loss_bbox": bbox_coef * 2,
                       "loss_giou": giou_coef}
        self.criterion = SetCriterion(max_txt_len, matcher, eos_coef, ["labels", "boxes"])
        self.criterion.to(device)

    def prepare(self, batch):
        from groundingdino.util.misc import nested_tensor_from_tensor_list
        images, targets = batch
        if isinstance(images, (list, tuple)):
            images = nested_tensor_from_tensor_list(images)
        images = images.to(self.device)
        captions = []
        for t in targets:
            t["boxes"] = t["boxes"].to(self.device)
            t["size"] = t["size"].to(self.device)
            t["labels"] = t["labels"].to(self.device)
            captions.append(t["caption"])
        return images, targets, captions

    def _loss(self, batch):
        images, targets, captions = self.prepare(batch)
        outputs = self.model(images, captions=captions)
        ld = self.criterion(outputs, targets, captions=captions,
                            tokenizer=self.model.tokenizer)
        total = sum(ld[k] * self.w_loss[k] for k in ld if k in self.w_loss)
        return ld, total

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        ld, total = self._loss(batch)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        out = {k: v.item() for k, v in ld.items()}
        out["total_loss"] = total.item()
        return out

    @torch.no_grad()
    def validate(self, val_loader):
        """repo에 있었지만 한 번도 불리지 않던 것. (RULES 4)"""
        self.model.eval()
        acc = defaultdict(float)
        n = 0
        for batch in val_loader:
            ld, total = self._loss(batch)
            for k, v in ld.items():
                acc[k] += v.item()
            acc["total_loss"] += total.item()
            n += 1
        return {k: v / n for k, v in acc.items()}

    def save(self, path):
        from peft import get_peft_model_state_dict
        torch.save({"model": get_peft_model_state_dict(self.model)}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, dest="set_name",
                    help="학습 세트. 예: --set 206  ->  dataset/train_206")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--run-id", required=True, help="예: FT-206v2")
    ap.add_argument("--dataset-root", default=str(DATASET))
    ap.add_argument("--val-dir", default=None,
                    help="고정 val. 기본 <dataset-root>/val (세 run 공용)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--results-root", default=str(BASE / "results"))
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()

    from groundingdino.util.train import load_model
    from groundingdino.util.lora import verify_only_lora_trainable
    from groundingdino.datasets.dataset import GroundingDINODataset

    set_seed(a.seed)

    run = Path(a.results_root) / a.run_id
    run.mkdir(parents=True, exist_ok=True)
    (run / "weights").mkdir(exist_ok=True)

    # ── 데이터: COCO -> CSV (병합은 make_dataset.py에서 이미 끝났다) ──────────
    ds_root = Path(a.dataset_root)
    train_dir = ds_root / f"train_{a.set_name}"
    val_dir = Path(a.val_dir) if a.val_dir else ds_root / "val"
    test_json = ds_root / "test200" / "annotations.json"
    for d in (train_dir, val_dir):
        if not (d / "annotations.json").exists():
            sys.exit(f"[ERROR] 데이터 폴더가 아니다: {d}")

    data_dir = run / "data"
    print(f"[data] train {train_dir.name} + val {val_dir.name} -> {data_dir}")
    rc = subprocess.run([sys.executable,
                         str(Path(__file__).parent / "coco_to_csv_4th.py"),
                         "--train-dir", str(train_dir), "--val-dir", str(val_dir),
                         "--test-json", str(test_json), "--out-root", str(data_dir)])
    if rc.returncode != 0:
        sys.exit("[ERROR] coco_to_csv_4th.py 실패")

    tr_csv, va_csv = data_dir / "train_annotations.csv", data_dir / "val_annotations.csv"
    tr_img, va_img = data_dir / "images/train", data_dir / "images/val"

    train_ds = GroundingDINODataset(str(tr_img), str(tr_csv), add_extra_classes=True)
    val_ds = GroundingDINODataset(str(va_img), str(va_csv))
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.num_workers,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1,
                            collate_fn=lambda x: tuple(zip(*x)))

    # repo 버그: len(dataset)//batch_size 는 마지막 자투리 배치를 빼먹는다
    steps_per_epoch = len(train_loader)
    print(f"[data] train {len(train_ds)}장 / val {len(val_ds)}장 | "
          f"steps/epoch {steps_per_epoch} (repo식이면 "
          f"{len(train_ds)//a.batch_size} — 이 차이가 OneCycleLR을 죽였다)")

    model = load_model(str(REPO / "groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                       str(REPO / "weights/groundingdino_swint_ogc.pth"),
                       use_lora=True)
    print(f"[model] LoRA만 학습되나? {verify_only_lora_trainable(model)}")
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] 전체 {n_all/1e6:.1f}M / 학습가능 {n_tr/1e6:.3f}M")

    trainer = Trainer(model, lr=a.lr, steps_per_epoch=steps_per_epoch,
                      num_epochs=a.epochs)

    gt, name2id, id2wh = build_val_gt(va_csv, va_img)

    cfg = {
        "run_id": a.run_id,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": a.seed,
        "set": a.set_name,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "fixed_val": True,          # 4차: 모든 run이 동일 val 299장을 쓴다
        "hyperparams": {
            "lr": a.lr, "epochs": a.epochs, "batch_size": a.batch_size,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": steps_per_epoch * a.epochs,
            "optimizer": "AdamW", "scheduler": "OneCycleLR",
            "pct_start": 0.1, "div_factor": 25, "final_div_factor": 1e4,
            "anneal_strategy": "cos", "max_grad_norm": 5.0,
            "class_loss_coef": 1.0, "bbox_loss_coef": 5.0, "giou_loss_coef": 1.0,
            "eos_coef": 0.1, "use_lora": True, "text_prompt": TEXT_PROMPT,
            "early_stopping": False,
        },
        "data": {"train": split_hash(tr_csv), "val": split_hash(va_csv)},
        "model": {"trainable_params": n_tr, "total_params": n_all},
        "code": code_version(),
        "val_ap": {"box_threshold": VAL_BOX_THR, "text_threshold": VAL_TEXT_THR},
    }
    (run / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    m_path, s_path = run / "metrics.csv", run / "steps.csv"
    m_cols = ["epoch", "train_loss", "train_loss_ce", "train_loss_bbox",
              "train_loss_giou", "val_loss", "val_ap50", "val_map", "lr"]
    mf = open(m_path, "w", newline="", encoding="utf-8")
    mw = csv.DictWriter(mf, fieldnames=m_cols); mw.writeheader()
    sf = open(s_path, "w", newline="", encoding="utf-8")
    sw = csv.DictWriter(sf, fieldnames=["step", "epoch", "lr"]); sw.writeheader()

    print(f"\n{'='*70}\n {a.run_id} | lr {a.lr:g} | {a.epochs} epoch | "
          f"seed {a.seed}\n{'='*70}\n")

    step = 0
    for epoch in range(1, a.epochs + 1):
        ep_losses = defaultdict(list)
        for bi, batch in enumerate(train_loader):
            # scheduler.get_last_lr()의 실측값. 의도한 수식이 아니라. (RULES 4)
            lr_now = trainer.scheduler.get_last_lr()[0]
            losses = trainer.train_step(batch)
            step += 1
            sw.writerow({"step": step, "epoch": epoch, "lr": f"{lr_now:.8e}"})
            for k, v in losses.items():
                ep_losses[k].append(v)
            if bi % 5 == 0:
                print(f"  ep {epoch:2}/{a.epochs} batch {bi:2}/{steps_per_epoch} "
                      f"loss {losses['total_loss']:.4f} lr {lr_now:.2e}", flush=True)
        sf.flush()

        tr = {k: sum(v) / len(v) for k, v in ep_losses.items()}
        vl = trainer.validate(val_loader)
        ap50, mAP = val_ap(model, gt, name2id, id2wh, va_img)

        mw.writerow({
            "epoch": epoch,
            "train_loss": f"{tr['total_loss']:.6f}",
            "train_loss_ce": f"{tr.get('loss_ce', 0):.6f}",
            "train_loss_bbox": f"{tr.get('loss_bbox', 0):.6f}",
            "train_loss_giou": f"{tr.get('loss_giou', 0):.6f}",
            "val_loss": f"{vl['total_loss']:.6f}",
            "val_ap50": f"{ap50:.6f}",
            "val_map": f"{mAP:.6f}",
            "lr": f"{trainer.scheduler.get_last_lr()[0]:.8e}",
        })
        mf.flush()

        trainer.save(run / "weights" / f"epoch_{epoch:02d}.pth")

        print(f"\n  [epoch {epoch:2}] train {tr['total_loss']:.4f} | "
              f"val {vl['total_loss']:.4f} | val AP50 {ap50:.4f} | "
              f"val mAP {mAP:.4f}\n", flush=True)

    mf.close(); sf.close()

    # ── best epoch: val mAP@.5:.95. AP50은 포화돼서 안 쓴다. (RULES 5) ────────
    rows = list(csv.DictReader(open(m_path, encoding="utf-8")))
    best = max(rows, key=lambda r: float(r["val_map"]))
    (run / "best_epoch.json").write_text(json.dumps({
        "best_epoch": int(best["epoch"]),
        "criterion": "val mAP@.5:.95 최대",
        "why_not_ap50": "1차 학습에서 AP50이 1.000으로 포화돼 epoch를 구분하지 못했다.",
        "selected": {k: float(best[k]) for k in
                     ["val_map", "val_ap50", "val_loss", "train_loss"]},
        "checkpoint": str(run / "weights" / f"epoch_{int(best['epoch']):02d}.pth"),
        "all_epochs": [{"epoch": int(r["epoch"]), "val_map": float(r["val_map"]),
                        "val_ap50": float(r["val_ap50"]),
                        "val_loss": float(r["val_loss"])} for r in rows],
        "note": "test는 이 epoch로 딱 한 번만 연다. (RULES 2)",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'='*70}")
    print(f" best epoch {best['epoch']} — val mAP {float(best['val_map']):.4f} "
          f"(AP50 {float(best['val_ap50']):.4f})")
    print(f" {run}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
규칙 4의 기록 계약(contract). repo의 train.py가 이것을 import해서 호출한다.

설계 의도:
  repo 의존 부분을 여기 한 곳으로 몰아넣는다. 학습 루프는 아래 3개만 호출하면 되고,
  그 아래 모든 것(곡선·best epoch·REPORT·검증)은 metrics.csv만 읽으므로
  repo와 무관하게 동작한다.

     logger = RunLogger(out_dir, config, data_root, rounds)
     logger.log_step(lr)                          # step마다 (LR 실측값)
     logger.log_epoch(epoch, train_loss, val_loss, val_ap50, val_map)   # epoch마다
     logger.close()

metrics.csv 컬럼 (이후 도구들이 의존하는 스키마 — 바꾸지 말 것):
  epoch, train_loss, val_loss, val_ap50, val_map, lr_epoch_end
steps.csv 컬럼:
  global_step, epoch, lr
"""

import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

METRICS_COLS = ["epoch", "train_loss", "val_loss", "val_ap50", "val_map", "lr_epoch_end"]
STEPS_COLS = ["global_step", "epoch", "lr"]


def data_fingerprint(data_root: Path, rounds: list[str]) -> dict:
    """어떤 데이터로 돌렸는지 고정. 규칙 4: config.json에 데이터 분할 해시를 반드시 박는다."""
    fp = {}
    for r in list(rounds) + ["test"]:
        for split in (["train", "val"] if r != "test" else ["."]):
            d = data_root / r / split if split != "." else data_root / r
            ann = d / "annotations.json"
            if not ann.exists():
                continue
            coco = json.loads(ann.read_text())
            names = sorted(i["file_name"] for i in coco["images"])
            # 파일명 목록의 해시 = 분할 정체성. 한 장이라도 바뀌면 해시가 바뀐다.
            h = hashlib.sha256("\n".join(names).encode()).hexdigest()[:16]
            key = f"{r}/{split}" if split != "." else r
            fp[key] = {
                "n_images": len(coco["images"]),
                "n_boxes": len(coco["annotations"]),
                "split_hash": h,
            }
    return fp


def _git_rev(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


class RunLogger:
    def __init__(self, out_dir, config: dict, data_root, rounds: list[str],
                 gdino_root=None):
        self.out = Path(out_dir)
        (self.out / "curves").mkdir(parents=True, exist_ok=True)
        (self.out / "viz").mkdir(exist_ok=True)
        (self.out / "attention").mkdir(exist_ok=True)

        self.metrics_path = self.out / "metrics.csv"
        self.steps_path = self.out / "steps.csv"
        self._mf = open(self.metrics_path, "w", newline="")
        self._sf = open(self.steps_path, "w", newline="")
        self._mw = csv.writer(self._mf); self._mw.writerow(METRICS_COLS)
        self._sw = csv.writer(self._sf); self._sw.writerow(STEPS_COLS)

        self.step = 0
        self.epoch = 0

        (self.out / "config.json").write_text(json.dumps({
            "run_id": self.out.name,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "rounds": list(rounds),
            "hyperparameters": config,
            "data": data_fingerprint(Path(data_root), rounds),
            "code_version": {
                "gdino_repo": _git_rev(Path(gdino_root)) if gdino_root else "unknown",
            },
        }, indent=2, ensure_ascii=False))

    def log_step(self, lr: float):
        """step마다 호출. LR은 반드시 scheduler.get_last_lr()[0] 실측값을 넘길 것.
        의도한 수식을 넘기면 warmup/decay가 실제로 동작했는지 확인할 수 없다."""
        self.step += 1
        self._sw.writerow([self.step, self.epoch, f"{lr:.10g}"])
        self._sf.flush()

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  val_ap50: float, val_map: float, lr: float):
        self.epoch = epoch
        f = lambda v: "" if v is None else f"{v:.6g}"
        self._mw.writerow([epoch, f(train_loss), f(val_loss),
                           f(val_ap50), f(val_map), f(lr)])
        self._mf.flush()

    def close(self):
        self._mf.close()
        self._sf.close()

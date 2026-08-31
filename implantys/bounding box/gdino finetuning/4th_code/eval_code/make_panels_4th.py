#!/usr/bin/env python3
"""
Attention 비교 패널. (4th_plan ④)

한 이미지에 대해 FT-206v2 / FT-306v2 / FT-506v2 / FT-1000 의 attention map을
가로로 붙여 1장으로 만든다. 라운드가 올라가며 "어디를 보는지"가 어떻게 변하는지 비교.

케이스 구분(FN/FP/TP)을 파일명 앞에 붙여 정렬하면 종류별로 모인다.
  fn_<이미지명>.jpg / fp_<...> / tp_<...>

  python make_panels_4th.py --attention-root <...>/4th/results/attention
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

MODELS = ["FT-206v2", "FT-306v2", "FT-506v2", "FT-1000"]
LABEL_H = 26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention-root", required=True)
    ap.add_argument("--out", default=None, help="기본 <attention-root>/panels")
    a = ap.parse_args()

    root = Path(a.attention_root)
    out = Path(a.out) if a.out else root / "panels"
    out.mkdir(parents=True, exist_ok=True)

    cases = json.loads((root / "cases.json").read_text())
    kind_of = {}
    for k in ["fn", "fp", "tp"]:
        for n in cases[k]:
            kind_of.setdefault(n, k)          # 겹치면 먼저 온 것(fn 우선)

    # 각 모델의 max score (attention_meta.json에 기록됨) — 패널 라벨에 넣는다
    scores = {}
    for m in MODELS:
        mp = root / m / "attention_meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
            scores[m] = {k: v.get("max_score") for k, v in meta.get("images", {}).items()}

    made, skipped = 0, []
    for name, kind in sorted(kind_of.items(), key=lambda x: (x[1], x[0])):
        stem = Path(name).stem
        imgs = []
        for m in MODELS:
            p = root / m / f"{stem}.jpg"
            if not p.exists():
                imgs = []
                break
            imgs.append((m, Image.open(p).convert("RGB")))
        if not imgs:
            skipped.append(name)
            continue

        h = min(im.height for _, im in imgs)
        resized = [(m, im.resize((int(im.width * h / im.height), h))) for m, im in imgs]
        W = sum(im.width for _, im in resized)
        canvas = Image.new("RGB", (W, h + LABEL_H), "white")
        d = ImageDraw.Draw(canvas)
        x = 0
        for m, im in resized:
            canvas.paste(im, (x, LABEL_H))
            s = scores.get(m, {}).get(name)
            tag = f"{m}" + (f"  max score {s:.3f}" if isinstance(s, (int, float)) else "")
            d.text((x + 6, 7), tag, fill="black")
            d.line([(x, LABEL_H), (x, h + LABEL_H)], fill="#DDDDDD", width=2)
            x += im.width
        d.text((W - 120, 7), f"[{kind.upper()}]", fill="#C00000")

        canvas.save(out / f"{kind}_{stem}.jpg", quality=92)
        made += 1

    print(f"패널 {made}장 -> {out}")
    if skipped:
        print(f"건너뜀 {len(skipped)}장 (일부 모델 attention 없음):")
        for n in skipped[:5]:
            print("  ", n)
    print("\n판정은 사람이 한다 (RULES 7):")
    print("  attention은 갔는데 점수 낮음 -> 점수 보정 문제")
    print("  attention이 아예 안 감      -> 표현 학습 문제 -> 데이터 확대")


if __name__ == "__main__":
    main()

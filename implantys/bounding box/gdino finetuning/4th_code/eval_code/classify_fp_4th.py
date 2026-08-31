#!/usr/bin/env python3
"""
FP(오검출) 유형 분류. "없는 걸 봤다"가 정말 몇 개인지 가른다.

FP는 한 덩어리가 아니다. 매칭 실패의 원인이 셋으로 나뉜다:
  [1] 진짜 오검출   : 어떤 GT와도 겹치지 않음 (IoU ~ 0)          -> "없는 걸 봤다"
  [2] 위치 부정확   : GT와 겹치긴 하나 IoU < 0.5 (near miss)      -> 찾긴 찾았다
  [3] 중복 검출     : 이미 다른 예측이 잡은 GT를 또 잡음(IoU>=0.5) -> 찾긴 찾았다

FN도 같이 분류한다 (attention 관찰의 정량 확인):
  [B] 위치 부정확형 : 그 GT에 겹치는 예측이 있었으나 IoU < 0.5
  [S] 점수 미달형   : IoU>=0.5인 예측이 있었으나 score < thr
  [X] 미검출형      : 겹치는 예측이 아예 없음

  python classify_fp_4th.py --results-root <...>/test200 \
      --gt <...>/test200/annotations.json --run FT-1000 --thr 0.30
"""

import argparse
import json
from pathlib import Path

IOU_MATCH = 0.5
IOU_NEAR = 0.10          # 이보다 작으면 "겹치지 않는다"로 본다


def iou_xywh(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    u = aw * ah + bw * bh - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--run", default="FT-1000")
    ap.add_argument("--thr", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.results_root)
    gt = json.loads(Path(a.gt).read_text())
    id2name = {im["id"]: im["file_name"] for im in gt["images"]}
    gts = {}
    for an in gt["annotations"]:
        gts.setdefault(an["image_id"], []).append(an["bbox"])

    all_preds = json.loads((root / a.run / "predictions.json").read_text())
    kept, low = {}, {}
    for p in all_preds:
        (kept if p["score"] >= a.thr else low).setdefault(p["image_id"], []).append(p)

    fp_rows, fn_rows = [], []
    for img_id, name in id2name.items():
        g = gts.get(img_id, [])
        ps = sorted(kept.get(img_id, []), key=lambda r: -r["score"])
        used = {}
        # 1차: 표준 greedy 매칭
        for k, p in enumerate(ps):
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if i in used:
                    continue
                v = iou_xywh(p["bbox"], gb)
                if v > best:
                    best, bi = v, i
            if best >= IOU_MATCH:
                used[bi] = k
            else:
                # FP — 유형 판정
                ious = [iou_xywh(p["bbox"], gb) for gb in g]
                mx = max(ious) if ious else 0.0
                arg = int(max(range(len(ious)), key=lambda i: ious[i])) if ious else -1
                if mx >= IOU_MATCH:
                    kind, why = "3_중복검출", f"이미 매칭된 GT와 IoU {mx:.2f}"
                elif mx >= IOU_NEAR:
                    kind, why = "2_위치부정확", f"가장 가까운 GT와 IoU {mx:.2f} (<0.5)"
                else:
                    kind, why = "1_진짜오검출", f"최대 IoU {mx:.2f} — 겹치는 GT 없음"
                fp_rows.append({"image": name, "kind": kind, "score": round(p["score"], 3),
                                "max_iou_with_gt": round(mx, 3), "why": why,
                                "bbox": [round(v, 1) for v in p["bbox"]],
                                "nearest_gt_idx": arg})

        # FN 유형
        for i, gb in enumerate(g):
            if i in used:
                continue
            over_kept = max((iou_xywh(p["bbox"], gb) for p in ps), default=0.0)
            lows = low.get(img_id, [])
            best_low = max(((iou_xywh(p["bbox"], gb), p["score"]) for p in lows),
                           default=(0.0, 0.0))
            if best_low[0] >= IOU_MATCH:
                kind = "S_점수미달"
                why = f"IoU {best_low[0]:.2f} 예측이 있었으나 score {best_low[1]:.3f} < {a.thr}"
            elif over_kept >= IOU_NEAR or best_low[0] >= IOU_NEAR:
                kind = "B_위치부정확"
                why = f"겹치는 예측 최대 IoU {max(over_kept, best_low[0]):.2f} (<0.5)"
            else:
                kind = "X_미검출"
                why = "겹치는 예측 없음"
            fn_rows.append({"image": name, "kind": kind, "gt_idx": i, "why": why})

    def tally(rows):
        c = {}
        for r in rows:
            c[r["kind"]] = c.get(r["kind"], 0) + 1
        return dict(sorted(c.items()))

    print(f"=== {a.run} @ thr {a.thr} ===")
    print(f"\nFP {len(fp_rows)}개 유형별:")
    for k, v in tally(fp_rows).items():
        print(f"  {k}: {v}개")
    print("\nFP 상세 (이미지 / 유형 / score / GT와 최대 IoU):")
    for r in sorted(fp_rows, key=lambda x: (x["kind"], -x["score"])):
        print(f"  [{r['kind']}] {r['image']}")
        print(f"      score {r['score']:.3f} · {r['why']}")

    print(f"\nFN {len(fn_rows)}개 유형별:")
    for k, v in tally(fn_rows).items():
        print(f"  {k}: {v}개")
    print("\nFN 상세:")
    for r in sorted(fn_rows, key=lambda x: (x["kind"], x["image"])):
        print(f"  [{r['kind']}] {r['image']} (GT#{r['gt_idx']}) — {r['why']}")

    out = Path(a.out) if a.out else root.parent / "attention" / "fp_fn_classification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"run": a.run, "threshold": a.thr, "iou_match": IOU_MATCH, "iou_near": IOU_NEAR,
         "fp_summary": tally(fp_rows), "fn_summary": tally(fn_rows),
         "fp": fp_rows, "fn": fn_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n->", out)
    print("\n* 유형 2·3은 '없는 걸 봤다'가 아니다. 1번만 진짜 오검출.")


if __name__ == "__main__":
    main()

"""
Score threshold 스윕 + AP (100장 검수본 기준). 재학습 없는 공짜 개선.

pred = predictions_batch_01.json (박스별 score 있는 모델 원본 예측)
gt   = pseudo_batch_01_result.json annotations (사람 검수 확정)

목적:
  1) score threshold를 바꿔가며 P/R/F1 @IoU0.5 를 재서, batch-2 pre-annotation을
     생성할 때 사람 수정을 최소화하는 "운영점(operating point)"을 찾는다.
  2) 100장 사람라벨 기준 AP50 / mAP@.5:.95 (test-13보다 표본이 큰 신뢰 지표).

한계: 모델 예측 score 하한이 0.31 → 그 아래 박스는 저장돼 있지 않음.
      따라서 PR curve는 score>=0.31 구간만 유효하고 recall 상단이 잘림 → AP는 하한값(lower bound).
NMS: 이 batch 예측은 겹치는 박스가 0개라 NMS는 no-op (참고용 인자만 둠).
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_review import img_name, to_boxes, iou, mfr_of


def nms(boxes, thr):
    """greedy NMS. boxes=[x1,y1,x2,y2,score]. thr=1.0이면 사실상 no-op."""
    if thr >= 1.0:
        return boxes
    keep = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(iou(b, k) <= thr for k in keep):
            keep.append(b)
    return keep


def match_count(preds, gts, iou_thr):
    """greedy IoU 매칭 -> (TP, FP, FN). preds는 score 내림차순 가정 불필요(greedy on iou)."""
    cand = sorted(((iou(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
                  reverse=True)
    up, ug, tp = set(), set(), 0
    for v, pi, gi in cand:
        if v < iou_thr or pi in up or gi in ug:
            continue
        up.add(pi); ug.add(gi); tp += 1
    fp = len(preds) - len(up)
    fn = len(gts) - len(ug)
    return tp, fp, fn


def ap_single(images, iou_thr):
    """COCO식 AP (단일 클래스). images=[(preds_all, gts)]. preds에는 score 포함, 필터 없음."""
    # 전체 pred 박스를 score 내림차순으로 늘어놓고 이미지별 gt에 매칭
    entries = []  # (score, is_tp)
    npos = 0
    gt_used = [ [False]*len(g) for _, g in images ]
    for ii, (preds, gts) in enumerate(images):
        npos += len(gts)
    # score 순 전역 정렬 (이미지 인덱스 포함)
    allp = []
    for ii, (preds, gts) in enumerate(images):
        for p in preds:
            allp.append((p[4], ii, p))
    allp.sort(key=lambda x: -x[0])
    tp = fp = 0
    prec_rec = []
    for score, ii, p in allp:
        gts = images[ii][1]
        best_v, best_g = 0, -1
        for gi, g in enumerate(gts):
            if gt_used[ii][gi]:
                continue
            v = iou(p, g)
            if v > best_v:
                best_v, best_g = v, gi
        if best_v >= iou_thr and best_g >= 0:
            gt_used[ii][best_g] = True
            tp += 1
        else:
            fp += 1
        rec = tp / npos if npos else 0
        prec = tp / (tp + fp)
        prec_rec.append((rec, prec))
    # 101-point interpolation
    if npos == 0:
        return 0.0
    ap = 0.0
    for r in [i / 100 for i in range(101)]:
        ps = [p for rec, p in prec_rec if rec >= r]
        ap += (max(ps) if ps else 0.0) / 101
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default='predictions_batch_01.json')
    ap.add_argument('--gt', default='pseudo_batch_01_result.json')
    ap.add_argument('--iou-match', type=float, default=0.5)
    ap.add_argument('--nms', type=float, default=1.0, help='NMS IoU (1.0=off; 이 batch는 겹침0라 무효)')
    args = ap.parse_args()

    pred_raw = json.load(open(args.pred, encoding='utf-8'))
    pred_map = {}
    for t in pred_raw:
        b = to_boxes((t.get('predictions') or [{}])[0].get('result'))
        pred_map[img_name(t)] = nms(b, args.nms)

    gt_raw = json.load(open(args.gt, encoding='utf-8'))
    gt_map = {}
    for t in gt_raw:
        anns = [a for a in t.get('annotations', []) if not a.get('was_cancelled')]
        gt_map[img_name(t)] = to_boxes(anns[0].get('result')) if anns else []

    names = sorted(gt_map)
    images_full = [(pred_map.get(n, []), gt_map[n]) for n in names]  # AP용 (필터 없음)

    # ---- AP ----
    ap50 = ap_single(images_full, 0.5)
    maps = [ap_single(images_full, thr / 100) for thr in range(50, 100, 5)]
    map5095 = sum(maps) / len(maps)

    print("=== 100장 검수본 기준 AP (score 하한 0.31이라 lower bound) ===")
    print(f"AP50        : {ap50:.3f}")
    print(f"mAP@.5:.95  : {map5095:.3f}   (per-IoU: " +
          " ".join(f"{v:.2f}" for v in maps) + ")")

    # ---- threshold 스윕 ----
    print(f"\n=== score threshold 스윕 (매칭 IoU {args.iou_match}) ===")
    print(f"{'thr':>5}{'pred':>7}{'TP':>5}{'FP':>5}{'FN':>5}{'Prec':>7}{'Rec':>7}{'F1':>7}")
    best = (0, -1)  # (thr, f1)
    rows = []
    for k in range(30, 91, 5):
        thr = k / 100
        TP = FP = FN = NP = 0
        for preds, gts in images_full:
            pf = [p for p in preds if p[4] >= thr]
            NP += len(pf)
            tp, fp, fn = match_count(pf, gts, args.iou_match)
            TP += tp; FP += fp; FN += fn
        prec = TP / (TP + FP) if TP + FP else 0
        rec = TP / (TP + FN) if TP + FN else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        rows.append((thr, NP, TP, FP, FN, prec, rec, f1))
        if f1 > best[1]:
            best = (thr, f1)
        mark = ''
        print(f"{thr:>5.2f}{NP:>7}{TP:>5}{FP:>5}{FN:>5}{prec:>7.3f}{rec:>7.3f}{f1:>7.3f}")

    bt = best[0]
    br = [r for r in rows if r[0] == bt][0]
    cur = [r for r in rows if abs(r[0] - 0.30) < 1e-6][0]  # 현재 운영점(0.31 하한≈0.30 bucket)
    print(f"\n현재 운영점(≈0.30, 전량)  : F1={cur[7]:.3f}  P={cur[5]:.3f}  R={cur[6]:.3f}  (박스 {cur[1]}개)")
    print(f"F1 최적 운영점 thr={bt:.2f} : F1={br[7]:.3f}  P={br[5]:.3f}  R={br[6]:.3f}  (박스 {br[1]}개)")
    dF1 = br[7] - cur[7]
    print(f"→ batch-2 생성 threshold를 {bt:.2f}로 올리면 F1 {cur[7]:.3f}→{br[7]:.3f} ({dF1:+.3f}), "
          f"헛박스(FP) {cur[3]}→{br[3]}개")
    print("주의: recall 상단이 0.31 하한에 막혀 있음. 진짜 낮은 threshold 이득은 재생성해야 확인 가능.")


if __name__ == '__main__':
    main()

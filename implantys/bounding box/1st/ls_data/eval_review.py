"""
모델 예측(pre-annotation) vs 사람 검수본 비교 리포트

사용법:
  python eval_review.py --json export.json --pred predictions_batch_01.json
  python eval_review.py --json export.json --pred predictions_batch_01.json \
      --iou-match 0.5 --iou-clean 0.9 --csv report.csv --score 0.0

용어:
  pred = pod에서 epoch20으로 만든 pre-annotation json (모델 예측 원본)   <- --pred
  gt   = LS export annotations (사람이 검수/확정한 것)  <- 정답 취급     <- --json

주의: LS export의 predictions 필드는 ID 숫자([1])만 담고 있어 박스 복원 불가.
      따라서 모델 박스는 반드시 --pred(원본 pre-annotation)에서 읽는다.
"""

import argparse, csv, json, os, re, urllib.parse
from collections import Counter, defaultdict


def mfr_of(name):
    """dentium254_jpg.rf.xxx.jpg -> dentium (제조사 prefix, 뒤 숫자 제거)"""
    head = name.split('_')[0].split('.')[0]
    m = re.match(r'[a-zA-Z]+', head)
    return m.group(0).lower() if m else head.lower()


def img_name(task):
    v = list(task.get('data', {}).values())[0]
    if 'd=' in v:
        v = v.split('d=', 1)[1]
    return os.path.basename(urllib.parse.unquote(v))


def to_boxes(result):
    """LS result(퍼센트) -> [(x1,y1,x2,y2,score)] 픽셀"""
    out = []
    for r in result or []:
        if r.get('type') != 'rectanglelabels':
            continue
        W, H = r['original_width'], r['original_height']
        v = r['value']
        x1 = v['x'] / 100 * W
        y1 = v['y'] / 100 * H
        x2 = x1 + v['width'] / 100 * W
        y2 = y1 + v['height'] / 100 * H
        out.append([x1, y1, x2, y2, r.get('score', 1.0)])
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua


def match(preds, gts, thr):
    """greedy IoU 매칭 -> (pairs[(pi,gi,iou)], unmatched_pred, unmatched_gt)"""
    cand = sorted(
        ((iou(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
        reverse=True)
    used_p, used_g, pairs = set(), set(), []
    for v, pi, gi in cand:
        if v < thr or pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); pairs.append((pi, gi, v))
    return (pairs,
            [i for i in range(len(preds)) if i not in used_p],
            [i for i in range(len(gts)) if i not in used_g])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='LS full export (사람 검수본=gt)')
    ap.add_argument('--pred', required=True, help='pod pre-annotation json (모델 예측)')
    ap.add_argument('--iou-match', type=float, default=0.5, help='이 이상이면 같은 박스로 간주')
    ap.add_argument('--iou-clean', type=float, default=0.9, help='이 이상이면 "손 안 댐"')
    ap.add_argument('--score', type=float, default=0.0,
                    help='이 미만 점수의 모델 박스는 무시 (검수 때 걸었던 threshold와 맞출 것)')
    ap.add_argument('--csv', help='이미지별 상세 저장 경로')
    args = ap.parse_args()

    tasks = json.load(open(args.json, encoding='utf-8'))

    # pod pre-annotation -> {basename: [pred boxes]}
    pred_raw = json.load(open(args.pred, encoding='utf-8'))
    pred_map = {}
    for t in pred_raw:
        name = img_name(t)
        boxes = to_boxes((t.get('predictions') or [{}])[0].get('result'))
        boxes = [b for b in boxes if b[4] >= args.score]
        pred_map[name] = boxes

    reviewed, skipped = [], 0
    for t in tasks:
        anns = [a for a in t.get('annotations', []) if not a.get('was_cancelled')]
        if not anns:
            skipped += 1
            continue
        reviewed.append((t, anns[0]))

    TP = FP = FN = CLEAN = ADJ = 0
    ious, rows = [], []
    clean_imgs = 0
    per_mfr = defaultdict(lambda: Counter())

    missing = []
    for t, ann in reviewed:
        name = img_name(t)
        mfr = mfr_of(name)
        if name not in pred_map:
            missing.append(name)
        preds = pred_map.get(name, [])
        gts = to_boxes(ann.get('result'))

        pairs, up, ug = match(preds, gts, args.iou_match)
        n_clean = sum(1 for _, _, v in pairs if v >= args.iou_clean)
        n_adj = len(pairs) - n_clean

        TP += len(pairs); FP += len(up); FN += len(ug)
        CLEAN += n_clean; ADJ += n_adj
        ious += [v for _, _, v in pairs]

        untouched = (n_adj == 0 and not up and not ug)
        clean_imgs += untouched

        per_mfr[mfr]['imgs'] += 1
        per_mfr[mfr]['clean'] += untouched
        per_mfr[mfr]['adj'] += n_adj
        per_mfr[mfr]['fp'] += len(up)
        per_mfr[mfr]['fn'] += len(ug)

        rows.append(dict(image=name, mfr=mfr, n_pred=len(preds), n_gt=len(gts),
                         matched=len(pairs), adjusted=n_adj,
                         deleted_fp=len(up), added_fn=len(ug),
                         mean_iou=round(sum(v for _,_,v in pairs)/len(pairs), 4) if pairs else 0,
                         untouched=int(untouched)))

    n = len(reviewed)
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    miou = sum(ious)/len(ious) if ious else 0
    total_pred = TP + FP

    print(f"\n=== 검수 현황 ===")
    if missing:
        print(f"[경고] pred에 없는 이미지 {len(missing)}장 (pred 0개로 처리): {missing[:5]}...")
    print(f"검수 완료 이미지 : {n}장   (미검수/스킵 {skipped}장)")
    print(f"손 하나 안 댄 이미지: {clean_imgs}장 ({clean_imgs/n*100:.1f}%)   <- 모델 신뢰도 체감 지표")

    print(f"\n=== 박스 단위 (검수본을 정답으로) ===")
    print(f"모델이 그린 박스 : {total_pred}개")
    print(f"사람 최종 박스   : {TP + FN}개")
    print(f"  그대로 유지(IoU≥{args.iou_clean})  : {CLEAN}개 ({CLEAN/total_pred*100:.1f}% of pred)")
    print(f"  위치만 조정              : {ADJ}개 ({ADJ/total_pred*100:.1f}%)")
    print(f"  삭제됨(오검출, FP)       : {FP}개 ({FP/total_pred*100:.1f}%)")
    print(f"  사람이 새로 추가(놓침, FN): {FN}개")

    print(f"\n=== 지표 @IoU {args.iou_match} ===")
    print(f"Precision : {prec:.3f}")
    print(f"Recall    : {rec:.3f}")
    print(f"F1        : {f1:.3f}")
    print(f"mean IoU (매칭된 박스) : {miou:.3f}   <- 박스 정합도")
    print(f"수정률 = {(ADJ+FP+FN)/max(total_pred,1)*100:.1f}%")

    print(f"\n=== 제조사별 ===")
    print(f"{'mfr':<12}{'imgs':>6}{'clean%':>8}{'adj':>6}{'FP':>5}{'FN':>5}")
    for m, c in sorted(per_mfr.items()):
        print(f"{m:<12}{c['imgs']:>6}{c['clean']/c['imgs']*100:>7.0f}%{c['adj']:>6}{c['fp']:>5}{c['fn']:>5}")

    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"\n이미지별 상세: {args.csv}")


if __name__ == '__main__':
    main()

"""
모델 예측(pre-annotation) vs 사람 검수본(COCO) 비교 리포트  [3rd_total / 794장]

1st/ls_data/eval_review.py 를 그대로 따르되, GT를 LS full-export 대신
COCO(result.json)에서 읽는다.

  pred = pod에서 만든 pre-annotation json (LS percent 형식, score>=0.30 이미 필터됨)  <- --pred
  gt   = LS에서 COCO로 export 한 사람 검수본 (정답 취급)                          <- --coco

  python eval_review_coco.py --coco 3rd_total/result.json --pred 3rd_total/predictions_ls.json
"""

import argparse, csv, json, os, re, urllib.parse
from collections import Counter, defaultdict


def mfr_of(name):
    head = name.split('_')[0].split('.')[0]
    m = re.match(r'[a-zA-Z]+', head)
    return m.group(0).lower() if m else head.lower()


def img_name_pred(task):
    v = list(task.get('data', {}).values())[0]
    if 'd=' in v:
        v = v.split('d=', 1)[1]
    return os.path.basename(urllib.parse.unquote(v))


def pred_to_boxes(result):
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


def load_coco_gt(path):
    """COCO -> {basename: [[x1,y1,x2,y2], ...]}  (중복 basename은 박스 합침)"""
    d = json.load(open(path, encoding='utf-8'))
    id2name = {im['id']: os.path.basename(im['file_name']) for im in d['images']}
    gt = defaultdict(list)
    for name in id2name.values():
        gt.setdefault(name, [])          # 배경(박스 0개) 이미지도 등록
    for a in d['annotations']:
        name = id2name[a['image_id']]
        x, y, w, h = a['bbox']
        gt[name].append([x, y, x + w, y + h])
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', required=True, help='COCO 사람 검수본 (gt)')
    ap.add_argument('--pred', required=True, help='pod pre-annotation json (모델 예측)')
    ap.add_argument('--iou-match', type=float, default=0.5)
    ap.add_argument('--iou-clean', type=float, default=0.9)
    ap.add_argument('--score', type=float, default=0.0)
    ap.add_argument('--csv', help='이미지별 상세 저장 경로')
    args = ap.parse_args()

    gt_map = load_coco_gt(args.coco)

    pred_raw = json.load(open(args.pred, encoding='utf-8'))
    pred_map = {}
    for t in pred_raw:
        name = img_name_pred(t)
        boxes = pred_to_boxes((t.get('predictions') or [{}])[0].get('result'))
        boxes = [b for b in boxes if b[4] >= args.score]
        pred_map[name] = boxes

    TP = FP = FN = CLEAN = ADJ = 0
    ious, rows = [], []
    clean_imgs = 0
    per_mfr = defaultdict(lambda: Counter())
    missing = []

    # 검수본(gt)에 있는 이미지들만 평가 대상
    for name in sorted(gt_map):
        mfr = mfr_of(name)
        if name not in pred_map:
            missing.append(name)
        preds = pred_map.get(name, [])
        gts = gt_map[name]

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
                         mean_iou=round(sum(v for _, _, v in pairs)/len(pairs), 4) if pairs else 0,
                         untouched=int(untouched)))

    n = len(gt_map)
    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
    miou = sum(ious)/len(ious) if ious else 0
    total_pred = TP + FP

    print(f"\n=== 검수 현황 (3rd_total / 794장) ===")
    if missing:
        print(f"[경고] pred에 없는 검수이미지 {len(missing)}장(pred 0개 처리): {missing[:5]}")
    print(f"검수 완료 이미지 : {n}장")
    print(f"손 하나 안 댄 이미지: {clean_imgs}장 ({clean_imgs/n*100:.1f}%)")

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
    print(f"mean IoU (매칭된 박스) : {miou:.3f}")
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

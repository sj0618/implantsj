#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coco_to_ls.py  —  COCO annotation.json(FT-1000 예측) → Label Studio import용 predictions json

infer_to_coco.py 가 만든 폴더별 COCO(annotation.json)를, Label Studio가 pre-annotation으로
읽는 predictions 형식으로 변환한다. (bbox 픽셀 xywh → LS % 좌표)

■ 폴더 전체 모드 (제조사 폴더들이 든 상위 폴더 지정) — 각 폴더에 predictions_ls.json 생성:
    python coco_to_ls.py --root "7+3 label_0721" \
        --image-prefix "/data/local-files/?d=7+3 label_0721/{sub}/images/"
  ({sub} = 제조사 폴더명으로 치환됨. LS Local Storage 세팅에 맞춰 prefix 조정)

■ 단일 파일 모드:
    python coco_to_ls.py --coco ADIN/annotation.json --out ADIN/predictions_ls.json \
        --image-prefix "/data/local-files/?d=7+3 label_0721/ADIN/"

주의: --image-prefix 는 본인 로컬 Label Studio의 Local Storage 경로(?d=...)에 맞춰야
      이미지가 뜬다. RectangleLabels name=label / Image name=image 설정과도 맞아야 함.
"""
import argparse, json, os, uuid


def coco_to_tasks(coco, prefix, model_version, label):
    id2meta = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in coco["images"]}
    by_img = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    tasks = []
    for iid, (fname, W, H) in id2meta.items():
        regions, scores = [], []
        for a in by_img[iid]:
            x, y, w, h = a["bbox"]
            sc = float(a.get("score", 1.0))
            scores.append(sc)
            regions.append({
                "id": uuid.uuid4().hex[:10],
                "type": "rectanglelabels",
                "from_name": "label", "to_name": "image",
                "original_width": int(W), "original_height": int(H),
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": max(0.0, x / W * 100.0), "y": max(0.0, y / H * 100.0),
                    "width": w / W * 100.0, "height": h / H * 100.0,
                    "rectanglelabels": [label],
                },
                "score": sc,
            })
        tasks.append({
            "data": {"image": prefix + fname},
            "predictions": [{
                "model_version": model_version,
                "score": (sum(scores) / len(scores)) if scores else 0.0,
                "result": regions,
            }],
        })
    return tasks


def convert_file(coco_path, out_path, prefix, model_version, label):
    coco = json.load(open(coco_path, encoding="utf-8"))
    tasks = coco_to_tasks(coco, prefix, model_version, label)
    json.dump(tasks, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    nbox = sum(len(t["predictions"][0]["result"]) for t in tasks)
    print(f"  {coco_path} -> {out_path}  (tasks={len(tasks)}, boxes={nbox})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="제조사 폴더들이 든 상위 폴더 (폴더 전체 모드)")
    ap.add_argument("--coco", default=None, help="단일 annotation.json (단일 파일 모드)")
    ap.add_argument("--out", default=None, help="단일 모드 출력 경로")
    ap.add_argument("--in-name", default="annotation.json", help="폴더모드: 읽을 COCO 파일명")
    ap.add_argument("--out-name", default="predictions_ls.json", help="폴더모드: 출력 파일명")
    ap.add_argument("--image-prefix", default="", help="data.image 접두어. {sub}=폴더명 치환")
    ap.add_argument("--model-version", default="gdino_ft1000_epoch9")
    ap.add_argument("--label", default="implant")
    args = ap.parse_args()

    if args.coco:
        out = args.out or os.path.join(os.path.dirname(args.coco), args.out_name)
        convert_file(args.coco, out, args.image_prefix.replace("{sub}", ""), args.model_version, args.label)
        return

    if not args.root:
        ap.error("--root 또는 --coco 중 하나는 필요")
    for sub in sorted(os.listdir(args.root)):
        d = os.path.join(args.root, sub)
        cp = os.path.join(d, args.in_name)
        if not os.path.isdir(d) or not os.path.exists(cp):
            continue
        prefix = args.image_prefix.replace("{sub}", sub)
        convert_file(cp, os.path.join(d, args.out_name), prefix, args.model_version, args.label)
    print("[done]")


if __name__ == "__main__":
    main()

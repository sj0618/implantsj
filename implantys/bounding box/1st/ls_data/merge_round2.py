"""
Round2 재학습 데이터 병합.

train(83) + pseudo_batch_01 검수본(100) -> dataset_coco_round2/train
val(23) / test(13) 은 그대로 복사 (절대 고정, 누수 방지 재검사 포함).

출력:
  dataset_coco_round2/{train,val,test}/annotations.json (+ images/)
  dataset_coco_round2/train_round2.csv   (image_name,bbox_x,bbox_y,bbox_width,bbox_height,label_name)

박스 0개 이미지는 이번 병합에서 제외.
"""
import json, os, re, shutil, csv, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'dataset_coco')
PSEUDO_COCO = os.path.join(ROOT, 'ls_data', 'pseudo_batch_01_coco', 'result.json')
PSEUDO_IMGS = os.path.join(ROOT, 'ls_data', 'pseudo_batch_01_images')
OUT = os.path.join(ROOT, 'dataset_coco_round2')
CAT = [{'id': 0, 'name': 'implant'}]


def rfhash(fn):
    m = re.search(r'rf\.([0-9a-f]+)', os.path.basename(fn))
    return m.group(1) if m else os.path.basename(fn)


def load(sp):
    return json.load(open(os.path.join(SRC, sp, 'annotations.json'), encoding='utf-8'))


def anns_by_img(coco):
    d = {}
    for a in coco['annotations']:
        d.setdefault(a['image_id'], []).append(a)
    return d


def write_split(name, images, ann_of_file, src_img_dirs):
    """images: list of (file_basename, [bbox,...]); ann_of_file already pixel [x,y,w,h]"""
    out_dir = os.path.join(OUT, name)
    img_dir = os.path.join(out_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    out_imgs, out_anns = [], []
    aid = 0
    for iid, (fn, boxes) in enumerate(images):
        out_imgs.append({'id': iid, 'file_name': fn, 'width': 640, 'height': 640})
        for b in boxes:
            out_anns.append({'id': aid, 'image_id': iid, 'category_id': 0,
                             'bbox': [round(x, 3) for x in b], 'area': round(b[2] * b[3], 3),
                             'iscrowd': 0})
            aid += 1
        # copy image
        for sd in src_img_dirs:
            cand = os.path.join(sd, fn)
            if os.path.exists(cand):
                shutil.copy2(cand, os.path.join(img_dir, fn))
                break
        else:
            print(f'  [경고] 이미지 파일 못 찾음: {fn}')
    json.dump({'images': out_imgs, 'annotations': out_anns, 'categories': CAT},
              open(os.path.join(out_dir, 'annotations.json'), 'w', encoding='utf-8'))
    return len(out_imgs), len(out_anns)


def main():
    os.makedirs(OUT, exist_ok=True)

    tr = load('train'); va = load('val'); te = load('test')
    val_test_hashes = {rfhash(i['file_name']) for i in va['images']} | \
                      {rfhash(i['file_name']) for i in te['images']}

    # --- pseudo 검수본 ---
    ps = json.load(open(PSEUDO_COCO, encoding='utf-8'))
    ps_by = anns_by_img(ps)
    pseudo_items, excluded_empty, leak = [], 0, []
    seen = {rfhash(i['file_name']) for i in tr['images']}
    for im in ps['images']:
        fn = os.path.basename(im['file_name'])
        h = rfhash(fn)
        if h in val_test_hashes:            # 누수 재검사 (병합 시점에도 한 번 더)
            leak.append(fn)
            continue
        boxes = [a['bbox'] for a in ps_by.get(im['id'], [])]
        if not boxes:                        # 박스 0개 제외
            excluded_empty += 1
            continue
        if h in seen:                        # train 중복 제외
            continue
        seen.add(h)
        pseudo_items.append((fn, boxes))

    if leak:
        print(f'[중단] 누수 발견 {len(leak)}장 → 병합 중단: {leak[:5]}')
        sys.exit(1)

    # --- train 원본 ---
    tr_by = anns_by_img(tr)
    train_items = [(os.path.basename(im['file_name']),
                    [a['bbox'] for a in tr_by.get(im['id'], [])]) for im in tr['images']]

    merged_train = train_items + pseudo_items
    tr_dirs = [os.path.join(SRC, 'train', 'images')]
    ps_dirs = [PSEUDO_IMGS]

    # write train (두 소스 이미지 폴더 모두 탐색)
    n_i, n_a = write_split('train', merged_train, None, tr_dirs + ps_dirs)
    va_i, va_a = write_split('val', [(os.path.basename(im['file_name']),
                    [a['bbox'] for a in anns_by_img(va).get(im['id'], [])]) for im in va['images']],
                    None, [os.path.join(SRC, 'val', 'images')])
    te_i, te_a = write_split('test', [(os.path.basename(im['file_name']),
                    [a['bbox'] for a in anns_by_img(te).get(im['id'], [])]) for im in te['images']],
                    None, [os.path.join(SRC, 'test', 'images')])

    # CSV (COCO [x,y,w,h])
    csv_path = os.path.join(OUT, 'train_round2.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['image_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'label_name'])
        for fn, boxes in merged_train:
            for b in boxes:
                w.writerow([fn, round(b[0], 3), round(b[1], 3), round(b[2], 3), round(b[3], 3), 'implant'])

    # 제조사 분포
    from collections import Counter
    def mfr(fn):
        head = fn.split('_')[0].split('.')[0]
        head = re.sub(r'^[0-9a-f]{8}-', '', head)   # coco의 8-hex prefix 제거
        m = re.match(r'[a-zA-Z]+', head); return m.group(0).lower() if m else head.lower()
    dist_new = Counter(mfr(fn) for fn, _ in pseudo_items)
    dist_all = Counter(mfr(fn) for fn, _ in merged_train)

    print('=== Round2 병합 결과 ===')
    print(f'신규 이미지(pseudo): {len(pseudo_items)}장  (박스0 제외 {excluded_empty}장, 누수 {len(leak)}장)')
    print(f'신규 박스          : {sum(len(b) for _,b in pseudo_items)}개')
    print(f'train 총계         : {n_i}장 / {n_a}박스   (기존 {len(train_items)}장 + 신규 {len(pseudo_items)}장)')
    print(f'val (고정)         : {va_i}장 / {va_a}박스')
    print(f'test(고정)         : {te_i}장 / {te_a}박스')
    print(f'CSV                : {csv_path}')
    print(f'신규 제조사 분포   : {dict(sorted(dist_new.items()))}')
    print(f'train 전체 분포    : {dict(sorted(dist_all.items()))}')


if __name__ == '__main__':
    main()

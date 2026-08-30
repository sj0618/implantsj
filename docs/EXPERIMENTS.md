# 실험 설정 · 실험 방법 (implantsj)

이 문서는 각 실험의 목적과 설정을 정리한다.
설치와 메뉴 사용법은 [`README.md`](../README.md)를 참고한다.

---

## 1. 실험 축

| 축 | 목적 | 진입점 |
|---|---|---|
| Supervised 분류 | closed-set 라벨 직접 분류 (3/7/10/11/27-label) | `run.sh` 메뉴 6 · `tools/run.py train` |
| Metric learning | SupCon / Triplet 임베딩 학습 | 메뉴 7 · `tools/run.py metric` |
| Prototype 평가 | seen 라벨 prototype 최근접 분류 | 메뉴 8 · `tools/run.py eval` |
| Unseen / open-set | 학습에 없던 라벨을 Mahalanobis 등으로 판정 | 메뉴 9 · `tools/run.py maha` |
| Transfer | source 라벨셋(7/10) → target 라벨셋(3) 전이 | `pipelines/eval/evaluate_3label_*` |
| YOLO 검출/crop | 원본 파노라마에서 임플란트 ROI만 잘라내기 | 메뉴 10·11 |
| VLM feature | 구조적 feature(13feature/21reason)를 ViT 패치로 주입 | `tools/run.py patchtrain` |

---

## 2. 데이터

### 2-1. 원본 구조 (폐쇄망)

```
TS1/h01/h01_0001/*.jpg    train 이미지
TL1/h01/h01_0001/*.json   train 라벨  (JSON 의 "모델명" 이 정답 클래스)
VS1/ ... VL1/ ...         test 이미지 / 라벨
```

`tools/run.py prepare` (메뉴 2)가 이걸 ImageFolder 로 변환한다.
JSON 이 없는 이미지는 건너뛰고 개수를 `summary.json` 에 남긴다.

### 2-2. 변환 결과

```
<이름>/imagefolders/data_42label/train/<클래스>/...      전체 라벨
<이름>/imagefolders/data_metric_seen27/...               학습용 seen
<이름>/imagefolders/data_metric_unseen15/test/...        평가용 unseen
```

라벨 수는 실제 데이터에서 세어 붙는다(`data_38label` 처럼). unseen 으로 뺄
희귀 라벨 개수는 변환 시 물어보며, `0` 이면 전부 학습에 쓴다.

### 2-3. manifest / split

| 항목 | 값 |
|---|---|
| manifest | `pipelines.data.make_manifest --data-root ... --out m.csv` |
| layout | `auto` / `class_folder` / `split_class_folder` |
| split 비율 | train : valid : test = **8 : 1 : 1** (`make_splits` 기본값) |
| split seed | **42** |
| 선언형 split | `manifests/splits/closed27_open15_seed42.yaml`: stratified_hash, 0.70 / 0.15 / 0.15 |

선언형 split 은 view 단위로 역할을 나눈다: `train`, `validation`, `closed_test`,
`gt_closed`(prototype), `open_test`(unknown), `all_train_test`(분석용).

### 2-4. 전처리 / augmentation (`src/data.py: build_transforms`)

| 구분 | 내용 |
|---|---|
| 공통 | `Resize(image_size, image_size)` → `ToTensor` → ImageNet Normalize (mean 0.485/0.456/0.406, std 0.229/0.224/0.225) |
| train strong_aug | `RandomHorizontalFlip(p=0.5)`, `RandomRotation(10°)`, `ColorJitter(brightness=0.15, contrast=0.15)` |
| 끄기 | `--no-strong-aug` |

---

## 3. 공통 하이퍼파라미터 (`exp.conf`)

```
EPOCHS=80  BATCH_SIZE=32  IMAGE_SIZE=224  LR=5e-5  SEED=42
NUM_WORKERS=8  DEVICE=cuda
DATA_ROOT=/workspace/data   OUTPUT_ROOT=/workspace/outputs
```

환경변수로 한 번만 값을 바꿀 수 있다: `EPOCHS=5 ./run.sh`. 메뉴 `14`에서도 임시 변경이 가능하다.

### 백본 후보 (`configs/backbones.txt`)

```
vit_base_patch16_224.augreg2_in21k_ft_in1k
vit_small_patch16_224.augreg_in21k_ft_in1k
resnet50.a1_in1k
resnet18.a1_in1k
convnext_tiny.fb_in22k_ft_in1k
efficientnet_b0.ra_in1k
```

> 모델을 추가하려면 timm 이름을 한 줄 추가한 뒤 이미지를 다시 빌드한다. pretrained weight는
> 빌드할 때 이미지에 저장되므로, 폐쇄망 안에서 이름만 추가하면 weight를 찾지 못한다.

---

## 4. Supervised 분류 설정

`pipelines/train/train_supervised.py` 기본값:

| 항목 | 기본값 |
|---|---|
| model-name | `vit_base_patch16_224` (pretrained=True) |
| image-size / epochs / batch-size | 224 / 80 / 32 |
| lr / weight-decay | 5e-5 / 0.05 |
| label-smoothing | 0.1 |
| dropout / drop-path-rate | 0.1 / 0.1 |
| AMP | on |
| early stopping | `--patience 12`, `--monitor macro_f1` |
| 불균형 대응 | `--use-class-weights` (기본 off) |
| 이어하기 | `--resume-checkpoint` / 가중치만 `--init-checkpoint` |

선언형 등가물은 `manifests/experiments/vit_closedft.yaml` (CE, AdamW, cosine
scheduler, epochs 10, batch 32, samples_per_class 2, projection 128/512).

---

## 5. Metric learning 설정

`pipelines/train/train_metric.py` 기본값:

| 항목 | 기본값 |
|---|---|
| loss-name | `supcon` (또는 `triplet`) |
| temperature (SupCon) | 0.07 |
| triplet-margin | 0.2 (batch-hard) |
| embedding-dim | 128 |
| lr / weight-decay | 3e-5 / 0.05 |
| epochs / batch-size | 80 / 32 |
| patience | 12 |

> **SupCon 주의**: 배치 안에 같은 라벨이 2장 이상 있어야 loss가 계산된다.
> 클래스가 많고 batch가 작으면 경고가 뜬다. `BATCH_SIZE`를 올린다.

---

## 6. 평가 프로토콜

### 6-1. 분류 평가 (`tools/run.py eval`)

`--checkpoint` + `--manifest` + `--split {train,valid,test}` (기본 test),
batch-size 64. model-name / image-size 를 생략하면 checkpoint 의 args 에서 읽는다.
지표는 accuracy, **macro-F1**, per-class report, confusion matrix.

### 6-2. Unseen / Mahalanobis (`tools/run.py maha`)

target-train 이미지로 클래스 prototype(mean)을 만들고 query 를 최근접으로 판정한다.

| 항목 | 값 |
|---|---|
| 임베딩 소스 | `--embedding projection` (기본) 또는 `backbone` |
| 거리식 | euclidean · manhattan · cosine · **mahalanobis** 동시 산출 |
| 공분산 추정 | Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`) |
| 지표 | accuracy, macro-F1, balanced accuracy, confusion matrix |

### 6-3. 해석 규칙

direct classification / legacy transfer / uncropped transfer / GT27 diagnosis 는
**split·protocol 이 서로 다르다**. 절대 수치로 순위를 매기지 말고 각 축 안에서
방향성만 비교한다. Accuracy 단독이 아니라 macro-F1 을 함께 본다.

---

## 7. YOLO 검출 · crop

| 단계 | 기본값 |
|---|---|
| 학습 (`yolotrain`) | epochs 120, batch 16, imgsz 640, workers 8, patience 20, mosaic/AMP 는 플래그로 on |
| crop (`yolocrop`) | imgsz 640, conf 0.25, iou 0.45, pad 0.15(박스 주변 여유), batch 16 |
| 박스 미검출 | 기본 스킵, `--save-empty` 면 전체 이미지를 리사이즈해 저장 |

crop 결과 데이터셋(`*_cropped`)을 다시 메뉴 3 에서 골라 분류/metric 학습에 쓴다.

---

## 8. 결과물 위치

```
outputs/<타임스탬프>_<모델>_<데이터셋>_<epoch>_<seed>_.../
  args.json            그 실행의 전체 설정 (재현 근거)
  class_to_idx.json    라벨 인덱스 매핑
  checkpoints/         best / last
  metrics/             accuracy·macro-F1·per-class
  confusion_matrices/
  embeddings/
  attention_maps/
  logs/
```

실행 폴더 이름에 데이터셋·모델이 들어가므로 조합을 바꿔 돌려도 서로 덮어쓰지 않는다.
모든 실행 로그는 `outputs/logs/` 에 날짜별로도 남는다.

---

## 9. 재현 절차 요약

```bash
./run.sh                      # 1 환경체크 → 2 데이터변환 → 3 데이터셋 → 4 모델 → 6/7 학습 → 8/9 평가
./run.sh train <데이터셋> <모델>   # 메뉴 없이 바로
```

재현에 필요한 최소 정보: **데이터셋 폴더명 · 백본 이름 · `args.json` · seed**.
seed 는 기본 42 이며, 결론을 내기 전에는 **최소 3 seed** 로 재검증한다.

---

## 10. 결과 · 후속 계획

- 실험별 정량 결과: [`../implant_experiment_summary_table.md`](../implant_experiment_summary_table.md)
- 3장 요약: [`../implant_experiment_summary_3slides.md`](../implant_experiment_summary_3slides.md)
- 상세 리포트: `docs/analysis/EXPERIMENT_SUMMARY.md`, `docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md`
- 다음 우선순위: ROI crop 우선, 384/512 multi-crop 비교, source-group balanced prototype,
  macro-F1 중심 평가, 3 seed 이상 반복.

# Stage 3. Classifier — band에 주목시키는 두 방식

코드는 `src/band_vit/`, `src/masked_vit/`.
설정은 `configs/04_band_vit.yaml`, `configs/04_masked_vit.yaml`.
노션 구조의 **3. Classifier** 브랜치.

같은 목표(임플란트 경계 band에 집중)를 서로 다른 방법으로 푼다.

| 방식 | 학습 입력 | test 입력 | 기본 조합 |
|---|---|---|---|
| `band-vit` | 원본 + band attention KL loss | 원본 | 18 (alpha{4,8,16} × lambda{0.5,0.8,0.95} × block{6,12}; prior = 1+α·mask 스케일) |
| `masked-vit` | band만 남기고 마스킹 | **원본** (+ 비교용 band) | 1 |

```bash
bash train_band_vit.sh
bash train_masked_vit.sh
PROTOCOL=full42 bash train_band_vit.sh
```

## band-vit

원본 이미지를 그대로 넣되, CLS→patch attention이 band 쪽으로 쏠리도록
KL prior loss를 CE에 더한다 (`prior = 1 + alpha * band_mask`).

| 옵션 | 위치 | 뜻 |
|---|---|---|
| `sweep.alpha` | prior 강도 | 클수록 band를 강하게 유도 |
| `sweep.lambda` | CE 대비 band loss 비중 | 0이면 일반 학습과 동일 |
| `sweep.block` | attention을 걸 **마지막 N개** block | block 인덱스가 아니다 |
| `backbone` | Stage 1 백본 이름 | 비우면 timm pretrained |
| `viz.attention_samples` / `rollout_samples` | test 시각화 장수 | −1=전부, 0=끄기 |

**입력 요구사항**: manifest에 `mask_path` 컬럼이 채워져 있어야 한다
(Stage 0 → Stage 0-a 재실행).

## masked-vit

학습할 때 SAM2로 런타임에 band만 남기고 중앙·배경을 가린다.
**test는 원본 이미지**로 한다 — 핵심 질문은
*"가리고 학습한 모델이 원본에서도 경계에 집중하는가"* 이고,
`focus_gain`(band 위 attention 비율 ÷ band 면적 비율)이 1을 넘으면 그렇다는 뜻이다.

| 옵션 | 뜻 |
|---|---|
| `test_variants` | `original`(핵심) / `band`(비교 기준) |
| `eval_valid` | valid도 평가할지. **끄면 Stage 5 fuse의 `weights.mode: learned` / `threshold.mode: auto` 를 쓸 수 없다** |
| `band.outer_radius` / `inner_radius` | 런타임 band 두께 |
| `band.max_failure_rate` | SAM2가 마스크를 못 뽑은 이미지를 이 비율까지 빼고 진행 (기본 0 = 하나라도 실패하면 중단) |
| `sam.*` | Stage 0에서 만든 세그 모델 경로 |

`mask_path` 컬럼은 쓰지 않는다(있으면 무시하고 로그로 알린다) — band를 런타임에 만들기 때문이다.

**`max_failure_rate`가 필요한 이유**: 데이터가 많아지면 SAM2가 마스크를 전혀
못 뽑는 이미지가 소수 나온다. 기본값 0은 하나라도 실패하면 중단해서
데이터가 조용히 빠지는 걸 막는다. 실패 목록은 항상
`preprocessing/band_preprocessing_report.csv` 에 남으므로, 확인한 뒤
`0.01` 같은 값으로 올려 진행하면 된다.

## 출력

```
outputs/classifier/band_vit/<protocol>_a<alpha>_l<lambda>_b<block>/
  checkpoints/best.pt
  evaluation/{valid,test}/scores.csv      ← Stage 5 fuse 입력
  evaluation/test/attention_maps/*.png
  rollout_test/*.png                      gradient rollout
  summary.json  history.csv  curves.png

outputs/classifier/masked_vit/<protocol>/
  eval_valid/scores.csv                   ← Stage 5 fuse 입력 (valid)
  eval_original/scores.csv                ← Stage 5 fuse 입력 (test, 핵심)
  eval_band/scores.csv                    비교용
  rollout_{original,band}/*.png
  preprocessing/band_preprocessing_report.csv
```

## 하나만 돌리기

sweep은 의무가 아니다. yaml의 리스트를 한 개로 줄이거나(스칼라도 허용),
yaml을 고치지 않고 `SET=` 으로 이번 실행만 좁힐 수 있다.

```bash
SET="sweep.alpha=8 sweep.lambda=0.8 sweep.block=6" bash train_band_vit.sh
SET="test_variants=original" bash train_masked_vit.sh
```

`SET`은 yaml 항목을 점으로 짚어 덮어쓴다 (`train.epochs=1`,
`band.max_failure_rate=0.01` 등). 적용된 값은 실행 로그에
`[config] override ...` 로 남는다.

# Stage 2. Metric learning 실험 목록

코드는 `src/metric/`, 설정은 `configs/03_*.yaml`.
노션 구조의 **3. Metric learning** 브랜치.

## 실험 3개

| 스크립트 | 설정 | 무엇을 보는가 |
|---|---|---|
| `run_metric_grid.sh` | `03_metric_grid.yaml` | **메인.** closed 27로 임베딩 학습 → 갤러리 42-way(closed 27 + open 15 support)로 분류. backbone × loss 그리드 |
| `run_clustering_full42.sh` | `03_clustering_full42.yaml` | 42클래스 전부 학습 프로토콜. metric learning 없이 임베딩 clustering만 |
| `run_metric_fewshot.sh` | `03_metric_fewshot.yaml` | **별개 실험.** 새 시스템을 1/5/10장만 등록해도 인식되는지, seed 반복으로 평균±표준편차 |

```bash
bash run_metric_grid.sh          # 메인 그리드
bash run_clustering_full42.sh    # full42 프로토콜
bash run_metric_fewshot.sh       # few-shot (그리드에서 best 뽑은 뒤 실행)
```

## 실험 축 (yaml에서 선택)

메인 그리드는 **backbone × embedding_loss**만 조합으로 돌고,
prototype × 거리는 재학습 없이 평가만 반복한다 (비용 절감).

| 축 | 위치 | 선택지 |
|---|---|---|
| backbone | `grid.backbone` | vit / band-vit / dinov2 / band-dinov2 / cnn (`base.yaml` 레지스트리) |
| embedding finetuning | `grid.embedding_loss` | supcon / triplet / arcface / **covaware_supcon**★ / none |
| prototype | `eval.prototype.methods` | mean / ss / medoid / **kmeans_multi**★ / **reference**★ |
| 거리 | `eval.distances` | cosine / euclidean / mahalanobis / **fractional**★ / **centered_angular**★ |
| mahalanobis 공분산 | `eval.mahalanobis.covariance` | ledoit_wolf / oas / diagonal / class_balanced / **rda**★ |
| 임베딩 변환 | `eval.transform.mode` | none / l2norm / **pca**★ |
| open-set 판정 | `eval.openset.mode` | none / **threshold**★ |
| 중복 제거 지표 | `eval.duplicate_free` | false / **true**★ |
| 거부 성능 지표 | `eval.reject_metrics.enabled` | false / **true**★ (AUROC·FPR@95) |
| 구면 평균 prototype | `eval.prototype.cosine_spherical_mean` | false / **true**★ |

★ = 팀원(implantsj)이 추가로 시도한 방법. 기본값은 전부 꺼져 있으므로
켜고 싶은 것만 yaml에서 바꾸면 된다.

### ★ 방법 한 줄 설명

- **covaware_supcon** — SupCon에 episodic 공분산 인식 loss를 더해, Mahalanobis
  거리에서 잘 분리되는 임베딩을 직접 학습한다.
- **reference** — 따로 준비한 **대표 이미지(GT)**의 임베딩을 prototype으로 쓴다.
  `data/reference/<클래스>/*.jpg` 구조. 팀원 코드의 `gt` prototype 정책과 같은 아이디어다.
- **kmeans_multi** — 클래스 하나를 여러 prototype으로 표현하고 최소거리로 판정.
  한 클래스 안에 촬영 각도 등으로 여러 군집이 있을 때 유리.
- **fractional / centered_angular** — 공분산 고유값 스펙트럼으로 축마다 가중치를
  주는 거리. `spectral.alpha`로 세기를 조절(0.5 / 0.75).
- **rda** — pooled 공분산과 클래스별 공분산을 섞고(`class_weight`) logdet 항으로
  보정. 클래스마다 분산 모양이 다를 때 유리.
- **pca** — 거리 계산 전에 임베딩을 저차원으로 줄인다. 샘플 수 대비 차원이 높아
  공분산 추정이 불안정할 때 유리.
- **threshold** — 최근접 score가 임계를 넘으면 `__unknown__`으로 판정. Fuse의
  open-set gate 실험 재료. `value`를 비우면 `percentile`로 자동 결정.
- **duplicate_free** — train과 파일 내용(sha256)이 같은 test 이미지를 빼고
  다시 계산한 지표를 함께 낸다 (데이터 위생 점검).

## 시각화

`eval.visualize.enabled: true`(기본)면 조합(prototype × distance)마다
`eval/visualization/`에 그림을 저장한다.

| 파일 | 내용 |
|---|---|
| `embedding_<reducer>_<tag>.png` | 임베딩 산점도. **세모 = open 클래스**, **별 = prototype** |
| `embedding_<reducer>_<tag>.csv` | 2D 좌표 (직접 다시 그리거나 다른 도구로 분석할 때) |
| `score_distribution_<tag>.png` | 정답/오답 best score 분포 — Stage 5 threshold를 어디 둘지 근거 |

```yaml
eval:
  visualize:
    enabled: true
    reducer: tsne          # tsne(기본) | umap(umap-learn 필요) | none
    show_prototypes: true
    max_samples: 2000      # t-SNE는 느리므로 클래스 비율 유지하며 표본 추출
    pca_dim: 50            # 축소 전 PCA (0이면 생략)
    perplexity: 30.0       # tsne
    n_neighbors: 15        # umap
    min_dist: 0.1          # umap
```

closed27_open15에서는 open 클래스를 다른 마커로 그리므로
**"학습에 없던 15개 클래스가 임베딩 공간에서 따로 뭉치는가"**를 바로 확인할 수 있다.
prototype을 임베딩과 **같은 좌표계**에서 축소해 겹쳐 그린다(따로 축소하면 좌표가 안 맞음).

그림 안 글자는 전부 영어다 — 폐쇄망 서버에 한글 폰트가 없어도 깨지지 않게.
시각화가 실패해도 평가 자체는 계속된다(로그에만 남김).

## open-set 거부 성능 (`reject_metrics`)

threshold를 정하고 unknown 비율만 보면 임계 선택에 따라 평가가 흔들린다.
이 지표는 **임계를 정하지 않고 순위만으로** "학습에 없던 클래스를 걸러낼 수 있는가"를 잰다.

```yaml
eval:
  reject_metrics:
    enabled: true
    scores: [min_dist, margin, dist_ent]   # logit_msp, logit_ent 도 가능
    temperature: 1.0
```

| 지표 | 의미 |
|---|---|
| `auroc` | unknown 점수가 known보다 높게 나올 확률 (1.0 = 완벽, 0.5 = 무작위) |
| `fpr_at_95tpr` | unknown을 95% 잡을 때 known을 잘못 버리는 비율 (낮을수록 좋음) |

**거부 점수 5종을 함께 비교**한다 — 무엇으로 판정하느냐가 성능을 크게 가른다:

| 점수 | 판정 근거 |
|---|---|
| `min_dist` | 최근접 prototype까지의 거리 (멀면 unknown) |
| `margin` | 1등과 2등 거리의 차 (작으면 애매) |
| `dist_ent` | 거리 softmax 분포의 엔트로피 |
| `logit_msp` | 거리 기반 확률의 최대값 |
| `logit_ent` | 같은 확률의 엔트로피 |

참고로 다른 팀원의 DINOv2 실험에서는 학습을 안 한 백본일 때
`margin`이 `min_dist`보다 크게 나았고(AUROC 0.394 → 0.645),
학습한 모델에서는 `min_dist`가 가장 좋았다. 조건마다 다르므로 함께 보는 게 낫다.

`closed27_open15`에서만 계산된다 — open 클래스가 있어야 known/unknown이 갈린다.
결과는 `rejection_<split>_<tag>.json`과 metrics json의 `reject` 블록에 저장된다.

## 구면 평균 prototype (`cosine_spherical_mean`)

cosine 거리는 방향만 보는데, 산술평균을 내면 **노름이 큰 샘플이 방향을 독점**한다.
각 벡터를 단위길이로 맞춘 뒤 평균 내고 다시 정규화하면 그 문제가 없다.

```yaml
eval:
  prototype:
    cosine_spherical_mean: true    # mean / ss / reference 에 적용
```

켜면 **cosine 조합에 대해 `_spherical` 태그가 붙은 결과가 따로 나온다** —
기존 결과를 덮어쓰지 않으므로 나란히 비교할 수 있다. 다른 거리(euclidean 등)에는
적용되지 않는다(방향만 보는 거리가 아니므로).

## 대표 이미지 prototype (`reference`)

제조사 카탈로그처럼 그 시스템을 가장 잘 보여주는 사진을 prototype으로 쓴다.
학습 데이터 평균 대신 **정답 이미지 자체**를 기준점으로 삼는 셈이다.

```yaml
eval:
  prototype:
    methods: [mean, reference]        # 다른 방법과 나란히 비교
    reference_dir: data/reference     # <root>/<클래스>/*.jpg
    reference_reduce: mean            # mean(여러 장 평균) | each(각 장을 별개 prototype)
    reference_extensions: [.jpg]      # null이면 png도
    reference_audit: true             # 누출 검사 (아래)
```

- 폴더명이 데이터셋 라벨과 조금 달라도 맞춰준다(대소문자·공백·앞 일련번호). 겹치면 중단
- **대표 이미지가 없는 클래스는 갤러리 평균으로 대체**하고 그 목록을 로그에 남긴다 —
  조용히 빠지면 그 클래스만 다른 기준으로 판정되는 걸 눈치채기 어렵다
- 대표 이미지도 query와 **똑같은 전처리·모델·변환**을 거쳐 임베딩한다
- `reference_reduce: each`면 대표 사진 한 장 한 장이 각각 prototype이 되고
  query는 그중 가장 가까운 것으로 판정된다 (각도가 다른 사진이 여러 장일 때)

### 누출 검사

대표 이미지가 학습·평가 데이터와 **바이트 단위로 같은지** sha256으로 대조한다.
같은 파일이 있으면 prototype이 정답을 미리 본 셈이므로 기본적으로 실행을 중단한다
(`reference_audit_strict: false`면 경고만). 결과는 `reference_audit.csv`,
겹친 항목은 `reference_collisions.csv`에 남는다.

t-SNE에서도 **대표 이미지에서 온 prototype은 ★, 갤러리 평균으로 대체된 것은 ✚**로
구분해 그린다 — 카탈로그 사진이 실제 X-ray 임베딩과 얼마나 떨어져 있는지 눈으로 확인할 수 있다.

## 출력

```
outputs/metric/<protocol>_<backbone>_<loss>/
  train/checkpoints/best.pt      학습된 임베딩
  train/metrics/history.csv
  eval/summary.csv                       prototype × 거리 조합별 지표 (test 기준)
  eval/metrics_<split>_<proto>_<dist>.json
  eval/predictions_<split>_<proto>_<dist>.csv
  eval/scores_<split>_<proto>_<dist>.csv [N,C] 거리 행렬 → Stage 5 Fuse 입력
  eval/confusion_<proto>_<dist>.png      (test만)
  eval/visualization/                   t-SNE/UMAP 임베딩 + score 분포
outputs/metric/grid_summary_<protocol>.csv    전체 조합 한 장 요약
```

`predictions_*.csv`의 `score`(작을수록 가까움)와 `rejected_unknown`이
Stage 5 Fuse의 open-set gate / threshold 입력이 된다.

**valid와 test 둘 다 평가한다** — Fuse가 브랜치 가중치와 임계값을 valid에서 정하기 때문이다.
지표·시각화·confusion matrix는 test 기준으로만 남기고, valid는 fuse 입력용 점수와 지표까지만 낸다.

## 하나만 돌리기

그리드는 의무가 아니다. yaml의 리스트를 한 개로 줄이거나(스칼라도 허용),
yaml을 고치지 않고 `SET=` 으로 이번 실행만 좁힐 수 있다.

```bash
# 학습 1개 조합만
SET="grid.backbone=dinov2 grid.embedding_loss=supcon" bash run_metric_grid.sh

# 평가 축까지 하나씩만 (prototype x distance 재평가 횟수가 준다)
SET="grid.backbone=dinov2 grid.embedding_loss=supcon eval.prototype.methods=mean eval.distances=cosine" bash run_metric_grid.sh
```

`SET`은 yaml 항목을 점으로 짚어 덮어쓴다 (`train.epochs=1`,
`eval.visualize.enabled=false` 등). 적용된 값은 실행 로그에
`[config] override ...` 로 남는다.

## `variant` 열 — 조합 이름

`eval/summary.csv` 의 `variant` 열이 그 행의 산출 파일 이름과 정확히 같다
(`scores_test_<variant>.csv`, `confusion_<variant>.png` …).

`eval.prototype.cosine_spherical_mean: true` 처럼 접미사가 붙는 옵션이 있어서
`prototype` + `distance` 로 이름을 조립하면 어긋난다. Stage 5 fuse가 최고 조합을
자동으로 고를 때도 이 열을 쓴다.

```
prototype       distance  variant                overall_macro_f1
mean            cosine    mean_cosine            0.378
mean_spherical  cosine    mean_cosine_spherical  0.352   ← 구면 평균은 대체가 아니라 추가
```

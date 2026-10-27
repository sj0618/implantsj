# Stage 5. Fuse

코드는 `src/fuse/`, 설정은 `configs/06_fuse.yaml`.
노션 구조의 **6. Fuse** 브랜치.

세 브랜치가 낸 `[N, C]` 점수를 합쳐 최종 분류한다 → **System ID 또는 `__unknown__`**.

## 브랜치 결과 지정 — `configs/06_fuse.yaml`

`run`에 그 브랜치의 결과 폴더를 적으면 valid·test 점수 파일을 알아서 찾는다.

```yaml
branches:
  metric:
    higher_is_better: false        # 거리라 방향 반전
    run: outputs/metric/closed27_open15_dinov2_supcon
    variant: null                  # 비우면 summary.csv에서 최고 조합 자동 선택
  classifier:
    higher_is_better: true
    run: outputs/classifier/band_vit/closed27_open15_a8_l0.8_b6
  attributes:
    higher_is_better: true
    run: outputs/attributes/closed27_open15_cnn_separate
```

```bash
bash run_fuse.sh
```

`run`은 **결과 폴더 · `best.pt` · scores CSV** 아무거나 받는다 —
`best.pt`를 주면 위로 거슬러 올라가 같은 run의 점수 파일을 찾는다.

| 브랜치 | run에 적는 것 | 찾아내는 파일 |
|---|---|---|
| metric | `outputs/metric/<tag>` | `eval/scores_{valid,test}_<proto>_<dist>.csv` |
| classifier | `outputs/classifier/.../<tag>` | `evaluation/{valid,test}/scores.csv` |
| attributes | `outputs/attributes/<tag>` | `scores_{valid,test}.csv` |

metric은 prototype×distance 조합이 여러 개라 `variant`로 고르고,
비워두면 그 run의 `eval/summary.csv`에서 `overall_macro_f1`이 가장 높은 조합을 자동으로 쓴다.

특정 파일을 직접 지정하려면 `run` 대신 `scores`를 쓴다 (`run`이 있으면 `run`이 우선):

```yaml
  metric:
    scores:
      valid: outputs/metric/<tag>/eval/scores_valid_mean_cosine.csv
      test:  outputs/metric/<tag>/eval/scores_test_mean_cosine.csv
```

### 한 번만 다르게 돌려보고 싶을 때

yaml을 고치지 않고 그때만 덮어쓰려면 인자로 넘길 수 있다 (yaml보다 우선).

```bash
bash run_fuse.sh --metric outputs/metric/<다른_tag> --metric-variant ss_cosine
```

`--metric` / `--classifier` / `--attributes` / `--metric-variant` /
`--labels-valid` / `--labels-test` 를 지원한다.

> fuse가 읽는 것은 체크포인트가 아니라 각 브랜치가 평가하며 남긴 **scores CSV**다.
> fuse가 모델을 직접 로드하려면 세 브랜치의 구조와 전처리(band crop, SAM2 런타임 등)를
> 전부 알아야 해서 패키지 독립성이 깨진다. 그래서 브랜치가 점수를 남기고 fuse는 그것만 읽는다.
> **각 브랜치를 먼저 평가해 두어야 한다** — 세 브랜치 모두 학습 스크립트가 평가까지 이어서 하므로,
> 정상 순서로 돌렸다면 이미 만들어져 있다.

## 입력 — 각 브랜치의 scores CSV

세 브랜치 모두 `path` + 클래스별 점수 컬럼으로 된 CSV를 낸다.

| 브랜치 | 파일 | 점수 방향 |
|---|---|---|
| metric | `outputs/metric/<tag>/eval/scores_<split>_<proto>_<dist>.csv` | **거리** (작을수록 가까움) |
| classifier | `outputs/classifier/.../evaluation/<split>/scores.csv` | softmax 확률 (클수록 좋음) |
| attributes | `outputs/attributes/<tag>/scores_<split>.csv` | 매칭 점수 (클수록 좋음) |

방향이 다르므로 yaml의 `higher_is_better`로 알려주면 코드가 부호를 맞춘다.
브랜치마다 이미지 순서·클래스 집합이 달라도 공통 축으로 자동 정렬한다.

## 처리 순서

```
브랜치 점수 → 방향 통일 → 행별 정규화 → 가중합
                                          ↓
                              open-set gate (metric이 open 판정 → metric 단독)
                                          ↓
                              threshold (기준 미만 → __unknown__)
                                          ↓
                                 top1 / top2 평가
```

## 옵션 4가지

### 1. 스케일 통일 — `normalize.method`

브랜치마다 점수 단위가 다르므로(거리 vs 확률) 결합 전에 행별로 맞춘다.

| 값 | 내용 |
|---|---|
| `softmax` | 확률로 변환 (기본). `temperature`가 작을수록 1등에 몰아준다 |
| `minmax` | 행별 0~1 |
| `rank` | 행별 순위 0~1 (분포 모양에 둔감) |
| `none` | 원값 그대로 |

### 2. 브랜치 가중치 — `weights.mode`

| 값 | 내용 |
|---|---|
| `learned` | **valid에서 자동 학습** (기본). 학습된 값은 `summary.json`에 저장 |
| `manual` | `weights.manual`에 적은 값을 그대로 사용 |

가중치 학습은 **valid만** 쓰고 test는 건드리지 않는다.

### 3. open-set gate — `openset` (노션 규칙)

> metric learning 결과 open-set 클래스로 분류되었다면
> classifier·attributes 모델은 최종결과에 미반영

`open_manifest`의 라벨을 open 클래스로 보고, **metric의 최근접이 open 클래스인 샘플은
다른 브랜치를 무시하고 metric 점수만으로 판정**한다.
`open_manifest`와 `open_classes`를 모두 비우면 gate가 꺼진다.

합성 데이터 검증에서 이 gate가 open 클래스 정확도를 0.828 → 0.968로 끌어올렸다
(전체 0.932 → 0.973).

### 4. threshold — `threshold.mode`

몇 점 이상이어야 그 클래스로 확정할지. 기준 미만이면 `__unknown__`(= open-set reject).

| 값 | 내용 |
|---|---|
| `auto` | **valid에서 자동 결정** (기본). 정답/오답 점수 분포가 가장 잘 갈리는 지점(Youden's J) |
| `manual` | `threshold.value`에 적은 값 |
| `none` | 임계 없음 — 항상 최고 점수 클래스로 판정, reject 안 함 |

`max_reject_rate`는 auto가 지나치게 많이 걸러내지 않도록 하는 상한이다
(기본 0.2 = 최대 20%). 브랜치 성능이 낮으면 auto가 공격적인 값을 고를 수 있어 둔 안전장치.
`null`이면 상한 없이 최적점을 그대로 쓴다.

### 5. 평가 방식 — `evaluation.top_k`

| 값 | 내용 |
|---|---|
| `[1]` | **top1** (기본) — 최종 1개가 정답이어야 맞음 |
| `[1, 2]` | top1과 top2를 함께 — top2는 점수 상위 2개 중 하나라도 정답이면 맞음 |

reject된 샘플은 top1·top2 모두 오답으로 센다(판정을 포기한 것이므로).
`accuracy_on_decided`로 **판정된 것만의 정확도**를 따로 보고한다.

## open-set 거부 성능 (옵션)

threshold와 무관하게, **최종 점수만으로 open 클래스를 걸러낼 수 있는지** 잰다.

```yaml
reject_metrics:
  enabled: true
  scores: [min_dist, margin, dist_ent]
```

`summary.json`의 `results.rejection`에 거부 점수별 AUROC / FPR@95가 들어간다.
threshold를 어디 두든 무관한 지표라 **설정끼리 비교**하기 좋다.
open 클래스가 지정돼 있어야 계산된다.

## 시각화

`visualize.enabled: true`(기본)면 `outputs/fuse/visualization/`에 자동 저장된다.

| 파일 | 내용 |
|---|---|
| `branch_weights.png` | 브랜치 가중치 (균등선과 함께 — 무엇에 의존하기로 했는지) |
| `branch_comparison.png` | 브랜치 단독 vs 결합 vs gate 적용 후 정확도 — **fuse의 실익** |
| `openset_breakdown.png` | closed/open 클래스별 정확도, **gate 전후 비교** |
| `score_distribution.png` | 최종 점수 분포 + threshold 선 (정답 vs 오답) |
| `final_confusion.png` | 최종 클래스 혼동행렬 (unknown 제외) |

```yaml
visualize:
  enabled: true
  which: [weights, comparison, openset, score, confusion]   # 원하는 것만 남겨도 됨
```

`branch_comparison`과 `openset_breakdown`이 이 스테이지의 핵심 질문에 답한다 —
**합치는 게 단독 브랜치보다 나은가**, 그리고 **open-set gate가 실제로 효과가 있는가**.

그림 안 글자는 전부 영어다. 개별 그림이 실패해도 나머지와 평가는 계속된다.

## 출력

```
outputs/fuse/
  predictions.csv   샘플별 최종 판정
  scores.csv        [N, C] 최종 점수
  summary.json      가중치 / 임계값 / 설정 / 지표 전체
  summary.csv       top1·top2 지표 요약
  visualization/    가중치·브랜치 비교·gate 효과·점수 분포·혼동행렬
```

`predictions.csv` 컬럼: `pred_label`(reject면 `__unknown__`), `score`,
`rejected_unknown`, `openset_gated`(metric 단독 판정 여부),
`correct_top1`, `top2_label`/`top2_score`, `correct_top2`.

보고 지표: `accuracy`(reject를 오답으로 포함), `accuracy_on_decided`(판정된 것만),
`reject_rate`, `reject_precision`(버린 것 중 실제로 틀렸을 비율 — reject가 제 역할을 했는지).

## 하나만 돌리기

그리드는 의무가 아니다. yaml의 리스트를 한 개로 줄이거나(스칼라도 허용),
yaml을 고치지 않고 `SET=` 으로 이번 실행만 좁힐 수 있다.

```bash
SET="evaluation.top_k=[1,2] threshold.mode=none" bash run_fuse.sh
```

`SET`은 yaml 항목을 점으로 짚어 덮어쓴다 (`train.epochs=1`,
`eval.visualize.enabled=false` 등). 적용된 값은 실행 로그에
`[config] override ...` 로 남는다.

## 브랜치 빼고 돌리기

`branches.<이름>.enabled: false` 로 그 브랜치를 빼고 결합한다.
블록을 통째로 지워도 되지만, `extends` 상속에서는 부모의 항목을 지울 수 없고
**"그 브랜치가 실제로 기여하는가"** 자체가 실험 축이라 플래그를 둔다.

```bash
SET="branches.attributes.enabled=false" bash run_fuse.sh
```

## ⚠ 브랜치 커버리지가 다르면 정확도를 비교하지 말 것

브랜치마다 다루는 이미지가 다르면 **교집합만** 평가된다.
예를 들어 attributes가 spec table에 붙는 클래스만 다루면, 그 부분집합에서만
정확도가 나온다. 이때 수치는 더 쉽거나 다른 표본에서 잰 값이라
브랜치 구성이 서로 다른 실행끼리 비교하면 안 된다.

실행 로그가 공통 장수와 함께 경고를 남긴다:

```
[test] 정렬 결과: 공통 62장 × 13클래스
[test] WARNING 공통 62장 < 최대 127장 — 교집합만 평가됩니다
       (더 넓은 브랜치: {'metric': 127, 'classifier': 119})
```

비교하려면 모든 브랜치가 같은 이미지를 덮게 만들거나,
같은 브랜치 구성끼리만 비교한다.

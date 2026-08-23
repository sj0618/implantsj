# finalSJ: 유연한 임플란트 실험 구조

`finalSJ`는 임플란트 영상의 분류 학습, 임베딩 추출, prototype 평가와 시각화를
각각 독립된 작업으로 실행하는 실험 프로젝트다. 상위 저장소의 `src/`, `pipelines/`,
`tests/`를 import하지 않으며 이 폴더만 복사해 사용할 수 있다.

한 명령은 한 조건의 작업만 수행한다. 모델, checkpoint,
seed, 데이터 view, prototype, 거리 함수를 명시적으로 전달하므로 실패한 단계만
다시 실행하거나 기존 산출물을 다른 실험에서 재사용할 수 있다.

## 1. 실험 목적

기본 실험은 다음 질문을 분리해 확인한다.

1. CE, SupCon, Triplet 학습이 임베딩 표현을 어떻게 바꾸는가?
2. Closed class만 학습한 모델과 전체 class를 학습한 모델은 어떻게 다른가?
3. GT, class mean, few-shot support prototype 중 어느 방식이 안정적인가?
4. Euclidean과 cosine 거리에서 분류 결과가 어떻게 달라지는가?
5. Closed query와 open query가 기존 label space에 어떻게 mapping되는가?
6. 같은 embedding에서 classifier, prototype, linear probe 결과가 일치하는가?

각 질문은 한 번에 여러 조건을 자동 실행하지 않는다. 비교할 조건마다 명령과
출력 디렉터리를 따로 만든 뒤 마지막에 두 결과를 비교한다.

## 2. 전체 구조

```text
finalSJ/
├── implantsj/                   # Python 구현
│   ├── data.py                  # dataset/loader
│   ├── models.py                # model·checkpoint adapter
│   ├── losses.py                # CE·SupCon·Triplet
│   ├── training.py              # 단일 학습 실행
│   ├── evaluation.py            # embedding·metric 계산
│   ├── visualization.py         # t-SNE 시각화
│   └── workflow/
│       ├── individual.py        # 개별 작업 CLI
│       ├── materialize.py       # 데이터 materialization
│       └── schema.py            # manifest 검증
├── manifests/
│   ├── data/                    # 데이터 위치와 label space
│   ├── splits/                  # split과 view 정의
│   ├── experiments/             # 학습 조건
│   └── models/                  # 추론 모델 구조
├── scripts/                     # 작업별 shell 진입점
├── tests/                       # 단위·통합 테스트
├── outputs/                     # 실행 시 생성, 버전 관리 제외
├── WORKFLOW.md                  # 명령별 상세 사용법
└── 123.md                       # 설계 원칙
```

## 3. 단계 연결

단계는 Python 객체가 아니라 파일 산출물로 연결된다.

```text
Dataset manifest
  └─ build-split
      └─ records.csv + view CSV + metadata.json
          └─ train
              └─ best/last checkpoint + training metadata
                  └─ extract-embeddings
                      └─ embeddings.npz + embeddings.json
                          ├─ build-prototype
                          │   └─ prototypes.npz + metadata.json
                          │       └─ evaluate-prototype
                          │           └─ metrics + predictions
                          ├─ visualize-tsne
                          ├─ diagnose-geometry
                          └─ linear-probe
```

모든 결과 디렉터리는 새 경로여야 한다. 정상 종료 시 `run.json`이 마지막으로
생성되며 다음 내용을 기록한다.

- `status: completed`
- 실제 실행 명령과 인자
- 완료 시각
- workflow 코드 SHA-256
- 산출물 상대 경로와 파일 크기

`run.json`이 없으면 중단되었거나 완료가 확인되지 않은 결과로 본다. JSON metadata는
임시 파일에 먼저 기록한 뒤 원자적으로 교체한다.

## 4. 설치

```bash
cd finalSJ
python3 -m pip install -r workflow-requirements.txt
```

폐쇄망에서는 의존성 wheel과 초기 checkpoint를 미리 준비한다. 로컬 checkpoint를
사용할 때 model manifest의 `pretrained`를 `false`로 설정하면 외부 다운로드를
시도하지 않는다.

## 5. 데이터 구조

기본 `manifests/data/implant_v1.yaml`은 `${DATA_ROOT}` 아래 세 역할을 읽는다.

```text
${DATA_ROOT}/
├── known/                       # 학습·validation·closed test 이미지
│   └── <label>/*.png
├── prototype/                   # GT prototype 이미지
│   └── <label>/*.png
└── unknown/                     # open/unseen query 이미지
    └── <label>/*.png
```

```bash
export DATA_ROOT=/absolute/path/to/implant-data
bash scripts/validate-data.sh \
  --dataset-manifest manifests/data/implant_v1.yaml
```

라벨은 폴더 이름에서 결정된다. alias가 필요한 경우 dataset manifest에 정확한
mapping을 명시한다. 잘못된 경로, 중복 샘플, 불완전한 label space를 암묵적으로
무시하지 않는다.

## 6. 기본 split과 view

`manifests/splits/closed27_open15_seed42.yaml`은 seed 42의 stratified hash split을
사용한다.

| View | 역할 |
|---|---|
| `train` | 학습 |
| `validation` | checkpoint 선택과 운영점 결정 |
| `closed_test` | 학습 label 범위의 최종 평가 |
| `gt_closed` | GT prototype 생성 |
| `open_test` | unseen/open mapping 분석 |
| `all_train_test` | 통합 embedding 시각화 |

```bash
bash scripts/build-split.sh \
  --dataset-manifest manifests/data/implant_v1.yaml \
  --split-manifest manifests/splits/closed27_open15_seed42.yaml \
  --output outputs/splits/seed42
```

원본 데이터와 split manifest의 hash는 결과 metadata에 기록된다. 다른 seed는 기존
결과를 수정하지 말고 별도 split manifest와 출력 경로로 실행한다.

## 7. 학습 실험

기본 예시는 `vit_base_patch16_224`, closed scope, CE, AdamW, cosine scheduler다.

```bash
bash scripts/train.sh \
  --dataset-manifest manifests/data/implant_v1.yaml \
  --split-dir outputs/splits/seed42 \
  --experiment-manifest manifests/experiments/vit_closedft.yaml \
  --init-checkpoint checkpoints/vit_base_patch16_224.pt \
  --seed 42 \
  --output outputs/train/vit_closedft_ce_seed42
```

학습 조건은 experiment manifest에서 독립적으로 바꾼다.

| 축 | 지원 값 또는 예시 |
|---|---|
| scope | `closed`, `all` |
| objective | `ce`, `supcon`, `triplet` |
| optimizer | `adamw`, `sgd` |
| scheduler | `cosine`, `step`, `none` |
| model | `timm` architecture와 adapter 설정 |
| seed | CLI의 단일 정수 |

권장 모델 비교 단위는 Base, ClosedFT, ClosedFT-SupCon, FullFT,
FullFT-SupCon이다. 각 모델과 seed는 명령을 별도로 호출한다. validation으로 선택된
`best` checkpoint와 마지막 epoch의 `last` checkpoint를 구분하고 test 결과로
checkpoint를 선택하지 않는다.

## 8. 임베딩 추출

checkpoint와 데이터 view 하나를 명시한다.

```bash
bash scripts/extract-embeddings.sh \
  --dataset-manifest manifests/data/implant_v1.yaml \
  --split-dir outputs/splits/seed42 \
  --model-manifest manifests/models/vit_closedft.yaml \
  --checkpoint outputs/train/vit_closedft_ce_seed42/checkpoints/best.pt \
  --view closed_test \
  --embedding-output encoder \
  --output outputs/embeddings/vit_closedft/closed_test
```

prototype을 만들려면 같은 checkpoint로 `gt_closed` 또는 `train` view를 별도
추출한다. `encoder`와 `projection`도 서로 다른 출력 디렉터리를 사용한다.

## 9. Prototype 생성

### GT prototype

```bash
bash scripts/build-prototype.sh \
  --embedding-dir outputs/embeddings/vit_closedft/gt_closed \
  --prototype-type gt \
  --distance-geometry euclidean \
  --output outputs/prototypes/vit_closedft/gt_euclidean
```

### Few-shot support prototype

```bash
bash scripts/build-prototype.sh \
  --embedding-dir outputs/embeddings/vit_closedft/train \
  --prototype-type ss \
  --samples-per-label 5 \
  --support-seed 42 \
  --distance-geometry cosine \
  --output outputs/prototypes/vit_closedft/ss5_seed42_cosine
```

지원 방식은 `gt`, `mean`, `ss`다. cosine geometry는 prototype 생성 전에 embedding을
정규화하며, 평가 시 다른 geometry를 요청하면 오류로 처리한다.

## 10. Prototype 평가

Closed query를 27개 closed candidate에 평가하는 예시다.

```bash
bash scripts/evaluate-prototype.sh \
  --query-embeddings outputs/embeddings/vit_closedft/closed_test \
  --prototype-dir outputs/prototypes/vit_closedft/gt_euclidean \
  --distance euclidean \
  --query-group closed \
  --candidate-space closed27 \
  --output outputs/evaluations/vit_closedft_gt_l2_closed27
```

평가 결과:

- `metrics.json`: accuracy, macro-F1, weighted-F1, 비교 signature
- `predictions.csv`: 이미지별 정답, 예측, 거리, top-2 margin
- `per_class_metrics.csv`: class별 precision, recall, F1, support
- `confusion_matrix.csv`: candidate label 기준 confusion matrix
- `metadata.json`, `run.json`: 입력 hash와 실행 정보

Open query는 `--query-group open`으로 실행한다. Open label이 candidate space에 없는
설정은 일반 closed-set accuracy로 해석하지 않고 mapping 또는 open-set 분석 조건으로
분리해야 한다.

## 11. 결과 비교

두 평가 결과의 입력 조건 signature가 같을 때만 비교할 수 있다.

```bash
bash scripts/compare-results.sh \
  --left outputs/evaluations/base_gt_l2_closed27 \
  --right outputs/evaluations/vit_closedft_gt_l2_closed27 \
  --comparison-id experiment_1a \
  --comparison-label Base_vs_ClosedFT \
  --output outputs/comparisons/experiment_1a
```

비교 명령은 누락된 평가를 자동 생성하거나 다른 결과 디렉터리를 검색하지 않는다.

## 12. 시각화와 진단

```bash
bash scripts/visualize-tsne.sh \
  --embeddings outputs/embeddings/vit_closedft/all_train_test \
  --geometry cosine \
  --seed 42 \
  --output outputs/visualizations/vit_closedft_tsne_cosine
```

추가 단일 작업:

```bash
bash scripts/diagnose-geometry.sh --help
bash scripts/linear-probe.sh --help
```

`t-SNE` seed는 학습 seed와 별도다. 최대 샘플 수를 제한할 경우 path와 seed로
결정적인 subset을 선택한다.

## 13. 권장 출력 이름

```text
outputs/
├── splits/<split-id>/
├── train/<model>-<objective>-seed<seed>/
├── embeddings/<model>/<view>-<embedding-output>/
├── prototypes/<model>/<prototype>-<geometry>/
├── evaluations/<model>-<prototype>-<distance>-<query>-<candidate>/
├── visualizations/<model>-<view>-<geometry>-seed<seed>/
├── diagnostics/<diagnosis-id>/
└── comparisons/<comparison-id>/
```

폴더 이름은 사람이 읽기 위한 식별자다. 동일 실험 여부는 이름이 아니라 metadata의
manifest, checkpoint, embedding hash와 실행 인자로 판단한다.

## 14. 확장 규칙

새 작업은 다음 조건을 만족할 때 별도 명령으로 추가한다.

1. 입력과 출력 파일 계약이 명확하다.
2. 독립적으로 재실행할 가치가 있다.
3. 모델이나 seed를 내부에서 자동 순회하지 않는다.
4. 기존 도메인 함수를 재사용하고 알고리즘을 workflow에 복제하지 않는다.
5. 정상 입력, 잘못된 입력, 기존 output 거부 테스트를 함께 추가한다.

반복 실험은 외부 shell script나 scheduler가 개별 명령을 호출하도록 한다. 개별
workflow에는 suite, 자동 checkpoint 검색, Cartesian product, output 덮어쓰기를
추가하지 않는다.

## 15. 검증

```bash
python3 -m pytest -q tests

for script in scripts/*.sh; do
  bash "$script" --help >/dev/null
done
```

현재 테스트는 manifest 검증, split, 단일 학습 호출, embedding 추출, prototype,
평가, 비교, t-SNE, geometry 진단, linear probe와 결과 완료 metadata를 확인한다.
명령별 세부 인자는 [WORKFLOW.md](WORKFLOW.md), 설계 근거는 [123.md](123.md)를
참조한다.

## 16. Metric learning 진단 프로토콜

canonical 연구 목적은 metric learning이 기대대로 동작하지 않는 원인을 다음 축으로
분리하는 것이다.

- 모델: Base, ClosedFT, ClosedFT-SupCon, FullFT, FullFT-SupCon
- prototype: GT, Mean, SS
- 거리: Euclidean, Cosine
- query: validation, test
- 결과 slice: all, closed, open

SupCon 모델은 pure supervised contrastive loss만 사용한다. CE+SupCon 결합이나
CE 이후 SupCon 단계는 이 프로토콜에 포함하지 않는다. 모든 학습 모델은 동일한
Base checkpoint, budget, seed를 사용하며 canonical embedding은 fixed-budget
`last.pt`의 encoder 출력이다.

canonical 예측은 항상 full42 candidate space에서 한 번 수행한 뒤 all/closed/open
세 행으로 나눈다. closed27 평가는 진단용일 뿐 canonical 비교에 포함하지 않는다.

## 17. 데이터와 실행 순서

seed 42/43/44는 각각 다음 CSV와 manifest에 고정된다.

```text
metric42_samples_seed42.csv → implant_metric42_seed42_v1.yaml
metric42_samples_seed43.csv → implant_metric42_seed43_v1.yaml
metric42_samples_seed44.csv → implant_metric42_seed44_v1.yaml
metric42_gt.csv             → 세 seed가 공유하는 독립 GT
```

각 CSV는 `image_uri,label,sample_id,group_id`를 가지며 sample CSV에는 `split`도
필수다. split은 `train`, `validation`, `test` 중 하나다. 세 assignment는 동일한
sample universe를 사용하되 assignment hash가 서로 달라야 한다. train, validation,
test, GT 사이 sample ID, 이미지 content hash, group ID가 하나라도 겹치면 canonical
split은 실패한다.

`metric_learning_v1.yaml`의 환경 변수에는 실제 파일 hash를 넣는다. 값이 없거나
64자리 SHA-256이 아니면 protocol 로딩부터 실패한다.

```bash
export BASE_CHECKPOINT=/workspace/checkpoints/vit_base_patch16_224.pt
export BASE_CHECKPOINT_SHA256=<actual-sha256>
export SEED42_ASSIGNMENT_CSV_SHA256=<actual-sha256>
export SEED42_ASSIGNMENT_SHA256=<normalized-sample-id-split-sha256>
export SEED43_ASSIGNMENT_CSV_SHA256=<actual-sha256>
export SEED43_ASSIGNMENT_SHA256=<normalized-sample-id-split-sha256>
export SEED44_ASSIGNMENT_CSV_SHA256=<actual-sha256>
export SEED44_ASSIGNMENT_SHA256=<normalized-sample-id-split-sha256>
```

실행 순서는 반드시 다음과 같다.

```text
split
→ 네 학습 조건을 각각 실행
→ Base 포함 다섯 모델의 train/validation/test/GT encoder embedding
→ GT/Mean/SS prototype
→ validation 평가 30개
→ freeze-validation
→ 동결된 test 평가 30개
→ scalar comparison 72개
→ audit-study
→ seed 42/43/44 결과 summarize-uncertainty
```

학습 중에는 test를 읽지 않는다. train/validation curve로 fixed budget 학습을
기록한 뒤 validation 결과 30개와 앞으로 생성할 test 경로 30개를
`freeze-validation`으로 동결한다. test 출력은 freeze에 정확히 등록된 경로에서
한 번만 만들 수 있다.

## 18. Freeze, audit, uncertainty

```bash
bash scripts/freeze-validation.sh \
  --protocol-manifest manifests/protocols/metric_learning_v1.yaml \
  --seed 42 \
  --pretest-ledger ledgers/seed42-pretest.csv \
  --output outputs/seed42/validation-freeze

bash scripts/audit-study.sh \
  --protocol-manifest manifests/protocols/metric_learning_v1.yaml \
  --seed 42 \
  --validation-freeze outputs/seed42/validation-freeze \
  --evaluation-ledger ledgers/seed42-evaluations.csv \
  --comparison-ledger ledgers/seed42-comparisons.csv \
  --output outputs/seed42/final-audit

bash scripts/summarize-uncertainty.sh \
  --protocol-manifest manifests/protocols/metric_learning_v1.yaml \
  --audit-dir outputs/seed42/final-audit \
  --audit-dir outputs/seed43/final-audit \
  --audit-dir outputs/seed44/final-audit \
  --output outputs/metric-learning-uncertainty
```

seed 하나당 정확한 canonical 총계는 evaluation 60개/metric 180행,
comparison 72개/delta 216행이다. uncertainty 결과는 세 paired seed의 평균,
표준편차, 표준오차와 자유도 2의 양측 Student-t 95% 구간을 기록한다. 명령은
누락된 실험을 검색하거나 실행하지 않으며, ledger에 명시된 경로와 hash만 검증한다.
`metric-learning-uncertainty`에는 검증한 세 seed의 원본 행
(`per_seed_results.csv`, `per_seed_comparison_results.csv`)과
`metric_uncertainty.csv`, `paired_delta_uncertainty.csv`, 입력 artifact hash가 함께
저장된다.

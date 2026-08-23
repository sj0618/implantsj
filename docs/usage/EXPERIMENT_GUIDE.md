# 실험 가이드 (폐쇄망)

처음부터 끝까지 실제 실험 순서대로 정리했다.

폐쇄망 *안에서는* 복붙이 된다. 안 되는 것은 망 경계를 넘는 복붙이다. 그래서
이 문서는 Docker 이미지에 함께 들어간다 — 폐쇄망 안에서 이 파일을 열어
명령을 복사해 쓰면 된다. 컨테이너 안 경로는 다음과 같다.

```bash
less /workspace/implantsj/docs/usage/EXPERIMENT_GUIDE.md
```

---

## 0. 구조 한눈에

```
run.sh            ← 사람이 쓰는 유일한 진입점 (대화형 메뉴)
  └ tools/run.py  ← 명령 이름 -> python 모듈 라우팅
      └ pipelines/, src/   ← 실제 학습·평가 코드 (건드릴 일 없음)

exp.conf              ← 경로·하이퍼파라미터를 한 번만 적어두는 곳
configs/backbones.txt ← 실험할 모델 목록 (여기만 고치면 메뉴에 반영됨)
```

경로 기본값 (`exp.conf`):

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `DATA_ROOT` | `/workspace/data` | 데이터가 mount 되는 곳 |
| `OUTPUT_ROOT` | `/workspace/outputs` | 결과가 쌓이는 곳 |
| `DEVICE` | `cuda` | 연산 장치 |
| `EPOCHS` / `BATCH_SIZE` / `IMAGE_SIZE` | 80 / 32 / 224 | 학습 기본값 |
| `LR` / `SEED` / `NUM_WORKERS` | 5e-5 / 42 / 8 | 학습 기본값 |

---

## 1. 최초 1회: 컨테이너 기동

망이 열려 있을 때(데이터 없음) 이미지를 만들어 반입하고, 망이 닫힌 뒤
(데이터 연결됨) 컨테이너를 올린다.

```bash
# [인터넷 O / 데이터 X] 외부에서
bash docker/build_and_export.sh          # 빌드 + implantsj_image.tar 생성

# [폐쇄망] 반입 후
docker load -i implantsj_image.tar
bash docker/run_container.sh             # /workspace/data, /workspace/outputs 를 mount
```

컨테이너 안에서 맨 먼저 자가진단을 돌린다. 5단계가 전부 통과하면 코드와
환경은 정상이다.

```bash
bash docker/smoke_test.sh
```

---

## 2. 실험 한 번의 전체 흐름

```bash
./run.sh
```

메뉴가 뜬다. 번호만 누르면 된다.

```
 1) 환경 체크 (GPU/패키지 확인)
 --- 데이터 ---
 2) 원본 -> 학습용 데이터셋 변환   (폐쇄망에서 맨 처음 1회)
 3) 데이터셋 선택
 4) 모델 선택
 5) manifest + split 만들기
 --- 학습/평가 ---
 6) 학습 · supervised 분류
 7) 학습 · SupCon metric
 8) 평가 · 분류 (checkpoint)
 9) 평가 · Mahalanobis (unseen)
 --- YOLO ---
10) YOLO · 검출 학습
11) YOLO · crop 전처리
12) 설정 보기/임시 변경
```

### 2-1. 환경 체크 (`1`)

패키지와 GPU 인식 여부를 찍는다. `cuda_available: True` 와 `ENV_OK` 를 확인한다.
여기서 실패하면 그 아래는 볼 필요 없다.

### 2-2. 원본 -> 학습용 변환 (`2`) — 맨 처음 딱 한 번

폐쇄망 원본은 병원/촬영건 폴더에 이미지(TS1/VS1)와 JSON 라벨(TL1/VL1)이
분리된 구조다. 그대로는 학습에 못 쓴다. 이 메뉴가 JSON 의 `모델명` 을 읽어
class 폴더 구조로 재배치한다.

물어보는 것은 셋뿐이고 전부 엔터로 기본값을 쓸 수 있다.

1. 원본 루트 (TL1/TS1/VL1/VS1 의 상위 폴더) — 기본 `/workspace/data/raw`
2. 만들 데이터셋 이름 — 기본 `closed_network`
3. unseen 으로 뺄 희귀 라벨 개수 — 기본 `15` (`0` 이면 전부 학습에 사용)

결과:

```
/workspace/data/closed_network/
├── manifests/closed_network.csv          ← path,label,split 전체 목록
├── summary.json                          ← 라벨 수, 이미지 수, 건너뛴 수
└── imagefolders/
    ├── data_42label/{train,valid,test}/<class>/   ← 전체
    ├── data_metric_seen27/{train,valid,test}/     ← 학습용 (seen)
    └── data_metric_unseen15/test/                 ← unseen 평가 전용
```

라벨 수가 42가 아니어도 된다. 실제 라벨 수를 세어 이름에 반영한다(`data_38label` 등).
파일은 hardlink 로 만든다 — symlink 는 mount 경로가 바뀌면 전부 끊어져
데이터가 통째로 사라진 것처럼 되기 때문이다.

### 2-3. 데이터셋·모델 고르기 (`3`, `4`)

`3` 을 누르면 `DATA_ROOT` 아래에서 `train/`·`test/` 를 가진 폴더를 자동으로
찾아 번호를 매겨 보여준다. 경로를 타이핑할 필요가 없다.

```
 1) 7label
 2) closed_network/imagefolders/data_42label
 3) closed_network/imagefolders/data_metric_seen27
 ...
```

`4` 는 `configs/backbones.txt` 의 모델 목록을 보여준다.

```
 1) vit_base_patch16_224.augreg2_in21k_ft_in1k
 2) vit_small_patch16_224.augreg_in21k_ft_in1k
 3) resnet50.a1_in1k
 4) resnet18.a1_in1k
 5) convnext_tiny.fb_in22k_ft_in1k
 6) efficientnet_b0.ra_in1k
```

**모델을 추가하려면 `configs/backbones.txt` 에 timm 이름 한 줄을 적으면 된다.**
단, 폐쇄망에서는 그 weight 가 이미지에 구워져 있어야 한다. 반입 전에
목록에 넣어야 `docker/prefetch_weights.py` 가 같이 받아둔다.

고른 값은 `OUTPUT_ROOT/.exp_state` 에 저장돼서 다음 실행 때도 유지된다.

### 2-4. manifest + split (`5`)

데이터셋 폴더를 훑어 `path,label,split` CSV 를 만든다. 폴더가 이미
`train/valid/test` 로 나뉘어 있으면 그 split 을 그대로 쓰고, 안 나뉘어 있으면
8:1:1 로 stratified split 을 만든다.

`6`·`7` 이 알아서 이 단계를 먼저 실행하므로 보통 따로 누를 필요는 없다.
데이터 개수만 미리 확인하고 싶을 때 쓴다.

### 2-5. 학습 (`6` supervised / `7` SupCon metric)

- `6` supervised: 일반 분류(cross-entropy). closed-set 성능 baseline.
- `7` SupCon metric: 임베딩 공간 학습. unseen 라벨 분석용.

실행 전에 실제 명령이 화면에 찍히고, 로그가 `OUTPUT_ROOT/logs/` 에 남는다.

결과 폴더 이름은 이렇게 만들어진다:

```
/workspace/outputs/20260823_163228_data_metric_seen27__resnet50.a1_in1k_supcon
                   └ 시각        └ 데이터셋          └ 모델        └ 학습종류
```

### 2-6. 평가 (`8` 분류 / `9` Mahalanobis)

체크포인트도 목록에서 번호로 고른다. `OUTPUT_ROOT` 아래 `.pt` 를 자동으로 찾는다.

- `8`: test split 에 대한 분류 정확도·F1·confusion matrix
- `9`: unseen 라벨이 seen 프로토타입 중 어디로 매핑되는지 Mahalanobis 거리로 분석

---

## 3. 통제변인 실험

핵심은 **한 번에 하나만 바꾸고, 나머지는 `exp.conf` 가 고정해 준다**는 것이다.
`exp.conf` 를 안 건드리는 한 epochs·batch·lr·seed·image_size 는 모든 실행에서
동일하므로, 바꾼 변수 하나만 성능 차이의 원인이 된다.

### 3-1. 모델을 바꿔가며 (데이터 고정)

메뉴에서: `3` 으로 데이터 한 번 고르고 → `4`(모델 변경) → `6` → `4` → `6` … 반복.

한 줄로 돌리려면:

```bash
for m in $(grep -v '^#' configs/backbones.txt | grep .); do
  ./run.sh train data_metric_seen27 "$m"
done
```

### 3-2. 데이터를 바꿔가며 (모델 고정)

```bash
for d in 7label 11label_crop_final_8_1_1 data_metric_seen27; do
  ./run.sh train "$d" resnet50.a1_in1k
done
```

### 3-3. crop 전/후 비교

YOLO 로 crop 한 데이터셋을 따로 만들어 같은 모델로 돌린다.

```bash
# 메뉴 11) 로 crop -> /workspace/data/<name>_cropped 생성
./run.sh train 7label          resnet50.a1_in1k
./run.sh train 7label_cropped  resnet50.a1_in1k
```

### 3-4. seed 를 바꿔가며 (분산 확인)

```bash
for s in 42 43 44; do
  SEED=$s ./run.sh train data_metric_seen27 resnet50.a1_in1k
done
```

환경변수는 `exp.conf` 기본값을 덮어쓴다. `EPOCHS=5 ./run.sh` 처럼
아무 값이나 임시로 바꿀 수 있고, 파일은 그대로 남는다.

### 3-5. 무엇이 기록되는가

**폴더 이름에는 데이터셋·모델·학습종류만 들어간다.** seed·lr·epochs 는
이름에 없다. 대신 각 결과 폴더의 `args.json` 에 전부 기록되고, 아래 집계가
그것을 컬럼으로 펼쳐준다. 즉 **seed sweep 은 폴더 이름으로 구분되지 않고
집계 CSV 의 `arg_seed` 컬럼으로 구분한다.**

---

## 4. 결과 확인

### 4-1. 결과 폴더 하나 안에 뭐가 있나

```
20260823_163228_<데이터셋>__<모델>_supcon/
├── args.json                  ← 이 실행의 모든 설정 (재현용)
├── class_to_idx.json          ← 라벨 -> 인덱스 매핑
├── checkpoints/
│   ├── best.pt                ← valid 기준 최고 성능
│   └── last.pt
├── metrics/
│   ├── history.csv            ← epoch 별 loss/accuracy/f1  ★ 학습 추이
│   ├── best_valid_prototype_metrics.json
│   ├── test_prototype_metrics.json      ★ 최종 성적표
│   └── test_prototype_predictions.csv   ← 샘플별 예측 (오답 분석용)
├── confusion_matrices/
│   ├── test_prototype.csv
│   └── test_prototype.png     ★ 어떤 클래스끼리 헷갈리는지
├── embeddings/
└── logs/
```

빠르게 성적만 보려면:

```bash
cat /workspace/outputs/<결과폴더>/metrics/test_prototype_metrics.json
```

### 4-2. 여러 실행을 한 표로 비교 (통제변인 실험의 핵심)

```bash
python3 tools/run.py summary \
  --runs-root /workspace/outputs \
  --out /workspace/outputs/experiment_summary.csv
```

`OUTPUT_ROOT` 아래 모든 결과 폴더를 훑어 한 줄씩 CSV 로 만든다. 컬럼은
`arg_*`(설정)와 성능 지표가 나란히 붙는다.

| 종류 | 컬럼 예 |
|---|---|
| 통제변인 | `arg_model_name`, `arg_manifest`, `arg_seed`, `arg_lr`, `arg_epochs`, `arg_batch_size`, `arg_image_size`, `arg_loss_name` |
| 성능 | `test_prototype_metrics_accuracy`, `test_prototype_metrics_macro_f1`, `best_valid_prototype_metrics_macro_f1`, `history_best_epoch` |

터미널에서 필요한 컬럼만 골라 보려면:

```bash
python3 -c "
import pandas as pd
d = pd.read_csv('/workspace/outputs/experiment_summary.csv').dropna(subset=['arg_model_name'])
print(d[['arg_model_name','arg_seed','test_prototype_metrics_macro_f1']]
      .sort_values('test_prototype_metrics_macro_f1', ascending=False)
      .to_string(index=False))
"
```

### 4-3. Mahalanobis 결과

`OUTPUT_ROOT/mahalanobis/<데이터셋>_<시각>/` 에 생기고, `protocol.json` 에
checkpoint 와 manifest 의 sha256 이 남는다. 어떤 모델·어떤 데이터로 낸
결과인지 이름이 아니라 해시로 확인할 수 있다.

---

## 5. 문제가 생기면

| 증상 | 확인 |
|---|---|
| 뭐가 잘못됐는지 모르겠다 | `bash docker/smoke_test.sh` — 5단계 중 어디서 깨지는지 알려준다 |
| 방금 실행이 실패했다 | `OUTPUT_ROOT/logs/` 의 최신 `.log` |
| 데이터셋 목록에 안 뜬다 | 그 폴더에 `train/` 또는 `test/` 하위 폴더가 있어야 한다 |
| 학습이 NaN 으로 죽었다 | 의도된 동작이다. 손상된 checkpoint 를 저장하지 않고 멈춘다. `LR` 을 낮추거나 `BATCH_SIZE` 를 올린다 |
| Mahalanobis 가 "test 가 train 과 동일" 이라며 멈췄다 | split 이 잘못 만들어진 것이다. 데이터셋을 다시 만든다 |

---

## 6. 지금 구조에서 아직 불편한 것

- **결과 집계(`summary`)가 메뉴에 없다.** 번호로 못 부르고 위 4-2 의 명령을 써야 한다.
  이 문서가 이미지에 같이 들어가므로 거기서 복사하면 되지만, 다른 단계처럼
  번호 하나로 끝나지는 않는다.
- `summary` 출력이 터미널에서 읽기 힘들 만큼 넓고, `logs/`·`manifests/` 같은
  실행 결과가 아닌 폴더도 빈 줄로 섞여 나온다.
- 여러 조건을 한 번에 돌리는 sweep 이 메뉴에 없다. `for` 문을 직접 쓴다.

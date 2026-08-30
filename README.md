# implantsj: 임플란트 분류와 metric learning 실험

폐쇄망 GPU 서버에서 **번호만 눌러 실험**하도록 만든 저장소다.
복사·붙여넣기가 안 되는 환경이라 긴 명령을 타이핑할 일이 없게 설계했다.

```bash
./run.sh
```

이 한 줄이 전부다. 나머지는 메뉴에서 번호로 고른다.

---

> 실험 설정·하이퍼파라미터·평가 프로토콜은 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) 에 있다.

---

## 1. 폐쇄망에서 하는 일 (순서대로)

### 1-1. 반입 전 준비 (인터넷이 되는 환경)

```bash
./docker/build_and_export.sh
```

`implantsj_<날짜>.tar.gz` 파일이 생성된다. 폐쇄망에는 이 파일만 반입한다.
이미지에 파이썬 패키지와 pretrained weight가 들어 있으므로,
폐쇄망에서 추가 다운로드가 필요하지 않다.

### 1-2. 반입 후 (폐쇄망)

```bash
gunzip -c implantsj_20260823.tar.gz | docker load
```

```bash
./docker/run_container.sh
```

데이터 경로가 다르면 이렇게 준다.

```bash
DATA=/mnt/실제데이터 OUT=/mnt/결과 ./docker/run_container.sh
```

### 1-3. 컨테이너가 뜨면 메뉴에서

```
  1) 환경 체크 (GPU/패키지 확인)          <- 먼저 이걸로 GPU 잡혔는지 확인
  --- 데이터 ---
  2) 원본 -> 학습용 데이터셋 변환          <- 폐쇄망에서 맨 처음 1회
  3) 데이터셋 선택
  4) 모델 선택
  5) manifest + split 만들기
  --- 학습/평가 ---
  6) 학습 · supervised 분류
  7) 학습 · SupCon metric
  8) 학습 · ViT + VLM feature 패치 주입
  9) 평가 · 분류 (checkpoint)
 10) 평가 · Mahalanobis (unseen)
 11) 평가 · ViT + feature 패치 (ablation)
  --- YOLO ---
 12) YOLO · 검출 학습
 13) YOLO · crop 전처리
  --- 13 feature 표 채우기 (VLM 미사용) ---
 15) 학습 · 클래스->13feature attribute
 16) 평가 · 처음 보는 클래스의 13feature 표 채우기
  ---
 14) 설정 보기/임시 변경
```

일반적인 순서는 `1 → 2 → 3 → 4 → 6` 또는 `7` 이후 `9` 또는 `10`이다.
13 feature 실험은 `1 → 2 → 3 → 4 → 15`를 실행한 뒤, 메뉴 3에서 unseen 데이터셋으로 바꾸고 `16`을 실행한다.

---

## 2. 모델·데이터를 바꿔가며 실험하기

메뉴에서 바꿀 항목은 두 가지다.

- **데이터** 바꾸기 → 메뉴 `3`
- **모델** 바꾸기 → 메뉴 `4`

선택한 뒤 `6`(분류) 또는 `7`(metric)을 실행하면 된다.
결과는 `데이터셋이름__모델이름_sup` 같은 별도 폴더에 저장되므로 기존 결과를 덮어쓰지 않는다.

쓸 수 있는 모델 목록은 `configs/backbones.txt` 에 있다.

```
vit_base_patch16_224.augreg2_in21k_ft_in1k
vit_small_patch16_224.augreg_in21k_ft_in1k
resnet50.a1_in1k
resnet18.a1_in1k
convnext_tiny.fb_in22k_ft_in1k
efficientnet_b0.ra_in1k
```

> 모델을 추가하려면 이 파일에 timm 이름을 한 줄 추가한 뒤 이미지를 다시 빌드한다.
> pretrained weight는 빌드할 때 이미지에 저장되므로, 폐쇄망 안에서 이름만 추가하면
> weight를 찾지 못해 실행이 실패한다.

---

## 3. 설정 바꾸기

epoch, batch size 같은 값은 `exp.conf` 에 있다. 한 번만 고치면 계속 쓰인다.

```bash
EPOCHS=80
BATCH_SIZE=32
IMAGE_SIZE=224
LR=5e-5
SEED=42
DEVICE=cuda
```

한 번만 바꾸려면 명령 앞에 환경변수를 붙인다.

```bash
EPOCHS=5 ./run.sh
```

메뉴 `14`에서도 임시로 바꿀 수 있다.

---

## 4. 원본 데이터 → 학습용 데이터셋 (메뉴 2)

폐쇄망 원본은 병원·촬영건 폴더에 이미지와 JSON이 나뉜 구조다.

```
TS1/h01/h01_0001/xxx.jpg     <- train 이미지
TL1/h01/h01_0001/xxx.json    <- train 라벨 (JSON 안의 "모델명" 이 정답)
VS1/... VL1/...              <- test 이미지 / 라벨
```

메뉴 `2` 가 이걸 학습용 구조로 바꾼다.

```
<이름>/imagefolders/data_42label/train/<클래스>/...
<이름>/imagefolders/data_metric_seen27/...
<이름>/imagefolders/data_metric_unseen15/test/...
```

물어보는 건 세 가지뿐이고 전부 엔터(기본값)로 넘길 수 있다.

1. 원본 루트 (TL1/TS1/VL1/VS1 의 상위 폴더)
2. 만들 데이터셋 이름
3. unseen으로 분리할 희귀 라벨 수: `0`이면 모든 라벨을 학습에 사용한다

라벨 수가 42가 아니어도 된다. 실제 데이터에서 센 개수를 그대로 쓰고,
폴더 이름도 거기에 맞춰 붙는다(`data_38label` 처럼).

---

## 4-1. 13 feature 표 채우기 (메뉴 15 / 16)

이 메뉴는 클래스 이름을 맞히는 실험이 아니다. 처음 보는 클래스의 이미지를 보고
13개 feature의 값을 표에 채운다.

정답은 `configs/class_features.csv`에서만 읽는다. VLM은 사용하지 않는다.
`aihub_all_13feature_21reason_vlm.jsonl`은 메뉴 8과 11에서만 사용한다.

```
idx, class_key, Company, Name, System, Connection_1, Connection_2, Flange,
     Collar, Microthread, Body_Shape, Body_Type, Thread_Shape, Apex_Shape, Apex_Hole, notes
```

표는 45행이다. `03_Osstem_SSII`, `SS II`, `SSII`처럼 표기가 다른 이름도
앞의 일련번호와 끝의 로마 숫자를 정리해 같은 행에 연결한다.
연결되지 않는 이름은 `configs/label_aliases.example.csv` 형식으로 별칭을 적고
`--label-aliases`로 전달한다.

### 학습 (메뉴 15)

attribute마다 head를 하나씩 둔 ViT다. 값이 서로 배타적이므로
"Bone Level이면서 Tissue Level"처럼 동시에 선택되는 예측은 나오지 않는다.

- `Name`은 클래스 이름 그 자체라 처음 보는 클래스에는 적용할 수 없다.
  기본적으로 제외하며, 필요하면 `--attributes`로 직접 지정한다.
- `Body_Type`은 45행 모두 `Threaded`라 정보량이 없다. 자동으로 제외한다.
  선택한 클래스에서 값이 하나뿐인 다른 attribute도 같은 방식으로 제외하고 로그에 남긴다.
- 표의 `확인 불가`는 `unknown`으로 저장하고 loss와 metric에서 제외한다.
  정답이 없는 칸은 오답으로 계산하지 않는다.

### 처음 보는 클래스 평가 (메뉴 16)

메뉴 3에서 학습에 사용하지 않은 클래스가 있는 데이터셋으로 바꾼 뒤 `16`을 실행한다.
생성되는 파일은 다음과 같다.

| 파일 | 내용 |
|---|---|
| `metrics/unseen_test_attribute_metrics.json` | attribute 별 accuracy / macro-F1, 전체 평균, `exact_row_match` |
| `metrics/unseen_test_filled_table.csv` | **채워진 표.** 클래스마다 한 행, attribute 마다 예측값 + 이미지 간 합의율(`_agree`) + 정답(`_true`) |
| `metrics/unseen_test_predictions.csv` | 이미지 한 장 단위 예측 |
| `unseen_setup.json` | 표에 못 붙은 클래스, 학습에 없던 클래스 목록, 학습 어휘에 없는 정답 값(`out_of_vocab`) |

`exact_row_match`는 한 이미지에서 13개 feature를 모두 맞힌 비율이다. 표 전체를
채우는 작업이므로 attribute 평균보다 이 지표가 실제 사용 상황에 가깝다.

학습에서 보지 못한 값은 예측할 수 없다. 예를 들어 학습 클래스에 `TRI-LOBE`가
없었는데 unseen 데이터에 등장하면 `out_of_vocab`으로 기록한다.

### 메뉴 없이

```bash
./run.sh featattr <데이터셋> <모델>
```

```bash
python3 tools/run.py featattr --manifest m.csv --feature-table configs/class_features.csv
```

```bash
python3 tools/run.py featattreval --checkpoint best.pt --manifest unseen.csv --split test
```

### 표를 고칠 때

`configs/class_features.csv`를 직접 수정하면 된다. 표기 변경 내용은 `notes` 컬럼에
기록한다. 예를 들어 `Hexagon->HEX`, `Dome->Rounded`, `Straight->Parallel`처럼 적는다.
표만 고치면 학습 코드는 수정할 필요가 없다.

---

## 5. 자주 막히는 곳

**GPU를 못 잡을 때**: 메뉴 `1`로 먼저 확인한다. `cuda_available: False`면
`docker/run_container.sh` 가 `--gpus all` 을 못 붙인 것이다. 호스트에
nvidia-container-toolkit 이 설치돼 있어야 한다.

**JSON이 없는 이미지가 있을 때**: 메뉴 `2`는 해당 이미지를 건너뛰고 개수를
`summary.json` 에 적는다. 몇 장이 빠졌는지 거기서 확인한다.

**SupCon loss가 이상할 때**: batch 안에 같은 라벨이 2장 이상 있어야 한다.
클래스가 많고 batch 가 작으면 경고가 뜬다. `BATCH_SIZE` 를 올린다.

**학습이 멈춘 것 같을 때**: 모든 실행 로그가 `outputs/logs/`에 날짜별로
남는다. 화면에 출력된 명령과 로그 경로를 그대로 다시 볼 수 있다.

---

## 6. 구조

```
run.sh                 유일한 진입점 (대화형 메뉴 + 짧은 CLI)
exp.conf               경로·하이퍼파라미터 기본값
configs/backbones.txt  실험에 쓸 모델 목록 (이미지 빌드 시 weight 함께 구움)
configs/class_features.csv  클래스 -> 13 feature 정답표 (메뉴 15/16)
tools/run.py           명령 -> python 모듈 디스패처
src/                   데이터·모델·학습·평가 공용 코드
pipelines/             단계별 실행 스크립트 (data/train/eval/yolo/visualize)
docker/                빌드·반입·실행 스크립트
legacy/                예전 중복 구조 (삭제하지 않고 보관)
```

메뉴를 안 쓰고 바로 실행하려면:

```bash
./run.sh train <데이터셋> <모델>
```

```bash
./run.sh --help
```

# implantsj — 임플란트 분류/metric learning 실험

폐쇄망 GPU 서버에서 **번호만 눌러 실험**하도록 만든 저장소다.
복사·붙여넣기가 안 되는 환경이라 긴 명령을 타이핑할 일이 없게 설계했다.

```bash
./run.sh
```

이 한 줄이 전부다. 나머지는 메뉴에서 번호로 고른다.

---

## 1. 폐쇄망에서 하는 일 (순서대로)

### 1-1. 반입 (인터넷 되는 바깥에서 미리)

```bash
./docker/build_and_export.sh
```

`implantsj_<날짜>.tar.gz` 하나가 나온다. **이 파일만 반입**하면 된다.
이미지 안에는 파이썬 패키지와 pretrained weight가 전부 들어 있어서,
폐쇄망에서 인터넷을 찾지 않는다.

### 1-2. 반입 후 (폐쇄망 안에서)

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
  8) 평가 · 분류 (checkpoint)
  9) 평가 · Mahalanobis (unseen)
  --- YOLO ---
 10) YOLO · 검출 학습
 11) YOLO · crop 전처리
 12) 설정 보기/임시 변경
```

일반적인 순서: **1 → 2 → 3 → 4 → 6** (또는 7) **→ 8** (또는 9).

---

## 2. 모델·데이터를 바꿔가며 실험하기

바꿀 게 두 개뿐이다.

- **데이터** 바꾸기 → 메뉴 `3`
- **모델** 바꾸기 → 메뉴 `4`

고르고 나서 `6`(분류) 또는 `7`(metric)을 다시 누르면 끝이다.
결과는 `데이터셋이름__모델이름_sup` 같은 폴더로 각각 쌓이므로 서로 덮어쓰지 않는다.

쓸 수 있는 모델 목록은 `configs/backbones.txt` 에 있다.

```
vit_base_patch16_224.augreg2_in21k_ft_in1k
vit_small_patch16_224.augreg_in21k_ft_in1k
resnet50.a1_in1k
resnet18.a1_in1k
convnext_tiny.fb_in22k_ft_in1k
efficientnet_b0.ra_in1k
```

> 모델을 추가하려면 이 파일에 timm 이름을 한 줄 적고 **이미지를 다시 빌드**한다.
> pretrained weight는 빌드할 때 이미지에 구워지므로, 폐쇄망 안에서 줄만 추가하면
> weight가 없어서 실패한다.

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

이번 실행만 바꾸고 싶으면 앞에 붙인다.

```bash
EPOCHS=5 ./run.sh
```

메뉴 `12` 에서도 임시로 바꿀 수 있다.

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
3. unseen 으로 뺄 희귀 라벨 개수 — `0` 이면 전부 학습에 쓴다

라벨 수가 42가 아니어도 된다. 실제 데이터에서 센 개수를 그대로 쓰고,
폴더 이름도 거기에 맞춰 붙는다(`data_38label` 처럼).

---

## 5. 자주 막히는 곳

**GPU를 못 잡을 때** — 메뉴 `1` 로 먼저 확인한다. `cuda_available: False` 면
`docker/run_container.sh` 가 `--gpus all` 을 못 붙인 것이다. 호스트에
nvidia-container-toolkit 이 설치돼 있어야 한다.

**JSON이 없는 이미지가 있을 때** — 메뉴 `2` 는 그런 이미지를 건너뛰고 개수를
`summary.json` 에 적는다. 몇 장이 빠졌는지 거기서 확인한다.

**SupCon loss 가 이상할 때** — batch 안에 같은 라벨이 2장 이상 있어야 한다.
클래스가 많고 batch 가 작으면 경고가 뜬다. `BATCH_SIZE` 를 올린다.

**학습이 멈춘 것 같을 때** — 모든 실행 로그가 `outputs/logs/` 에 날짜별로
남는다. 화면에 출력된 명령과 로그 경로를 그대로 다시 볼 수 있다.

---

## 6. 구조

```
run.sh                 유일한 진입점 (대화형 메뉴 + 짧은 CLI)
exp.conf               경로·하이퍼파라미터 기본값
configs/backbones.txt  실험에 쓸 모델 목록 (이미지 빌드 시 weight 함께 구움)
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

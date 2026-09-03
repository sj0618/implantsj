# 원격 서버 세팅 프롬프트 (폐쇄 전 마지막 창)

아래 `---` 사이의 내용을 원격 서버의 에이전트 세션에 그대로 붙여넣는다.
사람이 직접 할 경우에도 같은 순서로 따라가면 된다.

작성 기준: 이미지 `yahoexpert/implantsj:remote-20260903`, 2026-09-03.

---

## 상황

너는 치과 임플란트 분류 실험 서버를 세팅한다. 이 서버는 곧 폐쇄망(air-gapped)이 된다.

**지금**: 인터넷 사용 가능, 실제 데이터는 아직 없음
**폐쇄 후**: 인터넷 끊김, 실제 데이터가 mount 로 연결됨

그래서 **인터넷이 필요한 일은 전부 지금 끝내야 한다.** 폐쇄 후에 뭔가를 내려받으려 하면
그 실험은 그냥 못 하게 된다. 아래 순서대로 진행하고, 각 단계의 확인 결과를 그대로 보고한다.

실험 코드, 파이썬 의존성, timm pretrained 백본 6종, YOLO 가중치 2종이 모두 Docker 이미지
안에 구워져 있다. **너가 새로 설치하거나 내려받을 것은 없다.** 이 프롬프트의 목적은
호스트 쪽 전제조건(GPU 런타임)을 확보하고, 이미지가 실제로 도는지 검증하는 것이다.

## 폐쇄망에 들어올 데이터

`train` / `test` / `val` 로 나뉜 **42라벨** ImageFolder 데이터셋 하나가 전부다.
원본 JSON 라벨을 변환하는 전처리 단계는 필요 없다. 이미 정리된 상태로 들어온다.

---

## 1단계 — 호스트 전제조건 확인 (인터넷 필요할 수 있음)

이 단계에서 빠진 게 발견되면 **폐쇄 전에 반드시 설치해야 한다.** 폐쇄 후에는 못 한다.

```bash
docker --version
nvidia-smi
docker info | grep -i -A2 runtimes
df -h /var/lib/docker
```

확인할 것:

- `docker` 가 있고 데몬이 돌아간다
- `nvidia-smi` 가 GPU 를 보여준다
- `docker info` 의 runtimes 에 `nvidia` 가 있다 — **없으면 `--gpus all` 이 안 붙어 CPU 로만 돌아간다.**
  `nvidia-container-toolkit` 을 지금 설치한다. 이게 이 단계에서 가장 중요하다.
- 여유 디스크 30GB 이상 (이미지 전개 약 17GB + 반입용 tar.gz 약 6.2GB)

GPU 가 실제로 컨테이너에 보이는지는 이렇게 확인한다.

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

이 명령이 GPU 표를 출력해야 한다. 실패하면 toolkit 설치가 덜 된 것이다.

## 2단계 — 이미지 받기

```bash
docker pull yahoexpert/implantsj:remote-20260903
docker tag yahoexpert/implantsj:remote-20260903 implantsj:latest
```

`implantsj:latest` 태그를 반드시 만든다. `docker/run_container.sh` 가 기본으로 그 이름을 찾는다.

서버가 x86_64 가 아니면 이미지가 안 맞는다. 확인:

```bash
docker image inspect implantsj:latest --format '{{.Architecture}}/{{.Os}}'
```

`amd64/linux` 가 나와야 한다.

## 3단계 — 이미지에 구워진 가중치 확인

이 이미지에는 실행에 필요한 가중치가 **전부 들어 있다.** 폐쇄 후에 내려받을 것이 없다.
받아야 할 것은 없고, 들어 있는지만 확인한다.

```bash
docker run --rm implantsj:latest bash -c 'ls -l weights/; ls /opt/hf/hub | grep ^models'
```

기대 결과:

- `weights/yolov8s.pt` 약 21.5 MB, `weights/yolov8n.pt` 약 6.2 MB
- `models--timm--` 로 시작하는 디렉터리 6 개 (ViT-Base, ViT-Small, ResNet50, ResNet18,
  ConvNeXt-Tiny, EfficientNet-B0)

목록에 없는 모델을 쓰려면 **지금** 알려야 한다. 이미지를 다시 빌드해야 하고,
그건 인터넷이 있는 쪽에서만 가능하다. 폐쇄 후에는 되돌릴 수 없다.

## 4단계 — 반입용 파일 만들기 (선택)

이 서버가 그대로 폐쇄되는 경우에는 필요 없다. 이미지가 이미 로컬에 있다.

다른 서버로 옮겨야 하거나, Docker 상태가 초기화될 수 있으면 파일로 남긴다.

```bash
docker save implantsj:latest | gzip -c > implantsj_remote-20260903.tar.gz
ls -lh implantsj_remote-20260903.tar.gz
```

받는 쪽에서:

```bash
gunzip -c implantsj_remote-20260903.tar.gz | docker load
```

## 5단계 — 검증 (데이터 없이 돈다)

이게 세팅이 끝났다는 유일한 근거다. 합성 데이터로 10단계를 돌린다.

```bash
docker run --rm implantsj:latest ./docker/smoke_test.sh
```

**기대 출력**: 마지막 두 줄이 정확히 이래야 한다.

```
통과 10 / 실패 0
전부 통과. 코드와 환경은 정상입니다.
```

10단계는 환경 체크, 합성 데이터 생성, 데이터셋 변환, 학습 1 epoch, 평가,
얼굴인식(ArcFace) 전체, 동적 클래스 수 manifest, ArcFace 13feature fixture,
ArcFace 13feature 전체, YOLO 가중치 오프라인 로드다. 하나라도 실패하면 그 단계의
출력을 그대로 보고하고 멈춘다. **임의로 넘어가지 않는다.**

10 번 단계가 폐쇄망 대비의 핵심이다. YOLO 가중치가 이미지에 없으면 폐쇄 후
ultralytics 가 다운로드를 시도하다 실패하는데, 그 상황을 여기서 미리 잡는다.

GPU 가 컨테이너에서 보이는지도 따로 확인한다.

```bash
docker run --rm --gpus all implantsj:latest ./run.sh check
```

출력에서 `cuda_available` 이 `True` 인지 본다. `False` 면 1단계의 toolkit 문제다.

## 6단계 — 폐쇄 후 실행 방법을 파일로 남기기

폐쇄망 안에서는 이 문서를 다시 볼 수 없으므로, 서버에 메모를 남긴다.

```bash
cat > ~/implantsj_실행법.txt <<'EOF'
# 컨테이너 띄우기 (DATA/OUT 을 실제 경로로)
DATA=/실제/데이터경로 OUT=/실제/결과경로 \
  docker run --rm -it --gpus all --ipc=host \
  -v /실제/데이터경로:/workspace/data \
  -v /실제/결과경로:/workspace/outputs \
  -e DATA_ROOT=/workspace/data -e OUTPUT_ROOT=/workspace/outputs \
  implantsj:latest

# 뜨면 번호 메뉴가 나온다. 1 번으로 GPU 부터 확인한다.
# 문서는 컨테이너 안 README.md 와 docs/ 에 다 들어 있다.
EOF
cat ~/implantsj_실행법.txt
```

컨테이너 안에 `docker/run_container.sh` 도 있지만, 그건 저장소를 호스트에 풀어둔 경우에
쓰는 것이다. 이미지만 있으면 위 `docker run` 을 그대로 쓴다.

---

## 폐쇄 후 실험 가용성 (참고용, 지금 할 일 아님)

42라벨 데이터 기준이다. 폐쇄 후 사용자가 물어보면 이대로 답한다.

**그대로 돌아감**: 환경 체크, 모델 선택, manifest+split, supervised 분류 학습,
SupCon metric 학습, 분류 평가, 13feature attribute 학습, ArcFace 인식,
ArcFace 로 13feature 표 채우기, fine-tuning, attention·t-SNE 시각화, run 요약 리포트

**먼저 뭔가를 만들어야 함**:

- Mahalanobis unseen 평가와 unseen 표 채우기 평가는 학습에 없던 클래스가 필요한데,
  42라벨이 세 split 에 모두 있어 그런 클래스가 없다. 두 메뉴에는 holdout 옵션이 없으므로
  unseen 클래스가 빠진 manifest 를 따로 만들어야 한다.
- 메뉴 18(ArcFace 로 13feature 표 채우기)만 `ARCFACE_HOLDOUT=N` 으로 스스로 unseen 을
  만든다. 단일 데이터셋에서 바로 돌아가는 유일한 unseen 실험이다.

**막혀 있음**:

- 메뉴 8·11 (ViT + VLM feature 패치 주입)은 VLM 미사용 방침으로 닫혀 있다.
  같은 목적은 메뉴 18 이 VLM 없이 수행한다. **VLM 을 다시 끌어오지 않는다.**
- YOLO 검출 학습·crop 은 가중치가 이미지에 구워져 있어 열려 있다. 다만 기본 데이터
  경로(`data/Dental Implants 2.0.v3i.yolov8/data.yaml`)는 이미지에 없으므로 `--data` 는
  직접 준다. 검출용 bounding box 라벨이 있는 데이터가 따로 있어야 한다 —
  42라벨 분류 데이터로는 검출을 학습할 수 없다.

  ```bash
  python3 tools/run.py yolotrain --data <data.yaml>          # --model 기본값이 weights/yolov8s.pt
  python3 tools/run.py yolocrop --source <이미지루트> \
      --weights <학습된.pt> --output-root <출력경로>
  ```

**필요 없음**: 원본 → 학습용 데이터셋 변환(메뉴 2), 데이터셋 선택(메뉴 3).
데이터가 이미 정리돼 있고 하나뿐이다.

## 알려진 데이터 이슈

정답표 `configs/class_features.csv` 에 closed 27라벨 중 24개만 붙는다.
`Luna` 는 표에 `Luna1` 로 있어 별칭 한 줄로 해결되고, `Hexplant` 와 `IT` 는 표에 행이 없어
사람이 13 feature 값을 채워야 한다. open 15라벨은 이름 목록이 없어 매칭을 확인하지 못했다 —
메뉴 15 를 한 번 돌리면 실행 로그에 미매칭 목록이 그대로 찍힌다.

---

## 수용 기준

전부 이진 판정이다. 하나라도 아니면 세팅이 끝난 게 아니다.

1. `docker image inspect implantsj:latest` 가 `amd64/linux` 를 출력한다
2. `docker run --rm --gpus all implantsj:latest ./run.sh check` 에서 `cuda_available: True`
3. `docker run --rm implantsj:latest ./docker/smoke_test.sh` 가 `통과 10 / 실패 0`
4. `~/implantsj_실행법.txt` 가 존재한다

## 하지 말 것

- **폐쇄 후에 인터넷이 필요한 일을 남기지 않는다.** 남았다면 지금 끝내거나, 못 한다고 보고한다.
- `configs/backbones.txt` 에 없는 timm 모델 이름을 쓰지 않는다. 그 가중치는 이미지에 없다.
- 검증 단계를 건너뛰고 "세팅 완료"라고 하지 않는다. 3번 기준의 출력이 근거다.
- 실패를 성공으로 뭉개지 않는다. 실패한 단계의 출력을 그대로 붙여 보고한다.
- 이미지 안의 코드를 고치지 않는다. 수정이 필요하면 무엇이 왜 필요한지 보고한다.

## 보고 형식

끝나면 이렇게 보고한다.

```
호스트:      docker <버전> / GPU <모델> / nvidia runtime <있음|없음> / 여유디스크 <N>GB
이미지:      implantsj:latest <아키텍처> <크기>
GPU 확인:    cuda_available=<True|False>
smoke test:  통과 <N> / 실패 <N>
구워진 가중치: YOLO <N>개 / timm <N>개
반입 파일:   <경로> <크기> | 만들지 않음(이 서버가 그대로 폐쇄됨)
남은 인터넷 의존: <없음 | 목록>
```

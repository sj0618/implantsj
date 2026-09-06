# 통합 이미지 빌드 프롬프트 (x86_64 리눅스, 인터넷 필요)

아래 `---` 사이를 원격 x86 서버의 에이전트 세션에 붙여넣는다.
사람이 직접 할 경우에도 같은 순서로 따라가면 된다.

이 문서는 맥(ARM)에서 작성됐고 **빌드는 한 번도 실행되지 않았다.** Grounding DINO 의
CUDA 확장을 에뮬레이션에서 컴파일할 수 없어 x86 에 넘긴다. 그래서 아래 절차에는
"실패하면 이렇게 보고하라"가 함께 들어 있다.

---

## 상황

너는 치과 임플란트 연구 저장소 `tria-lab/ImplantID` 의 세 프로젝트를 하나의 Docker
이미지로 빌드한다. 이 이미지는 곧 폐쇄망(air-gapped) GPU 서버로 반입된다.

**지금**: 인터넷 사용 가능, 실제 데이터 없음
**반입 후**: 인터넷 끊김, 데이터가 volume 으로 연결됨

런타임에 내려받아야 하는 파일이 하나라도 남으면 그 실험은 폐쇄망 안에서 못 고친다.
모든 의존성과 pretrained 가중치는 빌드 타임에 이미지 안에 들어가야 한다.

## 이 이미지의 구조 — 왜 venv 가 셋인가

세 프로젝트의 핀이 서로 배타적이라 한 파이썬 환경에 넣을 수 없다.

```
              implantsj        implantys          implant_ny
  torch       2.4.1 cu121      2.4.1 cu124        2.5.1 cu124
  numpy       >=1.24,<2.0      1.26.4             2.4.4
  opencv      (ultralytics)    headless           opencv-python 5.0
```

- `numpy` 2.x 는 implantys 에서 금지다 — pycocotools ABI 에러(`setup_pod.sh` 기록)
- `torch` 2.6+ 는 implantys 의 `ms_deform_attn` CUDA 확장 컴파일을 깬다
- implantys 는 opencv GUI 빌드와 headless 의 동시 설치를 금지한다

그래서 `/opt/venv/{sj,ys,ny}` 세 개를 만들고 **어느 핀도 건드리지 않는다.** 코드와
가중치는 한 이미지에 있고, 실행할 때 해당 venv 로 들어간다.

베이스가 `runtime` 이 아니라 `devel` 인 이유는 `nvcc` 다. Grounding DINO 는 CUDA 확장을
소스에서 빌드해야 하고, nvcc 가 없으면 bounding box 실험이 통째로 막힌다.

## 1단계 — 호스트 확인

```bash
uname -m
docker --version
nvidia-smi
df -h /var/lib/docker
```

- `uname -m` 이 `x86_64` 여야 한다. ARM 이면 이 빌드를 하지 않는다.
- 여유 디스크 **80GB 이상** 을 권한다. devel 베이스 + torch 3벌 + 가중치라 크다.
- 빌드에는 GPU 가 필요 없다. nvcc 는 GPU 없이도 컴파일한다.
  (`TORCH_CUDA_ARCH_LIST` 가 Dockerfile 에 이미 박혀 있다.)

## 2단계 — 저장소 받기

```bash
git clone https://github.com/tria-lab/ImplantID.git
cd ImplantID
git log --oneline -3
```

이미 있으면 `git pull` 로 최신 `main` 을 맞춘다.

## 3단계 — 빌드

```bash
./docker/build_combined.sh
```

이 스크립트가 하는 일은 세 가지다. build context 크기를 보여주고, `docker build` 를
돌리고, 끝나면 자가진단을 실행한다.

**오래 걸린다.** 세 venv 에 torch 를 각각 설치하고, Grounding DINO 의 CUDA 확장을
컴파일하고, 가중치 약 1.5GB 를 받는다.

빌드 중 이미지에 굽는 가중치:

| 프로젝트 | 가중치 | 크기 |
|---|---|---|
| implantsj | timm 백본 6종 | ~1.5GB |
| implantsj | yolov8s.pt / yolov8n.pt | 27.7MB |
| implantys | groundingdino_swint_ogc.pth | ~660MB |
| implantys | bert-base-uncased (HF 캐시) | ~440MB |
| implantys | vit_base_patch14_dinov2.lvd142m | ~350MB |
| implant_ny | sam2.1_hiera_tiny.pt | 149MB |

## 4단계 — 가장 깨지기 쉬운 지점

**Grounding DINO 의 `ms_deform_attn` CUDA 확장 컴파일이다.** 이 빌드에서 유일하게
검증되지 않은 단계이고, 실패하면 여기서 멈춘다.

실패하면 로그의 어느 쪽인지 구분해서 보고한다.

- `nvcc: not found` → 베이스가 devel 이 아니다. Dockerfile 의 `FROM` 을 확인한다.
- `error: command '/usr/local/cuda/bin/nvcc' failed` → 컴파일 자체 실패.
  전체 에러 블록을 그대로 붙여 보고한다.
- `libc10.so: cannot open shared object file` → `LD_LIBRARY_PATH` 문제.
  Dockerfile 이 이미 걸어두었으니, 그래도 나면 그 사실을 보고한다.
- pip 이 다른 torch 를 받아가려 하면 → `--no-build-isolation` 이 빠진 것이다.

**임의로 핀을 바꾸거나 `--no-build-isolation` 을 떼서 우회하지 않는다.** 그 핀들은
이유가 문서에 적혀 있다. 막히면 막혔다고 보고한다.

gdino 만 빼고 나머지를 쓰기로 결정되면 그때 Dockerfile 에서 해당 블록을 지운다.
그건 사람이 결정할 일이지 네가 판단할 일이 아니다.

## 5단계 — 검증

빌드 스크립트가 자동으로 돌리지만, 따로 실행할 수도 있다.

```bash
docker run --rm implantid:latest smoke
```

**기대 출력**: 마지막 부분이 이래야 한다.

```
통과 5 / 실패 0
전부 통과.
```

5단계의 내용:

1. 환경 격리 — venv 3개의 torch/numpy 버전이 실제로 다른지
2. implantsj 전체 파이프라인 — 합성 데이터로 10단계 (학습·평가·ArcFace 포함)
3. implantys 환경 + gdino CUDA 확장 로드
4. implantys BERT 오프라인 로드 — 폐쇄망에서 텍스트 인코더를 받으러 나가지 않는지
5. implant_ny 환경 + SAM2 체크포인트

**2번만 실제 학습·평가를 돌린다.** 3~5 는 환경과 가중치가 살아 있는지만 본다.
implantys / implant_ny 에는 이 저장소에 자가진단이 없어서, 그 학습·평가 로직이
맞는지는 검사하지 않는다. 그렇게 보고한다 — "전부 검증됨"이라고 하지 않는다.

## 6단계 — 반입 파일

이 서버가 그대로 폐쇄되면 필요 없다. 다른 서버로 옮겨야 하면 만든다.

```bash
docker save implantid:latest | gzip -c > implantid_$(date +%Y%m%d).tar.gz
ls -lh implantid_*.tar.gz
```

받는 쪽에서:

```bash
gunzip -c implantid_YYYYMMDD.tar.gz | docker load
```

## 폐쇄망에서 쓰는 법

```bash
docker run --rm -it --gpus all --ipc=host \
  -v /실제/데이터경로:/workspace/data \
  -v /실제/결과경로:/workspace/outputs \
  -e DATA_ROOT=/workspace/data -e OUTPUT_ROOT=/workspace/outputs \
  implantid:latest <명령>
```

`<명령>` 자리에 오는 것:

```
menu          implantsj 실험 메뉴 (기본값)
sj  [인자...]  implantsj  venv 에서 실행
ys  [인자...]  implantys  venv 에서 실행
ny  [인자...]  implant_ny venv 에서 실행
smoke         자가진단
bash          셸
help          도움말
```

---

## 수용 기준

전부 이진 판정이다. 하나라도 아니면 빌드가 끝난 게 아니다.

1. `docker image inspect implantid:latest --format '{{.Architecture}}'` 가 `amd64`
2. `docker run --rm implantid:latest smoke` 가 `통과 5 / 실패 0`
3. `docker run --rm implantid:latest ys -c "from groundingdino import _C"` 가 조용히 성공
4. `docker run --rm implantid:latest help` 가 세 프로젝트를 나열

## 하지 말 것

- 핀을 바꿔 빌드를 통과시키지 않는다. 세 프로젝트의 버전 고정에는 문서화된 이유가 있다.
- 검증을 건너뛰고 "빌드 완료"라고 하지 않는다. 2번 기준의 출력이 근거다.
- 3~5 단계를 근거로 implantys / implant_ny 가 "동작 검증됐다"고 말하지 않는다.
  환경이 살아 있는 것과 실험이 맞게 도는 것은 다르다.
- 실패를 성공으로 뭉개지 않는다. 실패한 단계의 로그를 그대로 붙여 보고한다.

## 보고 형식

```
호스트:       <arch> / docker <버전> / 여유디스크 <N>GB
빌드:         성공 | 실패(<어느 단계>)
gdino 확장:   컴파일 성공 | 실패(<에러 첫 줄>)
이미지:       implantid:latest <크기>
smoke:        통과 <N> / 실패 <N>
가중치:       timm <N> / yolo <N> / gdino <크기> / bert <있음|없음> / sam2 <크기>
반입 파일:    <경로> <크기> | 만들지 않음
남은 인터넷 의존: <없음 | 목록>
```

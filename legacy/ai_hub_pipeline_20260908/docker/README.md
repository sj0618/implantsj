# 폐쇄망 반입 (ai_hub_pipeline)

바깥(인터넷 O, 데이터 X)에서 이미지를 굽고, 안(데이터 O, 인터넷 X)에서 실행한다.

## 이미지에 들어가는 것

| 항목 | 방법 | 위치 |
|---|---|---|
| pip 패키지 (torch 2.5.1/cu124, timm, …) / sam2·segment_anything | `requirements.txt` / `requirements-seg.txt`(build isolation 끔) | site-packages |
| timm pretrained (base.yaml·05_attributes.yaml 의 `backbones.*.timm` 전부) | `docker/prefetch_weights.py` | `/opt/hf`, `/opt/torch` |
| SAM1 `sam_vit_h_4b8939.pth` (2.5 GB) | prefetch 가 다운로드. `PREFETCH_SAM1=0` 으로 생략 가능 | `/opt/weights/sam/` |
| SAM2 `sam2.1_hiera_tiny.pt` | prefetch 가 다운로드 | `/opt/weights/sam2/` |
| 파인튜닝 adapter `best_sam2.1_tiny_adapter_unet.pth` | 빌드 전에 `weights/` 에 두면 복사 (없으면 경고) | `/opt/weights/` |

컨테이너가 뜰 때 `docker/entrypoint.sh` 가 `/opt/weights/*` 를 `checkpoints/` 아래 yaml 이 기대하는
경로에 놓는다. SAM1/SAM2 base 는 symlink, 파인튜닝 adapter 는 **복사**한다 — Stage 0 ⑤ 가 그 경로에
결과를 덮어쓰므로 symlink 면 재학습 결과가 컨테이너와 함께 사라진다. 이미 있는 파일은 건드리지 않는다.

## 절차

```bash
# 1. (선택) 팀 자산을 빌드 컨텍스트에 둔다
cp /어딘가/best_sam2.1_tiny_adapter_unet.pth weights/

# 2. 바깥에서 빌드 + 저장
./docker/build_and_export.sh              # → ai_hub_pipeline_<날짜>.tar.gz
PREFETCH_SAM1=0 ./docker/build_and_export.sh   # SAM1 생략 (adapter 있을 때)

# 3. 폐쇄망 안에서 로드
gunzip -c ai_hub_pipeline_<날짜>.tar.gz | docker load

# 4. 실행 — WORK 아래 data/ outputs/ checkpoints/ seg_data/ 가 붙는다
WORK=/workspace/ai_hub ./docker/run_container.sh bash scripts/00_data/prepare_data.sh
WORK=/workspace/ai_hub ./docker/run_container.sh bash scripts/01_backbone/finetune_backbones.sh
WORK=/workspace/ai_hub SET="grid.backbone=vit train.epochs=1" \
  ./docker/run_container.sh bash scripts/01_backbone/finetune_backbones.sh   # 짧은 점검
```

빌드는 마지막 레이어에서 스스로 검사한다: `src/data/` 패키지 존재, 패키지 import, timm 가중치가
오프라인 캐시에서 풀리는지, SAM 파일 존재, `verify_pipeline.py` 전체, `tests/test_smoke.py`
(임베딩 저장 + attributes 두 모드). 하나라도 실패하면 이미지가 만들어지지 않는다.

## 자주 걸리는 것

- 빌드가 `src/data/ 가 없습니다` 로 멈추면: 루트 `.gitignore` 의 `data/` 가 그 파이썬 패키지를
  무시해 저장소에 빠진 것이다. 원 작성자 로컬에서 `git add -f ai_hub_pipeline/src/data` 로 올린다
  (예외 규칙 `!ai_hub_pipeline/src/data/` 는 추가돼 있음). `run_prepare_data.py` 가 그 패키지를 쓰므로
  없는 이미지는 첫 단계부터 돌지 않는다 — 그래서 빌드에서 막는다.
- Apple Silicon 에서 빌드해도 `--platform linux/amd64` 로 x86_64 이미지가 나온다 (느리다).

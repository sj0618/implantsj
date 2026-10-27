# Stage 1. Backbone 제작

코드는 `src/backbone/`, 설정은 `configs/01_backbone.yaml`.
노션 구조의 **1. Backbone** 브랜치.

여기서 만든 백본을 Stage 2(metric) / Stage 3(classifier)이 **끌어다 쓴다**.

## 백본 4종

| 이름 | 베이스 | band-attention |
|---|---|---|
| `vit` | ViT-B/16 | ✗ |
| `band-vit` | ViT-B/16 | ✓ band 영역에 집중하도록 유도 |
| `dinov2` | DINOv2 ViT-B/14 | ✗ |
| `band-dinov2` | DINOv2 ViT-B/14 | ✓ band-vit와 동일 방식 |

`cnn` 백본은 현재 제외.

**band-\* 학습 방식**: 이미지와 band mask를 같은 기하로 augment한 뒤,
CLS→patch attention이 band 쪽으로 쏠리도록 KL prior loss를 CE에 더한다.
`prior = 1 + alpha * band_mask`, 최종 loss = `CE + loss_weight * KL`.
ViT 구조에만 의존하므로 DINOv2에도 그대로 적용된다.

## 실험 옵션 (yaml)

| 옵션 | 위치 | 선택지 |
|---|---|---|
| finetune 범위 | `grid.finetune_mode` | `full`(전체) / `partial`(마지막 n block + norm/head만, 나머지 frozen) |
| partial 범위 | `partial.trainable_blocks` | 학습할 마지막 block 수 (기본 4) |
| 프로토콜 | `protocol` | `closed27_open15`(27클래스만 학습) / `full42`(42클래스 전부) |
| band 강도 | `band.alpha`, `band.loss_weight` | band-* 백본에만 적용 |
| 클래스 가중 loss | `train.use_class_weights` | 빈도 역수를 CE 가중치로 — 이미지가 적은 클래스가 밀리지 않게 |

**프로토콜이 중요한 이유**: `closed27_open15` 실험에서는 open 15클래스가
백본에도 노출되면 안 되므로(data leakage), 백본을 27클래스 manifest로만 학습한다.
드라이버가 `protocol`에 따라 `closed_manifest` / `full_manifest`를 자동 선택한다.

```bash
bash finetune_backbones.sh                    # closed27_open15 (4 backbone x 2 mode = 8개)
PROTOCOL=full42 bash finetune_backbones.sh    # 42클래스 버전
```

## 입력

`data.closed_manifest` / `data.full_manifest` — `path,label,split` CSV.
**band-\* 백본은 `mask_path` 컬럼이 추가로 필요**하다
(Stage 0의 `06_make_band_masks.sh` 산출물 `manifest_band.csv`).
`vit`/`dinov2`는 mask 없이도 학습된다.

## 출력

```
checkpoints/backbones/
  <backbone>_<mode>_<protocol>.pt    분류 head를 제외한 백본 weight
  index_<protocol>.json              이번 실행이 만든 체크포인트 경로 목록
outputs/backbone/<backbone>_<mode>_<protocol>/
  config.json  history.csv  summary.json
```

학습이 끝나면 `index_<protocol>.json`의 경로를 `configs/base.yaml`의
`backbones.<name>.checkpoint` 에 넣는다. 그러면 Stage 2/3이 그 백본을 사용한다.

```yaml
# configs/base.yaml
backbones:
  band-dinov2:
    timm: vit_base_patch14_dinov2.lvd142m
    checkpoint: checkpoints/backbones/band-dinov2_full_closed27_open15.pt
    image_size: 224
```

## 하나만 돌리기

그리드는 의무가 아니다. yaml의 리스트를 한 개로 줄이거나(스칼라도 허용),
yaml을 고치지 않고 `SET=` 으로 이번 실행만 좁힐 수 있다.

```bash
SET="grid.backbone=band-vit grid.finetune_mode=partial" bash finetune_backbones.sh
```

`SET`은 yaml 항목을 점으로 짚어 덮어쓴다 (`train.epochs=1`,
`eval.visualize.enabled=false` 등). 적용된 값은 실행 로그에
`[config] override ...` 로 남는다.

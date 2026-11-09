# 백본 학습

| 코드 | 역할 |
|---|---|
| `core.py` | 이미지·mask 로딩, 정사각 padding, 모델·Band attention 구성 |
| `train.py` | full/partial fine-tuning, 검증·test, checkpoint와 특징 저장 |
| `__init__.py` | 패키지 연결 |

```bash
python src/run_backbone.py --config configs/01_backbone.yaml --protocol closed27_open15 --set grid.backbone=vit paths.output_dir=/new/run/outputs paths.checkpoints_dir=/new/run/checkpoints
```

입력은 manifest와 모델 레지스트리다. Band 계열은 이미지에 맞는 mask가 필요하다.
`closed27_open15`는 closed 클래스만 학습하고 `full42`는 전체 클래스를 학습한다.
모델·학습 설정과 결과·checkpoint를 저장한다. 정확한 과거 재현에는 당시 설정 변경값이 필요하다.

# Masked 이미지 분류

| 코드 | 역할 |
|---|---|
| `segmentation.py`, `band_preprocessing.py` | SAM2 adapter와 마스크 기반 전처리 |
| `data.py`, `core.py` | 이미지·manifest·공용 학습 처리 |
| `training.py`, `evaluation.py` | 학습 및 원본/마스크 입력 평가 |
| `gradient_rollout.py` | attention 확인 |
| `__init__.py` | 패키지 연결 |

```bash
python src/run_masked_vit.py --config configs/04_masked_vit.yaml
```

Manifest, SAM2 base checkpoint, fine-tuned adapter와 SAM2 패키지가 필요하다.
마스크 입력으로 학습한 뒤 설정에 따라 원본 이미지와 마스크 이미지의 평가를 저장한다.
원본 입력 평가와 마스크 입력 평가를 같은 조건의 결과로 합치지 않는다.

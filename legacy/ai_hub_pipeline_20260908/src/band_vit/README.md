# Band attention 분류

| 코드 | 역할 |
|---|---|
| `data.py` | 이미지·band mask 데이터셋 |
| `models.py`, `rollout_vit2.py` | 모델과 attention rollout |
| `losses.py` | 분류 손실과 band attention prior 손실 |
| `training.py`, `evaluation.py` | 학습·검증·예측 점수 저장 |
| `make_band_masks.py` | band 영역 생성 보조 함수 |
| `__init__.py` | 패키지 연결 |

```bash
python src/run_band_vit.py --config configs/04_band_vit.yaml --set sweep.alpha=8 sweep.lambda=0.8 sweep.block=6
```

Manifest의 `mask_path`와 실제 mask, 모델 초기 가중치를 준비한다.
alpha·lambda·block을 고정하거나 조합별로 학습하고 checkpoint·검증/test 점수·특징을 저장한다.
출력 경로는 새 경로로 지정한다. 분류기 학습은 backbone 학습 단계와 별도다.

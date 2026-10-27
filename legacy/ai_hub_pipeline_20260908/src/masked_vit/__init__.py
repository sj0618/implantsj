"""masked-vit: SAM2 런타임 band-crop으로 학습하고 원본 이미지로 평가하는 ViT 분류기.

band_vit_3(학습) + band_vit_3_eval(원본이미지 평가)를 하나로 합친 독립 패키지.
다른 패키지를 import하지 않는다 — 공용 헬퍼는 core.py에 내장.

  training.py         학습 (band-only crop) + test 평가/attention map
  evaluation.py       체크포인트 재평가 (--input-mode band|original) + focus_gain
  gradient_rollout.py gradient-weighted attention rollout 시각화
  data.py             런타임 band 전처리 캐시 + 데이터로더
  band_preprocessing.py  band(고리) crop 기하
  segmentation.py     frozen SAM2+Adapter+U-Net 세그멘터
  core.py             공용 헬퍼 (model/loss/metric/시각화)
"""

__all__ = [
    "training",
    "evaluation",
    "gradient_rollout",
    "data",
    "band_preprocessing",
    "segmentation",
    "core",
]

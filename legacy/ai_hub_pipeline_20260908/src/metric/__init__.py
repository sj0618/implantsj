"""metric: 백본 임베딩 + metric learning + prototype 분류 (노션 3번 브랜치).

implantsj의 metric learning 코드를 기반으로 만든 독립 패키지 —
다른 패키지를 import하지 않는다.

  core.py       manifest/데이터로더/P×K 샘플러/지표
  models.py     MetricModel(백본+projection), SupCon/Triplet/ArcFace, 체크포인트
  prototypes.py prototype 4종 + 거리 3종 (mahalanobis 공분산 4종) — yaml 선택
  train.py      supcon/triplet/arcface 학습 (none은 학습 생략)
  evaluate.py   closed27_open15(42-way 갤러리) / full42 프로토콜 평가
"""

__all__ = ["core", "models", "prototypes", "train", "evaluate"]

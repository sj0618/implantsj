"""attributes: feature별 모델 + spec table 매칭으로 최종 분류 (노션 5번).

독립 패키지 — 다른 패키지를 import하지 않는다.

  spec_table.py  클래스 → 13 feature 정답표 로딩/매칭 (원본 xlsx CSV 그대로 지원)
  core.py        데이터/백본/모델(shared·separate)/masked CE
  train.py       학습 + feature 확률 추출
  fuse.py        feature 확률 → 클래스 점수 (learned weights / min_match)
"""

__all__ = ["spec_table", "core", "train", "fuse"]

"""fuse: 세 브랜치(metric / classifier / attributes)의 점수를 합쳐 최종 분류 (노션 6번).

독립 패키지 — 다른 패키지를 import하지 않는다.

  core.py     브랜치 점수 CSV 로딩 + 방향/스케일 통일 + 축 정렬
  combine.py  가중 결합, open-set gate, threshold, top-k 평가
"""

__all__ = ["core", "combine"]

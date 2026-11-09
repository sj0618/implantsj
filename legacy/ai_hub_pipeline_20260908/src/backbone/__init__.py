"""backbone: 재사용 가능한 백본 4종을 만드는 스테이지 (노션 1번).

  vit / band-vit / dinov2 / band-dinov2
  band-* 는 band 영역에 attention을 집중시키는 KL prior loss로 학습한다.

독립 패키지 — 다른 패키지를 import하지 않는다.
산출물(분류 head 제외한 백본 weight)은 base.yaml의 backbones 레지스트리를 통해
Stage 2(metric) / Stage 3(classifier)이 끌어다 쓴다.

  core.py   데이터(band mask 동기화)/모델/finetune 범위/attention/저장
  train.py  학습 루프
"""

__all__ = ["core", "train"]

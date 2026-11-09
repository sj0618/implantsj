# 마스크와 Band 생성

아래 순서의 셸 스크립트는 `scripts/02_segmentation/`에 있다. 설정은 `configs/02_segmentation.yaml`이다.

| 순서 | Python 코드 | 입력 → 출력 |
|---|---|---|
| 01 | `sample_finetune_data.py` | 학습 이미지 → 클래스별 샘플 |
| 02 | `generate_sam1_masks.py` | 샘플+SAM1 → 초기 마스크 |
| 수동 | 코드 없음 | 사용할 이미지·마스크 쌍 선별 |
| 03 | `split_seg_dataset.py` | 선별 쌍 → train/val/test |
| 04 | `train_sam2_adapter.py` | 분할 쌍+SAM2 → adapter/decoder checkpoint |
| 05 | `predict_masks.py` | 전체 이미지+checkpoint → 예측 마스크 |
| 06 | `make_band_masks.py` | 예측 마스크 → band mask |

```bash
bash scripts/02_segmentation/01_sample_finetune_data.sh
```

각 단계는 별도로 실행한다. 출력 경로를 새로 지정하고 수동 선별을 생략하지 않는다.
SAM1/SAM2 설치와 원 가중치가 필요하다. 기존 adapter가 있으면 재학습 없이 예측 단계부터 실행할 수 있다.
Band 생성 후 데이터 준비를 다시 실행해 manifest에 mask 경로를 연결한다.

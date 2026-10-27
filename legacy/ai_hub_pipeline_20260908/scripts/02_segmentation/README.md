# Stage 0. Segmentation 파이프라인

설정은 전부 `configs/02_segmentation.yaml`, 코드는 `src/segmentation/`.
원본 코드: `implant_ny/make_seg_model/` (로직 동일, config 기반으로 리팩토링)

## 실행 순서

```
01_sample_finetune_data.sh   ① train data에서 n장 샘플링 (클래스별 라운드로빈)
02_generate_sam1_masks.sh    ② 순정 SAM(vit_h)으로 초기 마스크 생성 + 형태 필터
        │
        ▼  (수동) ③ 잘 생성된 마스크를 사람이 선별
        │         → seg_data/seg_dataset_new/{images,masks}/<클래스>/ 로 1:1 구성
        ▼
03_split_dataset.sh          ④ train/val/test 7:2:1 분할
04_train_sam2_adapter.sh     ⑤ SAM2+Adapter+U-Net 파인튜닝 → best .pth
05_predict_masks.sh          ⑥ 전체 데이터(42클래스) 마스크 예측 + 후처리
06_make_band_masks.sh        ⑦ band(고리) 추출 + manifest에 mask_path 추가
```

⑦의 출력(band mask + manifest)이 band-vit / band-dinov2 학습의 입력이 된다.

## 사전 준비
- SAM1 체크포인트: `sam_vit_h_4b8939.pth`
- SAM2 체크포인트: `sam2.1_hiera_tiny.pt` + sam2 repo 설치 (`pip install -e .`)
- 이미 파인튜닝된 체크포인트가 있으면(⑤ 결과물 `best_sam2.1_tiny_adapter_unet.pth`)
  ①~⑤를 건너뛰고 ⑥부터 실행 가능 — yaml의 `sam2_train.save_path`에 경로 지정

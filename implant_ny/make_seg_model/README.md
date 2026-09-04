# Segmentation 파이프라인 정리본

Downloads/download 폴더에 있던 여러 버전 중, 각 단계별로 실제 채택된 것으로 보이는 파일만 모았습니다.
(실험판/버그판/중복 파일은 제외 — 필요하면 원본 폴더에서 다시 찾을 수 있습니다.)

## 실행 순서

1. **seg_train_split.py** — AI Hub 원본 test셋에서 50장 샘플링 (seg_test_sample_2)
   - ※ train쪽 최초 1000장 샘플링 스크립트(`Seg_train_split_AIhub`)는 Documents 쪽 segmentation 루트 폴더에 있음, 여기엔 없음

2. **SAM1_new.py** — 순정 SAM(vit_h) automatic mask generator로 위 샘플 이미지들에 초기 정답 마스크 자동 생성

3. **seg_dataset_split_code.py** — (이미지, 마스크) 쌍을 train/val/test 7:2:1로 분할 → seg_dataset_split

4. **sam2_adapter_decoder_train.py** — SAM2(hiera-tiny) 인코더 프리즈 + Adapter/UNetDecoder 학습
   → 결과물: **best_sam2.1_tiny_adapter_unet.pth** (동봉)

5. **SAM_mask_produce_util.py** — 4번 체크포인트로 AI Hub 원본 전체(42클래스, data/datas)에 마스크 예측 + 후처리(구멍 메우기, 최대 덩어리만 남기기)
   → 결과: seg_masks_train/*_pred.png

6. **make_band_masks.py** — 5번 예측 마스크에서 임플란트 경계의 band(고리) 영역 추출
   → 이후 band-attention ViT 분류기 학습(supervised_seg.py, 별도 파일)의 입력으로 사용

## 제외한 파일과 이유
- SAM1.py — 리팩터링 중 버그 있음 (image_tensor_permute 오타)
- SAM_mask_produce_util_7label.py / _preprocessing.py — 작은 샘플 대상 실험판 (다른 체크포인트 unet2.pth 사용)
- sam2_adapter_decoder_train_7label.py — 7라벨 통합 데이터셋 실험판 (결과물 unet2.pth)
- make_band_masks copy.py — make_band_masks.py와 완전히 동일한 중복 파일
- best_sam2.1_tiny_adapter_unet2.pth — 위 7label 실험판의 결과물


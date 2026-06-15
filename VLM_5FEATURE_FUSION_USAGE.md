# VLM 5-Feature Fusion Experiment

이 실험은 VLM이 추출한 5개 구조 feature가 implant system classification에 실제 signal을 주는지 확인하기 위한 baseline입니다.

## 1. VLM Feature 추출

브랜드명이나 system class를 묻지 말고 ROI 이미지만 넣어 아래 prompt로 JSON만 저장합니다.

```bash
python train_vlm_fusion_5feature.py --write-prompt-only
```

출력 prompt:

```text
outputs/vlm_feature_prompt_5feature.md
```

권장 VLM 설정:

- temperature: `0`
- 모든 이미지에 동일 prompt 사용
- output은 설명문 없이 structured JSON only

## 2. VLM Feature 파일 형식

JSONL 권장:

```json
{"path": "/workspace/implant_python_only_final/data/7label/train/ADIN/example.jpg", "connection_type": {"value": "internal", "confidence": 0.91}, "platform": {"platform_visible": true, "platform_position_relative_y": 0.18, "platform_switching": "matching", "confidence": 0.82}, "thread": {"pitch": "medium", "depth": "medium", "micro_thread_zone": "present", "macro_thread_zone": "present", "confidence": 0.77}, "outline": {"body_shape": "tapered", "diameter_length_ratio": "medium", "confidence": 0.84}, "image_quality": {"thread_visibility": "clear", "abutment_present": true, "projection_distortion": "mild", "confidence": 0.88}}
```

CSV도 사용할 수 있습니다. `vlm_json` 컬럼에 위 JSON을 넣거나, `connection_type`, `thread_pitch`, `body_shape` 같은 flattened column을 사용할 수 있습니다.

## 3. 전체 실험 Suite 실행

```bash
python train_vlm_fusion_5feature.py \
  --manifest data/manifests/plain_vit_7label.csv \
  --vlm-features data/manifests/vlm_5feature.jsonl \
  --suite all \
  --epochs 40 \
  --batch-size 16
```

`--suite all`에 포함되는 실험:

- A. image-only
- B. VLM feature-only
- C. image + VLM simple sum
- D. image + shuffled VLM feature
- cumulative ablation
- leave-one-out ablation

## 4. 주요 옵션

```bash
--suite core
--suite cumulative
--suite leave_one_out
--suite single
--mode fusion_sum
--mode image_only
--mode vlm_only
--mode shuffled_fusion
--uncertain-confidence-threshold 0.7
--confidence-split-threshold 0.7
--finetune-image-encoder
```

기본값은 image encoder freeze입니다. `--finetune-image-encoder`를 주면 encoder도 함께 학습합니다.

## 5. 출력

각 run directory에 다음 파일이 저장됩니다.

- `metrics/valid_metrics.json`
- `metrics/test_metrics.json`
- `metrics/*_predictions.csv`
- `metrics/*_subgroup_metrics.csv`
- `confusion_matrices/*.csv`
- `confusion_matrices/*.png`
- `vlm_feature_schema.json`
- `vlm_feature_prompt_5feature.md`

suite 실행 시 전체 요약:

```text
outputs/vlm_5feature_fusion_suite_summary/summary.csv
```

판단 기준은 `image_only` 대비 `fusion_sum`의 Macro-F1/Top-3/per-class recall 개선, 그리고 `shuffled_fusion`에서 개선이 사라지는지입니다.

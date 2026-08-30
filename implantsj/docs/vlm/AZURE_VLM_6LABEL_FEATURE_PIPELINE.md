# Azure OpenAI VLM 5-feature pipeline for 6-label data

이 파이프라인은 6-label 이미지 전체를 Azure OpenAI vision deployment로 전송하고, 이미지별 구조 feature 5개와 숫자 feature vector를 JSONL로 저장한다.

## 1. Azure 설정

`.env.azure.example`을 참고해 실제 값은 `.env`에 넣는다. `.env`는 git에 올리지 않는다.

```bash
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE-NAME.openai.azure.com/
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=YOUR_VISION_CHAT_DEPLOYMENT_NAME
```

`AZURE_OPENAI_DEPLOYMENT`에는 Azure Portal에서 만든 vision-capable chat model deployment 이름을 넣는다.

## 2. Prompt 확인

```bash
python generate_vlm_features_azure_6label.py \
  --output data/manifests/vlm_5feature_6label.jsonl \
  --write-prompt-only
```

## 3. Dry run

실제 API를 호출하지 않고 manifest와 처리 대상만 확인한다.

```bash
python generate_vlm_features_azure_6label.py \
  --manifest data/manifests/plain_vit_6label.csv \
  --dry-run
```

manifest가 아직 없고 class-folder 또는 split/class-folder 데이터에서 만들려면:

```bash
python generate_vlm_features_azure_6label.py \
  --data-root /workspace/data/large_multiclass \
  --manifest data/manifests/plain_vit_6label.csv \
  --labels CLASS1 CLASS2 CLASS3 CLASS4 CLASS5 CLASS6 \
  --rebuild-manifest \
  --dry-run
```

## 4. 전체 데이터 처리

```bash
python generate_vlm_features_azure_6label.py \
  --manifest data/manifests/plain_vit_6label.csv \
  --output data/manifests/vlm_5feature_6label.jsonl \
  --summary-csv data/manifests/vlm_5feature_6label.summary.csv
```

중간에 작업이 끊겨도 기본값인 `--resume`이 이미 `status=ok`로 저장된 이미지를 건너뛴다.

## 5. 출력 JSONL

한 줄에 이미지 하나를 기록한다.

```json
{
  "path": "...jpg",
  "label": "osstem",
  "split": "train",
  "status": "ok",
  "feature": {
    "connection_type": {"value": "internal", "confidence": 0.82},
    "platform": {"platform_visible": true, "platform_position_relative_y": 0.21, "platform_switching": "matching", "confidence": 0.74},
    "thread": {"pitch": "medium", "depth": "medium", "micro_thread_zone": "present", "macro_thread_zone": "present", "confidence": 0.69},
    "outline": {"body_shape": "tapered", "diameter_length_ratio": "medium", "confidence": 0.78},
    "image_quality": {"thread_visibility": "clear", "abutment_present": true, "projection_distortion": "mild", "confidence": 0.86}
  },
  "vlm_feature_names": ["connection.connection_type=external", "..."],
  "vlm_feature_vector": [0.0, 1.0, 0.0, "..."],
  "mean_vlm_confidence": 0.778
}
```

생성한 파일은 fusion 실험에 바로 사용할 수 있다.

```bash
python train_vlm_fusion_5feature.py \
  --manifest data/manifests/plain_vit_6label.csv \
  --vlm-features data/manifests/vlm_5feature_6label.jsonl \
  --suite all
```

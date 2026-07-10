# Remote qwen_3B_VLM 실행 핸드오프

## 목적

이 문서는 로컬에서 수정한 `pipelines/vlm/generate_13feature_21reason_7label.py`를
원격 서버에서 `qwen_3B_VLM`으로 실제 실행할 때 필요한 최소 정보만 정리한 문서다.

현재 원칙:

- 목표는 **전체 42-label 데이터에 13feature 추가**
- `unseen15` 재학습은 범위 밖
- 로컬 머신에는 실제 VLM 모델이 없음
- 실제 추론은 **원격 서버**에서 수행
- 구현 경로는 오직 **in-script local transport/provenance lift**
- wrapper / 별도 local_qwen runner 재도입은 하지 않음

## 현재 코드가 기대하는 로컬 런타임 형태

현재 반영된 로컬 런타임 전제는 다음과 같다.

- Python API / library 사용
- `transformers` 계열
- `AutoProcessor + model.generate(...)` 호출
- 이미지 입력: **파일 경로 문자열**
- 프롬프트 입력: **일반 문자열**
- 모델 출력: **텍스트**
- 텍스트를 `json.loads(...)` 해서 dict로 변환

## 원격 서버에서 확인해야 할 필수 항목

실행 전에 아래만 확인하면 된다.

1. 실제 **로컬 모델 디렉터리 또는 이미 캐시된 모델 경로**
   - 현재 코드는 `local_files_only=True`로 model / processor를 로드하므로,
     단순 원격 모델 식별자 문자열만으로는 부족할 수 있다.
   - 예: `/models/Qwen2.5-VL-3B-Instruct`
2. `transformers` 설치 여부
3. `qwen_vl_utils` 설치 여부
4. `Pillow` 설치 여부
5. `torch` / CUDA 환경
6. 이미지 manifest 경로
7. 출력 디렉터리 경로

## 현재 추가된 실행 인자

`generate_13feature_21reason_7label.py`에 아래 인자가 추가되었다.

- `--local-model-path`
- `--local-torch-dtype`
- `--local-device-map`

환경변수도 사용 가능하다.

- `LOCAL_QWEN_MODEL_PATH`
- `QWEN_3B_VLM_MODEL_PATH`

## 원격 서버 subset 실행 예시

```bash
python3 pipelines/vlm/generate_13feature_21reason_7label.py \
  --input /path/to/closed_network_out/manifests/closed_network_42label.csv \
  --path-column path \
  --expected-label-count 42 \
  --output /path/to/nontracked_outputs/closed_network_42label_local_qwen13.jsonl \
  --json-output /path/to/nontracked_outputs/closed_network_42label_local_qwen13.json \
  --summary-csv /path/to/nontracked_outputs/closed_network_42label_local_qwen13.summary.csv \
  --prompt-output /path/to/nontracked_outputs/closed_network_42label_local_qwen13.prompt.md \
  --local-model-path /path/to/qwen_3B_VLM \
  --local-torch-dtype auto \
  --local-device-map auto \
  --workers 1 \
  --max-images 10 \
  --resume
```

## 원격 서버 full 실행 예시

```bash
python3 pipelines/vlm/generate_13feature_21reason_7label.py \
  --input /path/to/closed_network_out/manifests/closed_network_42label.csv \
  --path-column path \
  --expected-label-count 42 \
  --output /path/to/nontracked_outputs/closed_network_42label_local_qwen13.jsonl \
  --json-output /path/to/nontracked_outputs/closed_network_42label_local_qwen13.json \
  --summary-csv /path/to/nontracked_outputs/closed_network_42label_local_qwen13.summary.csv \
  --prompt-output /path/to/nontracked_outputs/closed_network_42label_local_qwen13.prompt.md \
  --local-model-path /path/to/qwen_3B_VLM \
  --local-torch-dtype auto \
  --local-device-map auto \
  --workers 1 \
  --resume
```

## 원격 실행 후 반드시 볼 것

성공 기록에서 아래를 확인한다.

- `provider == "qwen_3B_VLM"`
- `local_model_path` 존재
- `azure_deployment` 가 남아 있지 않음
- `status`
- `feature.api_status`
- `feature.feature_status.value`
- `feature.failure_reason`
- `feature.usable_for_feature_vector`
- `mean_confidence`
- `path`, `label`, `split`

## resume 재실행 확인

같은 subset 명령을 다시 돌렸을 때 아래를 확인한다.

- 이미 성공한 `path`는 skip 되는지
- 성공 row가 중복 append 되지 않는지
- provenance가 계속 `qwen_3B_VLM` 으로 남는지

## 산출물

원격 실행이 끝나면 최소 산출물은 아래 3개다.

- JSONL
- JSON
- summary CSV

## 비고

- 현재 로컬 머신에서는 모델이 없으므로 실제 추론 검증은 하지 않는다.
- 로컬에서는 코드/테스트/프롬프트/manifest 경로까지만 검증한다.
- 실제 모델 추론 성공 여부는 원격 서버 결과로 판단한다.

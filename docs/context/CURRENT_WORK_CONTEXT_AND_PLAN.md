# 현재 작업 맥락과 진행 계획

> 과거 작업 기록입니다. 실행 상태·경로·우선순위는 작성 당시 기준입니다. [문서 목차](../README.md)에서 현재 실행 가이드를 확인하세요.

> 사용자 직접 지시사항 자체는 `USER_DIRECTIVES_AND_REPO_RULES.md`에 별도로 정리했다.

## 1. 현재 작업 목적

지금 우선 목표는 다음 하나다.

> 폐쇄망 Ubuntu 환경에서 기존 코드 구조를 최대한 재사용하여, 전체 데이터에 대해 13feature를 추가하는 것

즉,

- `unseen15` class를 새로 학습시키는 것이 현재 1순위가 아니고
- 로컬 VLM을 사용하여 전체 데이터에 13feature를 붙이는 것이 핵심이다.

## 2. 환경 제약

현재 작업 환경 제약은 다음과 같다.

- 폐쇄망 Ubuntu 환경
- 코드 복사/붙여넣기 불가
- 긴 새 코드 작성이 어려움
- 따라서 타이핑량이 적어야 함
- 가능한 한 기존 함수 / Dataset / Model / transform / checkpoint loading / evaluation 로직을 재사용해야 함
- 이미 구현된 코드와 중복되는 코드는 새로 만들지 않는 것이 원칙

## 3. 작업 방식 원칙

대화 기준으로 정리된 원칙은 아래와 같다.

1. 기존 파일 구조와 기존 코드를 먼저 확인한다.
2. 새 파일을 길게 만들지 않는다.
3. 원본 파일을 임의로 수정하지 않는다.
4. 수정이 필요하면 전체 수정안을 먼저 보여주고 승인 후 적용한다.
5. 가능하면 원본을 건드리지 않고 wrapper script 또는 기존 실행 경로 재사용을 우선한다.
6. 직접 타이핑해야 하는 코드와 명령어는 최대한 짧게 유지한다.

## 4. 현재 레포에서 재사용할 수 있는 핵심 코드

### 데이터 준비

- `pipelines/data/prepare_closed_network_dataset.py`
  - TL1 / TS1 / VL1 / VS1 구조를 읽어 manifest를 생성
- `src/closed_network_data.py`
  - 이미지-JSON pair 탐색
  - JSON의 `모델명` 라벨 읽기
  - train / valid / test split 생성
  - manifest 저장

### 13feature 스키마 / 후처리

- `pipelines/vlm/generate_13feature_21reason_7label.py`
  - 13 core attributes + 21 supporting features prompt 정의
  - JSON 파싱
  - validation / completion / summary 로직 포함
- `docs/vlm/13feature_21reason.md`
  - 13feature 의미와 구조 설명

### metric 결과 해석

- `pipelines/eval/evaluate_metric_transfer.py`
  - seen metric checkpoint를 다른 manifest에 적용하는 평가 코드
- `src/metric_learning.py`
  - embedding 추출, prototype prediction, metric 평가 로직

## 5. 현재 정리된 사실

### unseen15 관련

- unseen15 성능이 낮은 것은 현재 구조상 충분히 예상 가능하다.
- 이 결과는 모델 오류보다 **실험 정의의 한계**를 먼저 의심하는 것이 맞다.
- 따라서 unseen15 평가는 분류 정확도보다 **seen space에서의 mapping 분석**으로 보는 것이 타당하다.

### 13feature 관련

- 현재 하고 싶은 일은 전체 데이터에 13feature를 추가하는 것이다.
- 이 목적에는 metric learning 재학습이 필수가 아니다.
- 먼저 전체 데이터 manifest를 만들고, 그 manifest 전체에 대해 13feature를 생성하면 된다.

## 6. local_qwen 관련 현재 상태

중요:

- `run_local_qwen_vlm.py`는 현재 레포에서 제거되었다.
- 요청에 따라 로컬 Qwen 전용 실행 파일은 로컬 저장소에서도 삭제했다.

따라서 현재 레포 상태는 다음과 같다.

- 13feature의 **prompt / schema / validation / summary 로직**은 남아 있음
- 하지만 폐쇄망 전용 로컬 Qwen 실행 진입점은 현재 없음

즉, 폐쇄망에서 13feature를 실제로 돌리려면 아래 두 방식 중 하나가 필요하다.

1. 폐쇄망 내부에 이미 있는 별도 local VLM 실행 스크립트를 사용
2. 나중에 승인 후, 기존 13feature 로직을 재사용하는 최소 wrapper를 다시 추가

현재 문서화 기준에서는 **원본 로직 재사용이 우선**이고, 중복 구현은 피해야 한다.

## 7. 지금 해야 할 실제 작업 순서

### 1단계. 전체 manifest 준비

목표:

- 전체 데이터를 하나의 42label manifest로 정리

재사용 코드:

- `pipelines/data/prepare_closed_network_dataset.py`

입력:

- `TL1`: train json
- `TS1`: train image
- `VL1`: test json
- `VS1`: test image

출력:

- `closed_network_42label.csv`

### 2단계. 전체 manifest 기준으로 13feature 생성

목표:

- train / valid / test를 포함한 전체 데이터 row마다 13feature 생성

재사용 코드:

- `pipelines/vlm/generate_13feature_21reason_7label.py`

필요 사항:

- 폐쇄망에서 실제로 돌릴 수 있는 local VLM 호출 진입점
- 기존 prompt, JSON schema, summary 로직 재사용

### 3단계. 결과 저장

최소 산출물:

- JSONL
- JSON
- summary CSV

### 4단계. 이후 선택 작업

13feature가 안정적으로 생성된 뒤에만 아래를 검토한다.

- ViT feature + VLM feature 결합
- confidence filtering
- partial usable / low confidence 처리
- unseen mapping 분석과 결합

## 8. 최소 타이핑 원칙에 맞는 실행 전략

현재 제약상 가장 좋은 방식은 다음이다.

- 긴 새 코드 작성 금지
- 기존 스크립트의 인자만 바꿔 실행
- 원본 소스 수정 최소화
- 필요하면 매우 짧은 wrapper 하나만 별도로 두고 실행
- 가능하면 한 줄 명령 또는 몇 줄 이내 명령으로 끝내기

## 9. 지금 문서 기준 핵심 결론

1. `unseen15` 저성능은 실험 구조상 자연스러운 결과다.
2. 현재 1순위는 `unseen15` 재학습이 아니라 **전체 데이터 13feature 추가**다.
3. 현재 레포에는 local Qwen 전용 진입점이 없다.
4. 따라서 폐쇄망 실행은 현지 실행 경로를 사용하거나, 승인 후 최소 wrapper를 다시 만드는 방향이 맞다.
5. 어떤 경우에도 기존 13feature 스키마와 후처리 로직은 재사용하는 것이 맞다.

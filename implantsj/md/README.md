# implantsj 작업 문서 모음

이 폴더는 현재 `implantsj`에서 진행 중인 작업 맥락을 짧게 다시 잡기 위한 정리 문서 모음이다.

## 문서 목록

- `USER_DIRECTIVES_AND_REPO_RULES.md`
  - 이번 대화에서 사용자가 직접 지시한 작업 원칙, 승인 규칙, 문서화 요구사항 정리
  - 현재 레포 작업 시 따라야 할 우선순위와 제약사항 명시

- `UNSEEN15_METRIC_RESULT_INTERPRETATION.md`
  - 27 seen class로 학습한 metric model을 15 unseen class에 적용했을 때 성능이 낮게 나오는 이유 정리
  - 보고서용 문장 포함

- `CURRENT_WORK_CONTEXT_AND_PLAN.md`
  - 지금까지 대화 기준 작업 목적, 제약사항, 재사용할 코드, 실제 해야 할 일 정리
  - 폐쇄망 Ubuntu 환경, 최소 타이핑, 기존 코드 재사용 원칙 반영
- `REMOTE_QWEN_3B_VLM_EXECUTION_HANDOFF.md`
  - 로컬에서 수정한 13feature 코드를 원격 서버의 `qwen_3B_VLM` 런타임으로 실행할 때 필요한 인자, 명령어, 검증 포인트 정리

## 현재 정리

1. `unseen15` 결과가 낮은 주된 이유는 코드 오류보다 실험 설정에 가깝다.
2. 현재 우선 목표는 `unseen15` 학습이 아니라 전체 데이터에 13feature를 추가하는 것이다.
3. 로컬 Qwen 전용 실행 파일(`run_local_qwen_vlm.py`)은 현재 레포에서 제거되었다.
4. 폐쇄망에서는 현지에 이미 있는 로컬 VLM 실행 방식을 사용하거나, 필요하면 별도 wrapper를 다시 설계해야 한다.
5. 가능하면 기존 `prepare_closed_network_dataset.py`, `closed_network_data.py`, `generate_13feature_21reason_7label.py`의 구조를 재사용한다.

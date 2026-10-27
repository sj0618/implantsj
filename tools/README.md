# 실험 재현 도구

| 코드 | 역할 |
|---|---|
| `run.py` | 기존 명령과 `reproduce` 명령 연결 |
| `reproduce_reports.py` | 설정에 따른 입력 검사·학습·평가·실행 기록 |
| `improve_unseen_mahalanobis.py` | Ledoit–Wolf/OAS/RDA/PCA·다중 prototype·스펙트럼 거리 후보 계산 |
| `validate_mahalanobis_methods.py` | validation으로 방법 선택 후 test·strict 5-shot 평가 |

## 실행

저장소 루트에서 실행한다. 목록·계획·경로 검사에는 Python 표준 라이브러리만 필요하다.
학습·평가에는 `requirements.txt`와 해당 PyTorch 환경이 필요하다.

```bash
python3 tools/run.py reproduce list
python3 tools/run.py reproduce plan --report target5_episodic5
python3 tools/run.py reproduce check --report target5_episodic5 --inputs inputs.local.json
python3 tools/run.py reproduce run --report target5_episodic5 --inputs inputs.local.json --output outputs/reproduction/run01
```

입력 JSON은 [예시](../configs/reproduction/inputs.example.json)를 참고해 필요한 경로만 지정한다.
원 보고서 문서는 출처 정보이며 실행 입력이 아니다.

| 명령 | 동작 |
|---|---|
| `list` | 24개 항목의 범위·단계 수 출력 |
| `plan` | 명령·필수 입력 출력; 학습하지 않음 |
| `check` | 입력 경로·미확정 조건 검사; 데이터 동일성 검증은 아님 |
| `run` | 새 폴더에서 실행; 실패·산출물 누락 시 중단 |

`--report all`은 미확정 조건이나 누락 입력이 있으면 어떤 학습도 시작하지 않는다.
학습에서 생성한 checkpoint를 다음 평가에 전달하며 원래 manifest는 덮어쓰지 않는다.
`run.json`에 명령, 입력·소스 해시, 패키지 버전과 성공·실패를 기록한다.
`executed`는 실행·산출물 확인이며 과거 수치 일치 판정이 아니다.

## 개별 도구

```bash
python3 -m tools.improve_unseen_mahalanobis --help
python3 -m tools.improve_unseen_mahalanobis --self-check
python3 -m tools.validate_mahalanobis_methods --help
```

첫 도구의 기본 경로는 과거 로컬 구조를 가리킨다. 새 작업에는 실행기나 명시적 입력·출력 인자를 사용한다.
두 번째 도구는 네 cache, strict cache와 seed별 support/query 배정 파일이 필요하다.
방법 선택에는 validation만 사용하며 test 점수로 방법을 바꾸지 않는다.
[평가 규격](../pipelines/eval/README.md)과 [테스트](../tests/README.md)를 참고한다.

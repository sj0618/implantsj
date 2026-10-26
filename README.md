# implantsj

임플란트 이미지 분류, SupCon 학습, unseen 클래스 평가를 위한 실험 코드입니다.
폐쇄망 GPU 서버에서는 다음 메뉴로 실행합니다.

```bash
./run.sh
```

- 처음 실행: 환경 확인 → 데이터 변환 → 데이터셋·모델 선택 → 학습 → 평가
- 기본 설정: [`exp.conf`](exp.conf)
- 모델 목록: [`configs/backbones.txt`](configs/backbones.txt)
- 반입·실행 절차: [시작 가이드](docs/usage/QUICKSTART.md)
- 전체 문서: [문서 목차](docs/README.md)
- 보고서별 재현: [실행 방법·재현 범위](docs/usage/REPRODUCE_REPORTS.md), `python3 tools/run.py reproduce list`

## 코드 구조

| 경로 | 역할 |
| --- | --- |
| `run.sh`, `tools/run.py` | 실행 메뉴와 명령 연결 |
| `src/` | 데이터, 모델, 학습, 평가 공용 코드 |
| `pipelines/` | 데이터 준비·학습·평가·VLM·시각화 실행 코드 |
| `two_stage/` | 혼동 클래스 재분류 실험 |
| `configs/`, `exp.conf` | 라벨·모델·실험 설정 |
| `docker/`, `Dockerfile` | 폐쇄망 반입용 이미지 빌드·실행 |
| `tests/` | 회귀 테스트와 smoke test |
| `docs/` | 실행 가이드, 분석, 결과, 과거 작업 기록 |
| `legacy/` | 이전 코드와 배포 압축본 보관 |

## 명령과 검사

저장소 루트에서 실행합니다.

```bash
./run.sh --help
python3 tools/run.py --help
python3 -m pytest -q tests
```

`pipelines/`의 개별 실행 파일은 `python3 -m pipelines.<분류>.<모듈>` 형식을 사용합니다.
실험별 기본값과 라벨 수가 다르므로 각 명령의 `--help`를 확인합니다.
모델 가중치와 데이터는 별도 보관하며, 과거 보고서의 수치는 현재 코드의 재검증 결과와 구분합니다.

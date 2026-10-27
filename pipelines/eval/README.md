# 재현 평가 코드

| 코드 | 역할 |
|---|---|
| `reproduce_gallery.py` | gallery·strict·episodic 평가, seed별 support·예측·지표 저장 |
| `evaluate_3label_twostage_metric_from_7label.py` | two-stage 평가; `--use-existing-manifest`로 원 분할 보존 |

기존 평가 파일의 옵션은 각 모듈의 `--help`를 참고한다.

```bash
python3 -m pipelines.eval.reproduce_gallery --help
python3 -m pipelines.eval.evaluate_3label_twostage_metric_from_7label --help
```

Checkpoint+manifest 또는 NPZ cache를 입력받는다.
Cache는 `labels`와 각 split의 `train_x/train_y/train_paths`,
`valid_x/valid_y/valid_paths`, `test_x/test_y/test_paths`가 필요하다.
`x`는 N×D, `y`는 labels의 정수 인덱스, paths는 문자열이며 pickle은 사용하지 않는다.

| 조건 | Prototype | 공분산 | Query |
|---|---|---|---|
| gallery | train 평균 / valid GT / train SS | 지정한 train 전체 또는 support | valid/test; GT는 test만 |
| strict | 클래스당 지정 장수 SS | 같은 support만 | 고정 test |
| episodic | 전체 pool에서 지정 장수 SS | 같은 support만 | support를 뺀 나머지 |

Strict/episodic은 고정 방법만 평가한다. Raw/L2, 공분산 범위, seed와 표본 순서를 명시한다.
경로·이미지 해시 중복을 검사하지만 환자 단위 독립성은 별도 확인한다.

새 출력 폴더에 지표·예측·표본 배정 CSV, 혼동행렬, `protocol.json`,
`embedding_cache.npz`, 현재 실행의 `REPORT.md`를 저장한다. 산출물은 Git에 올리지 않는다.

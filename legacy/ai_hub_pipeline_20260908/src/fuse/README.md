# 분류 결과 결합

| 코드 | 역할 |
|---|---|
| `resolve.py` | metric·classifier·attributes 결과 파일 탐색 |
| `core.py` | 점수 입력·정답·정규화 공용 처리 |
| `combine.py` | 가중치 학습·open-set gate·threshold·평가 |
| `visualize.py` | 결합 가중치·오류·점수 시각화 |
| `__init__.py` | 패키지 연결 |

```bash
python src/run_fuse.py --config /path/to/run_fuse.yaml
```

각 분류 경로의 valid/test 점수 CSV와 정답 manifest를 먼저 준비한다.
가중치와 임계값은 valid에서 정하고 test에 적용한다.
서로 다른 입력 집합이면 공통 표본만 평가하므로 평가 장수와 클래스 범위를 확인한다.
결합 점수·예측·지표와 설정에 따른 그림을 새 출력 경로에 저장한다.

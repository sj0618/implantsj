# 재현 코드 테스트

```bash
python3 -m pytest -q tests
python3 -m tools.improve_unseen_mahalanobis --self-check
```

| 파일 | 검사 범위 |
|---|---|
| `test_reproduce_reports.py` | 설정·CLI·입력·출력·실패·checkpoint 전달·문서 없는 실행 |
| `test_reproduce_gallery.py` | 표본 분리·이미지 중복·정규화·seed·거리·입력 규격 |
| `test_archived_server_source.py` | 전체 코드 포함·셸 문법·단계 연결·OAS·K-means·빌드 입력 |

서버 원본 CPU 테스트는 해당 디렉터리에서 별도로 실행한다.

```bash
cd legacy/ai_hub_pipeline_20260908
python src/verify_pipeline.py
python tests/test_smoke.py
```

합성 데이터와 작은 모델을 사용한다. GPU 학습·SAM 추론·원본 데이터 성능을 입증하지 않는다.

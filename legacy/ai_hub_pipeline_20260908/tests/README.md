# 원본 CPU 테스트

`test_smoke.py`는 합성 이미지와 사전학습하지 않은 작은 모델로 다음을 검사한다.

- split별 backbone/head 임베딩 저장과 해시·메타데이터.
- 설정에서 임베딩 추출 계획 생성.
- attributes separate/shared 학습·평가·checkpoint 재사용과 head 변경.

```bash
python tests/test_smoke.py
python src/verify_pipeline.py
```

프로젝트 루트에서 실행한다. 실제 가중치를 다운로드하지 않는다.
SAM·GPU 추론 및 실제 데이터 성능은 이 테스트의 범위가 아니다.

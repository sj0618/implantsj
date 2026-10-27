# 재현 실행 설정

- `reports.json`: 24개 항목, 57개 단계의 모듈·인자·필수 입력·산출물.
- `inputs.example.json`: manifest·checkpoint·cache·표본 배정 파일 경로 예시.
- 실제 경로는 Git 제외 파일 `inputs.local.json`에 필요한 항목만 넣는다.
- 원문·과거 점수는 업로드하지 않는다. `source`는 로컬 보고서의 출처 식별자다.

```bash
python3 tools/run.py reproduce plan --report strict_3label_10shot
python3 tools/run.py reproduce check --report strict_3label_10shot --inputs inputs.local.json
```

| 범위 | 의미 |
|---|---|
| `protocol` | 기록된 평가 방식 제공; 원본 수치 일치는 미확인 |
| `partial` | 일부 단계 제공; 나머지는 limitations에 명시 |
| `reconstruction` | 부족한 기록에서 재구성한 조건 |
| `blocked` | 실험 정의가 부족해 임의 실행 명령을 만들지 않음 |

May-26 fine-tuning, GT27, 서버 OAS, 서버 백본 비교, VLM 처리 현황에는 당시 기록이 추가로 필요하다.
서버 코드는 [전체 복구본](../../legacy/ai_hub_pipeline_20260908/README.md)을 제공한다.
VLM 처리 현황을 속성 분류 모델이나 새 API 호출 결과로 대체하지 않는다.

Manifest는 `path,label,split` CSV이며 split은 `train,valid,test`다.
원래 표본·순서·분할을 유지한다. [Cache·평가 규격](../../pipelines/eval/README.md)을 참고한다.

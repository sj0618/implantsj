# Mahalanobis spectrum 검증 결과

작성일: 2026년 8월 16일 일요일

## 판정

`ALL-IN-ONE COMPLETE`

Mahalanobis spectrum 작업은 계획, 구현, Ralph 검증, UltraQA를 모두 마쳤다. 작업 범위는 PASS로 판정됐고, UltraQA도 4 cycle 뒤 독립 test-engineer의 PASS를 받았다. 저장소 전체가 green인 것은 아니다. 범위 밖 기존 실패 2건은 남아 있다.

결과는 조건에 따라 달랐다. `centered_angular_0.5`는 fresh 3-label cache의 accuracy를 높였지만, legacy cache와 strict 5-label 5-shot에서는 pooled Ledoit-Wolf를 넘지 못했다. fresh backbone의 balanced accuracy도 소폭 낮아졌다. 따라서 이 방법을 새 기본값으로 바꿀 근거는 없다.

## 계획 심사

Stage 1 Architect 심사는 `azure-openai/gpt-5.6-luna` 기반 Codex v1 에이전트가 승인했다. 전용 Critic 역할은 배포 정책 때문에 세 차례 완료되지 않았다. 이후 같은 Azure Luna 기반의 독립 Codex v1 에이전트가 Critic 기준을 순서대로 검토해 승인했다. 이번 심사에 사용한 모델 식별자는 `openai/gpt-5.6-luna`가 아니라 `azure-openai/gpt-5.6-luna`다.

승인된 작업 범위는 공유 scorer factory에 네 실험 방법을 추가하고, 기존 자동 선택 경로와 pooled baseline은 그대로 두는 것이었다.

- `fractional_0.5`
- `fractional_0.75`
- `centered_angular_0.5`
- `centered_angular_0.75`

## 구현과 회귀 방지

`tools/improve_unseen_mahalanobis.py`의 `candidate_scorers()` 기본값은 `include_experimental=False`다. 새 방법은 `--fixed-method`를 명시한 CLI와 spectrum validator에서만 열거된다. 기존 자동 후보 탐색에서는 새 방법을 보지 않으므로 후보 집합과 선택 결과가 바뀌지 않는다.

새 scorer는 행 단위 L2 정규화, pooled Ledoit-Wolf residual covariance, `np.linalg.eigh`, 상대 eigenvalue floor를 사용한다. 같은 정규화 데이터와 같은 floor를 사용하면 fractional alpha=1 score는 Mahalanobis score의 `lambda_bar` 상수배가 된다. score 값은 달라도 class argmin은 같다. self-check는 이 관계와 zero query, empty query, 고차원 저표본 입력, alpha별 score 차이를 검사한다.

CLI에서도 네 fixed method가 모두 정상 동작했다. unknown, empty, whitespace method는 모두 명시적으로 실패했다. no-fixed CLI 네 조건은 pre-edit golden과 `atol=1e-12` 안에서 같았다. 기존 방법 선택 검증의 winner는 `rda_0.75_logdet_1`로 유지됐고 공통 후보는 24개였다.

## 검증 protocol

3-label 조건에서는 seed `0, 1, 2, 3, 4, 42`마다 train split에서 class당 support 10개를 뽑았다. 같은 30개 support로 prototype과 covariance를 만들고 전체 test split을 query로 사용했다.

strict 조건은 5-label 5-shot이다. seed마다 support 25개와 query 127개를 사용했다. 이 cache를 만든 10-shot 절차와 실제 평가에 사용한 저장된 5-shot episode는 서로 다른 protocol이다. 하네스는 두 protocol을 구분해 기록하고, 5-shot membership을 다시 만든 뒤 저장된 path와 label이 같은지 확인한다.

fresh cache에는 train과 test가 content 기준으로 겹치는 이미지가 5건 있었다. 그래서 전체 행 결과와 이 5건을 제외한 duplicate-free 결과를 함께 제시했다. legacy cache에서 제외한 이미지는 없다.

최종 증거는 all-row 150행, duplicate-free 120행, 총 270행과 membership 870행이다. 하네스는 독립적으로 두 번 생성한 결과를 byte 단위로 비교한다. 기존 final이 불완전하거나 새 결과와 다르면 파일을 덮어쓰지 않고 실패한다.

## 결과

아래 수치는 6개 seed 평균이다. 변화량은 pooled Ledoit-Wolf와 비교한 accuracy 차이다.

| 조건 | pooled accuracy | 최고 실험 accuracy | 변화 | 함께 봐야 할 결과 |
|---|---:|---:|---:|---|
| fresh backbone, all rows | 69.36% | centered angular 0.5, 74.22% | +4.86%p | macro-F1 +2.84%p, balanced accuracy -0.66%p |
| fresh projection, all rows | 66.18% | centered angular 0.5, 69.92% | +3.74%p | macro-F1 +2.27%p, balanced accuracy +0.73%p |
| legacy backbone | 38.99% | fractional 0.75, 38.07% | -0.92%p | 네 실험 방법의 accuracy가 모두 낮음 |
| legacy projection | 39.30% | centered angular 0.75, 37.77% | -1.53%p | 네 실험 방법의 accuracy가 모두 낮음 |
| strict 5-label 5-shot | 67.98% | fractional 0.75, 67.59% | -0.39%p | centered angular 0.5는 66.67%, pooled 대비 -1.31%p |

duplicate-free view에서도 fresh backbone의 centered angular 0.5는 74.23%, pooled는 69.43%였다. fresh projection은 각각 69.99%와 66.36%였다. 중복 5건을 제외해도 fresh 조건의 accuracy 개선 방향은 유지됐다.

## 67.06% 문구 확인

원문에는 "the best angular candidate reaches 67.06% accuracy"라고 적혀 있었지만 aggregation 공식은 없었다. 이번 검증에서는 두 angular 방법의 6-seed 평균 중 큰 값을 비교값으로 사용했다. 관측값은 66.67%로 67.06%보다 0.39%p 낮았다. seed별 최댓값을 다시 평균한 수치는 아니다.

따라서 67.06%는 지정된 네 방법과 이번 strict protocol에서는 재현되지 않았다.

## UltraQA

UltraQA는 정상 경로뿐 아니라 다시 서명한 위조 산출물도 검사했다. 다음 입력과 상태를 모두 거부하거나 기대한 결과로 처리했다.

- 필수 key가 없거나 NaN, Inf가 들어간 cache
- strict membership 누락과 path 또는 label 불일치
- 검증 anchor 밖의 경로
- incomplete final과 내용이 다른 stale final
- unknown, empty, whitespace fixed method
- zero query와 empty query
- 269행 metrics, 변조 membership, 잘못된 regression flag
- blank, NaN, 숫자, 문자열로 위조한 boolean 열

파일 hash만 새로 맞추는 것으로는 검증을 통과할 수 없었다. 하네스가 원본 cache와 episode에서 membership을 다시 만들고 metric, delta, regression flag, best flag를 다시 계산하기 때문이다. UltraQA 최종 판정은 `task-specific PASS after 4 cycles`다.

## 남은 테스트 실패

`python -m pytest -q tests` 결과는 `26 passed, 2 failed`다.

1. `tests/test_dependencies.py`는 작업 전부터 수정 상태였던 `tools/setup/dependencies.py`의 `safetensors` 계약과 맞지 않는다.
2. `tests/test_metric_diagnosis.py`는 작업 전부터 수정 상태였던 evaluator에 없는 `content_audit`을 호출한다.

두 실패는 Mahalanobis 경로와 연결되지 않는다. 관련 파일이 사용자 변경 상태였으므로 이번 작업에서는 덮어쓰지 않았다. 따라서 보고서의 PASS는 작업 범위에만 적용된다.

## 해석의 한계

이번 결과는 이미 살펴본 cache에서 후보를 고른 뒤 얻은 exploratory post-selection evidence다. fresh 조건의 개선은 재현됐지만 legacy와 strict 조건에서는 확인되지 않았다. 새로 고정한 데이터나 nested selection protocol 없이 confirmatory 성능 개선 또는 보편적 우위를 주장해서는 안 된다.

## 산출물과 무결성

- 검증 디렉터리: `docs/reports/mahalanobis_spectrum_validation_20260816/`
- QA 보고서: `.omx/qa/mahalanobis-spectrum-ultraqa.md`
- manifest SHA-256: `ae1600d9b4dfef3873e0c54d740b6cb85dd3e55ec3eb8004d9a5608e1b869fd9`
- source post SHA-256: `ea221ffbd401423fa3e308778c51a8ef40680c67a3691c0b346071b8c4b5782a`
- harness SHA-256: `7c43316de73137b55e9b0633fe06630b2fab36a30a524876dcc0a203a560c297`

Git commit은 만들지 않았다. 기존 dirty worktree의 관련 없는 파일도 reset, clean, revert하지 않았다.

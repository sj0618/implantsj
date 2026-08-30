---
title: "Mahalanobis Spectrum Validation Contracts"
tags: ["mahalanobis", "spectrum", "validation", "few-shot", "opt-in", "duplicate-free", "post-selection", "semantic-validation", "ultraqa"]
created: 2026-08-16T20:33:46.660Z
updated: 2026-08-16T20:33:46.660Z
sources: ["tools/improve_unseen_mahalanobis.py", "tools/validate_mahalanobis_spectrum.py", "docs/reports/MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md", ".omx/qa/mahalanobis-spectrum-ultraqa.md", "docs/reports/mahalanobis_spectrum_validation_20260816/SUCCESS.json"]
links: []
category: pattern
confidence: high
schemaVersion: 1
---

# Mahalanobis Spectrum Validation Contracts

2026년 8월 16일 Mahalanobis spectrum 실험에서 확인한 재사용 가능한 계약과 함정을 정리한다.

## Experimental scorer는 명시적으로만 열거한다

`tools/improve_unseen_mahalanobis.py`의 `candidate_scorers(..., include_experimental=False)`는 실험 방법을 기본 후보 집합에서 제외한다. `fractional_0.5`, `fractional_0.75`, `centered_angular_0.5`, `centered_angular_0.75`는 fixed-method CLI나 검증 하네스가 명시적으로 opt-in할 때만 보인다.

이 경계가 필요한 이유는 factory를 공유하는 기존 자동 ranking caller가 새 후보를 갑자기 평가하거나 선택하지 않게 하기 위해서다. 실험 scorer를 production factory에 한 번만 구현하면서도 기존 자동 선택 동작을 보존할 수 있다. 이번 회귀 확인에서 no-fixed CLI는 pre-edit golden과 `atol=1e-12`로 같았고 공통 후보 수는 24개로 유지됐다.

## alpha=1 불변식

같은 L2 정규화 데이터, 같은 class prototype, 같은 floored pooled covariance eigenbasis를 사용하면 fractional alpha=1 score는 다음 관계를 갖는다.

`fractional_1 / lambda_bar == same_data_mahalanobis`

따라서 fractional alpha=1의 raw score를 Mahalanobis와 직접 같다고 검사하면 안 된다. `lambda_bar`를 나눈 값의 수치 일치와 class `argmin` 일치를 함께 검사해야 한다. 상수배는 순위를 바꾸지 않는다.

## cache 생성 protocol과 평가 protocol을 구분한다

strict cache는 10-shot 절차에서 생성됐지만 최종 평가는 저장된 5-label 5-shot episode를 사용한다. seed마다 support 25개와 query 127개다. 검증기는 cache-generation protocol hash와 episode protocol hash를 별도로 기록하고, 5-shot membership을 재생성해 저장된 path와 label 순서가 정확히 같은지 비교한다.

3-label 평가는 다른 계약을 쓴다. seed `0, 1, 2, 3, 4, 42`마다 train split에서 class당 10개를 뽑고 같은 30개 support를 prototype과 covariance에 사용한다. 서로 다른 shot 수와 membership 규칙을 한 protocol로 뭉치면 재현성이 깨진다.

## content duplicate는 두 view로 보고한다

fresh cache에는 train과 test에서 content hash가 같은 이미지가 5건 있었다. 전체 query를 쓰는 `all_rows`와 해당 5건을 뺀 `duplicate_free`를 함께 보고했다. legacy cache의 제외 건수는 0이었다. 중복을 발견했다고 전체 결과를 버리거나, 중복을 숨긴 채 한 숫자만 남기는 대신 두 view의 방향이 같은지 확인하는 방식이다.

## 탐색 결과와 확증 결과의 경계

후보를 이미 살펴본 cache에서 고른 뒤 같은 cache로 비교했으므로 결과는 exploratory post-selection evidence다. fresh 3-label accuracy는 개선됐지만 legacy와 strict 5-label에서는 회귀했고 fresh backbone balanced accuracy도 낮아졌다. 이 결과만으로 universal default를 바꾸거나 confirmatory 성능 향상을 주장하면 안 된다. 새로 잠근 데이터나 nested selection protocol이 필요하다.

## hash 검증만으로는 위조를 막지 못한다

UltraQA에서 metrics나 membership을 변조한 뒤 artifact hash를 다시 계산하는 공격을 만들었다. 단순 hash 일치 검사는 이런 재서명 위조를 통과시킬 수 있다. 검증기는 원본 cache와 episode에서 membership을 다시 만들고 path, label, position, seed hash, set hash를 비교해야 한다. metric, delta, regression flag, selected status, best flag도 원시 값에서 재계산해야 한다. boolean 열은 문자열이나 숫자의 truthiness에 맡기지 말고 실제 bool dtype을 강제한다.

관련 결과는 `docs/reports/MAHALANOBIS_SPECTRUM_VALIDATION_FINAL_REPORT.md`와 `.omx/qa/mahalanobis-spectrum-ultraqa.md`에 있다.

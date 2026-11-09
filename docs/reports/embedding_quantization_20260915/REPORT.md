# 임베딩 균일·비균일 양자화 실험 (2026-09-15)

## 결론

중심부 간격을 8배 세밀하게 한 공분산 기준 비균일 양자화는 strict 5-shot에서 기존 마할라노비스와 모든 예측이 같았다. 균일 양자화의 하락을 피했지만 추가 성능 향상은 없었다. 현재 설정만으로 양자화 전체의 가능성을 부정할 수는 없다.

## 평가 조건

- 로컬에 저장된 ViT-Small 384차원 임베딩 재사용. 모델 재학습 및 원격 서버 실행 없음.
- 3-class gallery: Legacy cache의 train 1,765 / validation 201 / test 109장. 균일 비트 수는 validation Macro-F1로 선택 (동률이면 balanced accuracy, accuracy). PCA 결과와 같은 cache.
- 5-class strict 5-shot: 기존 cache의 train/valid/test를 합친 152장을 기존 evaluator 순서대로 사용. 클래스당 support 5장 (총 25), 나머지 query 127장. seed 0,1,2,3,4,42. 각 seed에서 원본처럼 L2 정규화. 분할은 path 기준으로 분리하며 patient/near-duplicate 독립성을 새로 검증하지 않음.
- support/train 클래스 평균에 대한 residual로 pooled Ledoit-Wolf 공분산을 적합. 그 역행렬의 대칭 제곱근(ZCA)으로 좌표를 변환. 원점은 전체 support/train 평균.
- 양자화한 support의 평균을 prototype으로 사용하고, 양자화한 query와 유클리드 거리로 분류. 양자화 이후 L2 정규화 없음.
- 모든 변환과 양자화 범위는 support/train만으로 계산. query는 적합·설정 선택에 사용하지 않음. strict 결과는 고정 설정 탐색 결과이며 최고 비트 수를 채택하지 않음.

## 방식

1. 균일 양자화: 좌표별 support/train 최솟값과 최댓값 사이를 8/4/2비트로 나누고 범위 밖은 경계값으로 제한.
2. 범위 제한 대조군: 같은 최솟값·최댓값으로 자르되 반올림은 하지 않음.
3. 비균일 8비트: arctan으로 중심 ±2 표준편차에 연속 코드 구간 80% 배정. 실제로는 256개 유한 복원값을 쓰므로 먼 꼬리의 오차는 남음. clipping_fraction 0은 명시적인 min-max 제한이 없다는 뜻이며 꼬리를 무손실 보존한다는 뜻이 아님.
4. 중심 세밀화: 표준화한 각 좌표의 ±2 안에서 기존 8비트 간격(range/255)의 1/8 사용. 바깥은 기존 간격에서 시작해 로그 방식으로 넓어짐. 같은 바깥 구간을 쓰고 중심만 기존 간격인 1배 대조군도 평가. 이는 고정 비트 수 압축이 아니며 저장 용량·추론 속도 개선을 측정하지 않음.

공분산은 유한한 경계 자체가 아니다. 이번 ±2는 실험상 고정한 범위이며, ZCA 좌표의 각 축에 적용한 상자 형태의 근사다. 클래스별 공분산 타원체의 안팎을 직접 양자화하는 방식은 아니다. 중심 8배 세밀화의 간격 오차는 self-test로 확인했다.

비균일 방식은 균일 실험 결과를 본 뒤 사용자 요청으로 추가한 탐색 실험이다. ±2, 80%, 8배는 비균일 결과를 보기 전에 고정했다. 새로운 독립 검증 결과로 해석하지 않는다.

## Strict 5-shot: 6개 seed 평균

| 방법 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| 균일 2비트 | 66.67% | 62.32% | 63.53% |
| 균일 4비트 | 66.93% | 62.61% | 63.72% |
| 균일 8비트 | 67.06% | 62.71% | 63.83% |
| 범위 제한만 적용 | 67.06% | 62.71% | 63.83% |
| 비균일 8비트 (arctan) | 67.85% | 63.10% | 64.35% |
| 중심 1배 + 넓어지는 바깥 구간 | 67.98% | 63.30% | 64.49% |
| 중심 8배 세밀 + 넓어지는 바깥 구간 | 67.98% | 63.30% | 64.49% |
| 마할라노비스 원본 | 67.98% | 63.30% | 64.49% |
| 원본 좌표 균일 8비트 | 55.25% | 52.19% | 53.71% |
| 원본 유클리드 | 55.64% | 52.50% | 54.05% |

마할라노비스 기준 균일 양자화에서 query 좌표의 35.96%가 support 범위를 벗어났다. 8비트와 범위 제한 대조군의 예측이 같으므로, 이 조건의 8비트 손실은 범위 제한으로 설명된다. 중심 1배 및 8배 비균일 방식은 6개 seed의 762개 query 평가에서 모두 원본 예측과 동일했다. 762개는 중복 없는 이미지 수가 아니라 반복 평가 횟수다.

## Legacy gallery: test 109장

| 방법 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| 원본 유클리드 | 29.36% | 28.64% | 52.43% |
| 마할라노비스 원본 | 63.30% | 48.23% | 50.77% |
| 범위 제한만 적용 | 64.22% | 48.82% | 51.32% |
| 원본 좌표 균일 8비트 | 29.36% | 28.64% | 52.43% |
| 균일 4비트 | 65.14% | 49.66% | 52.07% |
| 비균일 8비트 (arctan) | 63.30% | 48.23% | 50.77% |
| 중심 1배 + 넓어지는 바깥 구간 | 64.22% | 49.02% | 51.53% |
| 중심 8배 세밀 + 넓어지는 바깥 구간 | 63.30% | 48.23% | 50.77% |

균일 4비트는 validation에서 선택됐으며 Macro-F1 +1.43%p, 정확도 +1.83%p였다. 앞선 PCA 256차원 결과(Macro-F1 51.57%)보다는 낮다. 비균일 중심 8배 조건은 원본과 같은 성능이었다.

## 검증 및 재현

- 양자화 범위, 상수 좌표, 반올림 오차, 멱등성, 단조성, 중심 간격, query batch 독립성 self-test 통과.
- 양자화 없는 ZCA 거리와 기존 마할라노비스 거리의 수치 일치 확인.
- 기존 gallery baseline 지표 1e-12 이내 일치. 기존 float32 strict5 baseline의 예측도 6개 seed에서 모두 일치.
- 초기 기본 Python 환경에서 행렬 연산 경고가 발생. 최종 결과는 명시적 Python 3.11에서 RuntimeWarning을 오류로 처리해 재실행했고 경고 없이 완료. 초기 legacy_gallery/strict5는 사용하지 않고 최종 adaptive_gallery/adaptive_strict5를 사용.
- 코드·cache SHA-256과 방법 정의: 각 최종 폴더의 protocol.json. 이미지별 예측, 클래스별 성능, seed별 성능과 support 구성도 CSV로 저장.

```sh
# implantsj 패키지 디렉터리에서 실행
/opt/homebrew/opt/python@3.11/libexec/bin/python3 -W error::RuntimeWarning -m tools.evaluate_embedding_quantization --self-test
/opt/homebrew/opt/python@3.11/libexec/bin/python3 -W error::RuntimeWarning -m tools.evaluate_embedding_quantization --mode pool5 --cache docs/reports/vit6supcon_to_5label_strict10_20260814/strict10/backbone_seed42/embedding_cache.npz --output /tmp/quantization-new-run
```

출력 경로는 기존 결과를 덮어쓰지 않도록 새 디렉터리를 지정해야 한다.

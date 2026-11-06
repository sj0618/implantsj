# 양자화 없는 연속 좌표 변환 실험

사용자 우선순위: 저장 공간이나 속도보다 분류 성능. 양자화 실험은 종료하고 아래 연속 변환만 별도로 비교했다.

## 방법

384차원을 유지한다. support/train 평균 m과 클래스 내 pooled Ledoit-Wolf 공분산 Σ를 사용해 z=(x-m)Σ^(-1/2)로 변환하고, 각 좌표에 h(z)=asinh(z)를 적용한다. prototype은 h로 변환한 support의 클래스별 평균이며, query도 같은 변환을 거쳐 유클리드 거리로 분류한다. 양자화, 반올림, clipping, 차원 축소, 모델 재학습은 없다.

h'(z)=1/sqrt(1+z²)이므로 중심 부근 차이를 먼 영역의 차이보다 상대적으로 크게 반영한다. 변환은 연속적이며 가역적이다. 최종 scalar 거리만 단조 변환하면 분류 순위가 바뀌지 않으므로 좌표별로 변환했다. 중심은 클래스별 중심이 아닌 전체 support/train 평균이고, 전환 스케일은 공분산 표준화 좌표의 1로 고정했다. 이것은 가능한 연속 거리 변환 중 한 가지이며 최적화된 설정이 아니다.

## 결과

| 조건 | 방법 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---|---:|---:|---:|
| strict 5-shot, 6 seeds 평균 | 마할라노비스 | 67.98% | 63.30% | 64.49% |
| strict 5-shot, 6 seeds 평균 | 연속 asinh 변환 | 67.59% | 62.84% | 64.05% |
| Legacy test 109장 | 마할라노비스 | 63.30% | 48.23% | 50.77% |
| Legacy test 109장 | 연속 asinh 변환 | 64.22% | 48.83% | 51.32% |

Legacy test에서는 소폭 개선됐지만 validation Macro-F1은 55.52%에서 55.32%로 낮아졌다. strict 5-shot Macro-F1은 0.45%p 낮아졌다. 따라서 현재 설정을 기존 마할라노비스 대신 채택할 근거는 없다. 양자화 없이 성능을 우선한다는 방향은 유지하되, 이 고정 asinh 설정의 효과가 확인된 것으로 해석하지 않는다.

## 조건과 검증

5-shot은 152장 중 클래스당 5장 support, 나머지 127장 query. seed 0,1,2,3,4,42. 앞선 실험과 동일한 cache·support/query 순서·정규화. Legacy는 train 1,765 / validation 201 / test 109장. 자세한 cache 경로 및 SHA-256은 continuous_gallery/protocol.json과 continuous_strict5/protocol.json에 저장했다. query로 변환을 적합하거나 스케일을 선택하지 않았다. 앞선 결과를 본 후 추가한 탐색 실험이며 독립 확증 결과는 아니다.

원본 baseline 지표 재현, 모든 점수의 유한성, 연속 변환의 중심/외곽 차이 및 세밀한 값 구분 self-test를 통과했다. continuous-only 모드에서 생성되는 평가 방법은 raw_float, maha_float, maha_continuous_asinh 3개뿐임을 확인했다. 최종 실행은 Python 3.11에서 RuntimeWarning을 오류로 처리하고 경고 없이 완료했다.

```sh
# implantsj 패키지 디렉터리에서 새 output 경로로 실행
/opt/homebrew/opt/python@3.11/libexec/bin/python3 -W error::RuntimeWarning -m tools.evaluate_embedding_quantization --continuous-only --mode pool5 --cache docs/reports/vit6supcon_to_5label_strict10_20260814/strict10/backbone_seed42/embedding_cache.npz --output /tmp/continuous-metric-new-run
```

실행 도구의 파일 이름에 quantization이 남아 있지만 --continuous-only에서는 양자화 평가를 실행하지 않는다. 원본 모델과 기존 평가 코드는 변경하지 않았다.

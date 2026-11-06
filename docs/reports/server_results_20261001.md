# 폐쇄망 서버 실험 결과 정리

- 확인 시각: 2026-10-01 07:58~08:25 KST
- 확인 방법: Windows App 원격 화면에서 결과 파일과 로그를 열어 수동 기록
- 대상 경로: `/home/ubuntu/aihub_run/outputs`, `/home/ubuntu/aihub_run_new/outputs`, `/home/ubuntu/implant_pipeline_new/outputs`
- 제외: 원본 데이터, 임베딩, 체크포인트, 학습 모델

## 핵심 결과

| 실험 | 평가 방식 | 전체 정확도 | 전체 Macro-F1 | Seen/Closed 정확도 | Unseen/Open 정확도 |
|---|---|---:|---:|---:|---:|
| Gap-SupCon b256 | pooled OAS | **87.6705%** | 69.6511% | **90.6313%** | **50.4505%** |
| Band-ViT gallery covariance | k-means multi + centered angular | **92.7520%** | **78.4708%** | **94.3384%** | **72.8092%** |
| ViT SupCon 5ep | medoid + fractional | 87.3506% | 65.0698% | 90.2143% | 51.3514% |
| ViT SupCon 5ep | k-means multi + Mahalanobis OAS | 84.7495% | **77.0609%** | 85.4974% | **75.3481%** |
| 예측 속성 8개, 41클래스 | prototype + euclidean | 16.0396% | 5.1430% | 17.0110% | 3.5831% |

## 1. Gap-SupCon b256 최종 평가

결과 파일: `gap-supcon-b256-20260929/final_gap/test_metrics.csv`

| 거리/공분산 | 그룹 | 표본 수 | 정확도 | Macro Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| cosine | 전체 | 16,570 | 84.9728% | 68.6715% | 62.9275% |
| cosine | seen | 15,349 | 89.0025% | 88.0048% | 85.9461% |
| cosine | unseen | 1,221 | 34.3161% | 33.8716% | 31.1372% |
| euclidean | 전체 | 16,570 | 84.9909% | 68.7541% | 62.9470% |
| euclidean | seen | 15,349 | 88.9765% | 87.9927% | 85.9476% |
| euclidean | unseen | 1,221 | 34.8894% | 34.1247% | 31.1550% |
| pooled OAS | 전체 | 16,570 | **87.6705%** | **72.9216%** | **69.6511%** |
| pooled OAS | seen | 15,349 | **90.6313%** | **88.0209%** | **86.6579%** |
| pooled OAS | unseen | 1,221 | **50.4505%** | **45.7430%** | **48.3320%** |

- unseen 클래스의 `negative_gap_fraction`은 모든 방식에서 1.0이었다. 클래스 간 분리 여유가 음수인 상태다.
- 최종 선택 단계의 최고 확인값은 `select_04`, epoch 5, euclidean 기준 정확도 33.0584%, Macro-F1 47.7508%다.
- `select_04`의 euclidean 정확도는 epoch 1~5에서 32.7078%, 27.9027%, 30.2743%, 29.5112%, 33.0584%였다.

## 2. ViT SupCon 5 epoch 평가 그리드

로그: `followup-20260928-2228/vit-supcon5.log`

- 설정: ViT 전체 미세조정, SupCon, 5 epochs, batch 128, class당 8 samples, seed 42
- 분할: closed 27 / open 15
- prototype: mean, spherical mean, medoid, k-means multi
- 거리: cosine, euclidean, Mahalanobis, fractional, centered angular
- 공분산: Ledoit-Wolf, OAS, diagonal, class-balanced, RDA

| 조합 | 전체 정확도 | 전체 Macro-F1 | Closed 정확도 | Closed Macro-F1 | Open 정확도 | Open Macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| medoid + fractional | **87.3506%** | 65.0698% | **90.2143%** | 54.5502% | 51.3514% | 12.9736% |
| medoid + centered angular | 87.0368% | 64.8155% | 90.2339% | 54.5583% | 46.8468% | 12.7438% |
| k-means multi + Mahalanobis OAS | 84.7495% | **77.0609%** | 85.4974% | 53.0886% | **75.3481%** | **28.8661%** |
| k-means multi + centered angular | 84.7013% | 71.5915% | 86.4551% | 53.0362% | 62.6536% | 22.9834% |

- 전체 정확도 기준으로는 medoid + fractional이 가장 높았다.
- open 정확도와 전체 Macro-F1 기준으로는 k-means multi + Mahalanobis OAS가 가장 높았다.
- 로그 종료 문구: `Metric learning grid 완료` (2026-09-30 04:28:35)

## 3. 예측 속성 8개 기반 41클래스 평가

결과 파일: `gt-prototype-20260928/predattr8-41class-20260928/summary.csv`

| 거리 | 전체 정확도 | Macro-F1 | Seen 정확도 | Unseen 정확도 |
|---|---:|---:|---:|---:|
| cosine | 15.7175% | 4.8109% | 16.6723% | 3.4745% |
| euclidean | **16.0396%** | **5.1430%** | **17.0110%** | **3.5831%** |

- 현재 설정에서는 속성 8개만으로 41클래스를 구분하기 어렵다.

## 4. 반복 홀드아웃 및 실행 상태

`followup-20260928-2228/queue.log`에서 아래 작업이 모두 `exit_code: 0`으로 종료된 것을 확인했다.

- `galcov`
- `heldout`
- `heldout-frozen`
- `vit-supcon5`
- `dinov2-ft5`
- `heldout-s43`, `heldout-frozen-s43`
- `heldout-s44`, `heldout-frozen-s44`
- `repeat_summary`: `REPEAT_SUMMARY_OK 120`
- 최종 큐 상태: `finished`

홀드아웃 클래스 구성:

- seed 42: astra, US II, Himplant, IS III, IS I
- seed 43: Bone level, astra, IS II, internal, branemark
- seed 44: TS standard plus, TS standard, astra, internal, US II

화면에서 확인한 홀드아웃 로그 중 한 실행은 epoch 7에서 최고 validation Macro-F1 0.6029를 기록했다. 다만 마지막에 `test_manifest.csv`가 없어 임베딩 저장을 건너뛰었다. 프로세스 종료 코드는 0이지만, 해당 실행의 테스트 결과 파일 저장은 완료로 볼 수 없다.

## 5. 특징 공간 분석

`geometry41_verified.csv`에는 split별 euclidean/cosine 특징 공간 통계가 저장돼 있다.

- split 37 closed 예시: intra 0.1739, inter 1.1049, inter/intra 6.3536
- split 37 open 예시: intra 0.5712, inter 0.7084, inter/intra 1.2402
- split 38 open 예시: intra 0.8760, inter 0.8741, inter/intra 0.9978

open 그룹은 확인된 예시에서 클래스 간 분리가 closed보다 약했다.

## 6. 추가 metric 평가 결과

아래 수치는 각 `grid_summary`에서 화면에 보인 마지막 조합이다. 최고값으로 해석하지 않는다.

### 6.1 closed27/open15 결과

| 실험 | 마지막 조합 | 전체 정확도 | 전체 Macro-F1 | Closed 정확도 | Open 정확도 |
|---|---|---:|---:|---:|---:|
| DINOv2 covaware grad-checkpoint r2 | k-means multi + centered angular | 24.4056% | 19.4189% | 22.7637% | 45.0450% |
| DINOv2 FT 5ep | covaware SupCon, k-means multi + centered angular | 87.8636% | 65.8177% | 90.4815% | 54.9550% |
| Band-ViT gallery covariance | k-means multi + centered angular | **92.7520%** | **78.4708%** | **94.3384%** | **72.8092%** |
| ViT SupCon 5ep | k-means multi + centered angular | 84.7013% | 71.5915% | 86.4551% | 62.6536% |

`closed27_open15` 아래에서 확인한 전체 폴더:

- 수치 요약 확인: `dinov2_covaware_gradckpt_20260924_2124_r2`, `dinov2_ft5_20260928`, `vit_band_galcov_20260928`, `vit_supcon5_20260928`
- 결과 폴더만 확인: `dinov2_full_finetune_5ep`, `backbone_eval`, `full_finetune`, `dinov2_galcov_ft_5ep`, `backbone_eval_galcov`, `dinov2_covaware_gradckpt_20260924_2118`, `full_finetune_40e_p5`, `dinov2_galcov_timm_5ep`, `head_only`

### 6.2 41클래스 및 속성 기반 결과

| 실험 | 마지막 조합 | 전체 정확도 | 전체 Macro-F1 | Closed 정확도 | Open 정확도 |
|---|---|---:|---:|---:|---:|
| attr8 tokens 20ep p3 | k-means multi + fractional | 88.7206% | 69.9650% | 91.8692% | 49.1400% |
| attr8 tokens 20ep p3 | k-means multi + centered angular | 88.7387% | 69.5455% | 91.9995% | 47.7477% |
| new_41class | k-means multi + fractional | 83.8986% | **72.5410%** | 85.4453% | **64.4554%** |
| new_41class | k-means multi + centered angular | 83.7719% | 70.2422% | 85.5300% | 61.6708% |
| predattr8 | RDA | 73.7598% | 33.8553% | 79.6273% | 0.0000% |
| predattr8 | k-means multi + fractional | 80.8027% | 57.0121% | 85.3280% | 23.9148% |
| predattr8 | k-means multi + centered angular | 80.4285% | 56.3214% | 85.0153% | 22.7682% |

- 상위 폴더: `attr8_tokens_41class_20260927_20ep_p3`, `predattr8-41class-20260928`, `attr8_tokens_41class_20260927`, `new_41class`, `closed27_open15`
- `attr8_tokens_41class_20260927`은 폴더만 확인했고 별도 grid 요약 수치는 화면에서 찾지 못했다.

### 6.3 이전 metric 결과

| 출력 경로/설정 | 마지막 조합 | 전체 정확도 | 전체 Macro-F1 | Closed 정확도 | Open 정확도 |
|---|---|---:|---:|---:|---:|
| `metric_vit_vanilla_bs128` | Band-DINOv2, SS + cosine | 46.6626% | 33.4983% | 47.8533% | 31.6953% |
| `metric_vit_vanilla_bs128` | Band-DINOv2, SS + euclidean | 46.7230% | 33.6512% | 47.9510% | 31.2858% |
| `metric_vit_vanilla_bs128` | Band-DINOv2, SS + Mahalanobis | 28.7930% | 31.5918% | 30.0345% | 13.1859% |
| `metric-ft27` | ViT, SS + euclidean | **81.1768%** | 55.6113% | **85.6929%** | 24.4062% |
| `metric-ft27` | ViT, SS + Mahalanobis | 75.0392% | **59.9710%** | 79.5231% | 18.6732% |
| `pretrained-vit-full42` | ViT, k-means multi + Mahalanobis | 59.6620% | 49.6666% | 59.6620% | 해당 없음 |

이전 출력에서 확인한 실험 단위는 `vit_supcon`, `vit_covaware_supcon`, `vit_triplet`, `vit_arcface`, `band-dinov2_none`, `dinov2_supcon`, `vit_none`이다. `vit_covaware_supcon`과 `dinov2_supcon`은 폴더는 있었지만 별도 요약 파일을 화면에서 확인하지 못했다.

## 7. 속성 예측 결과

### 7.1 반복 held-out의 Apex Shape

| seed/설정 | heldout5 정확도 | heldout5 Macro-F1 | other37 정확도 | other37 Macro-F1 |
|---|---:|---:|---:|---:|
| seed 42, fine-tune | 76.3456% | 76.3455% | 82.6551% | 76.1539% |
| seed 42, frozen | 75.1908% | 74.9453% | 74.0466% | 70.3246% |
| seed 43, fine-tune | **94.2294%** | **93.8362%** | **92.4320%** | **90.5003%** |
| seed 43, frozen | 82.8265% | 81.6077% | 83.1895% | 79.7842% |
| seed 44, fine-tune | **96.7651%** | **95.8689%** | 92.0066% | 90.7439% |
| seed 44, frozen | 83.4485% | 80.5285% | 79.6803% | 77.7636% |

- Apex Hole의 heldout5는 6개 실행 모두 정확도와 Macro-F1이 100%였다.
- Apex Hole other37은 정확도 약 99.8%, Macro-F1 약 49.95%였다. 클래스 불균형 영향을 함께 봐야 한다.

### 7.2 closed27 속성 결합

| 방식 | split | Top-1 | Top-2 | Macro-F1 | Seen Top-1 | Unseen Top-1 |
|---|---|---:|---:|---:|---:|---:|
| separate, learned | test | 86.9221% | 90.2414% | 56.1326% | 93.5240% | 3.9312% |
| separate, equal | test | 86.9946% | 90.0121% | 56.0452% | 93.6282% | 3.6036% |
| shared, learned | test | 87.0791% | 89.7888% | 55.3105% | 93.8042% | 2.5389% |
| shared, equal | test | 87.0549% | 89.8310% | 55.4048% | 93.7781% | 2.5389% |

### 7.3 속성 tiny queue 최종 validation

- shared r3, epoch 26: 정확도 98.0451%, Macro-F1 93.9900%
- shared r3 속성별 Macro-F1: System 98.4995%, Connection 1 93.5527%, Connection 2 92.8881%, Flange 81.5201%, Collar 95.0971%, Microthread 98.5327%, Body Shape 97.1040%, Thread Shape 83.6195%, Apex Shape 99.0866%, Apex Hole 100%
- 개별 모델 최종값: Microthread 98.3844%, System 98.3058%, Thread Shape 87.8881%, Flange 81.7252% Macro-F1
- 초기 unit 실행 epoch 4: 정확도 37.5%, Macro-F1 27.2727%

## 8. 백본 학습 기록

아래 값은 각 `history.csv`의 마지막 epoch validation 값이다. 최고 epoch 값은 아니다.

### 8.1 기존 50 epoch 백본 16개

| 백본 | 학습 | split | Valid 정확도 | Valid Macro-F1 |
|---|---|---|---:|---:|
| DINOv2 | full | closed27/open15 | 93.1913% | 85.4431% |
| DINOv2 | full | full42 | 92.9447% | 86.6529% |
| DINOv2 | partial | closed27/open15 | 94.3967% | 87.5018% |
| DINOv2 | partial | full42 | 94.5259% | 88.8718% |
| ViT | full | closed27/open15 | 95.3544% | 90.0644% |
| ViT | full | full42 | 95.6002% | 91.1603% |
| ViT | partial | closed27/open15 | 93.6474% | 86.3450% |
| ViT | partial | full42 | 93.8801% | 87.5333% |
| Band-DINOv2 | full | closed27/open15 | 93.6409% | 87.1147% |
| Band-DINOv2 | full | full42 | 93.7775% | 85.5725% |
| Band-DINOv2 | partial | closed27/open15 | 94.4423% | 88.0631% |
| Band-DINOv2 | partial | full42 | 94.3811% | 88.8787% |
| Band-ViT | full | closed27/open15 | 95.5955% | 90.9381% |
| Band-ViT | full | full42 | 95.7028% | 91.2702% |
| Band-ViT | partial | closed27/open15 | 93.8754% | 86.7998% |
| Band-ViT | partial | full42 | 93.8017% | 87.2800% |

### 8.2 추가 백본 10개

| 실행 | 백본 | split | 마지막 epoch | Valid 정확도 | Valid Macro-F1 |
|---|---|---|---:|---:|---:|
| `20260918-170637-6ea9b483` | Band-DINOv2 | closed27/open15 | 48 | 95.4261% | 89.5781% |
| `20260918-170637-6ea9b483` | Band-ViT | closed27/open15 | 39 | 95.6216% | 90.5435% |
| `20260918-170637-6ea9b483` | DINOv2 | closed27/open15 | 50 | 93.3737% | 87.0435% |
| `20260918-170637-6ea9b483` | ViT | closed27/open15 | 45 | 95.5955% | 90.1111% |
| `20260919-113246-b2be95d6` | Band-DINOv2 | full42 | 50 | 95.7511% | 90.6006% |
| `20260919-113246-b2be95d6` | Band-ViT | full42 | 50 | 96.0770% | 91.5190% |
| `20260919-113246-b2be95d6` | DINOv2 | full42 | 50 | 93.6508% | 87.5584% |
| `20260919-113246-b2be95d6` | ViT | full42 | 50 | 95.8597% | **91.8477%** |
| `20260920-111629-fe6fc822` | Band-ViT-Large | full42 | 50 | 95.7994% | **91.8309%** |
| `20260920-111629-fe6fc822` | ViT-Large | full42 | 50 | 95.7692% | 91.5247% |

총 26개 백본 `history.csv`의 마지막 기록을 확인했다.

## 9. 분류기 학습 결과

| 실행 | 마지막 epoch | Valid 정확도 | Valid Macro-F1 | Valid Weighted-F1 |
|---|---:|---:|---:|---:|
| Band-ViT `full42_a16_l0.95_b4` | 47 | **95.8778%** | **91.0577%** | **95.8675%** |
| Band-ViT `full42_a4_l0.5_b6` | 28 | 94.3811% | 88.3971% | 94.3532% |

- `masked_vit` 폴더는 확인했지만 CSV 결과 파일은 화면에서 찾지 못했다.

## 10. 검증 파이프라인 결과

- `20260917-1830-image-fusion`: 16,570건, Top-1 정확도 95.3772%, 결정 건수 16,570, reject rate 0%, known-class coverage 100%
- `20260917-0005-real`: `summary.csv` 경로는 확인했지만 화면 입력 문제로 수치 행을 열지 못했다. 경로 존재만 기록한다.

## 11. 전체 실험 상태표

| 결과군 | 확인 범위 | 상태 |
|---|---|---|
| Gap-SupCon | b256 최종 평가, 선택 단계, geometry | 수치 확인 |
| ViT/DINOv2/Band metric | 신규 grid, 이전 grid, FT27, full42 | 수치 확인 |
| 41클래스/속성 prototype | attr8, predattr8, new_41class | 수치 확인 |
| 속성 모델 | closed27 결합, tiny queue, held-out seed 42/43/44 및 frozen | 수치 확인 |
| 백본 학습 | 기존 16개 + 추가 10개 | 마지막 epoch 수치 확인 |
| 분류기 | Band-ViT 2개, masked_vit | 2개 수치 확인, 1개 결과 파일 미확인 |
| 검증 파이프라인 | image-fusion, real | 1개 수치 확인, 1개 경로만 확인 |
| 후속 큐 | 9개 작업과 repeat summary | 종료 코드와 완료 로그 확인 |

`결과 폴더만 확인` 또는 `결과 파일 미확인`은 실행 실패를 뜻하지 않는다. 화면 탐색 범위에서 최종 요약 파일을 확인하지 못했다는 뜻이다.

## 서버에서 확인된 실험 폴더

- `20260919-1608-attention5-finished`
- `20260919-1749-gradient-rollout5-finished`
- `20260920-0955-full42-maps5-finished`
- `20260920-2014-vitlarge-full42-b32`
- `20260920-2128-band-dinov2-full42-rollout5`
- `20260921-2141-metric-vit-all-options-k5`
- `20260921-2230-metric-vit-headonly-allcov-k5`
- `20260923-1312-metric-headonly-resume-*`
- `20260923-1427-full-resume-queue-repair`
- `20260923-1655-dinov2-full-four-loss-5ep`
- `20260924-galcov-5ep`
- `20260924-2118-dinov2-covaware-gradckpt`
- `20260924-2250-attributes-cnn-groups`
- `20260924-2303-attributes-tiny-queue`, `-r2`, `-r3`
- `20260925-attr-closed27`
- `attr8_tokens_41class_20260927`
- `predattr8-41class-20260928`
- `20260928-0150-attribute-support5`
- `gt-prototype-20260928`
- `followup-20260928-2228`
- `gap-supcon-20260929`
- `gap-supcon-b256-20260929`
- `new_41class`

## 결과 해석 시 주의사항

- pooled OAS와 k-means/Mahalanobis 평가는 gallery의 prototype·공분산을 사용하므로 strict zero-shot 결과로 표기하면 안 된다.
- 화면에서 동일 조건의 기본 SupCon 대조 결과를 찾지 못했다. 따라서 Gap-SupCon이 SupCon보다 개선됐다고 직접 결론 내릴 수 없다.
- 이 문서는 화면에서 읽을 수 있었던 결과만 수동으로 기록했다. 서버 파일 자체는 로컬로 전송하지 않았다.
- 출력 경로에는 일반 파일 19,191개가 있었고, 그중 결과 관련 확장자 파일은 16,774개였다. 주요 구성은 CSV 5,514, JSON 5,195, TXT 188, LOG 130, Markdown 37, YAML 609, PNG 2,299, Python 2,670, Shell 132개다.

# 임플란트 실험 통합 보고서와 다음 학습안

작성: 2026-09-30 · 로컬 결과 조사 + 원격 서버 화면 확인 · GPT-6 Luna / max 서브에이전트 3개 분담

## 1. 핵심 결론

- **클래스 간 분리가 부족하다는 진단은 맞다. 특히 미학습 클래스에서 뚜렷하다.** 최신 Gap-SupCon의 unseen RMS 반지름은 0.4302로 seen의 2.25배이고, 최근접 유효 간격은 -0.5763이다. unseen 15개 클래스 모두 이 간격이 음수였다.
- **Gap-SupCon은 완료됐지만 개선을 입증하지 못했다.** unseen 정확도는 cosine 34.32%, pooled OAS 50.45%다. 같은 초기값·데이터·학습 조건의 SupCon 단독 대조군은 중단돼 결과가 없다.
- **prototype·거리 계산 방식도 성능에 크게 영향을 준다.** 새로 완료된 ViT의 Kmeans 5 + OAS는 unseen 85.18%, Band-ViT는 84.19%다. gallery로 공분산을 추정한 조건이므로 Gap 학습 효과와 직접 비교할 수 없다.
- **다음 학습은 동일 조건의 SupCon 대조군 → 기존 Gap 재현 → SupCon + episode별 prototype 분류 손실 순서로 권고한다.** 클래스 내부 거리와 최근접 클래스 간격을 함께 보되, 채택 기준은 미학습 클래스 분류 성능으로 둔다.

이번 작업에서 새 학습은 실행하지 않았다. 기존 결과를 이동·삭제하지 않고, 이 폴더에 보고서와 실험 목록을 만들었다.

## 2. 조사 범위와 근거

| 구역 | 확인한 자료 | 확인 수준 |
|---|---|---|
| 로컬 `implantsj` | outputs, 두 docs/reports 경로, 설정·프로토콜·학습 이력 | 결과 원문 확인. 기존 geometry에 연결된 NPZ 45개 SHA256 모두 일치. 배열 재계산은 하지 않음 |
| 로컬 다른 폴더 | Aihub, outputs_docker, implant_project, md파일모음 및 관련 문서 경로 | 결과 JSON·혼동행렬·manifest·체크포인트 메타데이터·노트북 저장 출력 확인 |
| 원격 `user07-gpu` | Gap-SupCon, followup queue, 41클래스 OAS, ViT/Band-ViT GalCov | Windows App에서 CSV·JSON·로그·실행 코드·Docker 상태 읽기. 9/30 12:28–12:46 KST 기준 |
| 과거 원격 결과 | 9/28·9/29 상세 보고서 | 기존 기록으로 인용. 모든 과거 원격 원본을 이번에 다시 읽은 것은 아님 |

‘전사 확인’은 원격 화면에서 읽은 수치를 뜻한다. ‘노트북 기록’은 저장된 출력이 있다는 뜻이며, 현재 체크포인트로 재현했다는 뜻은 아니다. 진행 중인 작업과 최종 결과가 없는 작업도 목록에 포함했다.

## 3. 결과 요약

### A. 최신 원격 결과

정확도 단위: %. 전체 16,570장 중 unseen은 1,221장으로 **7.37%**다. 전체 정확도만 보면 unseen 실패가 가려질 수 있다.

| 실험 | 평가 조건 | 전체 Acc | unseen Acc | 상태·해석 |
|---|---|---:|---:|---|
| Gap-SupCon, DINOv2 projection128 | Mean + cosine | 84.97 | **34.32** | 학습·평가 완료, 대조군 없음 |
| 같은 Gap 모델 | Mean + closed-only OAS | 87.67 | **50.45** | 같은 임베딩에서 분류기 변경 효과 |
| ViT backbone | Kmeans 5 + gallery OAS | **93.77** | **85.18** | grid 완료. 기존 backbone checkpoint 사용 |
| Band-ViT backbone | Kmeans 5 + gallery OAS | 93.72 | **84.19** | grid 완료. ViT와 같은 gallery 구성 |
| 41클래스 image-only | Kmeans + gallery OAS | 84.86 | **79.03** | 저장된 OAS 비교표 확인 |
| 41클래스 attribute token 5ep | Kmeans + gallery OAS | 85.43 | 78.13 | image-only보다 unseen 개선 없음 |
| 41클래스 attribute token 20ep/p3 | Kmeans + gallery OAS | 90.29 | 73.05 | 전체가 높아도 unseen은 낮음 |
| 41클래스 predicted attribute | Kmeans + gallery OAS | 82.65 | **37.51** | 학습 조건도 달라 속성만을 원인으로 단정 불가 |

**이 표는 전체 실험 순위표가 아니다.** backbone, 초기 checkpoint, 학습 클래스 수, 공분산 추정 자료가 다르다. Kmeans 5는 중심 5개이며 strict 5-shot이 아니다. GalCov의 중복 제외 성능과 기존 backbone checkpoint의 전체 학습 이력은 추가 확인 대상이다.

원격 GalCov의 `open_macro_f1`은 전체 클래스 목록과 unknown을 넣어 계산한 값이다. unseen 15개 클래스만의 Macro-F1로 해석하지 않았다. 41클래스 OAS 표의 `overall_macro_f1`도 open 행에서 반복된다. 상세 수치·계산 범위는 [원격 확인 기록](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/remote_results.md)에 있다.

### B. 과거 원격 실험에서 유지할 결론

다음은 [기존 보고서](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/docs/reports/professor_report_20260928/report.md)의 기록이다. 현재 원격 원문 전체를 재검증한 결과와 구분한다.

| 계열 | 기록된 결과 | 현재 판단 |
|---|---|---|
| M7 SupCon, 42클래스 | Kmeans + gallery Ledoit–Wolf unseen 약 **82.1%** | 기존 metric learning 기준 결과로 보존 |
| M8 ArcFace | 같은 계열 unseen 약 **74.1%** | 중심을 더 벌리는 것만으로 개선되지 않음 |
| M9 Triplet | Mean + gallery Mahalanobis 전체 Acc **10.5%**, F1 7.9% | 낮은 성능. 현재 설정을 재사용할 근거 부족 |
| M10 CovAware | Mean + Mahalanobis 전체 Acc **33.5%**, F1 24.5% | 공분산 loss를 더하는 것 자체가 해결책은 아님 |
| 속성 점수 융합 | 속성 가중치 증가 시 unseen 하락 기록 | source validation으로 가중치 선택 필요 |
| GT prototype | 25클래스·query 12,731장, cosine Acc 약 **15.72%** | GT/query 전처리·매핑·표현 차이 확인 필요 |
| Full42 closed-set 분류 | Band-ViT 약95.9% 등의 높은 정확도 | 모든 클래스가 학습에 포함된 별도 문제 |

### C. 로컬 few-shot 및 초기 실험

| 실험군 | 주요 결과 | 해석 |
|---|---|---|
| Track B: 15-way unseen, K=10 | PCA384 ViT-B + pooled Mahalanobis **58.80% Acc / 42.42% F1** | raw768 52.40/41.08, native384 53.45/39.37. 5개 분할 × 200 episode. train 역할 데이터로만 PCA 적합 |
| 같은 Track B cosine | CE ViT-S **52.3/31.8**, ArcFace **48.9/30.7** | CE가 두 지표 모두 높지만 unseen F1 자체는 낮음 |
| 별도 5클래스 strict 5-shot | SupCon backbone pooled LW **67.98/63.30** | 152장 중 support 25 / query 127, support seed 6개 |
| 같은 5클래스 CovAware | 초기 **66.40/62.65**; 이후 pooled-full 2학습 seed 평균 **66.60/63.32** | seed42 69.42/66.87, seed0 63.78/59.77. 안정적 개선 없음 |
| 7클래스 ResNet18, 1 epoch | 분류기 **39.94/15.01**; pooled Mahalanobis **55.92/44.27** | 초기 로컬 CPU 실험. 현재 42클래스 실험과 별도 |
| 11클래스 초기 실험 | **20.79/7.30**, 8개 클래스 F1=0 | 동일 checkpoint 복사본을 독립 실험으로 세지 않음 |
| 구형 ViT-SupCon 노트북 | 저장 출력 **94.99/84.48** | 대응 원본 checkpoint 미확인. 현재 확정 성능으로 채택하지 않음 |

앞 표의 Acc/F1 단위는 %. 3-label 잘못된 manifest로 평가가 실패한 결과, 차원·projection 표시가 맞지 않는 메타데이터도 별도 목록에 기록했다. 프로토콜이 다른 결과를 평균 내거나 하나의 최고 성능으로 합치지 않았다.

## 4. 거리 문제에 대한 판단

### 확인된 현상

최신 Gap-SupCon에서 support 중심 기준 query RMS 반지름과 최근접 유효 간격을 측정했다.

| 그룹 | 내부 RMS 반지름 | 최근접 유효 간격 | 간격 < 0 비율 |
|---|---:|---:|---:|
| Seen | 0.1910 | -0.1038 | 74.07% |
| Unseen | **0.4302** | **-0.5763** | **100%** |

유효 간격은 `두 클래스 중심 거리 − 두 클래스 반지름의 합`이다. 음수는 RMS 반지름을 고려했을 때 분리 여유가 부족하다는 뜻이다. 모든 표본이 겹치거나 오분류됐다는 뜻은 아니다. 실제 unseen 정확도도 0%가 아니다.

이전 M7의 unseen **pairwise 내부 평균 거리 0.5843 / 중심 간 평균 거리 0.6548**도 분리 여유가 작았다. M8은 각각 **0.9299 / 0.9619**로, 중심뿐 아니라 클래스 내부 거리도 증가했다. 이 pairwise 평균은 위 RMS 반지름과 다른 지표다.

### 원인 후보와 확인 방법

| 원인 후보 | 현재 근거 | 다음 확인 |
|---|---|---|
| 학습 클래스에 대한 과적합 | Gap 선택 단계 train gap +0.3426, heldout gap -0.4241 | source 클래스 holdout별 반복 결과·학습 곡선 |
| Gap loss 기여 부족 | 선택 기록의 가중 gap 손실은 SupCon의 약0.0045% | loss별 gradient norm과 active-gap 비율. 손실 크기만으로 확정하지 않음 |
| 클래스 내부 촬영 조건 차이 | unseen의 큰 RMS 반지름 | 클래스별 촬영기관·각도·배율·전처리별 산포, support/query 중심 이동 |
| 임베딩 공간과 분류기의 불일치 | 동일 Gap 임베딩의 unseen 34.32→50.45% | backbone/projection, 정규화, covariance fit 범위를 고정한 비교 |
| 라벨 오류·중복·다중 형태 | 41/42클래스 매핑 변경 및 로컬 manifest 실패 이력 | 혼동이 큰 클래스 쌍의 라벨과 환자/촬영 중복 확인 |

클래스 전체 평균이나 2차원 시각화만으로 원인을 확정하지 않는다. 가장 헷갈리는 클래스 쌍과 클래스별 recall을 함께 확인해야 한다.

## 5. 다음 학습안

### 우선 후보: SupCon + episode별 prototype 분류 손실

**목적:** 같은 클래스의 서로 다른 이미지를 모으면서, 소수 support로 만든 prototype을 사용해 query를 분류하도록 학습한다. support로 prototype을 만들고 별도 query를 분류하는 방법은 [Prototypical Networks](https://papers.nips.cc/paper_files/paper/2017/hash/cb8da6767461f2812ae4290eac7cbc42-Abstract.html)에 근거한다. 아래 SupCon 결합·샘플 수·가중치는 이번 데이터에 대한 실험 제안이다.

1. **데이터:** source 27개 클래스 중 5개를 모델 선택용으로 분리한다. 나머지 source 클래스만 gradient 학습에 쓴다. 최종 15개 unseen query는 선택 과정에 쓰지 않는다.
2. **배치:** 원격 Gap과 비교할 때 batch 256, 클래스당 서로 다른 원본 16장을 유지한다. 선택된 클래스마다 support 8장 + query 8장으로 나눈다. 같은 이미지의 다른 증강 view가 양쪽에 나뉘지 않게 한다. 실제 클래스 수와 고유 이미지 수를 기록한다.
3. **표현:** 동일 DINOv2 초기값과 동일 128차원 projection을 사용한다. 임베딩과 cosine prototype을 정규화한다. 768차원 동결 backbone은 별도 참고 기준으로 평가한다.
4. **손실:** `L = L_supcon + λ × L_episode`. `L_episode`는 query와 각 support prototype의 cosine similarity/temperature를 logit으로 둔 cross-entropy다. 정답 prototype 점수를 높이고 경쟁 prototype 점수를 낮추는 학습 신호를 준다.
5. **초기 비교:** 동일 optimizer·learning-rate schedule·5 epoch·증강·update 수로 SupCon과 비교한다. 기존 Gap loss는 이 신규 후보에 동시에 더하지 않는다. episode temperature는 대조군의 명시된 temperature에 먼저 고정한다.
6. **가중치 선택:** source holdout에서만 `λ∈{0.1,1.0}`을 비교하는 작은 탐색부터 시작한다. loss 크기뿐 아니라 gradient 기여를 기록한다. 이 값들은 아직 검증되지 않은 시작 후보이며 최적값이 아니다.

공분산 기반 분류 loss를 쓰는 기존 CovAware와 달리, 첫 후보의 episode loss는 cosine만 사용한다. 이미 있는 episode sampler와 SupCon 구현을 재사용하고, 평가에서는 cosine과 pooled OAS를 모두 비교한다. 학습에 공분산·center loss·ArcFace를 한꺼번에 추가하지 않는다.

### 최소 비교표

| ID | 조건 | 확인할 질문 |
|---|---|---|
| A | SupCon only | 동일 초기값과 학습 설정에서 기준 성능은 얼마인가? |
| B | SupCon + 기존 Gap 0.05/0.01 | 반지름을 고려한 Gap이 A보다 실제로 좋아지는가? |
| C | SupCon + prototype episode CE | support/query를 나눠 학습하면 새로운 클래스 분류가 좋아지는가? |

세 조건 모두 같은 support/query 구성과 seed를 사용한다. 먼저 seed42에서 재현·수치 안정성을 확인하고, source validation으로 선택한 설정을 seed43·44에서도 반복한다. 설정 선택이 끝나면 세 조건 모두 source 27개 클래스 전체로 다시 학습하고 같은 최종 unseen 평가를 수행한다. 이미 저장된 B 결과는 이 최종 단계의 설정과 데이터 해시가 일치하면 재사용한다.

증강으로 클래스 식별 부위가 잘리는 경우에만 약한 crop/회전 조건을 따로 비교한다. 학습 클래스 confusion에서 혼동 쌍이 배치에 거의 같이 나오지 않는다면, 그 쌍을 함께 넣는 sampler 변경을 후속으로 비교한다. 이때 라벨 오류를 먼저 확인한다.

### 채택 기준

- **주 지표:** source holdout Macro-F1·balanced accuracy. 최종 unseen에서도 같은 지표와 정확도·클래스별 recall을 보고한다.
- **보조 지표:** 클래스별 RMS 반지름, 최근접 중심 거리, 유효 간격, 음수 간격 비율. 분류 성능이 나빠지면 거리표만 좋아졌다는 이유로 채택하지 않는다.
- **공정한 평가:** 분류기·shot·공분산 적합 범위·표현 차원별로 따로 비교한다. support/query는 원본 이미지와 가능하면 환자 단위로 분리한다.
- **재현성:** 동일 3개 학습 seed에서 A 대비 차이를 보고하고, 기존 strict 5-shot support seed 반복을 적용한다. 여러 support seed를 여러 학습 seed로 세지 않는다.
- **독립 평가:** 이미 여러 번 결과를 본 현재 test는 개발 근거로 명시한다. 최종 성능 주장은 새로 잠근 class/patient holdout에서 확인한다.

전체 loss 조합 탐색보다 조건을 맞춘 소수 비교를 우선하는 이유는 metric learning 결과가 학습·평가 설정에 민감하기 때문이다. [A Metric Learning Reality Check](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf)도 비교 방법의 차이가 결론을 바꿀 수 있음을 지적한다. 실제 코드 경로와 후보별 주의점은 [학습 설계 상세](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/training_design.md)에 정리했다.

## 6. 아직 끝나지 않은 항목

- `vit-supcon5`: 원격 확인 시 실행 중. 최종 수치는 아직 없다.
- `heldout-frozen`: 종료 코드는 0이지만 manifest 누락으로 test 임베딩이 저장되지 않았다.
- heldout seed43/44, frozen seed43/44, DINOv2 후속 fine-tuning: 준비 파일은 있으나 최종 반복 결과는 확인되지 않았다.
- Gap의 동일 조건 SupCon baseline, GalCov의 중복 제외 성능과 정확한 F1 집계, 기존 checkpoint의 학습 범위는 추가 검증 대상이다.

## 7. 결과 파일

| 파일 | 용도 |
|---|---|
| [로컬 상세 보고서](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/local_results.md) | canonical implantsj의 프로토콜·결과·근거 링크 |
| [로컬 실험 목록](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/local_experiments.csv) | 실험군별 설정·checkpoint·결과 원문 경로 |
| [로컬 파일 목록](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/local_artifact_inventory.csv) | 결과 파일 4,598개의 경로·크기·시각 |
| [다른 폴더 상세 보고서](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/other_local_results.md) | 초기 Docker·노트북·manifest 오류·중복 checkpoint |
| [다른 폴더 실험 목록](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/other_local_experiments.csv) | implantsj 밖의 실험과 확인 수준 |
| [원격 상세 보고서](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/remote_results.md) | 현재 수치·완료 근거·실패·진행 상태 |
| [원격 실험 목록](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/remote_experiments.csv) | 모델·분류기·그룹별 63개 지표/상태 기록 |
| [학습 설계 상세](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/training_design.md) | 실제 코드 구조·loss 후보·누수 방지·논문 근거 |

문서 검증: CSV 4개 파싱, 원격 전체/seen/unseen 표본 수와 정확도 가중 평균, 로컬 Markdown 링크 32개, 표 열 수를 확인했다. 실험 목록의 행 수는 독립 학습 실행 횟수가 아니다.

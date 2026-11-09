---
type: experiment-ledger
project: implantsj
status: current
updated: 2026-08-07
---

# 임플란트 분류·Metric Learning 전체 실험 정리

기준일: 2026-08-07  
범위: 로컬에 보존된 결과 파일, 종합 보고서, 기존 Notion 실험 기록에서 수치와 조건을 확인할 수 있는 실험만 포함했다. 미보존 수치는 추정하지 않았다.

## 1. 핵심 결론

1. **폐쇄형 분류의 가장 안정적인 기준선은 10-label plain ViT**다. Test Accuracy **95.54%**, Macro-F1 **88.12%**를 기록했다.
2. **기존 7→3 direct metric transfer는 실제로 20~30%대였다.** 동일 epoch-15 SupCon checkpoint의 backbone cosine 결과는 **27.52% / 26.87%**, projection cosine은 **31.19% / 29.75%**였다.
3. **Mahalanobis는 target-train label로 covariance를 추정할 수 있을 때 가장 강했다.** Fresh SupCon 7→3의 backbone에서 **98.21% / 92.94%**, CE 7→3에서 **95.07% / 84.19%**였다. 다만 이는 strict zero-shot이 아니라 **target-supervised gallery 평가**다.
4. **공정하게 group-clean한 uncropped 7→3에서는 학습하지 않은 ViT-Base가 CE·SupCon·Triplet보다 좋았다.** Euclidean/Cosine 관측 최고는 **82.76% / 64.16%**였다.
5. **Prototype은 Mean이 가장 안정적이고 SS는 변동이 컸다.** Uncropped 전체 평균 Macro-F1은 Mean **48.26%**, GT **45.10%**, SS **36.77%**였다.
6. **Euclidean과 Cosine의 우열은 작고 조건 의존적이었다.** Uncropped 전체 평균 Macro-F1 차이는 **0.18%p**에 불과했다. 반면 Mahalanobis는 target covariance를 사용할 때 큰 폭으로 개선됐다.
7. **Accuracy 단독 해석은 위험하다.** Bicon 비율이 높은 3-label 데이터에서는 majority prediction만으로도 Accuracy가 높아질 수 있으므로 Macro-F1, Balanced Accuracy, 클래스별 recall을 함께 봐야 한다.

## 2. 용어와 해석 기준

- **폐쇄형 분류**: 학습과 test의 label 집합이 같다. 일반 classifier의 Accuracy/F1이다.
- **Unseen-label transfer**: encoder 학습에는 없던 target label을 평가한다.
- **Target-supervised gallery**: target-train 정답으로 prototype 또는 covariance를 만든다. Encoder 관점에서는 unseen label이지만 strict zero-shot은 아니다.
- **Strict zero-shot**: target label의 정답 이미지나 통계를 전혀 사용하지 않는다. 현재 보고된 7→3 prototype 실험에는 해당하지 않는다.
- **GT prototype**: 별도 GT 또는 target validation 이미지의 클래스별 평균 embedding이다. 보고서마다 GT의 출처가 다르므로 protocol을 함께 확인해야 한다.
- **Mean prototype**: target train 전체의 클래스별 평균 embedding이다.
- **SS prototype**: target train에서 소수 샘플, 주로 label당 5장을 뽑아 평균한 embedding이다.
- 표의 모든 결과는 **Accuracy / Macro-F1(%)** 순서다. 서로 다른 데이터·split·query 수의 수치는 직접 우열 비교하지 않는다.

## 3. 전체 실험 결과표

### 3.1 폐쇄형 분류 및 학습 실험

<table fit-page-width="true" header-row="true">
	<tr>
		<td>실험</td>
		<td>모델·학습</td>
		<td>데이터</td>
		<td>주요 결과</td>
		<td>판정·비고</td>
	</tr>
	<tr>
		<td>3-label plain ViT</td>
		<td>ViT-Base, CE, 80 epoch, strong augmentation, class weight</td>
		<td>3-label canonical</td>
		<td>Test **94.50 / 77.00**</td>
		<td>전용 3-label classifier 기준선. 불균형 때문에 Accuracy 대비 Macro-F1이 낮음.</td>
	</tr>
	<tr>
		<td>7-label plain ViT</td>
		<td>ViT-Base, CE, 25 epoch</td>
		<td>7-label canonical</td>
		<td>Test **94.40 / 88.29**</td>
		<td>안정적인 7-label supervised 기준선.</td>
	</tr>
	<tr>
		<td>10-label plain ViT</td>
		<td>ViT-Base, CE, 80 epoch, strong augmentation</td>
		<td>3-label+7-label 통합</td>
		<td>Test **95.54 / 88.12**</td>
		<td>현재 폐쇄형 supervised 최고 기준선.</td>
	</tr>
	<tr>
		<td>11-label cropped plain ViT</td>
		<td>ImageNet-pretrained ViT-Base/16, CE, 15 epoch, 224px, strong augmentation, class weight 없음, seed 42</td>
		<td>`11label_crop_final_8_1_1`; train/valid/test 797/102/101장</td>
		<td>Best epoch 11 Valid **81.37 / 81.89**; Test **82.18 / 75.67**</td>
		<td>Test 83/101 정답. `nucleoss` 0/2, `mis` 1/3으로 소수 클래스 변동성이 큼. 10-label 80-epoch 결과와 직접 비교 금지.</td>
	</tr>
	<tr>
		<td>7-label SupCon</td>
		<td>ViT-Base, SupCon, cosine prototype 평가</td>
		<td>7-label canonical</td>
		<td>Test **95.87 / 75.12**</td>
		<td>Accuracy는 높지만 소수 클래스 Macro-F1이 약함.</td>
	</tr>
	<tr>
		<td>10-label→3-label fine-tune</td>
		<td>10-label ViT checkpoint를 3-label로 FT</td>
		<td>3-label canonical</td>
		<td>Test **91.74 / 71.73**</td>
		<td>Validation은 개선됐지만 3-label plain 기준선보다 낮음.</td>
	</tr>
	<tr>
		<td>10-label two-stage</td>
		<td>10-label main + NORIS/osstem binary submodel</td>
		<td>10-label test 448장</td>
		<td>**95.09 / 87.86**</td>
		<td>Main 단독 95.54 / 88.12보다 하락. 기본 전략으로 비추천.</td>
	</tr>
	<tr>
		<td>Binary submodels</td>
		<td>NORIS/osstem, ADIN/MIS</td>
		<td>각 confusion pair</td>
		<td>NORIS/osstem **94.08 / 93.96**; ADIN/MIS **98.39 / 95.70**</td>
		<td>Submodel 자체는 좋지만 main handoff 오차 때문에 전체 이득 없음.</td>
	</tr>
	<tr>
		<td>10-label K-fold</td>
		<td>Plain CE vs SupCon</td>
		<td>10-label validation folds</td>
		<td>CE **87.13 / 83.56**; SupCon **96.02 / 90.54**</td>
		<td>Validation 평균이며 별도 test 결과와 직접 비교 금지.</td>
	</tr>
	<tr>
		<td>이전 7-label runs</td>
		<td>Class weight·strong augmentation 변형</td>
		<td>구형 6/7-label manifests</td>
		<td>Test Acc. **41.23%** run, Valid **46.15 / 58.54** 미완료 run, Test Acc. **98.39%** naming-mismatch run</td>
		<td>데이터명 불일치·미완료로 대표 결과에서 제외.</td>
	</tr>
	<tr>
		<td>27-label ClosedFT</td>
		<td>ViT, CE, 10 epoch</td>
		<td>Closed 27-label</td>
		<td>Test **95.43 / 92.67**</td>
		<td>동일 label classifier 성능은 높지만 GT27 metric 전이는 Base보다 낮았음.</td>
	</tr>
	<tr>
		<td>27-label ClosedFT-SupCon</td>
		<td>ViT, SupCon, 보존된 3/10 epoch</td>
		<td>Closed 27-label</td>
		<td>Test **87.65 / 79.30**</td>
		<td>부분 기록이라 10-epoch CE와 loss 효과 직접 비교 불가.</td>
	</tr>
	<tr>
		<td>42-label FullFT</td>
		<td>ViT, CE, 기존 54 epoch</td>
		<td>Full 42-label</td>
		<td>Valid 최고 **95.14 / 91.06**</td>
		<td>Per-epoch test 지표 미보존.</td>
	</tr>
	<tr>
		<td>VLM feature-only</td>
		<td>구조 feature + tabular ML</td>
		<td>7-label Effective10·AI Hub 변형</td>
		<td>Test Accuracy 대체로 **32~45%**</td>
		<td>ViT image baseline보다 낮음. Answer-key 식별 정보 사용 시 leakage 위험.</td>
	</tr>
</table>

### 3.2 동일 라벨 Prototype·거리 진단

<table fit-page-width="true" header-row="true">
	<tr>
		<td>실험</td>
		<td>모델·데이터</td>
		<td>방식</td>
		<td>주요 결과</td>
		<td>판정·비고</td>
	</tr>
	<tr>
		<td>42-class ImageNet baseline</td>
		<td>ViT-Large, 42-class test 16,570장</td>
		<td>Class당 train 10장 Mean</td>
		<td>Euc **17.53 / 12.84**; Cos **17.59 / 12.75**</td>
		<td>추가 학습 없는 42-way 기준선. Euc/Cos 사실상 동일.</td>
	</tr>
	<tr>
		<td>GT27 5-model benchmark</td>
		<td>ViT-Base, ClosedFT, ClosedFT-SupCon, FullFT, FullFT-SupCon; query 52장</td>
		<td>27-way GT few-shot, Euc/Cos</td>
		<td>Base Cos **51.92 / 52.80**; FullFT Euc **51.92 / 52.44**; ClosedFT Cos **44.23 / 42.46**</td>
		<td>Base가 ClosedFT보다 우세. Query가 작아 1장당 Accuracy 1.92%p.</td>
	</tr>
	<tr>
		<td>GT27 SupCon 비교</td>
		<td>ClosedFT-SupCon, FullFT-SupCon</td>
		<td>GT prototype, Euc/Cos</td>
		<td>Closed SupCon Euc **44.23 / 47.92**; Full SupCon Euc **42.31 / 42.79**</td>
		<td>현재 GT27에서는 SupCon의 일반 표현 개선 근거 없음.</td>
	</tr>
	<tr>
		<td>GT27 Band-loss 참고</td>
		<td>`vit_band_6b_lam0.8.pt`</td>
		<td>GT prototype, cosine</td>
		<td>Accuracy **61.54%**</td>
		<td>12개 checkpoint 중 최고 top-1이나 canonical 5-model 비교에서는 제외.</td>
	</tr>
	<tr>
		<td>42FT local 3-label</td>
		<td>Frozen ViT-Large 1024-D, cropped 3-label test 446장</td>
		<td>GT/Mean/SS × Euc/Cos</td>
		<td>Mean+Cos **91.70 / 76.93**</td>
		<td>Target-supervised gallery. Train-test exact duplicate 5장.</td>
	</tr>
	<tr>
		<td>42FT local 7-label</td>
		<td>동일 backbone, cropped 7-label test 141장</td>
		<td>GT/Mean/SS × Euc/Cos</td>
		<td>SS+Euc **26.24 / 15.88**</td>
		<td>Test에 nobel이 없어 6/7 label만 존재.</td>
	</tr>
	<tr>
		<td>42FT local 12-label</td>
		<td>동일 backbone, cropped 12-label test 512장</td>
		<td>GT/Mean/SS × Euc/Cos</td>
		<td>Mean+Cos **36.72 / 36.08**</td>
		<td>요청된 11-label이 아니라 로컬에 존재한 12-label 전체 평가.</td>
	</tr>
	<tr>
		<td>42FT Mean + 4거리</td>
		<td>동일 frozen backbone, cropped 3/7/12-label</td>
		<td>Euc/Man/Cos/Mahalanobis</td>
		<td>Mah: 3-label **98.43 / 95.65**; 7-label **61.70 / 26.89**; 12-label **77.34 / 70.16**</td>
		<td>세 데이터 모두 Mahalanobis 최고. Target-train covariance 사용.</td>
	</tr>
	<tr>
		<td>Attention map</td>
		<td>42FT frozen backbone, 3/7/12-label</td>
		<td>GT/Mean/SS × Euc/Cos, 각 방식 100장</td>
		<td>Overlay **1,800장**, 고유 attention 계산 300장</td>
		<td>시각적 진단 자료이며 성능 지표가 아님.</td>
	</tr>
	<tr>
		<td>Prototype t-SNE</td>
		<td>ClosedFT, ClosedFT-SupCon backbone/projection, FullFT</td>
		<td>GT+Mean+SS</td>
		<td>표현당 110점: GT 26 + Mean 42 + SS 42</td>
		<td>GT coverage 26/42, Mean·SS 42/42. 정성 시각화로만 사용.</td>
	</tr>
</table>

### 3.3 7-label 학습 → 3-label unseen-label transfer

<table fit-page-width="true" header-row="true">
	<tr>
		<td>실험</td>
		<td>모델·데이터</td>
		<td>방식</td>
		<td>Test 결과</td>
		<td>판정·비고</td>
	</tr>
	<tr>
		<td>과거 direct projection</td>
		<td>ViT-Small SupCon, canonical 3-label</td>
		<td>Mean prototype, cosine 계열</td>
		<td>**34.86 / 29.26**</td>
		<td>사용자가 기억한 20%대 Macro-F1 결과.</td>
	</tr>
	<tr>
		<td>과거 direct backbone</td>
		<td>동일 계열</td>
		<td>Backbone prototype</td>
		<td>**35.78 / 29.88**</td>
		<td>Projection보다 소폭 높지만 여전히 약함.</td>
	</tr>
	<tr>
		<td>후속 7-label SupCon direct</td>
		<td>ViT-Base SupCon</td>
		<td>3-label prototype</td>
		<td>**37.61 / 34.75**</td>
		<td>직접 전이는 여전히 전용 3-label classifier보다 낮음.</td>
	</tr>
	<tr>
		<td>Metric two-stage</td>
		<td>ViT-Small, 7→3</td>
		<td>Group KNN + group 내부 prototype</td>
		<td>**68.81 / 54.92**</td>
		<td>Direct transfer보다 개선됐지만 3-label plain보다 낮음.</td>
	</tr>
	<tr>
		<td>Full uncropped ViT-Base</td>
		<td>Untrained ViT-Base, group-clean target test 116장</td>
		<td>GT/Mean/SS × Euc/Cos</td>
		<td>GT+Cos **82.76 / 64.16**</td>
		<td>CE·SupCon·Triplet보다 최고. Majority baseline Accuracy 81.90%.</td>
	</tr>
	<tr>
		<td>Full uncropped CE</td>
		<td>ViT-Base, source 7-label CE 15 epoch</td>
		<td>GT/Mean/SS × Euc/Cos</td>
		<td>Mean+Euc **72.41 / 50.14**; GT+Euc **70.69 / 50.33**</td>
		<td>Source test 97.94%지만 unseen transfer는 Base보다 낮음.</td>
	</tr>
	<tr>
		<td>Full uncropped SupCon</td>
		<td>ViT-Base, source 7-label SupCon 15 epoch</td>
		<td>Backbone·projection</td>
		<td>Backbone **53.45 / 39.88**; projection **44.83 / 35.93**</td>
		<td>Projection이 더 낮아 source objective 특화 가능성.</td>
	</tr>
	<tr>
		<td>Full uncropped Triplet</td>
		<td>ViT-Base, source 7-label Triplet 15 epoch</td>
		<td>Backbone·projection</td>
		<td>Backbone **68.10 / 55.48**; projection **62.93 / 47.03**</td>
		<td>학습 모델 중 Macro-F1은 가장 높지만 Base 미달.</td>
	</tr>
	<tr>
		<td>Uncropped Base + Mahalanobis</td>
		<td>동일 group-clean test 116장</td>
		<td>GT/Mean/SS + pooled Ledoit-Wolf covariance</td>
		<td>GT **90.52 / 70.57**; Mean **93.10 / 79.69**; SS **84.48 / 67.46**</td>
		<td>동일 Base Euc/Cos보다 개선. Target support 1,839장 사용.</td>
	</tr>
	<tr>
		<td>Cropped CE 30 epoch</td>
		<td>ViT-Base CE, best epoch 11; target test 446장</td>
		<td>Mean + 4거리</td>
		<td>Euc **73.09 / 48.74**; Cos **73.32 / 48.83**; Mah **95.07 / 84.19**</td>
		<td>Exact duplicate 5장 제외 후 Mah **95.01 / 84.18**.</td>
	</tr>
	<tr>
		<td>Legacy 조건 재구성</td>
		<td>ViT-Small SupCon epoch 15; target test 109장</td>
		<td>동일 checkpoint Cos vs Mah</td>
		<td>Backbone Cos **27.52 / 26.87** → Mah **63.30 / 48.23**; Projection Cos **31.19 / 29.75** → Mah **60.55 / 44.77**</td>
		<td>Balanced Acc.는 backbone 51.34→50.77로 개선되지 않음. Bego recall 감소.</td>
	</tr>
	<tr>
		<td>Fresh SupCon 30 epoch</td>
		<td>ViT-Small SupCon, best epoch 26; target test 446장</td>
		<td>Mean + 4거리, backbone·projection</td>
		<td>Projection Cos **71.97 / 50.17** → Mah **93.27 / 81.46**; Backbone Cos **75.11 / 56.48** → Mah **98.21 / 92.94**</td>
		<td>Source test는 54.73 / 41.19. 높은 target 결과는 SupCon보다 target covariance 효과로 해석.</td>
	</tr>
</table>

## 4. 실험 방식별 특징·장단점

### 4.1 Plain supervised CE

- **특징**: Classifier head를 포함해 정답 label을 직접 예측하도록 학습한다.
- **장점**: 동일 label 폐쇄형 분류에서 가장 단순하고 안정적이다. 10-label에서 95.54 / 88.12를 기록했다.
- **단점**: 학습 label 경계에 특화되어 unseen label용 metric 표현이 약해질 수 있다. ClosedFT의 classifier 성능은 높았지만 GT27·7→3 transfer에서는 Base보다 낮았다.
- **사용 모델·데이터**: ViT-Base/ViT 계열, 3·7·10·27·42-label.
- **비고**: 실제 분류 성능 기준선으로 유지하고 metric transfer와 별도 표에서 비교한다.

### 4.2 SupCon metric learning

- **특징**: 같은 class embedding은 가깝게, 다른 class는 멀게 만들도록 학습한다. Projection head를 함께 쓰는 경우가 많다.
- **장점**: Prototype·retrieval·embedding 시각화에 자연스럽고, 7-label 동일 분포 Accuracy는 95.87%까지 나왔다.
- **단점**: 소수 class Macro-F1과 unseen-label transfer가 불안정했다. Fresh 실험에서도 projection보다 backbone이 높아 projection의 source objective 과특화가 관찰됐다.
- **사용 모델·데이터**: ViT-Base/ViT-Small, source 7·27-label, target 3-label.
- **비고**: 높은 Mahalanobis 결과를 SupCon 단독 효과로 주장하지 않는다.

### 4.3 Triplet learning

- **특징**: Anchor-positive-negative 간 상대 거리와 margin을 최적화한다.
- **장점**: Full uncropped 7→3에서 학습 모델 중 가장 높은 Macro-F1 55.48%를 기록했다.
- **단점**: Mining·margin·batch 구성에 민감하고 source validation이 CE/SupCon보다 낮았다. Projection 전이도 backbone보다 낮았다.
- **사용 모델·데이터**: ViT-Base, source 7-label, uncropped target 3-label.
- **비고**: Base를 넘지 못해 현재 기본 방식으로 선택하지 않는다.

### 4.4 GT·Mean·SS prototype

- **GT 특징**: 별도 GT나 validation support의 평균. 의미가 명확하지만 sample 수와 label coverage가 제한될 수 있다.
- **Mean 특징**: Target train 전체 평균. 가장 안정적이고 uncropped 평균 Macro-F1 48.26%로 세 방식 중 최고였다.
- **SS 특징**: Label당 약 5장 평균. 계산비용은 작지만 seed와 source 중복에 민감해 uncropped 평균 Macro-F1 36.77%로 가장 낮았다.
- **장점**: Encoder 재학습 없이 새 class gallery를 추가할 수 있다.
- **단점**: Label 내부가 여러 촬영 양식으로 나뉘면 단일 centroid가 class를 대표하지 못한다. Mean·SS는 target 정답을 사용하므로 zero-shot이 아니다.

### 4.5 Euclidean·Manhattan·Cosine·Mahalanobis

- **Euclidean**: 가장 단순한 L2 거리. 해석이 쉽지만 embedding scale과 축별 분산 차이에 민감하다.
- **Manhattan**: L1 거리. 일부 조건에서 미세한 차이는 있었지만 일관된 우위는 없었다.
- **Cosine**: Vector 방향을 비교해 크기 영향이 작다. Euc보다 약간 좋은 경우가 있었지만 uncropped 평균 차이는 0.18%p뿐이었다.
- **Mahalanobis**: Target train의 covariance로 축별 scale과 상관관계를 보정한다. 현재 cropped/uncropped 실험에서 가장 높은 결과를 냈다.
- **Mahalanobis 단점**: 충분한 labeled target train, covariance 추정과 shrinkage가 필요하다. Strict zero-shot이 아니며 class imbalance 또는 target 분포 적합 효과를 성능 향상으로 오해할 수 있다. 현재 방식에서 Mahalanobis는 **학습 loss가 아니라 평가 거리**다.

### 4.6 Two-stage 재분류

- **특징**: Main model이 혼동하는 label group만 별도 submodel 또는 group 내부 prototype으로 재판정한다.
- **장점**: Direct 7→3 metric transfer를 37.61 / 34.75에서 68.81 / 54.92 수준으로 개선했다.
- **단점**: 10-label NORIS/osstem에서는 main 단독보다 전체 성능이 하락했다. 1단계 trigger 오류가 2단계 성능을 상쇄하고 운영 복잡도가 증가한다.
- **사용 모델·데이터**: 10-label main+binary submodel, 7→3 group KNN+prototype.
- **비고**: Confidence threshold가 있는 제한적 보정으로만 재검토한다.

### 4.7 Frozen backbone + target gallery

- **특징**: Encoder는 고정하고 target train으로 prototype·covariance만 만든다.
- **장점**: 새 target class를 빠르게 추가하고 거리 함수를 공정하게 비교할 수 있다.
- **단점**: Target label을 사용하므로 strict zero-shot이 아니며, encoder 학습 데이터와 target overlap을 모르면 일반화 주장에 한계가 있다.
- **사용 모델·데이터**: 42-class FT ViT-Large 1024-D, 7-label CE/SupCon ViT, cropped/uncropped 3·7·12-label.
- **비고**: 현재 Mahalanobis 최고점은 이 protocol에서 나온 값이다.

### 4.8 Attention map·t-SNE

- **특징**: Attention rollout은 모델이 본 위치를, t-SNE는 embedding의 2-D 구조를 시각화한다.
- **장점**: ROI를 보는지, class가 섞이는지, backbone과 projection이 어떻게 다른지 설명하기 좋다.
- **단점**: Accuracy/F1의 대체 근거가 아니며 t-SNE 결과는 seed·sampling에 민감하다.
- **사용 모델·데이터**: 42FT 3/7/12-label attention 1,800장, ClosedFT/FullFT/SupCon/Triplet prototype t-SNE.
- **비고**: 정성 보조자료로만 사용한다.

### 4.9 VLM 구조 feature-only

- **특징**: 영상에서 구조 attribute를 텍스트/표 형식으로 추출한 뒤 tabular classifier를 학습한다.
- **장점**: 특징 의미를 설명하기 쉽고 임상 구조 attribute 분석에 활용할 수 있다.
- **단점**: 7-label Test Accuracy가 대체로 32~45%로 image ViT보다 낮고, answer-key 식별 정보가 들어가면 label leakage가 발생한다.
- **사용 모델·데이터**: Effective10·AI Hub 구조 feature 실험.
- **비고**: 단독 classifier보다 ViT 보조 feature 또는 오류 분석용이 적절하다.

## 5. 비교 시 반드시 지킬 조건

1. **폐쇄형 classifier와 unseen-label prototype 결과를 한 순위표에 섞지 않는다.**
2. **Uncropped 116장, legacy 109장, current cropped 446장은 서로 다른 target split이다.**
3. **34.86 / 29.26과 fresh 98.21 / 92.94는 checkpoint·target 데이터·covariance 사용이 모두 달라 직접 개선폭으로 보고하지 않는다.**
4. **Target train으로 prototype 또는 covariance를 만들었으면 zero-shot이라는 표현을 쓰지 않는다.**
5. **Accuracy와 함께 Macro-F1, Balanced Accuracy, per-class recall을 제시한다.**
6. **Exact duplicate 제거만으로 group·near-duplicate·환자 단위 독립성이 증명되지는 않는다.**
7. **GT27은 query 52장, uncropped ITI는 7장뿐이므로 작은 표본의 변동성을 명시한다.**
8. **Mac MPS에서는 CPU→MPS 전송에 `non_blocking=True`를 쓰지 않는다.** 2026-08-07 재현에서 label tensor 손상과 `adin` 단일 예측을 확인했으며, `src/supervised.py`가 비동기 전송을 CUDA에서만 사용하도록 수정된 뒤 11-label 결과를 다시 산출했다. 수정 전 결과는 무효 처리했다.

## 6. 교수님 발표용 최종 결론

> 동일 label을 직접 분류하는 문제에서는 plain ViT가 가장 안정적이었고, 10-label 모델이 95.54% Accuracy와 88.12% Macro-F1로 최고 기준선이었다. 반면 7-label에서 학습한 metric 모델을 새로운 3-label에 직접 전이하면 기존 성능은 Accuracy 27~38%, Macro-F1 27~35% 수준으로 낮았다. Target train label로 prototype과 covariance를 만든 gallery 평가에서는 Mahalanobis가 최대 98.21% / 92.94%까지 향상됐지만, 이는 strict zero-shot이 아니라 target-supervised metric 평가다. 따라서 현재 결론은 “SupCon이 unseen label을 해결했다”가 아니라 “target covariance를 반영한 Mahalanobis가 해당 target split에서 매우 효과적이었다”이다.

## 7. 근거 파일

- `docs/analysis/EXPERIMENT_SUMMARY.md`
- `docs/analysis/PRESENTATION_SUMMARY.md`
- `docs/reports/METRIC_DIAGNOSIS_FINAL_REPORT.md`
- `docs/reports/full_uncropped_7to3_metric/report/REPORT.md`
- `docs/reports/full_uncropped_7to3_metric/mahalanobis_addendum_20260807/REPORT_TABLE.md`
- `docs/reports/vit42ft_local_metric_20260806/REPORT.md`
- `docs/reports/vit42ft_local_mean_4dist_20260806/REPORT.md`
- `docs/reports/vit7ce_to_3label_mahalanobis_20260806/REPORT.md`
- `docs/reports/legacy_7to3_reproduction_20260807/REPORT.md`
- `docs/reports/vit7supcon_to_3label_mahalanobis_20260807/REPORT.md`
- `outputs/20260807_142042_vit_base_11label_crop_final_15e_seed42_mpsfix/metrics/test_metrics.json`
- `outputs/20260807_142042_vit_base_11label_crop_final_15e_seed42_mpsfix/confusion_matrices/test.csv`
- Notion 원본 기록: [1차 임플란트 결과](https://app.notion.com/p/39c0d5e377fd80c19fb5fe643d291f6c)

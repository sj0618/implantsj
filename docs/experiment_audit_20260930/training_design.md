# 임베딩 거리 문제: 학습 설계 점검

## 권고

1. **현재 Gap-SupCon은 채택하지 않는다.** 현재 원격 기록에 Gap-SupCon 단독 결과만 있고, 같은 초기 가중치·분할·평가 조건의 SupCon 단독 대조군이 없다. 기준선 작업은 `exit 143`으로 끝나 결과가 없다.
2. **표현 차원별 기준점을 분리한다.** pretrained backbone의 native 768차원 동결 기준선은 별도로 보고한다. SupCon과 기존 Gap-SupCon은 같은 초기화와 128차원 projection을 사용해 직접 비교한다.
3. **loss가 실제로 학습에 기여하는지 확인한다.** 현재 선택된 `margin=0.05`, `weight=0.01`은 기록된 평균 손실 기준으로 gap 항의 가중치가 SupCon 항에 비해 매우 작다. 수치만으로 기울기 기여까지 단정할 수 없으므로 active gap 비율과 각 항의 기울기도 기록한다.
4. **ArcFace margin을 키우는 실험은 우선하지 않는다.** M8에서 중심 거리는 늘었지만 클래스 내부 산포도 커졌고, unseen 정확도는 M7보다 낮았다.

## 확인한 구현과 결과

### 학습 코드 경로

현재 작업 worktree와 canonical 저장소의 코드가 서로 다르다. 실험을 시작하기 전 실제 실행 경로와 effective config를 한 번 더 맞춰야 한다.

| 경로 | 확인 내용 | 설계상 주의점 |
|---|---|---|
| 현재 worktree `src/metric_learning.py` | 학습 배치는 무작위 섞기이며, 두 개의 독립 증강 view로 SupCon을 계산한다. 지원하는 loss는 SupCon과 Triplet이다. 각 epoch에 test metric도 기록한다. | 한 클래스의 서로 다른 이미지가 같은 batch에 들어온다는 보장이 없어, 두 view가 같은 원본에서 나온 positive의 대부분일 수 있다. test 결과는 epoch별 선택에 쓰지 말고 최종 선택 뒤 한 번만 확인한다. |
| canonical `implantsj/src/metric_learning.py` 및 `implantsj/src/experiment_data.py` | `P×K` sampler, 두 view, 정규화된 projection 출력, SupCon/Triplet을 사용한다. valid Macro-F1로 checkpoint를 고르고 test는 학습이 끝난 뒤 평가한다. 기본 `K=2`다. | 이 경로를 쓴다면 manifest에 `K`장 이상 있는 클래스 수와 실제 `P`, `K`를 기록한다. |
| canonical `implantsj/src/arcface.py` | ArcFace는 별도 실행 경로다. 정규화된 128차원 embedding과 정규화된 클래스 weight에 target angular margin을 적용한 뒤 CE를 계산한다. 기본값은 `s=30`, `m=0.5`이며 일반 무작위 batch를 쓴다. | `m=0.5`는 각도 단위다. Gap의 L2 거리 margin `0.05`와 숫자만 비교할 수 없다. 원격 M8의 effective config가 이 local default와 같은지는 별도 확인해야 한다. |
| canonical `implantsj/tools/train_covariance_aware_supcon.py` | 5-shot support와 5-shot query를 train split에서 중복 없이 뽑는다. projection에서 SupCon을 계산하고, 기본 설정에서는 backbone embedding에 공분산 분류 loss를 더한다. `loss = supcon + weight × covariance`로 역전파한다. | 두 항이 서로 다른 표현 공간을 학습할 수 있다. M10 결과는 낮았으므로 Gap에 공분산 항까지 한꺼번에 더하지 않는다. |
| 원격 실행 snapshot `/home/ubuntu/aihub_run_new/outputs/experiment_records/gap-supcon-b256-20260929/`의 launcher 기록의 `driver.py`, `src/metric/gap.py`, `evaluation.py` | `z`를 L2 정규화하고 support 평균 `p_c`는 다시 정규화하지 않는다. 클래스 반지름은 query와 평균 중심 사이 RMS 거리다. 유효 간격은 `‖p_c-p_j‖ - r_c-r_j`이며, 가장 가까운 클래스 간격이 margin보다 작으면 제곱 hinge loss를 준다. | support/query는 모두 train role에서 뽑은 학습 자료다. 평가용 target query는 gradient, margin, loss weight, checkpoint 선택에 쓰면 안 된다. |

현재 worktree의 기본 metric 코드에는 `src/arcface.py`, CovAware 학습기, Gap 학습기가 없다. canonical 저장소에는 ArcFace와 CovAware 학습기가 따로 있다. 로컬 `/Users/yunseojune/TRIA_LAB/implant_project/Aihub/` 폴더에도 해당 학습 코드가 없고, 원격 학습 코드는 위 경로에서 실행됐다. 따라서 local trainer 기본값을 원격 M7/M8/Gap의 설정으로 간주하지 않는다.

### 저장된 거리 결과와 원격 실행 기록

아래 결과는 기준 조건이 다르므로 한 줄 순위표로 섞지 않는다.

| 근거 | 관찰 | 해석 범위 |
|---|---|---|
| 2026-09-29 M7/M8 보고서, 128차원 projection의 unseen 거리 (`docs/reports/professor_report_20260928/geometry-screen-transcription.csv`) | M7: 클래스별 pairwise Euclidean 내부 거리 평균 `0.584282`, 클래스 중심 간 평균 거리 `0.654811`. M8: 각각 `0.929945`, `0.961901`. Kmeans+Mahalanobis unseen 정확도는 M7 `0.821`, M8 `0.741`. | ArcFace는 중심 거리를 늘렸지만 pairwise 내부 거리도 커졌다. Gap의 RMS 반지름과 이 보고서의 pairwise 평균은 정의가 다르므로 수치로 직접 비교하지 않는다. M7/M8은 gallery covariance를 사용해 strict zero-shot 결과가 아니다. |
| 2026-09-30 원격 ViT galcov | `galcov.sh`가 `/home/ubuntu/aihub_run_new/checkpoints/backbones/20260918-170637-6ea9b483/vit_full_closed27_open15_c001.pt`와 `/home/ubuntu/aihub_run_new/checkpoints/backbones/20260918-170637-6ea9b483/band-vit_full_closed27_open15_c001.pt`를 읽는다. 추가 embedding loss는 `none`이다. ViT Kmeans5+OAS: 전체 정확도 `0.9377`, Macro-F1 `0.8299`, closed accuracy `0.9446`, open accuracy `0.8518`. Band-ViT open 정확도 `0.8419`. | closed27_open15 이름의 저장 checkpoint를 쓴 비교 조건이며, 원래 학습 범위 전체는 이번에 재검증하지 않았고, 별도 metric loss를 추가하지 않았다. unseen gallery로 공분산을 추정하므로 closed-only pooled OAS를 쓴 Gap 결과와 직접 비교할 수 없다. |
| 2026-09-30 Gap-SupCon | DINOv2 base 224, implant checkpoint 없이 offline pretrained 시작, 128차원 projection, batch 256, 5 epoch, seed 42. heldout 5개 클래스의 선택 단계에는 `P=16,K=16`이 기록됐다. 선택값은 `margin=0.05`, `weight=0.01`. final closed27 학습 각 batch의 실제 distinct-class `P`는 확인되지 않았다. | 단일 seed의 탐색 결과다. `selection.csv`는 Gap 후보 4개만 담고, `comparison.csv`에는 `final_gap` 결과 9행만 있다. SupCon 단독 baseline은 없다. 재현에서는 sampler가 실제 공급한 `P`와 `K`를 batch별로 기록한다. |
| 같은 Gap 실행의 학습 기록 | SupCon loss `3.62428665`, gap loss `0.01642270`, `selected.json`의 train nearest gap `0.3425928354263305`, train radius `0.1929574310779571`; heldout validation gap `-0.42414`, radius `0.42797`. | `weight×gap loss`와 SupCon loss의 단순 비는 약 `0.0045%`다. 다만 손실 크기 비율은 기울기 비율이 아니므로 실제 기여를 판정하려면 기울기와 active 비율을 봐야 한다. train과 heldout geometry 차이는 일반화 확인이 필요함을 보여준다. |
| 같은 Gap 실행의 unseen 평가 | `n=1,221`: cosine accuracy `0.3432`, macro accuracy `0.3387`, Macro-F1 `0.3114`; pooled OAS는 `0.5045`, `0.4574`, `0.4833`. unseen radius `0.4302`, effective gap `-0.5763`, negative-gap fraction `1.0`. | 같은 모델 안에서는 cosine보다 pooled OAS가 높았지만, 훈련 전 backbone의 같은 split·support/query baseline이 없어 Gap의 개선 또는 하락을 말할 수 없다. 전체 unseen 클래스에서 계산된 negative-gap fraction도 확인됐다. |
| 기존 M10 CovAware 보고서 | 42클래스 원격 비교에서 전체 정확도 `0.335`, Macro-F1 `0.245`; Kmeans+Mahalanobis 전체 정확도 `0.353`, Macro-F1 `0.292`. | 실험이 exit 0으로 끝난 기록은 있지만 개선 결과는 아니다. 정확한 9/24 effective config 없이 특정 covariance 설정을 실패 원인으로 단정하지 않는다. |

Gap 평가는 정규화 embedding과 재정규화하지 않은 평균 prototype으로 RMS 반지름을 계산한다. 이 값은 cosine 또는 Mahalanobis 점수와 같은 거리 정의가 아니다. distance scorer를 바꿔도 geometry 표가 같다면 그 표는 scorer별 geometry가 아니라 동일 embedding geometry를 재사용한 것이다. 보고서에는 scorer별 예측 점수와 공통 geometry 열을 구분한다.

GalCov의 Macro-F1은 전체 클래스와 unknown을 전달한 원문 집계값이며, unseen 15개만의 Macro-F1과 구분한다. [원격 지표 정의](/Users/yunseojune/.codex/worktrees/0550/implantsj/docs/experiment_audit_20260930/remote_results.md)를 참조한다.

## 손실 방법 선택

| 방법 | 원리와 이 데이터에 맞는 이유 | 현재 판단 |
|---|---|---|
| SupCon | batch 안에서 같은 label 예제를 당기고 다른 label 예제를 밀어낸다. 두 view와 클래스별 여러 원본 이미지가 함께 있어야 서로 다른 동일 클래스 예제도 positive가 된다. | 기존 M7과 가장 가까운 기준 방법이며, 우선 baseline으로 재현한다. 논문: [Khosla et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html). |
| ArcFace | 클래스 weight를 각 클래스의 방향 중심처럼 두고, 정답 클래스 logit에 각도 margin을 적용한다. | 이미 M8에서 내부 산포가 커졌고 unseen 수치가 낮았다. margin을 더 키우기 전에 endpoint, batch sampler, margin active 비율, gradient가 실제 맞는지 확인한다. 논문: [Deng et al., CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html). |
| Proxy-Anchor | 클래스 proxy와 batch 예제 관계를 함께 사용한다. ProxyAnchor 논문은 pair 기반 loss의 복잡도와 proxy 기반 loss의 관계 정보 부족을 함께 다룬다. | SupCon baseline이 제대로 작동하는데도 hard negative 분리가 부족할 때 한 번 비교할 후보. local code에는 없으므로 먼저 추가하지 않는다. 논문: [Kim et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.pdf). |
| Sub-center ArcFace | 클래스마다 여러 중심을 두어 한 클래스 안의 여러 모드나 label noise를 허용한다. | 클래스 내부 산포가 촬영 상태별 여러 군집으로 반복 확인될 때 후순위 실험으로 검토한다. 현재 목표인 하나의 작고 안정된 prototype과는 충돌할 수 있다. 논문은 대규모 noisy face 자료에 대한 방법이다: [Deng et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560715.pdf). |

손실 이름을 바꾸기 전에 같은 backbone, 초기 가중치, batch 크기, view 수, optimizer, epoch/step, validation 규칙을 고정한다. metric learning의 공정한 비교와 test-feedback 위험은 [Musgrave et al., ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf)도 지적한다.

## 최소 ablation 순서

| 순서 | 조건 | 한 번에 바꾸는 것 | 판정 |
|---|---|---|---|
| 0 | Frozen pretrained backbone: native 768차원 | 학습하지 않은 pretrained backbone의 native embedding | 별도 기준점으로 보고한다. 학습된 128차원 projection과 직접 성능 순위를 매기지 않는다. galcov 결과는 기존 backbone checkpoint와 gallery covariance를 쓴 별도 참고점이며, checkpoint 원래 학습 범위의 전체 검증은 남아 있다. |
| 1 | SupCon only: 128차원 projection | Gap 항을 끄고 공통 DINOv2 초기화와 projection head를 사용 | Gap 추가 효과를 재는 필수 대조군. 같은 source train classes, 5 epoch, batch 256, view/augmentation, optimizer, seed, 명시한 `P=16,K=16` sampler를 맞춘다. 실제 batch의 P와 K를 기록한다. 이 실행 결과는 아직 없다. |
| 2 | SupCon + current Gap: 동일 128차원 projection | 기존 `margin=0.05`, `weight=0.01`을 재현 | 1번과 같은 초기화, sampler, augmentation, step 수, 평가에서 비교한다. 저장된 final Gap 결과 재현은 별도 확인한다. |
| 3 | 하나의 신규 후보: support-query episodic CE + SupCon, 128차원 | train episode의 disjoint support로 unit-normalized prototype을 만들고, normalized query와의 cosine logits에 CE를 계산한다. `L = L_supcon + λ_epi L_episode` 두 항만 결합한다. | 1–2번 재현 및 원인 점검 뒤 추가한다. episode는 train class 안에서 뽑고 support/query 원본은 겹치지 않게 한다. heldout source-class validation에서 Macro-F1/accuracy를 우선 보고, pairwise 클래스 내부 거리와 최근접 다른 클래스 중심 거리·간격을 함께 기록해 checkpoint와 계수를 고른다. 이 측정은 heldout source-class validation만 사용하고 unseen final test는 잠근다. compactness를 처음부터 세 번째 loss로 더하지 않는다. |
| 4 | 중심 margin vs 반지름 포함 gap | SupCon+center-distance hinge와 SupCon+`distance-radius_i-radius_j`를 비교 | 1–2번에서 개선 가능성이 보일 때만 진행한다. 반지름 항이 unseen-class generalization을 돕는지 확인하고 계수는 heldout source classes에서만 고른다. |
| 5 | Sampler 또는 augmentation 점검 | actual training path의 sampler가 불명확하면 `P×K` 공급 여부부터 로그로 확인한다. sampler와 augmentation은 별도로 바꾼다. | selection 단계의 `P=16,K=16`을 final closed27 학습에 그대로 가정하지 않는다. batch 256에서 서로 다른 클래스 수와 클래스별 원본 수를 직접 기록하고, sampler 문제가 확인될 때 변경한다. augmentation 비교는 그 다음 별도 시행한다. |

순서 4는 순서 1–3에서 진단 근거가 있을 때만 진행한다. Sub-center나 Proxy-Anchor를 포함한 전체 조합 탐색은 하지 않는다. 각 조건은 우선 동일한 사전 지정 3 seeds로 확인하고, 개선이 반복되면 기존 strict 5-shot 평가 seed set으로 확대한다. 이 seed/조건 수는 제안이며 아직 실행 검증되지 않았다.

## 거리 측정과 평가 누수 방지

모든 지표를 실제 최종 분류에 사용할 표현에서 계산하고, `backbone`과 `projection` 결과를 섞지 않는다. Gap의 training objective가 projection 128차원에 적용됐다면 128차원을 주 분석으로, backbone은 별도 참고 결과로 보고한다. pretrained backbone의 native 768차원 frozen 기준선은 별도 운영 참고점으로 둔다. 이를 random 128차원 projector 출력과 같은 endpoint로 놓고 직접 비교하지 않는다. SupCon과 Gap의 학습 효과는 동일한 128차원 projection endpoint끼리 비교한다.

1. 유한하고 0이 아닌 embedding만 남기고 `z = h / ‖h‖₂`로 정규화한다. 같은 checkpoint, preprocessing, class/sample manifest, label mapping으로 모든 조건의 feature를 만든다.
2. 클래스 중심은 `μ_c = mean(z)`로 두고, Euclidean 거리는 이 평균을 재정규화하지 않은 prototype 기준으로 계산한다. Cosine 점수는 별도로 `normalize(μ_c)`를 사용한다. 두 판정은 일반적으로 일치하지 않는다.
3. 클래스 내부 산포는 `r_c = sqrt(mean_i ‖z_i - μ_c‖₂²)`로 계산한다. 중심 간 최근접 거리는 `d_c = min_{j≠c} ‖μ_c-μ_j‖₂`, 유효 간격은 `g_c = min_{j≠c}(‖μ_c-μ_j‖₂-r_c-r_j)`로 보고한다. 평균만 보지 말고 클래스별 median, 10/90 percentile, negative-gap fraction, 클래스별 표본 수를 저장한다.
4. 같은 support와 query에서 세 분류기를 따로 실행한다: cosine prototype, Euclidean nearest centroid, pooled OAS Mahalanobis. 과거 Ledoit–Wolf 결과는 별도 참고로 보존한다. Mahalanobis의 중심과 잔차 공분산은 train-only 또는 허용된 support만으로 적합한다. query feature/label은 공분산, prototype, margin, checkpoint, threshold 선택에 사용하지 않는다.
5. `pred_cosine`, `pred_l2`, `pred_maha`가 서로 얼마나 일치하는지와 각 confusion matrix를 저장한다. prototype 재정규화나 covariance fit 역할이 다르면 결과 차이가 분류 손실이 아니라 평가 정의 때문일 수 있다.
6. source 27개 class 중 5개를 validation으로 분리하고 나머지 22개로 학습한다. margin, lambda, augmentation, checkpoint 선택은 이 heldout source classes에서 한다. 설정을 고정한 뒤 최종 비교 모델은 27개 source class 전체로 다시 학습한다. 15개 unseen class는 최종 평가까지 잠근다. strict 5-shot 결과는 unseen 각 클래스에서 support 5장만 prototype/covariance에 쓰고 나머지는 query로 평가하며, 동일 support/query membership을 모든 모델에 재사용한다. unseen gallery covariance를 쓰면 반드시 few-shot/galcov로 표기한다.
7. 최종 판단은 overall accuracy만으로 하지 않는다. seen/unseen accuracy, Macro-F1, balanced accuracy, class별 recall, geometry 지표를 함께 본다. 선택에 쓴 validation class/query와 최종 unseen query를 분리하고, seed별 값과 평균/신뢰구간을 남긴다.

## 기존 margin이 효과가 없어 보일 때 확인할 순서

- **실제로 어떤 train 경로가 돌았는가:** `args.json`/effective config와 실행 명령을 checkpoint metadata에 맞춘다. canonical ArcFace가 SupCon `train_metric.py`와 별도 경로라는 점을 확인한다.
- **샘플러가 positive와 hard negative를 공급하는가:** 각 batch의 unique class 수 `P`, 클래스별 원본 이미지 수 `K`, 같은 클래스 다른 원본 positive 수를 기록한다. 두 view의 같은 원본만 positive인 batch가 많으면 class-balanced sampler로 재비교한다.
- **loss가 올바른 출력과 weight에 연결됐는가:** backbone과 projector의 `requires_grad`, optimizer parameter 포함, `loss.backward()` 뒤 각 경로의 유한한 gradient norm, optimizer step 수, `m` 적용 전/후 target logit 차이를 확인한다.
- **gap 항이 실제로 활성인가:** epoch별 `L_supcon`, `L_gap`, `weight×L_gap`, `gap < margin` 비율을 기록한다. 같은 fixed batch에서 `‖∇L_supcon‖`, `‖weight×∇L_gap‖`, 두 gradient 방향의 cosine을 측정한다. 제안 기준은 gap gradient가 SupCon의 대략 5–20% 범위에 들어오는 weight 후보를 비교하는 것이다. 이 비율은 실험 제안이지 현재 검증값이 아니다.
- **margin 단위를 맞췄는가:** ArcFace의 각도 margin과 Gap의 normalized L2 margin은 별도 단위다. Gap `margin`은 train-only source class episode의 유효 간격 분포를 보고 정하고 heldout source class validation에서만 고른다.
- **훈련 endpoint와 평가 endpoint가 같은가:** backbone `h`, projection `z`, normalization 여부, 128/768차원을 각각 기록한다. loss가 projection에 직접 적용된다고 backbone endpoint와 같은 성능을 가정하지 않는다.
- **test feedback이 없었는가:** 현재 worktree trainer는 매 epoch test metric을 쓴다. 분석 중 test metric을 보고 모델/epoch/threshold를 고르지 않는다. 이미 반복해서 봤다면 그 test set은 최종 독립 평가로 취급하지 말고 새 holdout을 쓴다.

## 검증 상태

**확인된 사실:** 9/29 로컬 보고서의 M7/M8/M10 결과, 9/30 원격 live 정보로 전달된 Gap run 및 galcov run 상태, 위 표에 연결한 local source 구조.

**제안만 된 내용:** SupCon 단독 대조군, 별도 native 768차원 frozen 기준선, 같은 128차원 projection의 SupCon/Gap 비교, radius-aware ablation, episodic CE 후보, gradient 비율 기준, 세 seed 반복, 위 geometry protocol. 아직 이 보고서로 학습이나 개선이 검증된 것은 아니다.

**원자료:** `docs/reports/remote_experiments_20260929.md`, `docs/reports/professor_report_20260928/report.md`, `docs/reports/followup_20260928_status.md`. 원격 Gap 결과 경로는 `/home/ubuntu/aihub_run_new/outputs/experiment_records/gap-supcon-b256-20260929/`; galcov 결과는 `/home/ubuntu/aihub_run_new/outputs/metric/closed27_open15/vit_band_galcov_20260928/`.

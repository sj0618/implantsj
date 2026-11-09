# 로컬 임플란트 실험 결과 — 2026-09-30

## 범위와 확인 상태

- 원본 저장소: `/Users/yunseojune/TRIA_LAB/implant_project/implantsj`
- 로컬 `outputs/`, `docs/reports/`, `implantsj/docs/reports/`의 실험 표, 설정, 프로토콜, 학습 이력을 확인했다. 학습이나 코드 변경은 하지 않았다.
- 조사 시작 시 원본 저장소에 Git 변경 경로 1,897개가 있었다. 원본 파일은 수정하지 않았고 감사 결과는 이 worktree에 새로 기록했다.
- 파일 목록은 4,598개다. 목록 파일은 경로·확장자·크기·수정 시각만 수집한다. 임베딩과 체크포인트 내용은 목록 생성 시 읽지 않았다.
- 로컬 임베딩 거리 결과는 2026-09-28 보고서의 기존 표를 확인했다. CSV는 90행, 임베딩 캐시 45개, 거리 종류 2개다. 45개 원본 파일의 SHA256을 다시 계산해 기록과 모두 일치하는 것을 확인했다. NPZ 배열은 다시 열지 않았다.
- `outputs/`의 마지막 저장 결과는 2026-09-18이다. 2026-09-28에 로컬 임베딩 기하 표가 추가됐다. 9/28~9/29의 다른 최신 문서는 원격 실험 정리이므로 로컬 성능 결과에 섞지 않았다.

## 핵심 결과

| 비교 | 로컬 결과 | 판단 |
|---|---|---|
| ImageNet 사전학습 ViT의 raw embedding, 42클래스 전체 구조 | ViT-B 768차원 cosine 분리비 0.649, ViT-S 384차원 0.732. 클래스 내부 거리가 중심 간 거리보다 컸다. | 거리표의 전체 평균상 내부 산포가 중심 간격보다 큰 상태다. 이 값만으로 분류 성능을 예측할 수는 없다. |
| 학습된 ViT-S 표현의 같은 전체 구조 | CE head 128차원 분리비 2.644, ArcFace 128차원 2.288. 두 값 모두 5개 분할의 평균이다. | 전체 클래스 기하는 개선됐지만, 이것만으로 미관측 클래스 일반화가 개선됐다고 볼 수 없다. |
| Track B 15-way unseen, K=10 | CE ViT-S cosine 52.3% Accuracy / 31.8% Macro-F1; ArcFace 48.9% / 30.7%. 각 값은 5개 그룹 홀드아웃 평균과 95% CI 반폭이다. | CE가 두 지표 모두 ArcFace보다 높지만, Macro-F1 절대값은 31.8%로 낮다. 어느 학습법도 확정 개선으로 판정하지 않는다. |
| ViT-B PCA 768→384, Track B K=10 pooled Mahalanobis | PCA 58.80% / 42.42%; raw 768차원 52.40% / 41.08%; native 384차원 53.45% / 39.37%. PCA는 분할별 train-role 행으로만 적합했고 평균 설명분산은 97.30%다. | 유용한 결과지만 pooled Mahalanobis에 한정된다. Cosine에서는 PCA Macro-F1이 raw보다 낮았다. 모델 재학습 결과가 아니라 임베딩 후처리다. |
| 5-class strict 5-shot에서 SupCon vs 기존 CovAware | SupCon backbone + pooled Ledoit–Wolf: 67.98% / 63.30% / 64.49% (Accuracy / Macro-F1 / balanced accuracy). CovAware seed42: 66.40% / 62.65% / 64.20%로 낮았다. | CovAware를 기본 학습법으로 채택할 근거가 없다. |
| 이후 episodic CovAware, 5-class strict 5-shot | pooled-full 설정 train seed42: 69.42% / 66.87%; train seed0: 63.78% / 59.77%. 두 학습 seed 평균은 66.60% / 63.32%다. | seed42는 높았지만 seed0는 낮다. 6개 support seed를 사용했어도 학습 seed가 2개뿐이므로 효과가 안정적이라고 말할 수 없다. |
| 5-class strict 5-shot 임베딩 변환 | 균일 8-bit: 67.06% / 62.71%; 원본 pooled Mahalanobis: 67.98% / 63.30%. 비균일 중심 세밀화는 원본과 762개 반복 query 예측이 모두 같았다. 연속 asinh는 67.59% / 62.84%였다. | 현재 시도한 변환은 원본 점수를 넘지 못했다. 이 5-class 프로토콜은 Track B와 별개다. |

거리표의 분리비는 클래스 중심 사이 평균 cosine 거리 ÷ 클래스 내부 평균 cosine 거리다. 모든 입력 임베딩을 L2 정규화했다. 클래스 내부 표본이 1개인 두 클래스는 내부 쌍거리 평균에서 제외하고 중심 거리에는 포함했다. 전체 클래스 평균은 가까운 특정 클래스 쌍의 실패를 가릴 수 있다.
첫 번째 행은 ImageNet 사전학습 ViT의 raw embedding 전체에 대한 사후 geometry다. 데이터·모델·평가 정의가 다른 원격 DINO 결과와 직접 비교하지 않았다.

## 프로토콜 구분

- **Track B few-shot:** 42개 후보 클래스, 27개 학습 클래스와 15개 unseen 클래스, 5개 그룹 홀드아웃, 분할마다 200 episode, K=1/3/5/10. 분할 seed는 0–4다. CE·ArcFace의 학습 설정은 각 분할에서 seed=0이다.
- **Strict 5-shot:** 별도 5-class target pool 152장, 클래스당 support 5장(총 25), query 127장, support 구성 seed 0/1/2/3/4/42. 학습 seed와 support seed는 다른 값이다. Track B 숫자와 합산하지 않았다.
- **Closed-set source train-to-test:** 6개 source 클래스의 train 676장을 support로, test 84장을 query로 사용했다. 여기서 RDA 결과 72.62% Accuracy / 70.52% Macro-F1은 unseen target 실험과 직접 비교할 수 없다.

## 다음 학습법 제안 — 아직 실행하지 않음

현재 로컬 결과만으로는 새 학습법의 안정적인 개선을 확인하지 못했다. 후속 비교안과 검증 기준은 [training_design.md](training_design.md)에 정리했다.

## 근거 파일

- PCA384 프로토콜과 수치: [protocol.json](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/pca384_in_vitb_20260918/protocol.json), [summary.csv](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/pca384_in_vitb_20260918/summary.csv), [metrics.csv](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/pca384_in_vitb_20260918/metrics.csv)
- Track B 5-way holdout 요약: [few-shot report](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/fewshot/REPORT.md), [runs.csv](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/fewshot/tables/runs.csv)
- 로컬 전체 geometry: [local-geometry.csv](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/docs/reports/professor_report_20260928/excluded-local-analysis/local-geometry.csv), [측정 스크립트](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/docs/reports/professor_report_20260928/excluded-local-analysis/measure_local_geometry.py)
- SupCon strict 5-shot 기준: [summary.csv](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_target_5shot/summary.csv), [protocol.json](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_all_target_5shot/protocol.json)
- 기존 CovAware: [target protocol](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/vit6covaware_supcon_to_5label_20260814/evaluation_target_pool_5shot/protocol.json), [backbone summary](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/vit6covaware_supcon_to_5label_20260814/evaluation_target_pool_5shot/backbone/summary.csv), [training args](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/vit6covaware_supcon_to_5label_20260814/training/20260814_023836_vit_small_covaware_supcon6_e30_seed42/args.json)
- pooled-full CovAware 반복: [seed42 args](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/local_evaluator_matched/20260827_194455_vit_small_epi_maha_full_backbone_e30_seed42/args.json), [seed42 target result](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/local_evaluator_matched/20260827_194455_vit_small_epi_maha_full_backbone_e30_seed42/eval_best/backbone/summary.csv), [seed0 target result](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/outputs/local_evaluator_matched/20260827_200011_vit_small_epi_maha_full_backbone_e30_seed0/eval_best/backbone/summary.csv)
- 별도 5-class 임베딩 변환 실험: [양자화 결과](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/embedding_quantization_20260915/REPORT.md), [연속 변환 결과](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/embedding_quantization_20260915/CONTINUOUS_REPORT.md)
- 닫힌 집합 결과: [source train-to-test report](/Users/yunseojune/TRIA_LAB/implant_project/implantsj/implantsj/docs/reports/vit6supcon_to_5label_strict10_20260814/episodic_source_train_to_test/REPORT.md)

세부 실험 행은 `local_experiments.csv`, 전체 파일 메타데이터는 `local_artifact_inventory.csv`에 있다.

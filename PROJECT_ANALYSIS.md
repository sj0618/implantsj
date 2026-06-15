# implant_python_only_final 전체 분석 보고서

- 생성일: 2026-05-26
- 분석 범위: `/workspace/implant_python_only_final`
- 산출물: 이 파일 (`PROJECT_ANALYSIS.md`)
- 보안 처리: `.env`는 존재와 변수명만 확인했고, 값은 분석 문서에 포함하지 않았다.

## 1. 한 줄 요약

이 폴더는 **치과 임플란트 ROI 방사선 이미지 분류 실험 플랫폼**이다. 핵심은 ViT 기반 이미지 분류/metric learning이고, 보조 축으로 Azure OpenAI VLM이 추출한 구조적 feature를 JSONL/CSV로 저장한 뒤 logistic regression, feature-only ML, image+VLM fusion, two-stage classifier 개선 실험을 수행한다.

## 2. 전체 구조와 역할

| 경로 | 크기/파일 수 | 역할 |
|---|---:|---|
| `src/` | 16 files, 약 118KB | 공통 라이브러리: 데이터셋/manifest, timm 모델 생성, supervised 학습, metric learning, 평가 지표, 유틸 |
| 루트 `*.py` | 40여 개 | 학습/평가/예측/VLM feature 생성/시각화 CLI 엔트리포인트 |
| `experiments/` | 4 files, 약 168KB | VLM 구조 feature만으로 분류 가능한지 확인하는 scikit-learn 실험 |
| `two_stage/` | 10 files, 약 40KB | 1차 ViT 예측 후 혼동 그룹만 하위 submodel로 재분류하는 two-stage 파이프라인 |
| `data/` | 5,919 files, 약 405MB | 3-label/7-label/AI Hub 이미지, COCO annotation, manifest, VLM JSONL/summary |
| `outputs/` | 542 files, 약 18.7GB | 학습 checkpoint, metrics, predictions, confusion matrix, t-SNE, attention map, ML 실험 결과 |
| `implant_outputs/` | 3 files, 약 1.65GB | 이전/별도 ViT highscore run artifact |
| `__pycache__/`, `src/__pycache__/`, `two_stage/__pycache__/` | 캐시 파일 | Python bytecode 캐시. 분석 대상은 아니며 재생성 가능 |
| `.git/` | 약 16GB | git 내부 객체. 실제 tracked file은 `README.md`뿐이며 대부분 파일은 untracked 상태 |

## 3. 파일 타입 통계 (`.git` 제외)

| 확장자 | 개수 | 총 크기 | 해석 |
|---|---:|---:|---|
| `.jpg` | 5,577 | 약 192MB | ROI 이미지 데이터 |
| `.png` | 492 | 약 91MB | confusion matrix, attention map, t-SNE 등 시각화 결과 |
| `.csv` | 205 | 약 17MB | manifest, prediction, metric summary, confusion matrix |
| `.json` | 93 | 약 46MB | args, metrics, VLM JSON array, label map, config |
| `.py` | 48 | 약 388KB | 핵심 코드 |
| `.pt` | 20 | 약 19.8GB | PyTorch checkpoint |
| `.joblib` | 12 | 약 99MB | scikit-learn model 저장물 |
| `.npz` | 9 | 약 20MB | embedding/feature cache 추정 |
| `.md` | 8 | 약 34KB | 사용법/실험 정리/프롬프트 문서 |
| `.jsonl` | 5 | 약 63MB | VLM feature extraction 결과 |
| `.onnx` | 1 | 약 343MB | ONNX model artifact |

## 4. 프로젝트의 큰 흐름

```text
이미지 데이터
  ├─ data/3label, data/7label, AI Hub Data
  ├─ make_manifest.py / make_splits.py / train_vit_* manifest builders
  ▼
Manifest CSV(path,label,split)
  ├─ Supervised ViT: train_supervised.py, train_vit_3/7/10label.py
  ├─ Metric learning/SupCon: train_metric.py, train_vit_7label_supcon.py
  ├─ Evaluation/prediction: evaluate.py, predict_vit.py
  ├─ Visualization: visualize_*attention.py, visualize_*tsne.py
  └─ Two-stage: two_stage/*, evaluate_3label_* scripts

VLM/Azure feature extraction
  ├─ generate_vlm_feature_presence.py
  ├─ generate_13feature_21reason_7label.py
  ├─ generate_vlm_features_azure_6label.py
  ▼
VLM JSONL/summary CSV
  ├─ train_vlm_fusion_5feature.py: image + VLM fusion ablation
  └─ experiments/run_*_ml.py: VLM feature-only ML/logistic regression 검증
```

## 5. 핵심 코드 모듈 분석

### 5.1 `src/data.py`

역할: 이미지 분류 데이터셋의 표준 입출력 계층.

- `detect_layout`: class-folder 또는 split/class-folder 구조를 판별한다.
- `build_manifest`: 이미지 폴더를 `path,label,split` CSV로 변환한다.
- `make_stratified_splits`: label별 stratified train/valid/test split을 생성한다.
- `ImageClassificationDataset`: PIL 이미지 로드 후 transform과 label index를 반환한다.
- `build_transforms`: timm/torchvision 스타일 resize/crop/augmentation pipeline 생성.
- `create_loader`: manifest와 split을 받아 DataLoader 생성.
- `class_weights_from_manifest`: class imbalance 대응용 weight 계산.

### 5.2 `src/models.py`

역할: timm 기반 ViT classifier와 metric model 생성/저장.

- `create_classifier`: timm model 생성 후 classifier head를 label 수에 맞춘다.
- `MetricModel`: backbone feature를 projection head로 보내 embedding을 만든다.
- `load_matching_weights`: checkpoint shape이 맞는 weight만 부분 로드한다.
- `save_checkpoint`, `load_checkpoint`, `get_class_to_idx_from_checkpoint`: run 재현용 checkpoint 입출력.

### 5.3 `src/supervised.py`

역할: 일반 supervised image classification 학습 루프.

- `SupervisedArgs`: 학습 hyperparameter dataclass.
- `evaluate_classifier`: split별 loss, prediction, score 계산.
- `train_supervised`: seed 고정, loader 구성, optimizer/scheduler/AMP, early stopping, best/last checkpoint, metrics/confusion matrix/predictions 저장.

### 5.4 `src/metric_learning.py`

역할: SupCon/triplet 기반 metric learning 실험.

- `TwoViewDataset`: SupCon용 같은 이미지 두 augmentation view 생성.
- `SupConLoss`: supervised contrastive loss.
- `BatchHardTripletLoss`: batch-hard triplet loss.
- `MetricArgs`: metric 학습 hyperparameter.
- `extract_embeddings`, `prototype_predict`, `evaluate_metric_model`: class prototype 기반 cosine/embedding 평가.
- `train_metric`: metric model 학습과 prototype 평가 artifact 저장.

### 5.5 `src/metrics.py`

역할: 공통 평가 지표 저장.

- accuracy, macro/weighted precision/recall/F1 계산.
- CSV confusion matrix와 PNG confusion matrix 저장.
- prediction CSV 저장.

### 5.6 `src/utils.py`

역할: seed, timestamp, run dir, JSON IO, logging, CUDA 요구 검사, parameter count 등 실행 보조.

## 6. 루트 Python 파일별 분석

| 파일 | 주요 역할 |
|---|---|
| `USAGE.py` | 사용 예시/가이드 성격의 Python 텍스트 파일. 실행 로직보다 문서에 가깝다. |
| `check_environment.py` | Python 버전, torch/timm/PIL/sklearn/pandas 등 필수 패키지 import 가능 여부, CUDA 여부 점검. |
| `install_requirements.py` | 기본 패키지를 pip로 설치하고, 옵션으로 torch CPU/CUDA variant를 설치. |
| `settings.py` | 프로젝트 루트, 기본 data/output/manifest 경로 상수. |
| `run.py` | 대표 workflow command 모음. `make_manifest`, `train_supervised`, `evaluate`, `predict` 등을 subprocess로 실행. |
| `make_manifest.py` | 이미지 폴더에서 manifest CSV 생성. |
| `make_splits.py` | 기존 manifest에 stratified split 추가/재생성. |
| `smoke_test.py` | 합성 이미지 데이터를 만들고 짧은 supervised 학습으로 pipeline smoke test 수행. |
| `finetune_small_data.py` | small dataset manifest를 만들고 large pretrain checkpoint에서 fine-tuning. |
| `train_supervised.py` | 범용 supervised ViT 학습 CLI. |
| `train_metric.py` | 범용 SupCon/triplet metric learning 학습 CLI. |
| `evaluate.py` | classifier checkpoint를 특정 manifest split에서 평가. |
| `predict_vit.py` | classifier checkpoint로 prediction CSV 생성. |
| `summarize_runs.py` | run 디렉터리의 metrics JSON을 모아 summary CSV 생성. |
| `train_vit_3label.py` | COCO/split 폴더 기반 3-label(Bego/Bicon/ITI) manifest 생성 및 supervised ViT 학습. |
| `train_vit_7label.py` | 7-label(ADIN, Dentium, DIONAVI, MIS, NORIS, nobel, osstem) manifest 생성 및 supervised ViT 학습. |
| `train_vit_10label.py` | 3-label + 7-label을 합쳐 10-label supervised ViT 학습. |
| `train_vit_6label.py` | large multiclass에서 6개 label만 필터링해 metric learning 학습. 이름은 6label이지만 내부는 `train_metric` 사용. |
| `train_vit_7label_supcon.py` | 7-label manifest 기반 SupCon metric learning 학습. |
| `evaluate_3label_metric_from_7label.py` | 7-label metric checkpoint를 3-label dataset에 적용해 prototype 방식으로 transfer 평가. |
| `evaluate_3label_metric_from_6label.py` | 위 7-label 평가 스크립트의 thin wrapper. |
| `evaluate_3label_metric_from_7label_supcon.py` | 위 7-label 평가 스크립트의 thin wrapper. |
| `evaluate_3label_twostage_metric_from_7label.py` | 1단계 group KNN + 2단계 group 내부 prototype으로 3-label transfer를 보정. |
| `test_7label_supcon_to_3label_metric.py` | fake timm/transform과 synthetic image로 7-label SupCon→3-label 평가 script를 검증하는 테스트. |
| `visualize_attention.py` | classifier ViT checkpoint의 attention rollout heatmap/overlay 저장. |
| `visualize_classifier_tsne.py` | classifier backbone feature를 추출해 PCA+t-SNE plot/CSV 저장. |
| `visualize_metric_attention.py` | metric model checkpoint의 attention rollout과 prototype prediction 기반 sample 분석. |
| `visualize_metric_tsne.py` | metric embedding t-SNE plot/CSV 저장. |
| `visualize_umap_from_tsne.py` | 기존 `tsne_*.npz` 임베딩 아티팩트를 재사용해 UMAP plot/CSV 저장. |
| `generate_vlm_features_azure_6label.py` | Azure OpenAI vision deployment로 6-label 이미지의 5개 구조 feature를 JSONL로 생성. |
| `generate_vlm_feature_presence.py` | 21개 supporting visual feature의 present/absent/not_assessable을 VLM으로 추출. |
| `generate_13feature_21reason_7label.py` | 13개 core attribute + 21개 supporting feature를 VLM으로 추출하고 decision rule로 feature_status/usable flag를 정리. |
| `sort_vlm_feature_presence_jsonl.py` | VLM JSONL을 원본 이미지 discovery 순서대로 정렬하고 backup 생성. |
| `train_vlm_fusion_5feature.py` | 5개 VLM feature group을 vector화해 image-only, VLM-only, fusion, shuffled, ablation suite 학습/평가. |

## 7. `experiments/` 분석

| 파일 | 역할 |
|---|---|
| `experiments/run_effective10_ml.py` | VLM core attribute 중 Effective 10 feature set만 categorical one-hot으로 만들어 logistic regression/random forest/boosting 등 tabular ML 평가. usable/partial/all 필터, shuffled control, leave-one-out/single-feature/unknown-only 실험 포함. |
| `experiments/run_aihub_logistic_regression.py` | AI Hub 13-feature VLM 결과와 answer-key를 비교/대체/보강해 logistic regression으로 label 예측 가능성을 검증. core12, structural11, supporting21 feature 조합 지원. |

## 8. `two_stage/` 분석

| 파일 | 역할 |
|---|---|
| `build_confusion_groups.py` | prediction CSV에서 true/pred 혼동을 보고 second-stage group 후보 JSON 생성. |
| `confusion_groups.json` | 기본 혼동 group 설정. 현재 `NORIS_osstem`, `ADIN_MIS` 그룹. |
| `confusion_groups_noris_osstem.json` | NORIS/osstem만 대상으로 하는 group 설정. |
| `train_subgroup_vit.py` | group별 subset manifest를 만들고 별도 supervised submodel 학습. |
| `predict_two_stage.py` | main model prediction 후 지정 group에 걸린 sample만 submodel checkpoint로 재예측. |
| `evaluate_two_stage.py` | two-stage prediction CSV의 accuracy/report/confusion matrix 산출. |

## 9. Markdown/문서 파일 분석

| 파일 | 내용 |
|---|---|
| `README.md` | 현재는 `# implantsj`만 있는 최소 README. 실제 사용법은 다른 문서에 분산됨. |
| `EXPERIMENT_SUMMARY.md` | 2026-05-07 기준 주요 ViT/metric/two-stage 실험 결과와 추천 baseline 정리. 가장 중요한 결과 문서. |
| `VLM_5FEATURE_FUSION_USAGE.md` | 5-feature VLM fusion 실험 목적, JSONL 형식, suite 실행법, 판단 기준 설명. |
| `AZURE_VLM_6LABEL_FEATURE_PIPELINE.md` | Azure OpenAI endpoint/API key/deployment 설정과 6-label VLM feature extraction 실행법. |
| `13feature_21reason.md` | 기존 고수준 VLM feature 실패를 계기로 13 core attributes + 21 supporting features 체계로 전환한 설계 문서. |
| `vlm_feature_prompt_5feature.md` | 5-feature VLM extraction prompt. brand/system 이름을 추측하지 말고 구조 feature JSON만 반환하도록 지시. |
| `data/manifests/vlm_5feature_6label.prompt.md` | 5-feature prompt 사본. |
| `data/manifests/vlm_feature_presence.prompt.md` | 21 supporting feature extraction prompt. |

## 10. 데이터와 manifest 분석

### 10.1 이미지 데이터셋

#### `data/3label`

| split | label별 개수 | 총계 |
|---|---|---:|
| train | bego 133, ITI 673, Bicon 959 | 1,765 |
| valid | bego 9, ITI 77, Bicon 115 | 201 |
| test | Bego 4, ITI 44, Bicon 61 | 109 |

주의: 폴더명에 `bego`/`Bego`처럼 대소문자 혼용이 있다. 코드에서는 canonical label 처리로 일부 보정하지만, 새 스크립트 작성 시 label normalization을 유지해야 한다.

#### `data/7label`

| split | label별 개수 | 총계 |
|---|---|---:|
| train | ADIN 850, Dentium 416, DIONAVI 12, MIS 114, NORIS 626, nobel 9, osstem 576 | 2,603 |
| valid | ADIN 118, Dentium 44, DIONAVI 3, MIS 18, NORIS 87, nobel 2, osstem 66 | 338 |
| test | ADIN 110, Dentium 59, DIONAVI 3, MIS 14, NORIS 67, nobel 1, osstem 85 | 339 |

강한 class imbalance가 있다. 특히 `nobel`, `DIONAVI`, `MIS`가 매우 적어서 macro-F1이 accuracy보다 낮게 나오는 원인이다.

#### `data/AI Hub Data`

42개 제조사/시스템 label folder가 있다. 이 데이터는 VLM 13-feature/answer-key/logistic-regression 실험의 기반으로 보인다.

### 10.2 주요 manifest

| 파일 | rows | label/split 요약 | 역할 |
|---|---:|---|---|
| `data/manifests/3label.csv` | 2,075 | Bicon 1,135 / ITI 794 / Bego 146 | 3-label canonical manifest |
| `data/manifests/7label.csv` | 3,280 | ADIN 1,078 / NORIS 780 / osstem 727 / Dentium 519 / MIS 146 / DIONAVI 18 / nobel 12 | 7-label canonical manifest |
| `data/manifests/plain_vit_3label.csv` | 2,075 | 3-label supervised ViT용 | `train_vit_3label.py` 출력 |
| `data/manifests/plain_vit_7label.csv` | 3,280 | 7-label supervised ViT용 | `train_vit_7label.py` 출력 |
| `data/manifests/plain_vit_10label.csv` | 5,355 | 3-label+7-label 통합 | `train_vit_10label.py` 출력 |
| `data/manifests/supcon_vit_7label.csv` | 3,280 | 7-label metric/SupCon용 | `train_vit_7label_supcon.py` 입력 |
| `data/manifests/NORIS_osstem.csv` | 1,507 | NORIS 780 / osstem 727 | two-stage submodel용 |
| `data/manifests/ADIN_MIS.csv` | 1,224 | ADIN 1,078 / MIS 146 | two-stage submodel용 |
| `data/manifests/13feature_21reason_7label_all.jsonl` | 3,312 lines | VLM 13+21 extraction 결과 | 7-label 이미지 + 일부 추가 row 포함 |
| `data/manifests/13feature_21reason_7label_all.summary.csv` | 3,312 | feature_status/usable/failure_reason 포함 | VLM 품질 분석용 |
| `data/manifests/vlm_feature_presence_7label_all.jsonl` | 3,280 lines | 21 supporting feature extraction | `generate_vlm_feature_presence.py` 결과 |
| `data/manifests/vlm_feature_presence_7label_all.summary.csv` | 3,280 | feature presence summary | VLM feature-only/fusion 입력 후보 |
| `data/manifests/vlm_5feature_7label.jsonl` | 3,279 lines | 5-feature VLM result | 과거 5-feature schema 결과 |
| `data/manifests/vlm_5feature_7label.summary.csv` | 225 | ADIN train만 summary | 부분 생성/초기 실험 artifact로 보임 |
| `data/manifests/aihub_all_13feature_21reason_vlm.jsonl` | 241 lines | AI Hub sample VLM extraction | AI Hub logistic regression 입력 |
| `data/manifests/aihub_answer_key_13feature.json` | 42 items | AI Hub answer key | answer-key 대체/보강 실험 |

## 11. VLM feature 체계 해석

### 11.1 기존 5-feature schema

`vlm_feature_prompt_5feature.md`와 `train_vlm_fusion_5feature.py`는 다음 5개 그룹을 사용한다.

1. `connection_type`
2. `platform`
3. `thread`
4. `outline`
5. `image_quality`

각 그룹은 categorical/boolean/numeric confidence를 vector화한다. `train_vlm_fusion_5feature.py`는 다음 비교군을 지원한다.

- image-only
- VLM feature-only
- image + VLM simple sum fusion
- shuffled VLM feature control
- cumulative ablation
- leave-one-out ablation
- single feature group 평가

### 11.2 개선된 13 core + 21 supporting schema

`13feature_21reason.md`와 `generate_13feature_21reason_7label.py`는 고수준 label 직접 예측이 실패한 뒤 설계를 바꾼 버전이다.

- 13 core attributes: `Company`, `Name`, `System`, `Connection_1`, `Connection_2`, `Flange`, `Collar`, `Microthread`, `Body_Shape`, `Body_Type`, `Thread_Shape`, `Apex_Shape`, `Apex_Hole`
- 21 supporting features: `implant_visible`, `connection_region_visible`, `thread_region_visible`, `abutment_or_prosthesis_present`, `positive_connection_region_present`, `straight_screw_entry_space_present`, `negative_internal_connection_space_present`, `platform_step_present`, `internal_small_void_present`, `smooth_no_thread_collar_present`, `machined_surface_present`, `platform_boundary_visible`, `platform_switching_step_present`, `thread_lines_visible`, `regular_thread_pitch_visible`, `deep_thread_relief_present`, `micro_thread_zone_present`, `macro_thread_zone_present`, `body_taper_present`, `parallel_body_wall_present`, `apex_boundary_visible`

핵심 설계는 `present/absent/not_assessable`을 구분하고, feature vector에는 `value`, `assessable_mask`, `confidence`를 넣어 not-assessable leakage를 줄이는 것이다.

## 12. 실험 결과 요약

이미 존재하는 `EXPERIMENT_SUMMARY.md`와 metrics JSON을 기준으로 보면 다음 결론이다.

| 카테고리 | 대표 run | Test Accuracy | Test Macro-F1 | 해석 |
|---|---|---:|---:|---|
| 3-label plain ViT | `outputs/20260507_104805_plain_vit_3label` | 0.9449 | 0.7700 | 전용 3-label classifier가 metric transfer보다 훨씬 좋다. |
| 7-label plain ViT | `outputs/20260507_095713_plain_vit_7label` | 0.9440 | 0.8829 | 현재 가장 안정적인 plain 7-label baseline. |
| 7-label SupCon metric | `outputs/20260507_101426_supcon_vit_7label` | 0.9587 | 0.7512 | accuracy는 높지만 minority class 때문에 macro-F1이 낮다. |
| 10-label plain ViT | `outputs/20260507_134640_plain_vit_10label` | 0.9554 | 0.8812 | 현재 전체 supervised baseline 중 가장 강함. |
| 10-label + NORIS/osstem two-stage | `outputs/20260507_134640_plain_vit_10label/two_stage_NORIS_osstem_test` | 0.9509 | 0.8786 | 적용은 됐지만 baseline보다 약간 낮아 기본 사용은 비추천. |
| 7-label metric → 3-label | `outputs/20260507_142544_metric_7label_supcon_to_3label` | 0.3761 | 0.3475 | 직접 prototype transfer는 약함. |
| 7-label metric → 3-label two-stage | `outputs/20260507_083124_twostage_metric_7label_to_3label` | 0.6881 | 0.5492 | direct transfer보다는 낫지만 전용 3-label ViT보다 낮음. |
| NORIS/osstem submodel | `outputs/submodels/20260507_051743_NORIS_osstem` | 0.9408 | 0.9396 | submodel 자체는 준수하지만 main handoff에서 전체 성능 개선 실패. |
| ADIN/MIS submodel | `outputs/submodels/20260507_051925_ADIN_MIS` | 0.9839 | 0.9570 | binary submodel 성능은 높음. |

### 12.1 VLM feature-only ML 결과 해석

`outputs/effective10_*`와 `outputs/aihub_*` 계열은 VLM 구조 feature만으로 label 예측이 가능한지 검증한다.

- 7-label VLM Effective10 feature-only 계열은 test accuracy가 대체로 0.32~0.45 수준으로, ViT image-only baseline(약 0.94+)에 크게 못 미친다.
- shuffled control이 함께 있어 feature leakage/우연 효과를 비교하려는 설계가 들어가 있다.
- AI Hub answer-key 기반 실험에서는 `Name` 같은 answer-key 직접 식별 정보가 들어가면 성능이 과도하게 높아진다. 이 경우는 구조 feature 검증이라기보다 label identity leakage에 가깝게 해석해야 한다.
- 구조 attribute만 쓰는 `structural11_supporting21`은 answer-key 사용 시 일부 개선이 있지만, 실제 VLM-only feature의 label 판별력은 제한적이다.

## 13. 출력 artifact 분석

`outputs/`는 크게 네 종류다.

1. **ViT run directory**  
   예: `20260507_095713_plain_vit_7label`, `20260507_134640_plain_vit_10label`  
   포함: `args.json`, `checkpoints/best.pt`, `metrics/*.json`, `metrics/*predictions.csv`, `confusion_matrices/*`, `embeddings/tsne_*`, `attention_maps/`.

2. **Metric/SupCon run directory**  
   예: `20260507_101426_supcon_vit_7label`, `20260507_142544_metric_7label_supcon_to_3label`  
   포함: prototype metrics, embedding t-SNE, metric attention map.

3. **Two-stage 결과**  
   예: `two_stage_predictions.csv`, `two_stage_metrics.json`, `two_stage_eval_7label_full/`, `20260507_134640_plain_vit_10label/two_stage_NORIS_osstem_test/`.

4. **Tabular/VLM ML 실험 결과**  
   예: `effective10_ml*`, `aihub_logistic_regression*`  
   포함: `metrics_summary.csv`, `classification_report_*.csv`, `confusion_matrix_*.csv/png`, `encoder_categories.json`, `feature_distribution.csv`, model `.joblib`.

## 14. 환경/의존성 분석

코드에서 확인되는 주요 의존성:

- Python 3.11 계열 bytecode 존재
- PyTorch / torchvision
- timm
- numpy
- pandas
- scikit-learn
- Pillow
- matplotlib
- joblib
- Azure OpenAI SDK (`openai.AzureOpenAI`) 또는 compatible client

주의사항:

- 현재 base Python에는 `pandas`가 없어서 전역 `python`으로 일부 분석 스크립트가 실패했다. 프로젝트 실행은 가상환경 또는 `install_requirements.py`로 dependency 설치 후 해야 한다.
- GPU 사용을 기본 가정하는 코드가 많으며, 대부분 `--allow-cpu` 옵션이 있다.
- `.env`에는 Azure OpenAI endpoint/key/deployment 관련 값이 들어 있다. gitignore에는 `.env`가 포함되어 있지만 현재 많은 파일이 untracked라 별도 commit 시 주의해야 한다.

## 15. 보안/품질 리스크

| 리스크 | 근거 | 권장 조치 |
|---|---|---|
| `.env` secret 노출 위험 | `.env`, `.env.*` ignore 설정은 있으나 파일이 실제 존재 | commit 전 `git status --ignored` 확인, secret rotation 고려 |
| 절대 경로 hardcoding | `/workspace/...` 기본값 다수 | `settings.py` 기반 상대 경로/환경변수로 정리 |
| 산출물/모델 크기 과대 | `outputs/` 약 18.7GB, `.pt` 약 19.8GB, `.git/` 약 16GB | 모델/출력은 Git LFS 또는 artifact storage로 분리 |
| tracked 상태 불명확 | git tracked file은 사실상 `README.md`뿐이고 대부분 untracked | 필요한 코드/문서만 선별해서 commit 전략 수립 |
| label 대소문자 혼용 | `bego`/`Bego`, `osstem`/brand case | canonical label 함수 유지, manifest 검증 추가 |
| `.gitignore` 오타 | `*.jpng` 존재 | `*.png` 의도였는지 확인 필요. 현재는 `.jpg`만 ignore됨 |
| VLM feature leakage 가능성 | answer-key `Name` 사용 시 성능 1.0 가능 | 구조 feature 검증과 label identity feature를 분리 |
| class imbalance | nobel/DIONAVI/MIS 매우 적음 | macro-F1 중심 평가, class-balanced sampling/thresholding 검토 |
| 테스트 부족 | 명시적 테스트는 `test_7label_supcon_to_3label_metric.py` 중심 | smoke/unit test 추가, CLI import/argparse test 추가 |

## 16. 실행 예시

### 환경 확인

```bash
cd /workspace/implant_python_only_final
python check_environment.py
```

### 3-label supervised ViT 학습

```bash
python train_vit_3label.py \
  --manifest data/manifests/plain_vit_3label.csv \
  --output-root outputs \
  --epochs 80 \
  --batch-size 16
```

### 7-label supervised ViT 학습

```bash
python train_vit_7label.py \
  --manifest data/manifests/plain_vit_7label.csv \
  --output-root outputs \
  --epochs 80 \
  --batch-size 16
```

### 10-label baseline 학습

```bash
python train_vit_10label.py \
  --manifest data/manifests/plain_vit_10label.csv \
  --output-root outputs \
  --epochs 80 \
  --batch-size 16
```

### checkpoint 평가

```bash
python evaluate.py \
  --checkpoint outputs/20260507_134640_plain_vit_10label/checkpoints/best.pt \
  --manifest data/manifests/plain_vit_10label.csv \
  --split test \
  --output-root outputs
```

### 21-feature VLM 추출 dry-run

```bash
python generate_vlm_feature_presence.py \
  --input data/7label \
  --output data/manifests/vlm_feature_presence_7label_all.jsonl \
  --dry-run
```

### VLM feature-only ML 평가

```bash
python experiments/run_effective10_ml.py \
  --input data/manifests/13feature_21reason_7label_all.jsonl \
  --out-dir outputs/effective10_ml \
  --feature-set effective10
```

### two-stage prediction

```bash
python two_stage/predict_two_stage.py \
  --main-checkpoint outputs/20260507_134640_plain_vit_10label/checkpoints/best.pt \
  --manifest data/manifests/plain_vit_10label.csv \
  --groups-json two_stage/confusion_groups_noris_osstem.json \
  --submodel-root outputs/submodels \
  --output outputs/two_stage_predictions.csv
```

## 17. 현재 기준 추천 사용 방향

1. **이미지 분류 baseline**: `outputs/20260507_134640_plain_vit_10label`를 10-label baseline으로 유지.
2. **7-label baseline**: `outputs/20260507_095713_plain_vit_7label`를 plain 7-label 기준점으로 유지.
3. **3-label 작업**: dedicated `20260507_104805_plain_vit_3label` 사용. 7-label metric transfer는 성능이 낮다.
4. **SupCon embedding**: classification replacement보다는 embedding/prototype/visualization 분석 용도로 활용.
5. **Two-stage**: 현재 NORIS/osstem two-stage는 baseline보다 낮으므로 기본 적용하지 말고 low-confidence trigger 조건을 추가한 후 재평가.
6. **VLM feature**: 최종 판단 모델보다는 구조 feature 설명/보조 signal/ablation 분석 용도로 쓰는 것이 안전하다.
7. **정리 우선순위**: README 확장, dependency 파일 작성, `.gitignore` 수정, output/model artifact 분리, CLI path 정리.

## 18. 전체 이해 결론

이 프로젝트는 단순한 학습 스크립트 모음이 아니라, 다음 질문을 단계적으로 검증한 실험 저장소다.

1. 임플란트 ROI 이미지만으로 ViT가 브랜드/시스템을 잘 분류하는가?  
   → 예. 7-label/10-label supervised ViT는 test accuracy 약 0.94~0.96 수준.

2. metric learning/SupCon embedding이 label transfer나 소수 class 문제를 해결하는가?  
   → 7-label 자체 prototype accuracy는 높지만 macro-F1과 3-label transfer는 제한적.

3. VLM이 구조 feature를 뽑아 분류 성능을 보조할 수 있는가?  
   → feature-only 성능은 제한적이며, VLM feature는 image-only ViT를 대체하기 어렵다. 다만 구조 설명과 보조/ablation 용도로 가치가 있다.

4. 혼동되는 class만 submodel로 재분류하면 좋아지는가?  
   → submodel 자체는 좋지만 현재 handoff 조건에서는 overall 성능이 약간 하락했다. confidence/threshold 기반 trigger가 필요하다.

따라서 현재 repo의 실용적 핵심은 **ViT image-only baseline + 실험적 VLM structural feature 분석 + 선택적 two-stage refinement 연구**로 이해하면 된다.

## 19. 검증 메모

- `.git` 내부와 `__pycache__` bytecode는 내용 분석 대상에서 제외했다.
- 이미지/PNG/checkpoint/joblib/onnx 같은 binary file은 직접 내용을 해석하지 않고 파일 수, 크기, 경로, 주변 metadata/metrics 기준으로 분석했다.
- `.env`는 값 노출을 피하고 변수명만 확인했다.
- Python 파일은 AST 수준으로 함수/class/import 구조를 훑고, 주요 entrypoint/CLI argument를 확인했다.
- Manifest/metrics는 CSV/JSON header와 row count, label/split 분포, 핵심 metric 값을 직접 집계했다.

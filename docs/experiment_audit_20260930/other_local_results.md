# implantsj 외 로컬 실험 결과 감사

조사일: 2026-09-30  
범위: `/Users/yunseojune/TRIA_LAB/implant_project`에서 `implantsj` 디렉터리를 제외한 결과물. 원격 접속과 학습 실행은 하지 않았다. 보고된 숫자는 로컬 예측·혼동행렬로 재계산한 값과 노트북에 저장된 값만 확인한 결과를 구분했다.

## 확인된 로컬 결과

### 7개 클래스 분류기

데이터 분할표 `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/manifests/7label.csv`: 총 3,279장, train 2,603 / valid 338 / test 338. train 클래스별 수는 ADIN 850, DIONAVI 12, Dentium 416, MIS 114, NORIS 626, nobel 9, osstem 576이다. valid/test 클래스 분포는 별도 표를 참조한다.

두 실행 모두 ImageNet 사전학습 ResNet18(`resnet18.a1_in1k`), 1 epoch, seed 42, 224px, CPU에서 끝났다. 체크포인트와 예측 CSV가 존재한다. 시험 지표를 `test_predictions.csv`의 정답/예측으로 다시 계산했고 JSON과 일치했다.

| 실행 | 주요 설정 | valid acc / Macro-F1 | test acc / Macro-F1 / weighted-F1 | 근거 |
|---|---|---:|---:|---|
| `20260828_064409_7label__resnet18.a1_in1k_sup` | batch 16, lr 5e-5, wd .05, label smoothing .1, 강한 증강 | .428994 / .171560 | .387574 / .150618 / .290847 | 예측 CSV에서 재계산, `Finished` 로그 |
| `20260828_115922_7label__resnet18.a1_in1k_sup` | batch 32, lr 5e-5, wd .05, label smoothing .1, 강한 증강 | .408284 / .148515 | .399408 / .150144 / .296399 | 예측 CSV에서 재계산, `Finished` 로그 |

정확한 근거 파일 예: `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/20260828_115922_7label__resnet18.a1_in1k_sup/args.json`, `metrics/test_predictions.csv`, `metrics/test_metrics.json`, `metrics/best_valid_metrics.json`, `checkpoints/best.pt`; 로그 `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/logs/20260828_115920_train.log`. 이 실행의 체크포인트 SHA-256은 `e05ae086f8783b2c82dc9603202ee1070768000199b102173ab15c7c205727ae`이다. 로그는 CPU, epoch 1, `Finished`를 기록하지만 OS 프로세스 종료 코드는 저장되어 있지 않다.

불균형 영향이 크다. batch32 실행은 DIONAVI, Dentium, MIS, nobel의 test F1이 0이다. 두 실행은 1 epoch·단일 seed라 성능 비교나 재현성 결론으로 쓰기 어렵다.

### 7개 클래스 임베딩 거리 평가

`/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/mahalanobis/7label_20260828_120413/metrics.csv`의 거리별 숫자를 해당 디렉터리 혼동행렬로 재계산했다. protocol 파일은 class mean prototype과 train split pooled within-class residual Ledoit–Wolf 공분산을 기록하고, valid/test를 적합에 쓰지 않았다고 명시한다. 중복 이미지 제외 수는 0이다. 기록된 manifest SHA-256(`983ca46a...16a40`)과 checkpoint SHA-256은 로컬 파일과 일치했다.

| split | 거리 | acc | Macro-F1 |
|---|---|---:|---:|
| valid | Euclidean | .381657 | .291403 |
| valid | Manhattan | .360947 | .288338 |
| valid | cosine | .375740 | .287665 |
| valid | Mahalanobis | .482249 | .364914 |
| test | Euclidean | .428994 | .331640 |
| test | Manhattan | .417160 | .321620 |
| test | cosine | .426036 | .331707 |
| test | Mahalanobis | .559172 | .442687 |

test Mahalanobis balanced accuracy는 .556311이다. 시험 수치는 혼동행렬 재계산과 일치한다. 단, `protocol.json`은 embedding을 `projection`으로 적었지만 `model_metadata`는 `backbone_only`, `use_projection=false`, projection 가중치 4개 누락으로 기록한다. 지표는 재검산했으나 어떤 feature space로 산출했는지는 파일 간 기록이 충돌해 확정할 수 없다. 핵심 파일은 `protocol.json`, `metrics.csv`, `confusion_test_*.csv`, `confusion_valid_*.csv`, `predictions.csv.gz`다.

### 11개 클래스 분류기

분할표 `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/manifests/11label_crop_final_8_1_1.csv`: 총 1,000장, train 797 / valid 102 / test 101. test의 클래스별 표본은 2–26장으로 작다.

`20260828_122908_11label_crop_final_8_1_1__resnet18.a1_in1k_sup`는 사전학습 ResNet18, 1 epoch, batch32, seed42, CPU, class weighting off다. valid acc/Macro-F1 `.186275/.067734`, test `.207921/.072957`, test weighted-F1 `.173810` (n=101)이다. `metrics/test_predictions.csv`에서 acc와 Macro-F1을 재계산해 일치했다. test에서 Bego, Bicon, Bilim, Dentium, Implance, ITI, MIS, Nucleoss 8개 클래스의 F1이 0이다. 상세 파일은 실행 폴더의 `args.json`, `metrics/{test,valid}_predictions.csv`, `metrics/{test,valid}_metrics.json`, `checkpoints/best.pt`다.

`20260828_123430_11label_crop_final_8_1_1__resnet18.a1_in1k_sup`는 위 실행과 best checkpoint SHA 및 test의 정답/예측/정오답 열이 같다. 독립 반복으로 세지 않는다. 6 epoch로 설정한 `20260828_123103`은 `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/logs/20260828_123100_train.log`가 첫 epoch 전에 끝나고 지표/체크포인트/종료 상태가 없어 미완료로 분류했다.

### 실패하거나 유효하지 않은 실행

- 7개 클래스 `20260828_115521`: `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/logs/20260828_115518_train.log`에 DataLoader worker bus error와 shared-memory 부족이 기록됐다. 지표/체크포인트 없음.
- 3개 클래스 Bego/Bicon/ITI 평가: manifest `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/manifests/Bego_Bicon_ITI_3label.csv`의 2,796행 `label` 값이 전부 문자열 `images`다. 평가 로그 `/Users/yunseojune/TRIA_LAB/implant_project/outputs_docker/logs/20260828_122319_eval.log`는 `KeyError: 'images'`로 중단됐다. 결과 지표 없음.

## 노트북에 저장된 결과: 로컬 재검증 불가

노트북은 `workspace` 경로를 사용한다. `/Users/yunseojune/TRIA_LAB/implant_project/md파일모음/README.md`도 실행 경로가 RunPod Jupyter라고 설명한다. 이번 조사 범위에서 해당 체크포인트와 원시 예측을 찾지 못했고 원격에 접속하지 않았다. 아래는 노트북의 저장된 출력만 읽은 값이며 재검증된 로컬 결과가 아니다.

| 원본 노트북 | protocol / split | 저장된 지표 | 제한 |
|---|---|---|---|
| `notebooks/vit/ViT_ another.ipynb` | ViT-Small SupCon, 7 classes, seed 100, batch64, 100 epochs 설정, temp .07, 64-D projection, train class mean + cosine | best valid acc .9438 (epoch66); test acc .9499, Macro-F1 .8448, weighted-F1 .9491, n339 | 출력 confusion matrix에서 n/acc/Macro-F1 재계산 일치. 클래스별 test 표본 수는 ADIN110, Dentium59, DIONAVI3, MIS14, nobel1, NORIS67, osstem85. 저장 경로 `/workspace/vit_supcon_prototype_model.pth`는 로컬에서 확인하지 못함. |
| `notebooks/vit/train_supcon_vit.ipynb` | Bego/Bicon/ITI 3-class; ViT-Small, seed42, batch64, lr3e-4, temp .07, config 150 epochs, 64-D projection; train+valid embedding bank cosine | test n111, acc .927928, Macro-F1 .74, weighted-F1 .92 | 노트북 출력만 확인. 클래스별 test 표본 수/F1: Bego 5/.33, Bicon 62/.97, ITI 44/.91. 150 epochs는 설정값이며 완료 여부를 보여주는 trace는 이 출력에서 확인되지 않음. |
| `notebooks/vit/vit_another2.ipynb` | 7-class ViT-SupCon 설정: 35 epochs, seed100, lr1e-4, wd2e-4, temp .07; 별도 Bego/Bicon/ITI protocol, 201 reference images, test 1,765 | best valid acc .9438 (epoch10); 3-class test acc .3428, Macro-F1 .3293, weighted-F1 .3819 | 클래스별 test 표본 수 Bicon959, ITI673, Bego133. 7-class와 분리된 평가라 직접 비교 불가. 저장 파일명이 `vit_supcon_prototype_model_epoch1.pth`여서 best epoch 설명과 이름이 다름. |
| `notebooks/metric_learning/metric_learn_3images.ipynb`, `metric_learn_5images.ipynb` | ResNet50 NT-Xent self-supervised, seed42, batch64, 50 epochs, lr1e-5, temp .2, wd1e-4; 1,791 unlabeled train; 512-D backbone h prototype, 256-D projection z loss; valid reps/class + cosine | 3-shot: test n109, acc .412844, Macro-F1 .34, weighted-F1 .44. 5-shot: .357798/.31/.40 | 노트북 저장값. Bego/Bicon/ITI 클래스별 test 표본 수 4/61/44. 각 1회 결과라 shot 증가 효과를 결론낼 수 없음. |
| `notebooks/kfold/resnet_50_kfold.ipynb` | ResNet50 classifier, seed42, stratified 5-fold, 15 epochs, lr1e-5; 동일한 109장 test 반복 평가 | best fold4 valid acc .89313; test acc .908257, Macro-F1 .6140, weighted-F1 .8919; fold test acc 평균 .893578 ± .012445 | test는 매 fold에서 같은 고정 109장을 사용. fold별 test 평균은 독립 test-fold 평가가 아님. |
| `notebooks/kfold/resnet18_kfold_noized.ipynb`, `resnet_50_kfold_noized.ipynb` | 서로 다른 root 설정이나 output cell 12/14/16 SHA가 서로 같음 | n219, 선택 test acc .890411, Macro-F1 .6680, weighted-F1 .8815; 평균 test acc .870320 ± .015445 | Bego/Bicon/ITI test 표본 수 8/123/88. 두 파일의 결과가 동일해 독립 모델 비교 자료로 볼 수 없음. |

`metric_learn_resnet_metric.ipynb`, `metric_learn_v1.ipynb`에는 loss trace가 있으나 test 결과는 확인되지 않았다.

## 탐색 범위와 증거 한계

- `Aihub/`에서 결과 지표나 체크포인트를 찾지 못했다.
- `md파일모음/`은 README 및 실험 관련 문서만 살폈고, 추가 수치 결과는 찾지 못했다.
- 중첩된 `/Users/yunseojune/TRIA_LAB/implant_project/implant_project`에서 별도의 result/output/run/checkpoint/metric 디렉터리는 찾지 못했다.
- Desktop/Downloads/Documents는 `implant`/`aihub` 이름이 있는 폴더만 검색했다. Downloads의 `AI Hub Data/08_Dentium_Implantium` 및 `GT_data_42label/AI Hub Data/08_Dentium_Implantium`은 데이터 폴더로 보이며 실험 결과 폴더는 찾지 못했다. 파일 내용은 열지 않았다.
- Docker 로그에는 `cuda_available:false` 및 CPU 실행이 표시된다. `Finished`는 로그의 완료 문구로만 취급했으며 OS 종료 코드를 확인하지 못했다.

행별 근거·파일 경로 목록은 같은 폴더의 `other_local_experiments.csv`에 있다.

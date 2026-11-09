# 원격 실험 종합 기록

- 기준일: 2026-09-29
- 대상: Windows App `localhost:3389`를 통해 V100 32GB 원격 서버에서 수행한 실험
- 현재 상태: 원격 세션은 2026-09-29 확인 시 연결이 끊겨 있었다. 아래 내용은 마지막으로 확인된 원격 화면·로그·산출물 기준이다.
- 원격 코드: `/home/ubuntu/implant_pipeline_new`
- 원격 결과: `/home/ubuntu/aihub_run_new`
- 공통 조건: AI-Hub 임플란트 X-ray, test split, seed 1개가 기본이다. 신뢰구간이 없는 결과가 많다.
- 작업 방식: 원격 화면·화상키보드·원격 내부 복사만 사용했다. 로컬 SSH나 로컬 파일 전송 결과로 보지 않는다.

## 0. 유의미한 결과만

### 핵심 수치

| 질문 | 결과 | 해석 | 제한 |
|---|---|---|---|
| 모든 학습 클래스 분류가 잘 되는가 | 42클래스 Band-ViT Acc `0.9589`, ViT-Large Macro-F1 `0.9200` | Seen 분류 성능은 높다 | Unseen 인식 근거가 아님 |
| 전체 정확도가 가장 높은 metric 조건은 무엇인가 | M6 백본 + galcov Mahalanobis: 전체 `0.909`, Unseen `0.702` | 전체 정확도 기준 최고 | Unseen train gallery를 사용해 strict zero-shot이 아님 |
| Unseen 인식이 가장 높은 조건은 무엇인가 | M7 SupCon + galcov Kmeans/Mahalanobis: Unseen `0.821`, 전체 Macro-F1 `0.772` | 현재 기록상 Unseen 기준 최고 | Seen 정확도 `0.857`, 전체 정확도 `0.854`로 하락 |
| gallery 공분산의 영향은 무엇인가 | 백본 Unseen `0.269` → galcov `0.702` | 공분산과 gallery 구성이 결과에 큰 영향 | 같은 조건이 아니므로 단순 모델 개선으로 해석 금지 |
| 속성이 Unseen을 개선했는가 | 10속성 조합 매칭 Unseen `0.025`; 속성 점수 결합도 Unseen 하락 | 현재 속성 입력은 Unseen에 효과 없음 | GS III·TS III 라벨/속성 오류 영향 확인 필요 |
| 예측 속성 토큰이 개선했는가 | 41클래스 기록값 전체 `0.8265`, Unseen `0.3751` | 예측 속성 노이즈가 성능을 크게 낮출 가능성 | 최종 로그·조건 원본 재확인 필요 |
| 학습 가능한 속성 토큰은 어떤가 | 20epoch/patience 3 기록값 전체 `0.9029`, Unseen `0.7305` | 기준선보다 높아 보이는 탐색 결과 | 평가 조합과 checkpoint 원본 대조 전 채택 금지 |

### 확정할 수 있는 판단

1. **현재 Unseen 목표의 1순위 후보는 M7**이다. 다만 galcov 조건과 Seen 성능 하락을 함께 기록해야 한다.
2. **전체 정확도만 필요하면 M6**가 가장 높다. 이 결과를 zero-shot 성능으로 부르면 안 된다.
3. **속성 결합은 채택하지 않는다.** Seen 상승만으로는 부족하고 Unseen이 반복해서 낮아졌다.
4. **Unseen 실패의 구조적 원인은 확인됐다.** M7에서 Seen 내부 거리 `0.188`, Unseen 내부 거리 `0.584`; Seen 중심 거리 `1.092`, Unseen 중심 거리 `0.655`로, Unseen이 더 퍼지고 서로 가까운 상태다.
5. **Triplet·CovAware는 제외한다.** 전체 정확도가 각각 `0.105` 수준과 `0.335`로 낮아 현재 기준선이 될 수 없다.
6. **41클래스 실험은 42클래스 결과와 분리한다.** `it`와 `internal` 통합 이후의 같은 조건 기준선과만 비교해야 한다.

### 다음 실험 전에 반드시 고칠 것

- 스펙표와 manifest 불일치: `45/49행 대 42라벨`, `dataset_label` 공백으로 `0/42` 연결된 기록이 있다.
- 속성 라벨 확인: GS III가 TS III 속성으로 예측되는 현상과 TS/SS/US III 스펙표를 원본 자료와 대조한다.
- 비교 조건 고정: 41클래스, 같은 gallery, 같은 공분산 방식, 같은 prototype·거리 함수를 사용한다.
- seed·holdout 반복: 단일 seed 수치로 채택하지 말고 seed 42/43/44와 class holdout 결과를 확인한다.

### 유의미한 결과에서 제외한 항목

- GPU 사용률, 실행 중인 PID, 열린 tmux 창, smoke 통과만으로는 학습 완료나 성능 개선으로 보지 않는다.
- GT mean 결과(`15.7~16.0%`, 25클래스)는 전처리·매핑 점검이 끝나기 전에는 모델 결론으로 사용하지 않는다.
- 로컬 few-shot·PCA384 결과는 이 문서의 원격 결과 요약에서 제외했다.

## 0-M. MICCAI 목표 기준

### 논문 주제 후보

**학습하지 않은 치과 임플란트 시스템을 등록 이미지와 임베딩 거리로 인식하는 open-world 인식 방법**

방법 이름은 임시로 `gallery-calibrated covariance-aware prototype inference`로 둔다. 핵심은 새 클래스를 다시 학습하지 않고, 등록 이미지로 prototype과 공분산을 만들며, Seen·Unseen 성능을 함께 평가하는 것이다.

### 현재 결과의 논문 가치

- **가장 쓸 수 있는 결과:** M7의 Unseen `82.1%`, 전체 Macro-F1 `0.772`.
- **핵심 분석:** Unseen은 같은 클래스 내부 산포가 크고 클래스 중심이 가깝다. 단순히 전체 정확도를 높이는 것보다 이 구조를 해결하는 것이 중요하다.
- **속성 결과:** 속성 결합이 Unseen을 낮춘다는 실패 결과가 있다. 속성을 주기만 하면 좋아진다는 주장을 반박하는 분석으로 사용한다.
- **데이터 기여:** 42클래스와 41클래스의 라벨·스펙표 차이를 드러냈다. 단, 라벨 통합 근거를 먼저 확정해야 한다.

### MICCAI main track 기준에서 부족한 것

현재 결과만으로는 제출 가능한 상태가 아니다. 다음 네 가지가 없으면 핵심 수치를 논문 결론으로 쓰지 않는다.

1. **누수 없는 데이터 분할:** 환자·기관·촬영일·촬영 위치 기준의 중복 점검과 기관 제외 테스트.
2. **반복 측정:** 여러 Unseen 클래스 분할과 seed 42/43/44 이상을 사용하고 평균·표준편차·95% 신뢰구간을 보고한다.
3. **공정한 비교:** frozen DINOv2/ViT, CE 분류기 feature, cosine·Euclidean·shrinkage Mahalanobis NCM, KNN 또는 ProtoNet을 같은 split·shot·예산으로 비교한다.
4. **주 지표 고정:** 전체 정확도 대신 Seen/Unseen 조화평균(H), Unseen Macro-F1, balanced accuracy를 1차 지표로 둔다. 전체 정확도는 보조 지표다.

### 논문 실험 구성

| 단계 | 논문에 넣을 내용 | 현재 상태 |
|---|---|---|
| P0 | 41/42클래스 라벨 확정, manifest·스펙표·환자/기관 중복 점검 | 미완료 |
| P1 | 같은 조건의 cosine, Euclidean, Mahalanobis, Kmeans prototype 비교 | 일부 완료 |
| P2 | closed-only 공분산과 gallery 공분산 분리 보고 | 결과 있음, 반복 필요 |
| P3 | 3~5개 seed × 여러 Unseen class split | 큐 등록, 최종 결과 없음 |
| P4 | 기관 제외 또는 외부 데이터 검증 | 아직 없음 |
| P5 | 실패 분석: internal↔Hexplant, Anyone internal↔external, GS III↔TS III | 일부 완료 |

### 논문에서 제외할 주장

- 42클래스 closed-set `95%대`를 Unseen 일반화 성능으로 쓰지 않는다.
- M6/M7 galcov 수치를 strict zero-shot으로 부르지 않는다. Unseen train gallery가 들어간 inductive 설정이다.
- 속성 토큰·Gap-SupCon은 반복 실험과 원본 산출물 확인 전에는 개선 방법으로 확정하지 않는다.
- GT prototype의 낮은 수치는 전처리·매핑 검증 전 모델 결론으로 사용하지 않는다.

### MICCAI 진행 판정

- **현재 판정: 조건부 진행.** 연구 질문과 예비 결과는 있으나, 반복·기관 제외·누수 검증이 부족하다.
- **Go 기준:** 같은 41클래스 프로토콜에서 M7 계열이 여러 split·seed에서 H와 Unseen Macro-F1을 일관되게 높이고, 기관 제외 테스트에서도 유지될 것.
- **Stop 또는 주제 변경 기준:** 반복 결과에서 개선이 사라지거나 기관 제외 성능이 무너지면 방법론 논문으로 밀지 않고, open-world 평가 프로토콜·실패 분석 중심으로 다시 설계한다.

세부 문헌 조사와 MICCAI 요구사항 메모는 `docs/handoff_20260924/paper_research_20260924.md`에 있다. 해당 문헌 인용은 최종 제출 전에 원문을 다시 확인한다.

### 공식 기준 참고

MICCAI 2026 공식 지침 기준으로는 8쪽 본문 + 최대 2쪽 참고문헌, 이중 익명 심사, 데이터 출처·사용권·윤리 정보 기재, 재현 가능한 데이터·구현·평가 설계가 중요하다. 다음 연도 규정은 달라질 수 있으므로 제출 직전에 공식 지침을 다시 확인한다.

- [MICCAI 2026 Paper Submission Guidelines](https://conferences.miccai.org/2026/en/PAPER-SUBMISSION-GUIDELINES.html)
- [MICCAI 2026 Reviewer Guidelines](https://conferences.miccai.org/2026/en/REVIEWER-GUIDELINES.html)

## 0-W. 웹·연구 스펙표 대조 (2026-09-29)

### 비교 범위

- 로컬 기준표: [`implantsj/configs/class_features.csv`](../../implantsj/configs/class_features.csv) 45행, Company·Name을 제외한 11개 속성. 2026-09-29 웹 대조로 19행을 수정했고, 변경 근거를 각 행의 `notes`에 기록했다. 행 수와 `class_key`는 유지했다.
- 수정 후보표: [`docs/handoff_20260924/implant_attributes_corrected_conservative.csv`](../handoff_20260924/implant_attributes_corrected_conservative.csv) 49행.
- 원격 42개 라벨: `RESULTS.md`에 남은 목록을 기준으로 대조했으며, 원격 세션이 끊겨 현재 `manifest_full42.csv`와 `spec_table.csv` 원본은 재조회하지 못했다. 따라서 42개 목록은 **보고서에 기록된 원격 관찰값**이고 최종 확정값이 아니다.

### 42개 원격 라벨과 로컬 45행의 차이

`RESULTS.md`의 42개 라벨을 제품명 alias로만 맞춰 보면 다음 차이가 난다. 이 비교는 현재 원격 파일을 다시 읽은 결과가 아니므로 최종 라벨 확정에 사용하지 않는다.

| 구분 | 라벨 |
|---|---|
| 원격 42개에 있고 로컬 기준표에 없는 이름 | `it`, `iu`, `Ankylos`, `Stella` |
| 로컬 기준표에 있고 원격 42개 목록에 없는 이름 | `SM`, `SM_Submerged`, `Atlas_External`, `SwissPlus`, `Semados RS/RSX`, `Integra-CP`, `Standard Plus` |
| alias 확인이 필요한 이름 | `Hexplant↔Explant`, `Biohorizon external↔External`, `Simpleline↔SimpleLine`, `luna↔Luna1` 등 |

원격에서 `it`와 `internal`을 합쳐 41개로 만든 것은 같은 제품일 가능성을 확인하기 위한 데이터 처리다. `RESULTS.md`에는 같은 일련번호·규격·식립일·치아 위치가 다른 병원에서 서로 다른 이름으로 기록된 사례가 있어, 제품 alias와 기관별 라벨 오류를 분리한 뒤 병합해야 한다. 제조사 페이지는 Warantec IT와 IU를 별도 제품으로 구분하므로, 병합을 제조사 근거로 확정할 수 없다.

### 연구 표준과 현재 표의 차이

Sahiwal 계열의 임플란트 식별 연구는 연결부·플랜지·몸체·나사산·첨부부를 치관부/중간부/첨부부로 나눠 기록한다. 최근 스코핑 리뷰도 제조사·브랜드만 쓰는 분류보다 고유한 macrogeometry와 prosthetic connection을 함께 기록할 것을 제안한다 ([Sahiwal et al.](https://pubmed.ncbi.nlm.nih.gov/12070518/), [2026 scoping review](https://pc.isdent.org/DOIx.php?id=10.5624/isd.20250240)).

| 현재 필드 | 외부 자료에서 보통 분리하는 항목 | 판정 |
|---|---|---|
| `Connection_1` + `Connection_2` | 연결 계열, indexing(hex/octagon 등), engagement(Morse/conical/CrossFit 등) | `HEX`, `OCTA`, `CONICAL`을 한 칸에 넣으면 서로 다른 구조가 섞인다. 3개 필드로 분리해야 한다. |
| `Flange` + `Collar` | neck shape(직선/flare), neck surface(machined/smooth/microthread) | `Collar=Yes/No`만으로는 제조사 사양과 방사선 관찰 기준을 구분할 수 없다. |
| `Body_Shape` | straight, tapered, mixed taper | `Tapered body`, `Tapered apex`, `Non-tapered`의 경계를 고정하고 복합형을 별도 값으로 둬야 한다. |
| `Body_Type` | threaded, non-threaded, plateau/fin, locking taper | Bicon처럼 일반 나사산이 아닌 제품을 표현할 값이 없다. |
| `Thread_Shape` | V, square, buttress, reverse buttress, 부위별 혼합 | 상부·하부 모양이 다른 제품은 기준 부위를 함께 저장해야 한다. |
| `Apex_Shape` + `Apex_Hole` | flat/curved, orifice, apical chamber, grooves | 현재 두 칸만으로는 첨부부 구조를 충분히 표현하지 못한다. |

### 외부 자료와 직접 대조해 확정할 수 있는 불일치

| 로컬 행 | 외부 자료 대조 | 판정·조치 |
|---|---|---|
| 5 `TSIII`, 7 `SSIII`, 8 `USIII` | Osstem 공식 카탈로그는 세 모델을 `1.5° tapered body`로 설명한다 ([Osstem TS catalogue](https://www.osstemuk.com/catalogs1/TS-CATALOG.pdf)). | 현재 `Body_Shape=Non-tapered`는 틀렸다. `Tapered body`로 고치되, 원본 제품 변형을 먼저 고정한다. |
| 6 `USII` | FDA 510(k) 표와 식별 DB는 USII를 external hex·straight body로 설명한다 ([FDA K161103](https://www.accessdata.fda.gov/cdrh_docs/pdf16/K161103.pdf), [ImplantDB USII](https://implantdb.org/implants/osstem-implant-company-usii-sa-5-x-13-mm-wide-platform)). | 현재 `Tapered apex`는 틀렸다. `Straight/Non-tapered`로 정정한다. |
| 11 `SimpleLine` | Dentium 자료는 SimpleLine II를 tissue-level 제품으로 설명하고, `8° Morse Taper`와 internal octagon을 명시한다 ([Dentium SimpleLine II](https://dentiumjp.com/product/Implant/)). | 현재 `Bone Level + HEX`는 틀렸다. 다만 행명이 I/II를 구분하지 않으므로 모델 변형을 먼저 확정한다. |
| 33 `TS_StandardPlus`, 46 `Standard Plus` | Standard Plus는 1.8 mm machined neck 제품이다 ([Straumann Standard Plus](https://www.straumann.com/en/dental-professionals/dental-implants/tissue-level-implant-line/standard-plus-implant.html), [ImplantDB](https://implantdb.org/implants/straumann-tl-standard-plus-3.3-x-10-mm-nnc-3-5-platform)). | 같은 제품 변형이면 두 행을 하나로 합친다. `Collar=No/Yes` 충돌은 collar를 machined neck 기준으로 정의할 경우 33행이 틀렸다. |
| 34 `BoneLevel` | Straumann CrossFit은 단순 원뿔이 아니라 15° conical-cylindrical connection과 내부 4개 groove다 ([Straumann CrossFit](https://www.straumann.com/be/en/dental-professionals/prosthetics/connections/straumann-crossfit.html)). | `Connection_2=CONICAL` 하나로 끝내면 정보가 빠진다. `engagement_angle`, `anti_rotation_grooves`를 분리한다. |
| 45 `Integra-CP` | Bicon은 locking taper와 plateau design을 사용하며 일반적인 screw thread 제품이 아니다 ([Bicon MAX catalog](https://bicon.com/pdf/Bicon_MAX_Catalog.pdf)). | `Body_Type=Plateau-fin`, `Thread_Shape=N/A`로 수정했다. 현재 `Connection_2`를 indexing으로 해석하면 `NONE`이 맞고, 실제 체결 방식 `LOCKING-TAPER`는 별도 필드로 분리해야 한다. |
| 원격 `it`·`internal` | Warantec 공식 제품 페이지는 IT와 IU를 별도 제품으로 나누고, IT를 tapered body·square progressive thread·400 μm microthread로 설명한다 ([Warantec products](https://warantec.com/implant/)). | `it+internal`을 41개로 합친 것은 제조사 확인이 아니라 데이터 라벨 결정이다. 원격 manifest의 제품명·기관 코드·이미지 중복을 확인하기 전에는 병합하지 않는다. |

### 현재 판정

- **스펙표의 큰 범주는 연구 표와 맞지만, 현재 값은 그대로 고정하면 안 된다.** 특히 연결부, collar, Bicon, TS/SS/US III가 핵심 오류다.
- **41/42 라벨은 아직 확정하지 않는다.** 원격 재접속 후 `manifest_full42.csv`, `spec_table.csv`, `dataset_label`을 같은 시점의 파일로 읽고 행 수·라벨 집합·alias를 한 번에 대조해야 한다.
- MICCAI용 기준표는 제조사 제품명과 데이터셋 라벨을 분리하고, `제품 변형 → 정규화 스펙 → 데이터셋 라벨` 매핑표와 변경 이력을 남긴 뒤 다시 평가해야 한다.

## 1. 상태 표기

| 상태 | 의미 |
|---|---|
| 완료·수치 확인 | 원격 결과 파일, 종료 코드, 로그 또는 화면 수치를 확인함 |
| 부분 확인 | 실행·일부 결과는 확인했지만 최종 로그·산출물·원본 대조가 남음 |
| 준비·미실행 | 설정·코드·사전 검사는 있으나 학습/평가 완료를 확인하지 못함 |

## 2. 전체 결론

| 실험 | 확인된 핵심 결과 | 상태 |
|---|---|---|
| 27/42클래스 백본 분류 | 27클래스 최고 Band-DINOv2 Acc `0.9524`, 42클래스 최고 Band-ViT Acc `0.9589` | 완료·수치 확인 |
| 42클래스 metric learning | 전체 Acc 최고 M6 `0.909`, Unseen Acc 최고 M7 `0.821`, 전체 Macro-F1 최고 `0.772` | 완료·수치 확인 |
| 10개 속성 모델 | Seen 속성 정확도 `0.959~1.000`, Unseen 속성 조합 매칭 `0.025` | 완료·수치 확인 |
| 속성 점수와 이미지 임베딩 결합 | Seen은 올라갔지만 Unseen은 내려감 | 완료·수치 확인 |
| `it`+`internal` 41클래스 기준선 | Mean OAS 전체 `89.6198%`, Kmeans OAS Unseen `79.0336%` | 결과 기록 있음 |
| 학습 가능한 8개 속성 토큰 | 20epoch/patience 3 기록값: 전체 `90.2897%`, Unseen `73.0549%` | 부분 확인 |
| 예측 속성 8개 토큰 | 20epoch/patience 3 기록값: 전체 `82.6494%`, Unseen `37.5102%` | 부분 확인 |
| GT 프로토타입 평가 | 25클래스, GT 232장·query 12,731장, 컨테이너 exit 0 | 완료·수치 확인 |
| 후속 OAS·holdout·반복 실험 | 일부 사전 검사와 큐 등록 확인, 전체 종료 미확인 | 부분 확인 |
| Gap-SupCon | 설계·코드·사전 실행 준비 확인, 최종 성능 없음 | 준비·미실행 |

## 3. 백본 분류 실험

분류 헤드를 사용한 closed-set 평가다. 학습에 없는 클래스를 인식하는 실험이 아니므로 metric learning 결과와 직접 비교하면 안 된다.

| 모델 | 27클래스 Acc / Macro-F1 | 42클래스 Acc / Macro-F1 |
|---|---:|---:|
| Band-DINOv2 | `0.9524 / 0.9024` | `0.9565 / 0.9188` |
| ViT | `0.9515 / 0.9012` | `0.9563 / 0.9194` |
| Band-ViT | `0.9494 / 0.8981` | `0.9589 / 0.9197` |
| DINOv2 | `0.9377 / 0.8754` | `0.9369 / 0.8900` |
| ViT-Large | - | `0.9544 / 0.9200` |

주요 혼동은 `internal↔Hexplant`, `Anyone external↔Anyone internal`, `it→internal`, `IS II→IS I`에 집중됐다. 표본이 23장 이하인 Unseen 클래스는 정확도 변동이 크다.

### 3-1. 초기 원격 백본 학습 16개

아래는 `/home/ubuntu/aihub_run/outputs/backbone`에서 확인한 완료 학습의 최종 테스트 값이다. `closed27_open15`는 27개 클래스 학습, `full42`는 42개 클래스 학습이다.

| 모델·학습 방식 | 학습 범위 | 최종 Macro-F1 | Acc |
|---|---|---:|---:|
| Band-ViT full fine-tune | closed27 | `0.9128` | `0.9562` |
| ViT full fine-tune | closed27 | `0.9056` | `0.9535` |
| ViT partial fine-tune | closed27 | `0.8766` | `0.9363` |
| Band-ViT partial fine-tune | closed27 | `0.8717` | `0.9353` |
| Band-ViT full fine-tune | full42 | `0.9122` | `0.9565` |
| ViT full fine-tune | full42 | `0.9138` | `0.9560` |
| ViT partial fine-tune | full42 | `0.8975` | `0.9389` |
| Band-ViT partial fine-tune | full42 | `0.8968` | `0.9363` |
| DINOv2 full fine-tune | closed27 | `0.8720` | `0.9320` |
| DINOv2 full fine-tune | full42 | `0.8658` | `0.9289` |
| DINOv2 partial fine-tune | closed27 | `0.8835` | `0.9448` |
| DINOv2 partial fine-tune | full42 | `0.8924` | `0.9434` |
| Band-DINOv2 full fine-tune | closed27 | `0.8699` | `0.9337` |
| Band-DINOv2 partial fine-tune | closed27 | `0.8850` | `0.9429` |
| Band-DINOv2 full fine-tune | full42 | `0.8809` | `0.9340` |
| Band-DINOv2 partial fine-tune | full42 | `0.8943` | `0.9397` |

추가로 Band-ViT 분류 실험(`alpha=16`, `lambda=0.95`, `block=4`)은 47epoch에서 조기 종료됐고, Acc `0.953772`, Macro-F1 `0.920510`, Weighted-F1 `0.953851`, 컨테이너 종료 코드 0으로 기록됐다. 과거 중복 컨테이너의 exit 143과 섞지 않는다.

### 3-2. ViT 임베딩 거리 평가

백본을 새로 학습하지 않은 평가와 closed27로 미세조정한 ViT 평가를 구분한다.

| 모델·조건 | 프로토타입/거리 | 전체 Acc | 전체 Macro-F1 | Seen Acc | Unseen Acc |
|---|---|---:|---:|---:|---:|
| 사전학습 ViT, closed27+open15 | Mean + Mahalanobis | `0.6021` | `0.4687` | `0.5943` | `0.6994` |
| 사전학습 ViT, full42 | Mean + Mahalanobis | `0.6034` | `0.4741` | - | - |
| 사전학습 ViT, full42 | Kmeans multi + Mahalanobis | `0.5966` | `0.4967` | - | - |
| ViT full fine-tune, closed27 | Mean + Mahalanobis | `0.8760` | `0.7072` | `0.8999` | `0.5758` |

이 표는 최신 DINOv2 metric 실험의 M1~M10과 별도 실행이다. 거리 함수와 학습 범위가 다르므로 하나의 순위표로 합치지 않는다.

## 4. Metric learning: 27클래스 학습 → 15클래스 Unseen 인식

`closed-only`는 Seen 학습 이미지로만 공분산을 계산한다. `gallery covariance(galcov)`는 Seen·Unseen 등록용 train 이미지를 함께 사용한다. 따라서 두 조건은 같은 실험으로 비교하면 안 된다.

| 실행 | 학습 | 공분산 | 전체 Acc | 전체 F1 | Seen Acc | Unseen Acc | 상태 |
|---|---|---|---:|---:|---:|---:|---|
| M1 | 학습 없는 DINOv2 백본 | closed-only | `0.886` | `0.616` | `0.935` | `0.269` | 완료 |
| M2 | SupCon 5epoch | closed-only | `0.884` | `0.688` | `0.911` | `0.539` | 완료 |
| M3 | ArcFace 5epoch | closed-only | `0.860` | `0.639` | `0.881` | `0.601` | 완료 |
| M4 | Triplet 5epoch | closed-only | `0.124` | `0.092` | `0.115` | `0.243` | 학습 붕괴 |
| M5 | ViT head-only SupCon | closed-only | `0.557` | `0.435` | `0.558` | `0.539` | 완료 |
| M6 | 학습 없는 DINOv2 백본 | galcov | `0.909` | `0.724` | `0.926` | `0.702` | 완료 |
| M6 | 같은 백본 | galcov + Kmeans | `0.907` | `0.756` | `0.920` | `0.749` | 완료 |
| M7 | SupCon 5epoch | galcov | `0.898` | `0.733` | `0.911` | `0.731` | 완료 |
| M7 | SupCon 5epoch | galcov + Kmeans | `0.854` | **`0.772`** | `0.857` | **`0.821`** | 완료 |
| M8 | ArcFace 5epoch | galcov | `0.868` | `0.677` | `0.883` | `0.681` | 완료 |
| M8 | ArcFace 5epoch | galcov + Kmeans | `0.787` | `0.668` | `0.790` | `0.741` | 완료 |
| M9 | Triplet 5epoch | galcov | `0.105` | `0.079` | `0.096` | `0.217` | 학습 붕괴 |
| M10 | CovAware SupCon | - | `0.335` | `0.245` | `0.326` | `0.454` | 실패 |

판정:

- 전체 정확도만 보면 M6 `0.909`가 가장 높다.
- Unseen 정확도와 전체 Macro-F1을 함께 보면 M7 Kmeans+Mahalanobis가 가장 높다.
- M7은 Seen 정확도가 `0.857`로 내려가므로 전체 운영 기준의 단일 우승 모델로 확정하면 안 된다.
- Triplet, CovAware, RDA 계열은 성능이 무너졌고 원인 수정·재검증 전 채택하지 않는다.
- M6/M7의 galcov 결과는 Unseen train gallery를 사용하므로 strict zero-shot 결과가 아니다.

## 5. 속성 모델과 속성 점수 결합

### 5-1. 10개 속성 모델

ConvNeXt-Tiny를 27개 Seen 클래스에 학습하고 15개 Unseen을 평가했다.

| 항목 | 결과 |
|---|---:|
| Seen 속성 정확도 범위 | `0.959~1.000` |
| Unseen 속성 조합 매칭 | 전체 `0.871`, Seen `0.938`, Unseen `0.025` |
| 42클래스 조합 채점 | 전체 `0.798`, Seen `0.860`, Unseen `0.011` |
| Unseen에서 낮은 속성 | `Body_Shape 0.257`, `Apex_Shape 0.287`, `Microthread 0.484` |

GS III가 Unseen의 약 41%를 차지하며, `Body_Shape`·`Apex_Shape`를 Seen의 TS III 속성으로 예측한 것이 주요 실패 원인으로 확인됐다. TS/SS/US III 스펙 라벨 오류 가능성은 별도 검증 대상이다.

### 5-2. 속성 점수와 이미지 거리 결합

42클래스 M7 Kmeans+Mahalanobis 기준이다. 수치는 `전체 / Seen / Unseen` 정확도다.

| 속성 가중치 λ | 결과 |
|---:|---:|
| `0` | `0.854 / 0.857 / 0.821` |
| `0.03` | `0.901 / 0.919 / 0.681` |
| `0.1` | `0.893 / 0.934 / 0.375` |
| `1` | `0.874 / 0.939 / 0.060` |

속성 점수를 넣으면 Seen은 좋아지지만 Unseen은 낮아진다. λ를 test 결과로 고른 기록은 최종 채택 근거가 아니다. validation에서 정해야 한다.

## 6. 41클래스 기준선과 속성 토큰

`it`와 `internal`을 통합한 별도 `new_41class` 경계를 사용했다. 기존 42클래스 결과와 직접 비교하지 않는다.

| 실험 | 평가 방식 | 전체 Acc | Seen Acc | Unseen Acc | Macro-F1 |
|---|---|---:|---:|---:|---:|
| 이미지 전용 5epoch | Mean + Mahalanobis OAS | `89.6198%` | `91.0874%` | `71.1712%` | `0.727992` |
| 이미지 전용 5epoch | Kmeans + Mahalanobis OAS | `84.8582%` | `85.3215%` | `79.0336%` | `0.767481` |
| 학습 가능한 속성 토큰, 최대 20epoch/patience 3 | 평가 조합 원본 대조 필요 | `90.2897%` | `91.6607%` | `73.0549%` | `0.764502` |
| 예측 속성 토큰, 최대 20epoch/patience 3 | Kmeans + Mahalanobis OAS | `82.6494%` | `86.2401%` | `37.5102%` | `0.605398` |

### 6-1. 학습 가능한 8개 속성 토큰

- 속성: `System`, `Connection_1`, `Flange`, `Collar`, `Microthread`, `Body_Shape`, `Thread_Shape`, `Apex_Shape`
- 제외: `Connection_2`, `Apex_Hole`, `Body_Type`
- 입력: 이미지 패치 256개 + CLS + 학습 가능한 속성 토큰 8개
- 손실: SupCon + 속성 보조 CE
- 정답 속성은 보조 손실 target으로만 사용하고, 추론 입력에는 넣지 않는다.
- 실제 GPU smoke는 `SMOKE_OK`, forward/backward, gradient, `-1` 마스킹, checkpoint 저장/복원을 확인했다.
- 9/27 초기 실행은 최대 5epoch/patience 2로 시작이 확인됐다. 이후 20epoch/patience 3 별도 실행의 최종 원본 로그·best checkpoint 재확인이 남아 있다.

### 6-2. 예측 속성 토큰

- 고정 ConvNeXt-Tiny가 이미지에서 출력한 8개 속성 softmax 확률을 사용한다.
- 각 확률 벡터를 DINOv2 토큰 차원으로 투영한 뒤 속성 토큰에 더한다.
- 학습·검증·테스트 모두 이미지에서 얻은 예측 확률을 사용한다.
- 정답 속성은 보조 CE target으로만 사용한다. test에 정답 속성을 넣은 oracle 실험이 아니다.
- smoke는 `SMOKE_OK`, predictor 무gradient, adapter gradient, checkpoint 저장/복원을 확인했다.
- 기록된 최종 평가는 전체 `82.6494%`, Unseen `37.5102%`다. 학습 조건과 원본 종료 로그를 다시 읽기 전에는 속성 자체의 원인으로 단정하지 않는다.

## 7. GT 프로토타입과 임베딩 거리 분석

### 7-1. GT 프로토타입 평가

- 컨테이너: `gt-prototype-predattr-v2`
- 종료 코드: `0`
- GT 등록 이미지: `232장`
- query: `12,731장`
- 매핑 후 평가 클래스: `25개`
- GT와 query의 SHA-256 중복 검사는 통과했다.
- 같은 환자의 다른 촬영본까지 제외한 환자 단위 중복 검사는 아니다.

| 프로토타입·거리 | 전체 Acc | Seen Acc | Unseen Acc | Macro-F1 |
|---|---:|---:|---:|---:|
| GT mean + cosine | `15.7175%` | `16.6723%` | `3.4745%` | `0.048109` |
| GT mean + euclidean | `16.0396%` | `17.0110%` | `3.5831%` | `0.051430` |

이 결과만으로 GT 데이터 자체가 나쁘다고 결론 내릴 수 없다. GT 전처리, 클래스 매핑, 촬영 조건 차이를 추가 확인해야 한다.

### 7-2. 전체 41클래스 임베딩 거리

| 임베딩 | 출력 | Cosine 내부 거리 | Cosine 클래스 간 거리 | 분리 비율 |
|---|---:|---:|---:|---:|
| 이미지 전용 41클래스 | 128 | `0.093474` | `0.533075` | `5.702951` |
| 예측 속성 토큰 | 128 | `0.139069` | `0.626711` | `4.506462` |
| 학습 가능한 속성 토큰 20epoch | 128 | `0.115702` | `0.621502` | `5.371559` |

예측 속성 토큰은 클래스 간 거리가 늘었지만 클래스 내부 산포도 더 커져 분리 비율이 낮아졌다. 이는 Unseen 성능 하락의 가능성을 보여주지만, Seen/Unseen을 같은 조건으로 나눈 추가 비교 없이 원인으로 확정할 수 없다.

### 7-3. M7·M8 geometry 관찰

- M7 SupCon galcov: Seen 내부 거리 `0.188`, Unseen 내부 거리 `0.584`; Seen 중심 간 거리 `1.092`, Unseen 중심 간 거리 `0.655`.
- M8 ArcFace galcov: Unseen 중심 간 거리는 `0.962`로 늘었지만 Unseen 내부 거리는 `0.930`으로 더 커졌다.
- M8 Unseen 정확도 `0.741`은 M7의 `0.821`보다 낮다. 중심 거리 하나만으로 모델을 고르면 안 된다.

## 8. 초기 폐쇄망 파이프라인·mask 준비 기록

아래 항목은 원격 실행 준비와 실패 원인 확인 기록이다. 최종 학습 성능으로 세지 않는다.

- V100 32GB용 P27(Seen 27개 학습)·P42(42개 학습) S0~S8 실행 절차와 `ai_hub_pipeline:emb3` Docker 이미지 사용법을 정리했다.
- P27 중 한 번의 Docker 실패는 VRAM 부족이 아니라 shared memory 부족이었다. 재실행 조건으로 `--ipc=host`가 확인됐다.
- 화면에서 관찰된 작업 이름은 `p27-backbone`, `p27-masked`, `p42-backbone`, `p42-masked`였다. 각 작업의 최종 로그·exit code가 없는 상태를 완료로 기록하지 않는다.
- `configs/02_segmentation.yaml`의 데이터 경로를 `data/prepared/imagefolders/data_42label`로 맞춘 기록이 있다.
- 완전 검정 원본 이미지 3장과 대응 prediction/band mask를 `/home/ubuntu/aihub_run/excluded_blank3`로 옮긴 뒤 manifest를 다시 생성했다. mask 품질이나 최종 분류 성능을 검증한 것은 아니다.
- Band-ViT mask 설정은 기대 조합 8개가 아니라 18개로 노출되어 일시 중지된 기록이 있다.
- 이 단계의 acceptance는 이미지·mask 개수 일치와 빈 mask 0개 확인이며, mIoU 검증 결과는 없다.

## 9. 데이터·라벨·환경 점검

- 원격 `implant_pipeline_new`에서 스펙표 행 수와 42클래스 manifest가 맞지 않는 상태가 관찰됐다. 초기 기록은 `45행 대 42라벨`, 이후 다른 기록은 `49행`을 보고해 현재 수치는 다시 읽어야 한다.
- `dataset_label`이 비어 기존 matcher가 `0/42`를 연결한 상태가 관찰됐다. 이 문제는 학습 성능 결과가 아니라 라벨 연결 문제다.
- `Company` 제외, `Body_Type` 추가, 속성 순서·라벨 매핑 검사가 설정에 반영된 기록이 있다. 실제 현재 파일은 원격 재접속 후 다시 확인해야 한다.
- 세 장의 완전 검정 원본 이미지와 대응 mask를 `/home/ubuntu/aihub_run/excluded_blank3`로 옮긴 뒤 manifest를 다시 만든 기록이 있다. 최종 데이터 준비 acceptance는 별도 확인이 필요하다.
- `it`+`internal` 통합은 기존 데이터·코드를 덮지 않는 별도 `new_41class` 경계로 관리했다.

## 10. 2026-09-28 후속 큐 상태

원격 화면에서 2026-09-28 23:16 KST 기준으로 확인된 상태다. 큐 전체 완료는 확인하지 못했다.

| 작업 | 마지막 확인 |
|---|---|
| 동일 OAS 조건 비교 | 4개 모델 × mean/Kmeans 비교 파일 생성 확인 |
| ViT·Band-ViT galcov | 일부 실행 중·계산 중 |
| 속성 A/B class holdout·frozen backbone | 사전 검사 통과, 큐 대기 |
| ViT SupCon 5epoch | model load/forward 검사 통과, 큐 대기 |
| fine-tuning checkpoint 기반 metric 5epoch | 별도 사본에서 assert 우회 후 preflight 통과, 큐 대기 |
| 속성 A/B seed 반복 | seed 42/43/44, 총 6회 큐 등록 |
| M10 CovAware | 9/24 재실행 exit 0, 성능 저하 확인 |
| 41클래스 거리 분석 | `geometry41_verified.csv` 확인 |
| M1/M6 cache 연결 | `m1_m6_geometry_link.csv` 확인 |
| 속성 결합 입력 연결 | `fusion_inputs_verified.csv` 확인 |

최종 판단에는 각 작업의 `queue_status.json`, `queue.log`, 컨테이너 exit code, `history.csv`, checkpoint, `metrics_test_*.json`을 함께 읽어야 한다.

## 11. Gap-SupCon 후속 설계

- 목적: Unseen 클래스의 큰 내부 산포와 작은 중심 간 거리를 줄이는 것.
- M7/M8 geometry 분석과 `gap.py`, `evaluation.py`, `driver.py`, launcher가 준비된 기록이 있다.
- margin·loss weight 후보와 smoke 실행 준비는 확인됐지만, 최종 `selection.csv`, `comparison.csv`, 학습 로그, 저장 metric은 확인되지 않았다.
- 따라서 Gap-SupCon은 현재 제안·준비 단계이며 성능 개선 실험으로 기록하지 않는다.

## 12. 비식별 결과 반출 준비

- 반출 준비 폴더: `/home/ubuntu/workspace/nasw/download/`
- 확인 수량: CSV 3,876개 + checkpoint `.pt` 109개 = 3,985개
- 파일명과 경로 식별자는 임의 ID로 치환했다. 내부 매핑은 서버에만 보관했다.
- CSV·checkpoint 검사와 결과 열 복구 작업이 진행됐다.
- 공식 반출 신청·승인, 최종 배치 완료는 확인되지 않았다.
- 이 작업은 실험 성능 결과가 아니라 결과물 정리·반출 준비다.

## 13. 최종 판단

1. 현재까지 가장 높은 전체 정확도는 M6 `0.909`지만, Unseen train gallery를 공분산에 사용한 조건이다.
2. Unseen 정확도와 전체 Macro-F1 기준 최고는 M7 `0.821 / 0.772`지만, Seen 정확도는 `0.857`로 낮다.
3. 속성 점수 결합과 예측 속성 토큰은 Seen 성능을 높일 수 있지만, 현재 기록에서는 Unseen 개선이 확인되지 않았다.
4. 41클래스 실험은 42클래스 결과와 분리해 보고해야 한다. `it`·`internal` 통합의 라벨 근거와 매핑을 먼저 고정해야 한다.
5. 후속 큐·속성 토큰·Gap-SupCon은 원격 세션 재연결 후 로그, exit code, checkpoint, 평가 CSV를 읽은 뒤 완료 여부를 확정한다.

## 14. 원격 재접속 후 확인 순서

1. Windows App에서 원격 터미널을 다시 열고 `pwd`, 컨테이너 목록, GPU 프로세스를 화면으로 확인한다.
2. 각 실행의 effective config와 manifest/spec-table class order를 확인한다.
3. `train.log`, `history.csv`, `best.pt`, `metrics_test_*.json`, `queue_status.json`, `queue.log`, container exit code를 읽는다.
4. 41클래스와 42클래스, closed-only와 galcov, 이미지 전용과 속성 입력을 표에서 분리한다.
5. 원본 파일과 결과 파일의 checksum·행 수·클래스 수를 다시 맞춘 뒤 `RESULTS.md`를 갱신한다.

## 15. 근거 파일

- `RESULTS.md`
- `docs/handoff_20260924/metric_galcov_results_20260925.md`
- `docs/handoff_20260924/metric_results_summary_20260924.md`
- `docs/attribute_tokens_41class_20260927.md`
- `docs/predicted_attribute_tokens_41class_20260928.md`
- `docs/reports/followup_20260928_status.md`
- `docs/reports/professor_report_20260928/report.md`
- `docs/reports/closed_network_experiment_results_2026-09-12.md`

범위에서 제외한 항목: 로컬 few-shot, 로컬 PCA384, 로컬 임베딩만 사용한 분석. 이 문서는 원격 수행·원격 산출물 기준으로만 정리했다.

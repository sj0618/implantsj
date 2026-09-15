# 논문 명세 — 13개 물리 특징으로 구조화한 임베딩으로 처음 보는 임플란트 시스템을 식별하고 특징표를 채운다

- 작성일: 2026-09-15
- 상태: v1 (폐쇄망 본실험용 명세 + 로컬 파일럿 매핑)
- 코드 위치: `implantsj/implantsj/` (이 문서의 상대경로 기준)

## 0. 주장과 기여

**주장.** 13개 물리 특징(class_features.csv)으로 구조화된 임베딩을 학습하면, 학습에 없던
임플란트 시스템을 1~5장의 서포트(K-shot)만으로, 또는 특징표 한 행만으로(0-shot) 식별하고,
그 시스템의 특징표를 이미지에서 채울 수 있다.

**기여 후보.**
1. 치과 임플란트 시스템 식별에 attribute 기반 zero/few-shot 프로토콜을 처음 도입한다
   (label-unseen ∧ patient-unseen, 5회 이상의 unseen 분할, 신뢰구간 포함).
2. ArcFace 임베딩에 13-feature 보조 헤드를 붙인 방법과 일반 FSL 기준선을 같은 분할에서 비교한다.
3. 진단 지표로 "cosine 대 Mahalanobis(seen 클래스 내 공분산 화이트닝) 정확도 간격 Δ_W"를 정의하고,
   이를 "학습이 치우지 못한 클래스 내 잡음의 양"으로 해석해 보고한다.

**1차 지표(프로젝트 목적과 일치).** 화이트닝 없는 cosine 최근접 프로토타입의 unseen K-shot(K=1,3,5,10)
정확도와 macro-F1, 그리고 unseen 이미지의 13 feature별 특징표 채우기 정확도.
Mahalanobis 계열은 기준선(Simple CNAPS 헤드)과 진단(Δ_W)으로만 쓴다.

---

## 1. 데이터와 분할

### 1.1 폐쇄망 본실험 데이터

| 항목 | 내용 | 상태 |
|---|---|---|
| 라벨 | 42 시스템 = seen 계약 27 (`configs/closed_labels.txt`) + open 15 | 확정 |
| 분할 | 폐쇄망 반입본은 train/val/test 로 이미 나뉨. 논문 프로토콜은 이 split 을 쓰지 않고 §2 규칙으로 다시 나눈다 | 확정 |
| 출처 | AI-Hub 공개 데이터라면 데이터셋 이름·버전·다운로드 일자를 재현성 절에 적는다. 기관 데이터가 섞였다면 IRB 승인번호 문구가 필요하다 | **미확인(사람 확인 필요)** |
| 환자 ID | `group` 열의 1순위. AI-Hub 메타데이터에 환자 ID가 있으면 그것을 쓴다 | **미확인** |
| 특징표 | `configs/class_features.csv` 45행 × 13 feature. 학습 대상은 `Name` 제외 12개 (`src/class_features.py: DEFAULT_TRAIN_ATTRIBUTES`) | 확정 |

**group 열 우선순위.** ① 환자 ID → ② 원본 방사선 사진 ID(한 장에서 나온 크롭은 같은 그룹) →
③ 근접 중복 클러스터. 상위 정보가 없으면 하위로 내려가며, 어느 수준을 썼는지 보고서에 적는다.
②·③은 환자 단위보다 약한 보장이다(같은 환자의 다른 촬영이 다른 그룹이 될 수 있다).

### 1.2 근접 중복 제거

- 해시: 256-bit dHash(회색조 17×16), **dihedral 8변환(회전 0/90/180/270 × 좌우반전) 중 최소 해밍거리**를 거리로 쓴다.
  Roboflow 증강본의 반전·90도 회전을 잡기 위해서다. v1 초안의 64-bit(9×8, ≤5)는 파노라마끼리 저해상도에서 닮아
  무관한 사진을 과병합해 버렸다(2026-09-15 리뷰 재계산: 교차 group 원본 쌍 154,226개가 합쳐진다).
- 원본 이미지 수준: 거리 ≤ 10 (256 중)이면 같은 group 으로 **합친다**(union-find). 분할 경계를 넘지 못하게 하는 규칙이다.
  임계는 16/12/10/8/6/4 sweep(`dedup_report.json`)과 육안 점검으로 골랐다(12 에서는 무관한 사진이 사슬로 이어진 큰 component 가 생겼다).
- 크롭 수준: 같은 라벨 안에서 거리 ≤ 8 (256 중)이면 **하나만 남긴다**. 증강이 없는 split(valid/test) 사본을 우선 보존한다.
- 환자키: 파일명에 환자 정보가 있는 출처군은 stem 보다 먼저 쓴다(§1.1 ①). `implant_yolo26_12label` 은 8자리 이상 숫자
  (환자/기록 번호, `-2`·`-pa` 꼬리 무시), `dddd` 는 환자 이름(꼬리 촬영번호 `-N`·`N` 제거, 남는 글자가 3자 미만이면 중단).
  출처군·수준별 건수를 `dedup_report.json` 의 `initial_group_level` 에 남긴다.
- 파일명 stem 규칙: Roboflow 파일명 `<stem>_<ext>.rf.<hash>.jpg` 의 `<stem>` 을 정규화(소문자, `-copy-N`·`_bmp`·`_png` 접미 제거)해
  같은 출처군(dataset family) 안에서 같은 stem 은 같은 group 으로 둔다. 증강본은 해시로 안 잡히는 회전·줌을 포함하기 때문이다.
- 제거·병합 건수를 `dedup_report.json` 으로 남긴다.

### 1.3 클래스별 장수 표 (필수 산출물)

`class_counts.csv`: `label, n_items, n_groups, n_items_by_domain, datasets`. 논문 표 1의 원천이다.
unseen 후보 자격(§2.1)과 K별 평가 가능 클래스 수가 이 표에서 결정된다.

### 1.4 로컬 파일럿 데이터 (2026-09-15 관측)

폐쇄망 42라벨은 로컬에 없다. 사용자 지시("임시로 로컬 정보로 실험")에 따라 두 트랙으로 대체한다.

| 트랙 | 원천 | 라벨 단위 | 특징표 | 용도 |
|---|---|---|---|---|
| **B** (brand) | `data/dddd`(7 brands, panoramic), `data/FULL_Bego_Bicon_ITI_3label` + `data/Dental Implants 2.0.v3i.yolov8`(3 brands, panoramic, 같은 출처군), `data/implant_yolo26_12label`(12 brands, periapical), `data/realyolo26`(36 brands, 카탈로그 이미지) — YOLO bbox 로 크롭 | 제조사(brand) | 없음(브랜드 ≠ 시스템) | 식별 프로토콜, 기준선, margin·백본·crop ablation, Δ_W 진단 |
| **S** (system) | `implantys/data/AI Hub Data` 27 시스템 241장(팬텀 촬영, 시스템당 3~30장, specimen 1~6개) | 시스템 | 있음(27개 모두 표에 존재) | 특징표 채우기, 0-shot, feature 헤드 유무 ablation |

트랙 B 라벨은 제조사이므로 같은 제조사의 다른 이름은 하나로 합친다(`alias_map.csv`): Hiossen → osstem(Osstem 미국 브랜드,
파일명에 Osstem 제품코드), ITI·Straumann ITI → straumann, IMPLANCE → ags_medikal(AGS Medikal 파일명에 `implance`).
`swiss`(y12 `SWISS`)가 AGS Medikal 의 ImplantSwiss 인지는 확인하지 못해 따로 둔다(미해결).

관측된 누수: Roboflow 내보내기에서 같은 stem 이 여러 split 에 걸쳐 있다.
dddd 는 stem 727개 중 435개, realyolo26 은 747개 중 370개, FULL_Bego_Bicon_ITI_3label 은 518개 중 236개가
2개 이상 split 에 존재한다. 기존 폴더 split 을 쓴 과거 실험 수치는 낙관적으로 편향되었을 수 있다.
`data/7label_new` 는 `data/dddd` 와 같은 내보내기(개수 동일)이므로 쓰지 않는다.

트랙 S 의 group 은 specimen(파일명에서 `(n)` 제거, 공백·크기 표기 정규화)이다. 근접 중복 제거 뒤 121장(241장 중 120장 제거,
같은 specimen 을 반복 촬영한 사진)·specimen 51개가 남는다. 16/27 클래스가 specimen 1개뿐이라
group-disjoint K-shot 은 K=1(11개 클래스), K=2(8개 클래스)까지만 가능하다. 0-shot 과 이미지 단위 특징표 채우기는 서포트가 필요 없어 27개 모두 쓸 수 있다.

---

## 2. 프로토콜

### 2.1 분할 (label-unseen ∧ patient-unseen)

- R ≥ 5 개의 무작위 unseen 분할(seed 0..R-1). 폐쇄망에서는 추가로 계약 분할(seen 27 / open 15)을 split `fixed` 로 따로 보고한다.
- `n_unseen = round(N_classes × 15/42)` (폐쇄망 비율 유지). unseen 은 `n_groups ≥ min_groups_unseen` 인 클래스 중에서만 뽑는다.
  - 폐쇄망·트랙 B: `min_groups_unseen = K_max + 2 = 12` (K=10 서포트 + query group 2개 이상).
  - 트랙 S: `min_groups_unseen = 1` (0-shot 이 주 용도). K-shot 은 K+1 개 이상 group 을 가진 unseen 클래스만 평가하고 그 수를 기록한다.
- 항목 역할(role) 규칙 — 모든 항목은 정확히 하나의 역할을 갖는다.
  1. `label ∈ unseen` → `unseen`
  2. 그 외, 같은 group 안에 unseen 항목이 하나라도 있으면 → `excluded` (unseen 환자의 사진은 학습에 들어가지 않는다)
  3. 그 외 seen 항목은 **group 단위로** `train / val / test = 80 / 10 / 10` (seed 고정)
- 불변식(assert): 어떤 group 도 두 역할에 걸치지 않는다(`excluded` 와 `unseen` 의 공존만 허용). unseen 클래스 항목은 학습·모델선택에 한 번도 쓰이지 않는다.
- 모델 선택(early stopping, 하이퍼파라미터)은 seen `val` 만으로 한다. unseen 정보는 쓰지 않는다.

### 2.2 에피소드 (K-shot, unseen all-way)

- K ∈ {1, 3, 5, 10} (트랙 S: {1, 2}), 분할당 에피소드 E = 200.
- 각 unseen 클래스에서 서로 다른 group K개를 고르고 group 마다 항목 1개를 서포트로 쓴다.
  query 는 그 클래스의 항목 중 **서포트 group 이 아닌 group** 의 모든 항목이다.
- N-way = 그 분할의 unseen 클래스 전부(all-way). K 에서 자격이 없는 클래스는 그 K 에서 빠지고 `n_classes` 에 반영된다.
- 보조(GFSL): seen 프로토타입(seen `train` 평균) ∪ unseen K-shot 프로토타입, query = seen `test` ∪ unseen query.
  acc_seen, acc_unseen, 조화평균 H 를 cosine 분류기로만 보고한다.

### 2.3 0-shot (특징표만)

unseen 클래스마다 특징표 한 행만 쓴다. query 이미지의 attribute 확률 p_a 로
`score_c(q) = mean_{a: row_c[a] 알려짐} log p_a(q)[row_c[a]]` 를 계산하고 unseen 클래스 중 argmax 를 고른다.
특징표가 같은 unseen 클래스들(예: SSII 와 SSIII 는 Name 을 빼면 행이 같다)은 원리상 구분이 불가하므로
**구분 가능 클래스 비율(ceiling)** 을 같이 보고한다.

### 2.4 지표

| 지표 | 정의 | 등급 |
|---|---|---|
| unseen acc | query micro 정확도 | 1차 |
| unseen macro-F1 | unseen 클래스 macro-F1 (sklearn `average='macro'`) | 1차 |
| 특징표 채우기(이미지) | unseen 이미지마다 feature 별 예측 vs 클래스 정답행, feature 별 정확도 + 상수 아닌 feature 평균(모든 클래스에서 값이 같은 feature 는 빼고 보고서에 적는다). `unknown` 칸 제외 | 1차 |
| 특징표 채우기(클래스, K) | K개 서포트의 확률 평균 → 클래스 행 예측, feature 별 정확도 | 2차 |
| 0-shot acc / macro-F1 | §2.3 | 1차 |
| GFSL H | §2.2 보조 | 2차 |
| Δ_W, trace ratio | §5 | 진단 |

**통계.** 분할 r 의 값은 E개 에피소드 평균이다. 보고값은 R개 분할 평균 ± 95% CI (t 분포, df = R−1)이다.
분할 변동(어떤 클래스가 unseen 이 되는가)이 에피소드 변동보다 크므로 분할이 1차 단위다. 에피소드 수준 CI 는 부록에 둔다.
방법 비교는 같은 분할 위에서 짝지은 차이(paired difference)의 평균과 95% CI, 분할별 승수로 보고한다.

---

## 3. 기준선 (표현 × 분류기 행렬)

표현(representation)과 분류기(classifier)를 분리해, 모든 표현에 모든 분류기를 적용한다.

| 표현 id | 내용 | 학습 | 레포 현황 |
|---|---|---|---|
| `in_vitb` | ViT-B/16 `vit_base_patch16_224.augreg2_in21k_ft_in1k`, 동결 | 없음 | 가중치 있음 |
| `dino_vitb` | DINOv2 ViT-B/14 `vit_base_patch14_dinov2.lvd142m`(img 224), 동결 | 없음 | 폐쇄망 FT 이력 있음, 동결 평가 코드 없음 |
| `in_vits` | ViT-S/16 동결 (학습된 ViT-S 의 기준점) | 없음 | — |
| `ce_*` | seen CE 분류 학습 | seen | 분류 학습은 있음, SimpleShot 평가 없음 |
| `proto_*` | ProtoNet 에피소드 학습 | seen | **없음** |
| `arc_*` | ArcFace (s=30, margin m) | seen | `src/arcface.py` 있음 |
| `arcattr_*` | ArcFace + 13-feature 보조 헤드 (제안) | seen | `src/arcface_attributes.py` 있음 |

| 분류기 id | 정의 (서포트 s_i, query q, L2(x)=x/‖x‖) | 기준선 이름 |
|---|---|---|
| `cos` | μ_c = L2(mean L2(s_i)); argmax L2(q)·μ_c | cosine 프로토타입 / ArcFace 갤러리 |
| `euc` | μ_c = mean s_i; argmin ‖q−μ_c‖² | ProtoNet 추론 |
| `ss` | x' = L2(x − m_base), m_base = seen train 평균; argmin ‖q'−mean s'_i‖² | SimpleShot (CL2N) |
| `mt` | L2 공간. Q_c = λ_c Σ_c + (1−λ_c) Σ_task + βI, λ_c = K/(K+1), β = 1; argmin (q−μ_c)ᵀQ_c⁻¹(q−μ_c) | 퇴화 변형(아래 주) |
| `mtr` | `mt` 와 같은 식을 **정규화 전** 특징 공간에서 | Mahalanobis (Simple CNAPS 헤드) |
| `mw` | L2 공간. Σ_W = seen train 의 클래스 내 공분산 풀링 + Ledoit-Wolf 축소; argmin (q−μ_c)ᵀΣ_W⁻¹(q−μ_c) | 진단 전용 |

논문 기준선 표의 칸은 다음 조합이다: ImageNet ViT-B + `cos`, DINOv2 + `cos`, ProtoNet(`proto`) + `euc`,
SimpleShot(`ce`) + `ss`, Simple CNAPS 헤드 = 각 표현 + `mt`, ArcFace 갤러리(`arc`) + `cos`, 제안(`arcattr`) + `cos` 및 0-shot.
`mt` 주(2026-09-15 관측): L2 공간에서는 공분산 최대 고유값이 0.07(동결 ViT-B)~0.14(ArcFace)로 β=1 보다 한 자릿수 작아
βI 가 지배하고, `mt` 는 사실상 euc 프로토타입이 된다. 그래서 기준선 칸은 원 논문처럼 정규화 전 특징에 쓰는 `mtr` 로 한다.
정직성 주: Simple CNAPS 원 논문의 FiLM 적응은 구현하지 않는다. 쓰는 것은 계층 정규화 Mahalanobis 분류 헤드뿐이며 표에 그렇게 적는다.

**특징 확률의 출처.** `head` = 제안 모델 자신의 attribute 헤드. `probe` = 모든 표현에 대해 seen train 의 L2 임베딩으로
feature 별 다항 로지스틱 회귀를 학습(학습에 없던 값은 도달 불가로 오답 처리, 개수 보고).
특징표 기준선: `majority`(seen 클래스 단위 최빈값), `nearest_seen`(가장 가까운 seen 클래스 centroid 의 행 복사).

---

## 4. 제안 방법과 ablation

**제안.** 백본 → projector(128d) → L2 → ArcFaceHead(s=30, m=0.5) + feature 별 선형 헤드(12개, 입력은 L2 임베딩).
`loss = CE_arc + w · mean_a CE_a(ignore unknown)`, w = 1.0. 추론: K-shot 은 `cos` 프로토타입, 0-shot 은 §2.3, 특징표는 헤드.
(구현 근거: `src/arcface_attributes.py` 의 `head_source="embedding"` 설계와 같다.)

| ablation | 수준 | 폐쇄망 | 로컬 트랙 |
|---|---|---|---|
| 특징 헤드 유무 | w ∈ {0, 1} (w=0 은 `arc` 와 같다) | ✓ | S |
| margin | m ∈ {0.0, 0.2, 0.5} (0.0 = normalized softmax) | ✓ | B |
| 백본 크기 | ViT-S/16 vs ViT-B/16 | ✓ | B |
| crop 유무 | 임플란트 크롭 vs 원본 사진 전체 | ✓ | B |
| 0-shot 대 K-shot | 특징표만 vs K ∈ {1,3,5,10} | ✓ | S (K ≤ 2) |

---

## 5. 진단 지표

- **Δ_W(K) = acc_mw(K) − acc_cos(K)** (percentage point). seen 클래스 내 공분산으로 화이트닝했을 때 얻는 이득이다.
  학습이 클래스 내 잡음 방향을 억제했다면 Δ_W → 0 이다. Δ_W 가 크면 임베딩이 클래스와 무관한 변동(촬영 조건·각도 등)을
  여전히 큰 분산 방향에 싣고 있다는 뜻이다. 표현별·K별로 보고하고, 학습 전(동결)과 학습 후를 비교한다.
- **trace ratio** = tr(S_W)/tr(S_B), unseen query 의 L2 임베딩, 클래스 균형 가중. 작을수록 unseen 클래스가 잘 뭉친다.
- **Σ_W 유효 순위** = exp(정규화 고유값 엔트로피), seen train 의 **축소 없는** pooled Σ_W 기준(LW 의 identity 성분이
  순위를 부풀리므로 `mw` 분류기만 LW 를 쓴다. 축소 강도는 `lw_shrinkage` 로 따로 기록). 잡음이 몇 차원에 퍼져 있는지 본다.
- GFSL H 의 분할 값 = 그 분할 에피소드 평균 acc_seen·acc_unseen 의 조화평균(에피소드별 H 의 평균이 아니다).

---

## 6. 구현 계약

모든 경로는 `implantsj/implantsj/` 기준. 새 의존성은 추가하지 않는다(numpy, pandas, scikit-learn, scipy, torch, timm, PIL, matplotlib 은 이미 설치됨).

### 6.1 manifest CSV (트랙·폐쇄망 공통)

필수 열 `path`(절대경로), `label`, `group`. 선택 열 `class_key`(특징표 `class_key`; 있으면 특징 평가 활성화), `domain`, `dataset`, `source_image`.

### 6.2 `tools/fewshot_build_local.py` — 로컬 manifest 생성

```
python tools/fewshot_build_local.py --data-root <.../implant_project/data> \
    --aihub-root "<.../implantys/data/AI Hub Data>" --out <.../implant_project/data/fewshot_local>
```
산출물:
- `trackB/crops/<label>/<id>.jpg` — 정사각 크롭(pad 0.1, `pipelines/yolo/crop_yolo.py: expand_square_crop` 재사용), 256×256 JPEG
- `trackB/full/<label>/<id>.jpg` — 원본 전체를 256×256 으로 리사이즈. 라벨이 하나인 원본만
- `trackB/manifest_crop.csv`, `trackB/manifest_full.csv` — 같은 `group` 체계를 공유
- `trackB/class_counts.csv`, `trackB/dedup_report.json`, `trackB/alias_map.csv`(원 라벨 → 정규 브랜드)
- `trackS/images/<class_key>/<id>.jpg`(자동 크롭 후 256×256), `trackS/manifest.csv`(`class_key` 포함), `trackS/class_counts.csv`, `trackS/dedup_report.json`, `trackS/autocrop_montage.jpg`(육안 점검용)
- 16px 미만 박스는 버리고 건수를 기록한다. polygon 라벨은 min/max 로 bbox 를 만든다.

### 6.3 `src/fewshot.py` — 프로토콜 코어 (numpy/sklearn, torch 불필요)

```
make_splits(manifest_df, n_splits=5, n_unseen=None, min_groups_unseen=12, seed=0,
            val_frac=0.1, test_frac=0.1) -> list[dict]
    # dict: {split_id, seed, unseen_classes, seen_classes, unseen_groups: [...],
    #        group_roles: {group: 'train'|'val'|'test'}}
    # 역할을 path 가 아니라 (label, group) 으로 정의해 crop/full manifest 가 같은 분할을 공유한다.
item_roles(manifest_df, split) -> pd.Series      # label∈unseen→unseen; group∈unseen_groups→excluded; 그 외 group_roles[group]
check_split(manifest_df, split) -> None          # §2.1 불변식 assert
sample_episode(unseen_df, K, rng) -> (support_idx, query_idx, classes)
classify(name, S, yS, Q, ctx) -> yhat            # name ∈ {cos, euc, ss, mt, mw}; ctx: base_mean, seen_cov
evaluate(npz_path, manifest_csv, split, Ks, episodes=200, feature_table=None, seed=0) -> dict
aggregate(values: list[float]) -> {mean, std, ci95_lo, ci95_hi, n}
selftest()                                       # 합성 데이터로 누수 불변식·Δ_W 부호·0-shot > 우연·CI 검사
```
CLI: `python -m src.fewshot split|evaluate|selftest ...`.

### 6.4 `src/fewshot_train.py` — seen 학습과 임베딩 추출 (torch)

```
python -m src.fewshot_train train  --manifest m.csv --splits s.json --split-id r --loss {ce,protonet,arcface,arcface_attr}
       --model-name <timm> --margin 0.5 --scale 30 --attr-weight 1.0 --epochs 10 --epoch-size 3000
       --batch-size 32 --lr 5e-5 --head-lr 1e-3 --seed 0 --device auto --out <run_dir>
python -m src.fewshot_train export --manifest m.csv (--ckpt <run_dir>/model.pt | --pretrained <timm>) --out emb.npz
```
- 인코더: `src.models.MetricModel` 재사용. export 하는 `emb` 는 projector 출력(정규화 전, 128d). 동결 표현은 백본 pooled feature.
- ArcFace 는 `src.arcface.ArcFaceHead` 재사용. 특징 타깃은 `src.class_features.load_table/match_labels/build_targets` 를 manifest 전체 `class_key` 로 만든다.
- 샘플러: 클래스 균형 `WeightedRandomSampler(1/n_c)`, epoch 당 `epoch_size` 개. ProtoNet 은 N=min(10, seen), K=5, Q=5 에피소드.
- 모델 선택: 매 epoch seen `val` 의 cosine 프로토타입 정확도(centroid = 클래스당 train 최대 30장). 최고 state 저장,
  동률이면 뒤 epoch. val 이 20장 미만이면 선택이 잡음이므로 마지막 epoch 를 쓴다(트랙 S 는 val 5~9장이다. v1 은 동률에서
  앞 epoch 를 골라 트랙 S 학습 모델 10개가 epoch 1~4 에서 뽑혔다. 그 결과는 `outputs/fewshot_v1_S_earliest_tie/` 에 보관).
- 증강: `src.data.build_transforms` 재사용(기하·증강은 그대로, Normalize 만 `timm.data.resolve_model_data_config(backbone)`
  의 mean/std 로 바꾼다 — augreg ViT 는 0.5/0.5, DINOv2 는 ImageNet). 쓴 값은 summary.json·model.pt·export 로그에 남긴다.
  역할이 `train` 인 항목만 학습, `val` 만 선택에 쓴다.
- ProtoNet 은 train 항목이 K+Q 장 미만인 seen 클래스를 에피소드에서 뺀다(복원추출하면 서포트·query 가 같은 사진이 된다).
  뺀 클래스를 summary.json `protonet.ineligible` 과 보고서 커버리지에 적는다.
- `model.pt` 는 임시 이름으로 쓴 뒤 옮긴다(suite 가 있는 출력을 완료로 보고 건너뛰기 때문이다).
- emb.npz: `path`(N, str), `emb`(N, D, float32), 선택 `attrprob__<Attribute>`(N, V_a), `attrvocab__<Attribute>`(V_a, str).

### 6.5 `tools/fewshot_suite.py` — 실행·집계

```
python tools/fewshot_suite.py plan   --track {B,S} --root <fewshot_local> --out <results_root>
python tools/fewshot_suite.py run    --track {B,S} --methods <id,...|all> --splits 0-4 ...
python tools/fewshot_suite.py report --out <results_root>  -> REPORT.md, kcurve_<track>.png, tables/*.csv
```
- 결과 JSON: `<results_root>/<track>/<method>/split<r>.json`. 이미 있으면 건너뛴다(재시작 가능).
- 방법 레지스트리(§7 표)는 이 파일 한 곳에만 둔다.

---

## 7. 로컬 파일럿 실행 계획

측정 처리량(M4 Pro MPS, batch 32, 224px, 2026-09-15): 학습 ViT-S 105 img/s, ViT-B 36 img/s, DINOv2-B 24 img/s.
추론 ViT-S 343, ViT-B 109, DINOv2-B 71 img/s. bf16/fp16 autocast 는 속도 차이가 없었다.

| 트랙 | 방법 id | 표현 | 목적 |
|---|---|---|---|
| B | `in_vitb`, `dino_vitb`, `in_vits` | 동결 | 일반 pretrained 기준선 |
| B | `ce_vits` | CE | SimpleShot |
| B | `proto_vits` | ProtoNet | ProtoNet |
| B | `arc_vits_m05` | ArcFace m=0.5 | ArcFace 갤러리(주 방법; 트랙 B 는 특징표가 없어 제안=ArcFace) |
| B | `arc_vits_m02`, `arc_vits_m00` | ArcFace | margin ablation |
| B | `arc_vitb_m05` | ArcFace ViT-B | 백본 크기 ablation |
| B | `arc_vits_m05_full` | ArcFace, 원본 전체 입력 | crop ablation |
| S | `in_vitb`, `dino_vitb`, `in_vits` | 동결 + probe | 특징표·0-shot 기준선 |
| S | `arc_vits_m05` | ArcFace (w=0) + probe | 특징 헤드 없음 |
| S | `arcattr_vits_m05` | ArcFace + 특징 헤드 (w=1) | 제안 |

공통: 분할 R=5 (seed 0..4), 학습 seed 0, epochs 10, epoch_size 3000 (트랙 S 는 600), lr 5e-5 / head 1e-3,
AdamW wd 1e-4, cosine schedule, grad clip 5. 예상 시간: 트랙 B ViT-S 1회 ≈ 5분, ViT-B 1회 ≈ 14분 → 전체 ≈ 4시간.

---

## 8. 수용 기준 (파일럿)

| id | 기준 |
|---|---|
| AC1 | 두 트랙의 manifest 에 `group` 이 있고, 모든 분할에서 §2.1 불변식 assert 가 통과한다 |
| AC2 | §7 의 모든 (트랙, 방법, 분할 r∈0..4) 결과 JSON 이 존재하고, K 별 acc·macro-F1 이 분할 평균 ± 95% CI 로 표에 나온다 |
| AC3 | 트랙 S 에서 상수 아닌 feature 별 특징표 채우기 정확도가 `head`·`probe`·`majority`·`nearest_seen` 과 함께 나오고, 뺀 상수 feature 가 보고서에 적힌다 |
| AC4 | 표현별·K별 Δ_W 와 trace ratio 가 나온다 |
| AC5 | `python -m src.fewshot selftest` 와 학습 코드 smoke test 가 통과한다 |
| AC6 | 보고서에 못 한 것(커버리지 공백)이 명시된다 |

## 9. 파일럿 결정 기록

| 결정 | 선택지 | 권고(채택) | 근거 | 확신 | 확신을 낮추는 것 |
|---|---|---|---|---|---|
| 트랙 B 데이터 범위 | (a) 5개 데이터셋 합집합 (b) 임상 사진만 | (a), domain 별 층화 보고 | 클래스 수가 42에 가장 가깝다 | 0.60 | 카탈로그(realyolo26) 클래스는 도메인이 달라 unseen 난이도가 섞인다 |
| 로컬 주 백본 | (a) ViT-S (b) ViT-B | (a), ViT-B 는 ablation | ViT-B 학습이 3배 느리다(36 vs 105 img/s) | 0.75 | 폐쇄망 본실험은 ViT-B 가 주력이라 순위가 달라질 수 있다 |
| 환자 대용 group | (a) 환자키(y12 기록번호·dddd 이름)+stem+해시 union (b) 해시만 | (a) | 증강 회전·줌은 해시로 안 잡히고, 같은 환자의 다른 촬영은 파일명으로만 이어진다 | 0.70 | 짧은 stem(`1`, `R10`)과 dddd 제품명+번호(`dentium-superline-12`)는 무관한 사진을 과병합한다(보수적 방향). dddd 의 이름 변형(`sheela`/`sheela-badam`)은 못 잇는다 |
| FSL 기준선 구현 | (a) 레포 안에 최소 구현 (b) 외부 라이브러리 | (a) | 폐쇄망 이미지에 의존성 추가 불필요 | 0.80 | 원 구현과 세부가 다를 수 있다(§3 정직성 주) |

## 10. 범위 밖 / 미해결

- 폐쇄망 42라벨 manifest 의 `group`(환자 ID) 확보 — 데이터 접근이 필요하다.
- 데이터 출처(AI-Hub 여부)와 IRB 문구 — 사람 확인이 필요하다.
- Simple CNAPS 의 FiLM 적응, K-shot 과 특징 점수의 결합(score fusion) — v2 후보.

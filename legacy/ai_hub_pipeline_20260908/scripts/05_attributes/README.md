# Stage 4. Attributes

코드는 `src/attributes/`, 설정은 `configs/05_attributes.yaml`.
노션 구조의 **5. Attributes** 브랜치.

이미지에서 11개 feature를 예측하고, 그 결과를 spec table
(`configs/spec_table.csv` = 클래스 → feature 정답표)과 비교해 최종 클래스를 정한다.

```bash
bash train_attributes.sh                    # closed27_open15
PROTOCOL=full42 bash train_attributes.sh    # 42클래스 전부
```

## 모델 구조

`train.mode` 로 고른다. 둘 다 구현돼 있어 같은 데이터로 비교할 수 있다.

| mode | 구조 | 학습 횟수 | 언제 |
|---|---|---|---|
| `separate` (기본) | feature마다 백본+head 를 통째로 하나씩 | 11 | feature 간 간섭을 배제하고 싶을 때 |
| `shared` | 백본 1개 + feature마다 head (multi-head) | 1 | 비용을 1/11 로 줄이고 표현을 공유하고 싶을 때 |

출력 폴더 이름에 mode 가 붙어(`<protocol>_<backbone>_<mode>`) 서로 덮어쓰지 않는다.
예전 이름(`<protocol>_<backbone>`)으로 남은 폴더는 `_separate` 를 붙여 옮기거나 다시 돌린다 —
`06_fuse.yaml` 의 `branches.attributes.run` 이 새 이름을 가리킨다.

제외한 컬럼: `Name`(데이터셋 클래스 이름 그 자체 = 라벨),
`Body_Type`(표의 모든 클래스가 `Threaded`로 동일해 분류에 기여하지 않음).

## 선택 옵션 5가지

### 1. 실행 단위 — `run`

모델은 항상 feature마다 따로다. 이번 실행에서 **몇 개를 학습할지**만 정한다.

```yaml
run:
  features: all              # train:true인 feature 전부를 한 번에
  # features: [System, Flange]   # 지정한 것만 → 나눠서 여러 번 실행
  skip_existing: true        # 이미 학습된 체크포인트는 재사용(재학습 안 함)
```

`skip_existing`이 켜져 있으면 폐쇄망에서 며칠에 걸쳐 feature를 나눠 돌린 뒤
마지막에 한 번 더 실행해 전부 모아 fuse할 수 있다.

### 2. feature별 사용 여부 — `features.<이름>`

11개 각각에 플래그 2개가 있다.

```yaml
features:
  System:  {train: true,  use_in_fuse: true}    # 학습하고 최종 판정에도 반영
  Company: {train: true,  use_in_fuse: false}   # 학습은 하되 최종 판정에서 제외
  Collar:  {train: false, use_in_fuse: false}   # 아예 사용 안 함
```

표에서 값이 1종뿐인 feature가 생기면 `drop_constant: true`가 자동으로 뺀다.

### 3. 클래스 가중 loss — `train.use_class_weights`

feature마다 **값 빈도의 역수**를 CE 가중치로 준다. 드문 값(예: `확인 불가`가 아닌
소수 값)이 loss에서 밀리는 걸 막는다. 기본은 꺼져 있다.

### 4. 최종 점수 산출 — `fuse.mode`

| 값 | 내용 |
|---|---|
| `learned` | feature별 가중치 `w_f`를 **valid에서 자동 학습**.<br>`score(c) = Σ_f w_f · P_f(spec[c,f] \| x)`. 학습된 가중치는 `feature_weights.json`에 저장 |
| `min_match` | 사용자가 정한 **n개 이상의 feature가 일치**해야 그 라벨로 인정.<br>일치 = `argmax P_f == spec[c,f]`. n 미만이면 `__unknown__` 판정 |

`min_match`도 점수는 계속 나온다 (일치 개수는 `matched_features` 컬럼에 별도 저장).
`learned`의 가중치 학습은 **valid만** 사용하고 test는 건드리지 않는다.

### 5. 백본 — `grid.backbone`

노션의 두 가지: `cnn`(ConvNeXt-Tiny) / `dinov2`. 둘 다 나열하면 비교 실험이 된다.
`dinov2`는 `base.yaml` 레지스트리를 그대로 쓰므로 Stage 1에서 만든 백본을 끌어다 쓸 수 있다.

## spec table

`configs/spec_table.csv` — 원본 표(임플란트 라벨링.xlsx) 형식을 **그대로** 읽는다.

- 컬럼명 자동 정규화: `Connection(1)`→`Connection_1`, `Body Shape`→`Body_Shape`,
  `Conection(2)`(오타)→`Connection_2`
- **`확인 불가`는 unknown으로 처리** → 그 칸은 학습·평가에서 제외한다
  (정답을 모르는 칸을 틀렸다고 셀 수 없다). `spec.unknown_values`로 조정
- **값 수정은 CSV를 직접 고치는 것이 원칙**이다. 어떤 값이 같은 값인지는 담당자만
  판단할 수 있으므로 코드가 임의로 통합하지 않는다. 표에 적힌 문자열이 그대로 값이 된다
  (`spec.value_aliases`는 기본 비어 있음)

현재 표에서 확인이 필요해 보이는 칸 (수정 여부는 담당자 판단):

| 클래스 | feature | 값 | 비고 |
|---|---|---|---|
| Cybermed Core1 | Thread_Shape | `Sharp double / compact / open` | 슬래시로 여러 값 |
| Straumann BoneLevel | Connection_2 | `CrossFit / Conical` | 슬래시로 여러 값 |
| BioHorizons External | Thread_Shape | `Square / V-shaped*` | 슬래시 + `*` |
| Bicon Integra-CP | Connection_2 | `No anti-rotational / Morse taper` | 슬래시로 여러 값 |
| Straumann Standard Plus | Connection_2 | `Hexagon` | `HEX`(30건)와 같은 값인지 |
| Bicon Integra-CP 외 | Connection_2 | `Octagon` | `OCTA`(8건)와 같은 값인지 |

수정 후 값 목록 확인:
```bash
python src/check_spec_table.py --config configs/05_attributes.yaml --labels "..."
```

### 클래스 매칭 — `dataset_label` 열

데이터셋 폴더명과 표의 `Name`이 미묘하게 다를 수 있으므로,
표에 **`dataset_label` 열**을 두어 폴더명을 직접 타이핑해 등록할 수 있다.

| 우선순위 | 방식 |
|---|---|
| 1 | `dataset_label` 열에 적은 이름 (여러 개면 `;`로 구분) |
| 2 | yaml의 `spec.label_aliases` |
| 3 | `Company`+`Name` 자동 매칭 |
| 4 | `Name` 단독 자동 매칭 |

**폐쇄망 들어가기 전에 점검 도구를 돌려 미리 채워두면 된다** (안에서는 AI를 못 쓰므로):

```bash
# 매칭 상태 확인 — 못 붙는 클래스를 찾아준다
python src/check_spec_table.py --config configs/05_attributes.yaml

# 데이터가 아직 없으면 클래스 이름만 넘겨서 점검
python src/check_spec_table.py --config configs/05_attributes.yaml --labels "01_Osstem_GSII,02_..."

# 자동 매칭된 결과를 dataset_label 열에 채워 저장 → 못 붙은 것만 손으로 채우면 됨
python src/check_spec_table.py --config configs/05_attributes.yaml --fill
```

### 표 45행 vs 데이터 42클래스

표에만 있고 데이터에 없는 행은 **그냥 쓰이지 않는다** (문제 없음).
학습·분류는 매칭된 클래스만으로 이뤄지고, 남는 행은 로그와 `spec_table.json`의
`unused_spec_rows`에 따로 보고되므로 "매칭 실수"와 구분된다.
반대로 데이터에 있는데 표에 못 붙는 클래스는 경고로 표시된다.

## 시각화

`visualize.enabled: true`(기본)면 실행할 때 `<run_dir>/visualization/`에 자동 저장된다.

| 파일 | 내용 |
|---|---|
| `feature_performance_<split>.png` | feature별 정확도·macro-F1 막대 — **어떤 feature가 약한지** |
| `feature_weights.png` | learned 모드에서 학습된 가중치 (균등선과 함께) |
| `feature_confusions_<split>.png` | feature별 혼동행렬을 한 장에 격자로 |
| `match_counts_<split>.png` | 일치 feature 수 분포 (정답 vs 오답) — **min_match 기준 근거** |
| `score_distribution_<split>.png` | 최종 클래스 점수 분포 (정답 vs 오답) |
| `final_confusion_<split>.png` | spec table 매칭 후 최종 클래스 혼동행렬 |

```yaml
visualize:
  enabled: true
  which: [performance, weights, confusions, match, score, final]   # 원하는 것만 남겨도 됨
```

그림 안 글자는 전부 영어다 — 폐쇄망 서버에 한글 폰트가 없어도 깨지지 않게.
개별 그림이 실패해도 나머지와 평가는 계속된다.

## 출력

```
outputs/attributes/<protocol>_<backbone>_<mode>/
  checkpoints/<feature>.pt     separate: feature마다 독립 모델
  checkpoints/shared.pt        shared: 백본 1개 + head 전부
  embeddings/                  학습 종료 시 전체 이미지 backbone/head 임베딩 npz (separate 는 <feature>/ 아래)
  checkpoints.json             체크포인트 경로 + valid macro-F1
  history_<feature>.csv        feature별 학습 곡선
  attribute_metrics.json       feature별 정확도/macro-F1
  feature_weights.json         learned 모드의 학습된 가중치
  predictions_{valid,test}.csv 최종 판정 (아래 참조)
  scores_{valid,test}.csv      [N, C] 전체 점수 행렬  ← Stage 5 입력
  spec_table.json              매칭 결과/vocab/unknown 리포트
  visualization/               feature 성능·가중치·혼동행렬·점수 분포
outputs/attributes/summary_<protocol>.csv
```

`predictions_*.csv` 컬럼: `pred_label`, `score`(클수록 좋음), `score_normalized`(합=1),
`matched_features`(일치한 feature 수), `rejected_unknown`, `top2_label`/`top2_score`.

**Stage 5 Fuse용**: `scores_*.csv`가 `[N, C]` 점수 행렬이고 **클수록 좋음** 방향이다.
metric 브랜치의 `score`는 거리라 작을수록 좋으므로, Fuse에서 방향을 맞춰 결합해야 한다.

## 하나만 돌리기

그리드는 의무가 아니다. yaml의 리스트를 한 개로 줄이거나(스칼라도 허용),
yaml을 고치지 않고 `SET=` 으로 이번 실행만 좁힐 수 있다.

```bash
SET="grid.backbone=cnn" bash train_attributes.sh

# feature도 나눠 돌 수 있다 (run.features — 이미 학습된 것은 재사용)
SET="grid.backbone=cnn run.features=[System,Connection_1]" bash train_attributes.sh
```

`SET`은 yaml 항목을 점으로 짚어 덮어쓴다 (`train.epochs=1`,
`eval.visualize.enabled=false` 등). 적용된 값은 실행 로그에
`[config] override ...` 로 남는다.

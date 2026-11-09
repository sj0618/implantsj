# Stage 0-a. 데이터 준비

코드는 `src/data/`, 설정은 `configs/00_data.yaml`.
**파이프라인에서 가장 먼저 실행할 것** — 이후 모든 스테이지가 여기서 만든 manifest를 쓴다.

```bash
bash prepare_data.sh
```

## 만들어지는 것

| 파일 | 내용 | 쓰는 곳 |
|---|---|---|
| `data/manifest_full42.csv` | 42클래스 전체 | full42 프로토콜 |
| `data/manifest_closed27.csv` | open을 제외한 클래스 | closed27_open15의 학습 전부 |
| `data/manifest_open15.csv` | open으로 뺀 클래스 | 갤러리 등록 support + open query |
| `data/manifest_labels.json` | 어떤 클래스를 open으로 뺐는지 기록 | 재현·감사용 |

컬럼은 `path, label, split, mask_path`. **파이프라인 전체가 이 네 개만 쓴다.**

## 설정

```yaml
source:
  root: data/datas            # <root>/<클래스>/*.jpg 또는 <root>/{train,valid,test}/<클래스>/*.jpg
  layout: auto

split:
  keep_existing: true         # 폴더에 이미 split이 있으면 그대로
  train: 8, valid: 1, test: 1 # 없으면 이 비율로 (클래스마다 각 split 최소 1장 보장)

band:
  mask_root: data/masks       # Stage 0 ⑥의 산출물
  suffix: _band.png

openset:
  labels:                     # 여기에 open 클래스를 한 줄에 하나씩
    - 28_Warantec_Explant
    - ...
  count: 15                   # 검산용 — 지정 개수와 다르면 경고
  auto: null                  # 목록이 없을 때 임시 자동 선정 (명시해야 동작)
```

## K-fold 교차검증 (옵션)

seed 3개 반복보다 견고한 검증이 필요할 때. `train+valid`를 k번 다르게 나눈
manifest를 추가로 만든다. **`test`는 fold마다 동일하게 유지**해 최종 평가 기준을 고정한다.

```yaml
kfold:
  enabled: true
  k: 5
  subdir: folds        # data/folds/manifest_closed27_fold01.csv ...
```

각 스테이지 yaml의 manifest 경로를 fold 파일로 바꿔 k번 돌리면 된다.
원래 split은 `source_split` 컬럼에 남고, `fold` 컬럼으로 몇 번째인지 알 수 있다.
클래스마다 최소 k장이 없으면 중단한다.

## band mask는 두 번 돌린다

band mask가 아직 없어도 실행된다 (`mask_path`가 빈 칸으로 남을 뿐).
Stage 0(segmentation)을 돌려 band를 만든 뒤 **이 스크립트를 다시 실행하면** 채워진다.

```
prepare_data.sh          → manifest 3종 (mask_path 비어 있음)
   ↓
Stage 0 (01~06)          → data/masks/<stem>_band.png
   ↓
prepare_data.sh (다시)   → mask_path 채워짐 → band-* 백본과 band-vit 사용 가능
```

`vit` / `dinov2` 백본과 `masked-vit`은 mask_path 없이도 학습된다
(masked-vit은 SAM2로 런타임에 band를 만든다).

## open 클래스 지정

**실제 데이터셋에서 open-set으로 쓸 클래스는 정해져 있으므로 명시 지정이 원칙**이다.
목록 없이 실행하면 중단된다 — 엉뚱한 클래스가 open이 되면 실험 전체가 어긋나기 때문이다.

`configs/00_data.yaml`의 `openset.labels`에 데이터셋 폴더명을 한 줄에 하나씩 적는다.
주석 처리된 자리를 만들어 뒀으니 주석만 풀고 채우면 된다.

```yaml
openset:
  labels:
    - 28_Warantec_Explant
    - 29_Warantec_Internal
    - 30_Cybermed_Core1
    # ... 15개
  count: 15
```

**이름이 조금 달라도 맞춰준다** — 대소문자·공백·구분자·앞 일련번호를 무시하고 한 번 더 찾는다
(`Company0 Model0` → `01_Company0_Model0`). 보정이 일어나면 로그에 남는다.
못 찾는 이름이 있으면 **비슷한 후보를 보여주며 중단**한다.

데이터를 받으면 돌리기 전에 목록부터 확인:

```bash
python src/check_open_labels.py --config configs/00_data.yaml --list   # 전체 라벨 보기
python src/check_open_labels.py --config configs/00_data.yaml          # 목록 대조
```

클래스별 이미지 수와 closed/open 분할 결과를 미리 보여준다.

### 목록을 아직 모를 때

임시로 자동 선정할 수 있다. 다만 **명시해야만** 동작한다.

```yaml
openset:
  auto: rarest      # rarest(이미지 적은 순) | random
  count: 15
```

실행하면 임시 선정이라는 경고가 뜬다. 실제 목록이 정해지면 `labels`에 적을 것.
선정 결과는 `manifest_labels.json`에 남아 다음 실행에서 재현된다.

## 서버 폴더 구조를 모를 때

manifest를 만들기 **전에** 구조부터 확인한다. 아무것도 쓰지 않고 읽기만 한다.

```bash
python src/run_prepare_data.py --config configs/00_data.yaml --inspect
python src/run_prepare_data.py --config configs/00_data.yaml --inspect \
    --set source.root=/mnt/aihub/데이터
```

깊이별 폴더·이미지 분포, 확장자, 두 layout으로 각각 스캔한 결과(클래스 수·장수 분포)를
보여준다. **클래스 수가 예상과 다르면 거기서 멈추고 판단하면 된다.**

### 자동으로 처리되는 것

| 항목 | 지원 |
|---|---|
| 레이아웃 | `<root>/<클래스>/…` · `<root>/<split>/<클래스>/…` (auto 감지) |
| split 폴더 이름 | `train/valid/test`, `Training/Validation`, `val`, `dev`, `testing` … (대소문자 무관) |
| 이미지 확장자 | jpg, jpeg, png, bmp, tif, tiff, webp |
| 클래스 폴더 **아래**의 추가 중첩 | 재귀 탐색하므로 상관없음 |
| 라벨 이름 표기 흔들림 | open 클래스 지정 시 대소문자·공백·구분자·앞 일련번호를 무시하고 매칭 |

### 자동으로 안 되는 것 — 그때는 CSV를 직접 만든다

- 클래스 폴더 **위에** 병원·기관 같은 단계가 하나 더 있는 경우 → 그 단계가 클래스로 잡힌다
- 라벨이 폴더명이 아니라 json/xml/csv 안에 있는 경우
- 라벨이 파일명 안에 인코딩된 경우

**이 파이프라인의 진짜 입력은 폴더가 아니라 manifest CSV다.** 컬럼 네 개면 된다:

```csv
path,label,split
/mnt/aihub/data/xxx/0001.jpg,Osstem_TSIII,train
/mnt/aihub/data/xxx/0002.jpg,Osstem_TSIII,valid
```

`mask_path`는 Stage 0을 돌린 뒤 채워지므로 처음엔 없어도 된다.
구조가 특이하면 그 서버에서 CSV만 만들어 `data/manifest_full42.csv` 등으로 두면
`run_prepare_data.py` 없이도 이후 스테이지가 그대로 돈다
(closed/open 분리만 직접 하거나, `--inspect`로 확인한 뒤 layout을 지정해 재실행).

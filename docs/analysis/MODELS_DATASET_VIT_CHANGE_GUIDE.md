# `/home/ubuntu/workspace/data/models` 구조 대응용 ViT 최소 수정 가이드

> 상태: 코드는 아직 변경하지 않았다.
>
> `implantsj`에서 수정할 파일과 위치, 이유를 정리했다.

## 1. 현재 판단

현재 최종 데이터셋 구조:

```text
/home/ubuntu/workspace/data/models/
├── train/
│   ├── 1.모델명/
│   │   ├── images/*.jpg
│   │   └── labels/*.json
│   └── ...
└── test/
    ├── 1.모델명/
    │   ├── images/*.jpg
    │   └── labels/*.json
    └── ...
```

이 구조에서 해결할 문제는 2개다.

1. `valid/` 폴더가 없음
2. 클래스 폴더명이 `1.모델명` 형식이라 라벨명이 그대로 들어감

다음 구조는 그대로 사용해도 된다.

- `images/` 아래 JPG가 있는 점
- `labels/` 아래 JSON이 따로 있는 점

이유:

- `src/data.py:25-27`, `src/data.py:40-42` 는 이미지 확장자만 manifest에 넣는다.
- `train_vit_3label.py:81-83`, `train_vit_7label.py:45-46`, `train_vit_7label_supcon.py:45-46` 도 마찬가지로 재귀 탐색 중 이미지 파일만 사용한다.

따라서 `labels/` 폴더를 따로 둔 현재 구조는 바꿀 필요가 없다.

---

## 2. 지금 그대로 두면 어디서 깨지는가

### 근거 1: `valid` split을 강하게 가정함

- `src/supervised.py:116-118`
  - 학습은 무조건 `train`
  - 검증은 무조건 `valid`
- `src/supervised.py:214-218`
  - 마지막에 `test` 평가를 따로 시도

즉 manifest에 최소한 `train`, `valid` 가 있어야 한다.

### 근거 2: 3/7-label 전용 runner도 `valid` 폴더를 직접 기대함

- `pipelines/train/train_vit_3label.py:71-74`
- `pipelines/train/train_vit_7label.py:27-30`
- `pipelines/train/train_vit_7label_supcon.py:27-30`

모두 `train`, `valid`, `test` 폴더를 직접 돈다.

### 근거 3: 클래스 폴더명을 그대로 라벨로 씀

- `src/data.py:22`
- `src/data.py:39`
- `train_vit_7label.py:44-50`
- `train_vit_3label.py:77-83`

그래서 `1.모델명` 이면 라벨도 그대로 `1.모델명` 이 된다.

3/7/10 label 전용 runner에서는 이게 특히 문제다.

- `train_vit_3label.py:21-30`
- `train_vit_7label.py:20`
- `train_vit_7label_supcon.py:20`

이 스크립트들은 정해진 라벨명과 정확히 일치해야 한다.

---

## 3. 재현 결과

임시 샘플 구조로 확인했을 때:

- `src.data.build_manifest(...)` 는 `train/test` 는 읽지만 라벨을 `1.ADİN`, `2.MIS` 처럼 그대로 잡음
- `train_vit_7label.build_filtered_split_manifest(...)` 는 폴더명이 `1.ADIN` 식이면 라벨 불일치로 실패
- `train_vit_3label.build_3label_manifest(...)` 는 `valid/` 가 없어서 바로 실패

현재 구조에서는 코어 학습 루프보다 manifest 생성부를 먼저 고쳐야 한다.

---

## 4. 최소 변경 경로

### 추천 경로

`/home/ubuntu/workspace/data/models`처럼 모델 폴더 수가 고정되지 않은 데이터셋에는,
`train_vit_3label.py`, `train_vit_7label.py`, `train_vit_10label.py` 같은 라벨 수 고정 wrapper보다

- manifest 생성부만 고치고
- 기존 `pipelines/train/train_supervised.py`

를 그대로 사용하는 편이 변경 범위가 작다.

우선 수정할 파일은 2개다.

1. `implantsj/src/data.py`
2. `implantsj/pipelines/data/make_manifest.py`

`src/supervised.py` 자체는 바꿀 필요가 없다.

---

## 5. 파일별 변경 지시서

## A. `implantsj/src/data.py`

### A-1. 라벨 폴더명 정규화 helper 추가

대상 위치

- 현재 `src/data.py:19-43` 주변

왜 바꿔야 하나

현재는 `class_dir.name` 을 그대로 라벨로 사용한다.
그래서 `1.모델명` → `1.모델명` 으로 들어간다.

어떻게 바꿔야 하나

`1.`, `2.`, `10.` 같은 숫자 접두어를 제거하는 helper를 추가한 뒤:

- `_iter_images_under_class_folder()`
- `_iter_images_under_split_class_folder()`

둘 다 `class_dir.name` 대신 정규화된 이름을 쓰게 바꾼다.

현재 기준 변경 포인트

- `src/data.py:22`
  - `label = class_dir.name`
- `src/data.py:39`
  - `label = class_dir.name`

이 두 줄이 직접 변경 포인트다.

의도

- `1.모델명` → `모델명`
- `02.모델명` → `모델명`
- 앞뒤 공백 제거

---

### A-2. `train/test`만 있을 때 `train`에서 `valid`를 분리하는 helper 추가

대상 위치

- 현재 `src/data.py:53-76` (`build_manifest`)
- 또는 `src/data.py:79-136` (`make_stratified_splits`) 위/아래에 새 helper 추가

왜 바꿔야 하나

최종 데이터 구조는 `train/`, `test/` 만 있고 `valid/` 가 없다.
하지만 학습 코드는 `valid` split이 있어야 돌아간다.

어떻게 바꿔야 하나

새 helper 하나를 추가해서:

- manifest 안에 split이 `train`/`test` 만 있고 `valid` 가 없으면
- `train` rows만 대상으로 stratified split 수행
- 예: `train`의 10%를 `valid` 로 분리
- 기존 `test` 는 그대로 유지

이 helper는 DataFrame을 받아서 DataFrame을 돌려주는 형태가 최소 변경이다.

권장 동작

- 입력 split: `train`, `test`
- 출력 split: `train`, `valid`, `test`
- seed 사용
- 클래스별 표본이 너무 적으면 기존 `make_stratified_splits()` 처럼 fallback 허용

주의

전체 데이터를 다시 8:1:1로 나누면 안 된다.
이미 `test/` 가 따로 있기 때문에 `train`만 쪼개서 `valid` 를 만드는 것이 맞다.

---

### A-3. `images/` 우선 사용은 선택 사항

대상 위치

- `src/data.py:25-27`
- `src/data.py:40-42`

판단

이 부분은 필수 수정 아님.

현재도 `rglob('*')` + 이미지 확장자 필터 때문에:

- `images/*.jpg` 는 읽고
- `labels/*.json` 은 무시한다.

그래서 당장 막히지는 않는다.

다만 더 안전하게 하려면

클래스 폴더 아래에 `images/` 가 있으면 그 하위만 읽고,
없으면 기존처럼 전체 재귀 탐색하는 방식으로 좁힐 수 있다.

이 변경은 선택 사항이다.

---

## B. `implantsj/pipelines/data/make_manifest.py`

### B-1. `valid-from-train` 옵션 추가

대상 위치

- 현재 `make_manifest.py:15-21`

왜 바꿔야 하나

`src/data.py` 에서 helper를 추가해도,
CLI 쪽에서 그 helper를 호출하지 않으면 `/data/models` 용 manifest를 편하게 만들 수 없다.

어떻게 바꿔야 하나

parser에 아래 성격의 옵션을 추가한다.

- `--valid-from-train` (예: 기본값 `0.1`)
- `--seed` (예: 기본값 `42`)

그리고 `build_manifest(...)` 호출 뒤에:

1. DataFrame을 받고
2. `train/test` only 구조면 `train -> train+valid` 로 변환하고
3. 최종 CSV를 다시 저장

하도록 바꾼다.

현재 기준 변경 포인트

- `make_manifest.py:15-18`  
  parser 인자 추가
- `make_manifest.py:20`  
  `build_manifest(...)` 결과를 변수로 받아 후처리하도록 변경
- `make_manifest.py:21`  
  후처리된 결과 기준으로 summary 출력

---

## C. `implantsj/src/supervised.py`

### C-1. 수정 불필요

근거 라인

- `src/supervised.py:102-118`
- `src/supervised.py:214-218`

이 파일은 manifest만 정상적으로 `train/valid/test` 를 가지면 그대로 동작한다.

이 파일은 수정하지 않고 manifest 생성부만 맞추면 된다.

---

## 6. 3/7/10-label 전용 runner를 계속 쓰고 싶다면 추가로 바꿔야 할 파일

아래는 선택 사항이다.
앞으로도 `train_vit_3label.py`, `train_vit_7label.py`, `train_vit_10label.py`, `train_vit_7label_supcon.py`를 계속 쓸 때만 필요하다.

generic supervised 학습만 사용한다면 이 절은 건너뛰어도 된다.

---

## D. `implantsj/pipelines/train/train_vit_3label.py`

### D-1. 숫자 접두어 제거 후 alias 매핑

대상 위치

- 현재 `train_vit_3label.py:33-34`

왜 바꿔야 하나

지금 `_canonical_label()` 은 문자열 strip만 한다.
그래서 `1.Bego` 가 들어오면 `Bego` 로 바꾸지 못한다.

어떻게 바꿔야 하나

`1.`, `2.` 같은 접두어를 제거한 뒤 `LABEL_ALIASES` 를 조회하도록 바꾼다.

---

### D-2. `valid/` 가 없을 때 `train` 에서 `valid` 분리

대상 위치

- 현재 `train_vit_3label.py:67-110`
- CLI 인자 연결은 `train_vit_3label.py:113-146`

왜 바꿔야 하나

지금은 `for split in ["train", "valid", "test"]` 고정이라 `valid/` 없으면 바로 실패한다.

어떻게 바꿔야 하나

- `valid/` 가 있으면 기존 동작 유지
- `valid/` 가 없고 `train/` 과 `test/` 만 있으면
  - 먼저 `train` 과 `test` 만 읽고
  - 이후 `train` rows 일부를 stratified split 해서 `valid` 생성

추가할 CLI 인자

- `--valid-ratio-from-train`
- `--split-seed`

---

## E. `implantsj/pipelines/train/train_vit_7label.py`

### E-1. 라벨 폴더명 정규화

대상 위치

- 현재 `train_vit_7label.py:23-63`

왜 바꿔야 하나

지금은 `label` 폴더명이 `ADIN` 과 정확히 같아야 한다.
하지만 실제 폴더는 `1.ADIN` 형태일 수 있다.

어떻게 바꿔야 하나

`label` 비교 전에 숫자 접두어를 제거한 이름으로 비교한다.

즉 아래 구간이 수정 포인트다.

- `train_vit_7label.py:32-42`
- `train_vit_7label.py:44-50`

---

### E-2. `valid/` fallback 추가

대상 위치

- 현재 `train_vit_7label.py:27-30`
- 현재 `train_vit_7label.py:66-97`

왜 바꿔야 하나

지금 구조는 `train/test` 만 있는데 이 스크립트는 `valid/` 를 직접 요구한다.

어떻게 바꿔야 하나

3-label 쪽과 동일하게:

- `valid/` 있으면 그대로 사용
- 없으면 `train` 에서 stratified valid 생성

추가할 CLI 인자

- `--valid-ratio-from-train`
- `--split-seed`

---

## F. `implantsj/pipelines/train/train_vit_7label_supcon.py`

### F-1. 7-label supervised와 같은 manifest 로직 재사용

대상 위치

- 현재 `train_vit_7label_supcon.py:23-63`
- 현재 `train_vit_7label_supcon.py:66-95`

왜 바꿔야 하나

이 파일도 7-label supervised와 동일한 문제를 중복으로 가지고 있다.

최소 변경안

여기서 manifest builder를 또 따로 고치기보다:

- `train_vit_7label.py` 의 수정된 `build_filtered_split_manifest()` 를 import 해서 재사용

하는 편이 가장 작다.

변경 범위:

- 중복 manifest builder 함수는 제거 또는 축소
- `main()` 에서 7-label supervised와 같은 인자(`valid-ratio-from-train`, `split-seed`)를 전달

---

## G. `implantsj/pipelines/train/train_vit_10label.py`

### G-1. 직접 수정은 보통 불필요

근거 라인

- `train_vit_10label.py:22-23`
- `train_vit_10label.py:41-42`

이 파일은:

- `build_3label_manifest`
- `build_filtered_split_manifest`

를 재사용한다.

따라서 3label/7label builder만 고치면 10label도 대부분 자동 반영된다.

---

## H. `implantsj/pipelines/train/train_vit_6label.py`

### H-1. 이 runner까지 쓸 거면 추가 수정 필요

대상 위치

- 현재 `train_vit_6label.py:76-100`
- 현재 `train_vit_6label.py:136-149`

왜 바꿔야 하나

이 파일은 generic `build_manifest()` 를 쓰므로
`1.모델명` 접두어 문제는 `src/data.py` 수정으로 해결 가능하다.

하지만 여전히 `_validate_6label_manifest()` 는 `valid` 존재를 요구한다.

어떻게 바꿔야 하나

`layout == "split_class_folder"` 경로에서:

- manifest 생성 후 split이 `train/test` 뿐이면
- `train` 일부를 `valid` 로 분리
- 그 다음 `_validate_6label_manifest()` 호출

---

## 7. 바꾸지 않아도 되는 파일

아래는 이번 구조 대응만 놓고 보면 우선 수정 불필요다.

- `implantsj/src/supervised.py`
- `implantsj/src/metric_learning.py`
- `implantsj/pipelines/train/train_vit_10label.py`
- `implantsj/pipelines/train/train_vit_10label_supcon.py`

단, 이들은 입력 manifest가 이미 정상(`train/valid/test`)이라는 전제에서만 그대로 사용 가능하다.

---

## 8. 적용 순서

변경 순서는 다음과 같다.

1. `src/data.py` 수정
   - 숫자 접두어 제거
   - `train/test` only → `train/valid/test` helper 추가
2. `pipelines/data/make_manifest.py` 수정
   - 새 helper를 CLI에서 호출 가능하게 연결
3. 우선 generic 학습으로 사용
   - `pipelines/train/train_supervised.py` 는 그대로 사용
4. 필요할 때만 3/7/6-label wrapper 수정

최소 변경 범위는 다음과 같다.

- 필수 수정: `src/data.py`, `pipelines/data/make_manifest.py`
- 선택 수정: `train_vit_3label.py`, `train_vit_7label.py`, `train_vit_7label_supcon.py`, `train_vit_6label.py`

---

## 9. 판단

현재 폴더 구조에서 이미지와 라벨을 분리한 방식은 문제가 없고,
실제 blocker는 `valid` split 부재와 `1.모델명` 접두어 라벨명이다.

코어 ViT 학습 코드는 유지하고 manifest 생성부만 수정하면 된다.

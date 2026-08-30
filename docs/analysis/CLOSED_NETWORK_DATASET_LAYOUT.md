# 폐쇄망 원본 데이터셋을 학습용 ImageFolder로 정리하는 방법

원격 폐쇄망에 저장된 원본 임플란트 데이터를 학습용 구조로 바꾸는 방법을 정리했다.

---

## 1. 원본 데이터 구조

원본 데이터는 이미지와 라벨(JSON)이 분리된 폴더 구조로 존재한다.

- `TL1`: Train JSON 라벨 파일
- `TS1`: Train JPG 이미지 파일
- `VL1`: Test JSON 라벨 파일
- `VS1`: Test JPG 이미지 파일

각 폴더 내부 구조는 동일하다.

```text
TS1/
└── h01/
    └── h01_0001/
        └── 5185-...-h01.jpg

TL1/
└── h01/
    └── h01_0001/
        └── 5185-...-h01.json
```

이미지와 JSON은 파일명이 같고 확장자만 다르다. 실제 class label은 폴더명이 아니라 JSON 내부 `"모델명"` 값으로 결정된다.

총 라벨 수는 42개이다.

---

## 2. 원본 구조와 학습용 구조의 차이

원본 구조는 병원별 / 촬영건별 보관 구조이고,  
학습용 구조는 모델이 읽기 쉬운 class 중심 구조이다.

### 원본 구조
- 병원(`h01 ~ h15`) 중심
- 촬영 건(`h01_0001`) 중심
- 이미지와 JSON이 분리되어 있음
- 같은 class가 여러 병원 폴더에 흩어져 있음

### 학습용 구조
- `train / val / test` 중심
- 그 아래에 class folder(라벨 폴더) 배치
- 각 이미지가 자기 class 폴더 안으로 이동/복사됨

예:

```text
data_42label/
├── train/
│   ├── 1.implantium(im)/
│   ├── 2.TS III/
│   └── ...
├── val/
│   ├── 1.implantium(im)/
│   ├── 2.TS III/
│   └── ...
└── test/
    ├── 1.implantium(im)/
    ├── 2.TS III/
    └── ...
```

---

## 3. 왜 train / val / test 구조로 바꾸는가

이유는 모델 학습과 성능 검증을 분리하기 위해서다.

- `train`: 모델이 실제로 학습하는 데이터
- `val`: 학습 중 성능을 확인하고, 과적합 여부를 판단하는 데이터
- `test`: 최종 성능을 독립적으로 평가하는 데이터

즉, 같은 데이터를 학습과 평가에 동시에 쓰지 않기 위해 분리한다.  
이렇게 해야 성능 수치가 더 공정하고 재현 가능하다.

> 참고: 설명에서는 `val`로 표현하지만, 이 레포의 manifest 내부 split 이름은 보통 `valid`로 저장된다.

---

## 4. 왜 라벨별 폴더를 만드는가

이미지 분류 파이프라인은 대개 다음 구조를 사용한다.

```text
split/class_name/image.jpg
```

이미지가 들어 있는 상위 폴더 이름이 정답 라벨이 된다.

라벨별 폴더를 만들면 다음 작업이 쉬워진다.

- 데이터 로딩이 단순해지고
- class index 생성이 쉬워지며
- PyTorch/ImageFolder 계열 구조와 잘 맞고
- 추후 manifest CSV를 만들 때도 일관성이 유지된다.

---

## 5. 왜 JSON의 `"모델명"` 값을 기준으로 class folder를 생성하는가

원본 데이터에서 실제 정답은 병원명도 아니고, 하위 폴더명도 아니다.  
실제 분류 대상은 임플란트 모델명이므로 JSON의 `"모델명"` 값이 가장 직접적인 class label이다.

- `h01`, `h02` 등은 데이터 출처(병원) 정보
- `h01_0001` 같은 값은 촬영 단위 관리 정보
- `"모델명"`은 학습할 정답 class

따라서 학습용 폴더 구조는 병원 기준이 아니라 `"모델명"` 기준으로 재정렬하는 것이 맞다.

---

## 6. 42라벨 전체 학습 구조

42개 라벨 전체를 그대로 학습할 때는 모든 `"모델명"`을 class로 사용한다.

```text
data_42label/
├── train/
│   ├── 1.implantium(im)/
│   ├── 2.TS III/
│   ├── 3.XXX/
│   └── ... (총 42개 class)
├── val/
│   ├── 1.implantium(im)/
│   ├── 2.TS III/
│   ├── 3.XXX/
│   └── ... (총 42개 class)
└── test/
    ├── 1.implantium(im)/
    ├── 2.TS III/
    ├── 3.XXX/
    └── ... (총 42개 class)
```

이 구조는 42-class closed-set classification 실험에 대응한다.

---

## 7. 데이터 수가 적은 15라벨을 제외한 27라벨 학습 구조

42개 전체 중 데이터가 너무 적은 15개 라벨은 학습 안정성이 떨어질 수 있다.  
그래서 표본 수가 상대적으로 충분한 27개 라벨만 seen class로 두고 metric learning을 학습한다.

구조 예시는 다음과 같다.

```text
data_27label_seen/
├── train/
│   ├── seen_label_01/
│   ├── seen_label_02/
│   └── ... (총 27개 class)
├── val/
│   ├── seen_label_01/
│   ├── seen_label_02/
│   └── ... (총 27개 class)
└── test/
    ├── seen_label_01/
    ├── seen_label_02/
    └── ... (총 27개 class)
```

이 구조는
- metric learning 모델 학습
- seen class 내부 prototype 평가
- 일반 supervised/metric baseline 비교

를 위한 기본 구조다.

---

## 8. 제외된 15라벨을 unseen test용으로 따로 분리하는 구조

학습에 사용하지 않은 15개 라벨은 unseen class로 별도 보관한다.

이 15개 라벨은 학습(train)에 넣지 않는다. 학습을 마친 뒤 seen 27개 prototype과의 거리 또는 유사도를 비교할 때만 사용한다.

예시 구조:

```text
data_15label_unseen/
└── test/
    ├── unseen_label_01/
    ├── unseen_label_02/
    └── ... (총 15개 class)
```

분석 편의를 위해 `val`을 둘 수는 있지만, 학습에 쓰지 않은 class를 평가용으로만 유지해야 한다.

---

## 9. 최종적으로 아래 구조가 되는 이유

```text
data_42label/
├── train/
│   ├── 1.implantium(im)/
│   ├── 2.TS III/
│   └── ...
├── val/
│   ├── 1.implantium(im)/
│   ├── 2.TS III/
│   └── ...
└── test/
    ├── 1.implantium(im)/
    ├── 2.TS III/
    └── ...
```

이 구조가 필요한 이유는 다음 네 가지로 요약된다.

1. 모델이 읽기 쉬운 표준 분류 구조이기 때문
2. class label을 폴더명으로 직접 표현할 수 있기 때문
3. train / val / test 분리로 공정한 평가가 가능하기 때문
4. 이후 42-label / 27-label / unseen-15 실험을 일관된 방식으로 관리할 수 있기 때문

원본 데이터는 수집과 보관을 위한 구조이며, 변환된 데이터는 학습과 평가를 위한 구조다.

---

## 10. 27라벨 metric learning 학습용 구조와 unseen 15라벨 테스트용 구조

### 10-1. 27라벨 metric learning 학습용

```text
data_metric_seen27/
├── train/
│   ├── seen_label_01/
│   ├── seen_label_02/
│   └── ... (27개)
├── val/
│   ├── seen_label_01/
│   ├── seen_label_02/
│   └── ... (27개)
└── test/
    ├── seen_label_01/
    ├── seen_label_02/
    └── ... (27개)
```

역할:
- 27개 seen label만 사용하여 metric learning 모델 학습
- train split의 embedding으로 class prototype 생성
- val/test에서 seen-class 성능 확인

### 10-2. unseen 15라벨 테스트용

```text
data_metric_unseen15/
└── test/
    ├── unseen_label_01/
    ├── unseen_label_02/
    └── ... (15개)
```

역할:
- 학습에 쓰지 않은 15개 라벨의 embedding 추출
- seen 27개 prototype과 cosine similarity 계산
- 각 unseen sample을 가장 가까운 seen label에 대응

seen 27은 모델이 학습한 class 집합이고, unseen 15는 학습하지 않은 class 집합이다. 이 구분을 통해 metric learning 표현 공간에서 새로운 class가 어느 seen class 근처에 놓이는지 분석할 수 있다.

---

## 11. 요약

현재 폐쇄망 원본 데이터는 병원/촬영건/JSON 분리 구조이고,  
학습용 데이터는 이를 JSON의 `"모델명"` 기준 class folder 구조로 재배치한 것이다.

그 결과:

- 42라벨 전체 학습 구조
- 27라벨 seen 학습 구조
- 15라벨 unseen 평가 구조

를 각각 명확하게 분리하여 사용할 수 있다.

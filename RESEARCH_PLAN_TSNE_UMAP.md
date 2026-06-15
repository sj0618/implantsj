# t-SNE / UMAP 시각화 연구계획

## 1. 목표

- 분류기 / metric model의 임베딩이 클래스별로 얼마나 잘 분리되는지 시각적으로 확인한다.
- t-SNE와 UMAP을 같은 임베딩 소스에 적용해, 국소 군집과 전역 구조를 함께 비교한다.
- 잘못 분류된 샘플이 어떤 군집 경계에 놓이는지 확인해 다음 실험 방향을 정한다.

## 2. 현재 준비 상태

- `visualize_classifier_tsne.py`
  - classifier checkpoint에서 backbone feature를 추출해 t-SNE 시각화를 만든다.
- `visualize_metric_tsne.py`
  - metric/SupCon checkpoint에서 embedding을 추출해 t-SNE 시각화를 만든다.
- `visualize_umap_from_tsne.py`
  - 이미 저장된 `embeddings/tsne_*.npz` 아티팩트를 재사용해 UMAP 시각화를 생성한다.

즉, 현재 구조는 다음 흐름이다.

1. checkpoint에서 feature / embedding 추출
2. t-SNE 결과와 메타데이터 저장
3. 같은 feature를 다시 읽어 UMAP으로 재투영

## 3. 실험 가설

- t-SNE는 클래스 간 국소 분리와 이상치 확인에 강하다.
- UMAP은 전역 구조와 클래스 간 거리 관계를 더 안정적으로 보여줄 가능성이 있다.
- 두 방법을 함께 보면,
  - 클래스 내부 응집도
  - 클래스 경계의 혼동
  - split 별 분포 차이
  를 더 빠르게 판단할 수 있다.

## 4. 실행 계획

### 4.1 t-SNE 생성

- classifier run과 metric run 각각에 대해 `embeddings/tsne_*.csv`, `embeddings/tsne_*.npz`, `embeddings/tsne_*.png`를 생성한다.
- 주요 확인 항목
  - train / valid / test 분포 차이
  - 클래스별 클러스터 분리
  - 오분류 샘플의 위치

### 4.2 UMAP 생성

- 같은 `tsne_*.npz` 아티팩트를 입력으로 받아 `umap_*.csv`, `umap_*.npz`, `umap_*.png`를 생성한다.
- 주요 확인 항목
  - t-SNE에서 뚜렷했던 군집이 UMAP에서도 유지되는지
  - 클래스 간 상대적 거리 구조가 일관적인지
  - 샘플 수가 적은 클래스에서도 과도한 붕괴가 없는지

### 4.3 비교 기준

- 시각적으로 분리도 높은 클래스
- 경계가 겹치는 클래스 쌍
- 오분류 샘플의 집중 위치
- split 별 분포 차이

## 5. 하이퍼파라미터 후보

### t-SNE

- `perplexity`: 20 / 30 / 50
- `pca_dim`: 30 / 50
- `max_samples`: 클래스 균형을 유지하는 범위에서 조절

### UMAP

- `n_neighbors`: 10 / 15 / 30
- `min_dist`: 0.0 / 0.1 / 0.3
- `pca_dim`: 30 / 50

## 6. 산출물

- `outputs/<run>/embeddings/tsne_*.png`
- `outputs/<run>/embeddings/tsne_*.csv`
- `outputs/<run>/embeddings/tsne_*.npz`
- `outputs/<run>/embeddings/umap_*.png`
- `outputs/<run>/embeddings/umap_*.csv`
- `outputs/<run>/embeddings/umap_*.npz`

## 7. 다음 단계

1. 현재 존재하는 checkpoint run들에 대해 t-SNE 아티팩트를 확인한다.
2. `visualize_umap_from_tsne.py`로 동일 아티팩트의 UMAP을 생성한다.
3. class confusion이 큰 run을 우선 비교한다.
4. 시각적으로 가장 안정적인 파라미터 조합을 기록한다.
5. 최종적으로 research summary 문서에 t-SNE / UMAP 비교 결과를 남긴다.


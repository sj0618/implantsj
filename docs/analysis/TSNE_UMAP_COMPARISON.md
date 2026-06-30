# t-SNE / UMAP 비교 요약

## 실행 결과

- 대상: `outputs/` 아래 기존 `embeddings/tsne_*.npz`
- 생성된 UMAP 수: **9개**
- 생성 위치: 각 run의 `embeddings/umap_*.png|csv|npz`

## 비교한 run

- `20260507_104805_plain_vit_3label`
- `20260507_095713_plain_vit_7label`
- `20260507_101426_supcon_vit_7label`
- `20260507_081650_metric_7label_to_3label`
- `20260507_134640_plain_vit_10label`
- `20260507_143425_plain_vit_10label_attention`
- `20260507_145414_supcon_vit_7label_tsne`

## 관찰 요약

### 1) 3-label classifier

- t-SNE와 UMAP 모두 **세 클래스가 매우 잘 분리**된다.
- UMAP에서도 Bego / Bicon / ITI 군집이 각각 안정적으로 떨어져 보인다.
- 현재 시각화 기준으로는 **가장 해석이 쉬운 run** 중 하나다.

### 2) 7-label plain classifier

- t-SNE에서 클래스별 큰 덩어리는 보이지만, 일부 클래스는 **내부가 여러 subcluster로 갈라진다.**
- UMAP은 그 구조를 더 압축해서 보여주지만, 클래스 간 경계 자체는 여전히 꽤 선명하다.
- `nobel`, `osstem`, `ADIN` 쪽은 상대적으로 잘 묶인다.
- `DIONAVI` / `Dentium` 계열은 내부 분산이 조금 더 크다.

### 3) 7-label SupCon / metric

- 클래스별 군집은 유지되지만, t-SNE와 UMAP 모두 **같은 클래스 내부의 길게 늘어진 연결 구조**가 보인다.
- 완전한 점 군집보다 **연속적인 manifold 형태**가 강해서, embedding geometry를 보기 좋다.
- classifier보다 경계 해석은 조금 덜 직관적이지만, prototype 기반 구조를 보기에는 유리하다.

### 4) 3-label transfer metric

- 3-label 전이 결과는 **군집이 가장 섞여 보이는 편**이다.
- Bego / Bicon / ITI 사이에 경계가 넓게 겹치는 구간이 많아, transfer embedding의 클래스 분리가 약하다.
- 이 run은 분류 성능보다는 **전이 실패 양상 확인용**으로 보는 게 적절하다.

### 5) 10-label classifier / attention run

- t-SNE와 UMAP 모두 **라벨별로 매우 깨끗한 분리**를 보여준다.
- `Bego`, `ITI`, `NORIS`, `osstem` 등이 비교적 독립적인 섬처럼 나타난다.
- 전반적으로 현재 생성된 그림들 중 **가장 안정적인 분리 품질**을 보인다.

## 결론

- **가장 해석이 쉬운 구조**: `plain_vit_3label`, `plain_vit_10label`
- **구조 확인에 좋은 run**: `supcon_vit_7label`
- **전이/혼동 분석용 run**: `metric_7label_to_3label`
- UMAP은 t-SNE보다 전체 배치가 더 압축되어 보이지만, 클래스별 큰 구조는 대체로 일관적이었다.

## 참고

UMAP은 `tsne_*.npz` 안의 **저장된 feature payload**를 재사용해 계산했다.  
즉, t-SNE 좌표를 UMAP 입력으로 넣은 것이 아니라, 같은 임베딩 feature를 다시 투영한 결과다.


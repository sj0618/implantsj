# UMAP 재계산

`reproduce_umap.py`는 기존 `tsne_*.npz`와 같은 이름의 CSV를 새 폴더에 복사하고,
`pipelines.visualize.visualize_umap_from_tsne.process_artifact`로 UMAP을 계산한다.

```bash
python3 -m pipelines.report.reproduce_umap --source /path/to/features --output-dir outputs/umap/run01
```

기대 입력은 기본 9쌍이며 `--expected-count`로 지정한다.
입력 수 불일치·CSV 누락·이미 존재하는 출력 폴더는 오류다.
seed 42, PCA 50차원, neighbors 15, min_dist 0.1을 사용하고 입력 해시를 기록한다.
원본은 변경하지 않는다. 생성된 그림·표·기록은 로컬 산출물이다.

# 데이터 준비

`build.py`가 이미지 폴더 스캔, 기존 split 확인, 클래스별 분할, open 라벨 선택,
band mask 연결, k-fold manifest를 담당한다. `__init__.py`는 패키지 연결용이다.

```bash
python src/run_prepare_data.py --config configs/00_data.yaml --inspect --set source.root=/data/images
python src/run_prepare_data.py --config /path/to/run_data.yaml
```

입력은 클래스별 이미지 폴더 또는 train/valid/test별 클래스 폴더다.
설정에서 실제 open 라벨을 명시한다. 자동 선택은 `openset.auto`를 지정한 경우만 수행한다.
출력은 full/closed/open CSV와 선택한 라벨 JSON이다. Mask가 있으면 `mask_path`를 기록한다.
과거 실험을 재현할 때는 이미 확정된 manifest를 우선 사용한다.

# 이미지 속성 분류

| 코드 | 역할 |
|---|---|
| `core.py` | 이미지·manifest 공용 처리 |
| `spec_table.py` | 클래스별 속성 정답표 읽기·라벨 매칭·unknown 처리 |
| `train.py` | 속성별 독립 모델 또는 공유 backbone/multi-head 학습 |
| `fuse.py` | 속성 확률을 클래스 점수로 결합 |
| `visualize.py` | 속성별 성능·가중치 시각화 |
| `__init__.py` | 패키지 연결 |

```bash
python src/check_spec_table.py --config configs/05_attributes.yaml
python src/run_attributes.py --config configs/05_attributes.yaml --set train.mode=separate
python src/run_attributes.py --config configs/05_attributes.yaml --set train.mode=shared
```

Manifest와 사용자가 제공하는 `configs/spec_table.csv`가 필요하다.
`spec.csv` 설정으로 다른 경로를 지정할 수 있다. 정답표 자체는 Git에 포함하지 않는다.
클래스 연결에는 dataset_label 또는 Company/Name을 사용하며,
속성은 System·Connection_1/2·Flange·Collar·Microthread·Body_Shape·Thread_Shape·Apex_Shape·Apex_Hole·Company다.
Unknown 값은 손실과 지표에서 제외한다. 속성 확률·클래스 점수·checkpoint·특징을 저장한다.
이 코드는 이미지 분류 학습이며 VLM API 처리 기록의 재현과는 별도다.

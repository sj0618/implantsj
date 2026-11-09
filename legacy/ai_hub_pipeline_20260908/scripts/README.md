# 단계별 실행 스크립트

각 하위 폴더의 README에 해당 단계의 명령과 입력이 있다.
`common.sh`는 프로젝트 루트·Python·설정 경로를 정하고 Python 드라이버를 실행한다.

```bash
PROTOCOL=full42 SET="grid.backbone=vit" bash scripts/01_backbone/finetune_backbones.sh
CONFIG=/path/to/run.yaml PY=python3 bash scripts/03_metric_learning/run_metric_grid.sh
```

`CONFIG`로 전체 설정을 바꾸고 `SET`으로 공백으로 구분한 key=value 변경값을 전달한다.
쉘 스크립트는 순서 제어만 하며 학습·평가는 `src/`가 수행한다.
전체를 한 번에 실행하지 않는다. 선행 입력과 마스크 수동 선별을 완료한 뒤 다음 단계로 진행한다.

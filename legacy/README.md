# legacy — 예전 실행 구조 보관소

여기 있는 것은 **삭제하지 않고 옮겨둔** 예전 구조다. git 에 추적되지 않던
디렉터리라 지우면 복구할 수 없어서 보관만 한다. 필요 없다고 확신이 서면
그때 지우면 된다.

## 왜 옮겼나

리팩터링 전에는 실행 진입점이 세 개였고 어느 것을 써야 하는지 알 수 없었다.

- `finalSJ/` — `implantsj/` 의 거의 동일한 사본 (`run.sh`, `config.sh`, `WORKFLOW.md`)
- `implantsj/` — 중첩된 독립 baseline 패키지. 자체 `run.sh` 와 `implantsj.workflow.individual`
  CLI 를 갖고 있었고, 상위 저장소의 `src/`·`pipelines/` 와 기능이 겹쳤다.

두 사본은 서로도 달라서(README, WORKFLOW, evaluation.py 가 각각 다름) 어느 쪽이
최신인지 판단할 수 없었다.

## 지금 쓰는 것

저장소 루트의 진입점 하나만 쓴다.

    ./run.sh

설정은 `exp.conf`, 모델 목록은 `configs/backbones.txt` 에 있다.

## legacy/scripts/

`scripts/*.sh` 10개도 함께 옮겼다. 전부 `implantsj.workflow.individual` 을
호출하는데 그 모듈이 `legacy/implantsj/` 안에 있어서, 루트에 남겨두면
"있는데 실행하면 깨지는" 스크립트가 된다. 그 워크플로가 다시 필요하면
`legacy/implantsj/` 안에서 실행한다.

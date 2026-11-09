"""
config_utils.py — yaml 설정 로더 (모든 스테이지 공용)

- `extends: base.yaml` 상속 처리 (자식 값이 부모 값을 덮어씀, dict는 deep merge)
- --protocol 으로 yaml의 protocol 덮어쓰기
- --set 으로 yaml의 아무 항목이나 덮어쓰기 (한 번만 돌리고 싶을 때)
- paths.* 는 프로젝트 루트(ai_hub_pipeline/) 기준 상대경로를 절대경로로 변환

사용 예:
    from config_utils import load_config_from_cli
    cfg = load_config_from_cli()          # --config xxx.yaml [--protocol full42]
    print(cfg["paths"]["data_dir"], cfg["protocol"])

**그리드는 의무가 아니다.** grid 항목은 리스트가 아니어도 되고(`backbone: vit`),
한 개짜리 리스트여도 된다(`backbone: [vit]`). 둘 다 조합 1개로 처리된다.
yaml을 고치지 않고 이번 실행만 하나로 줄이려면:

    python src/run_metric.py --config configs/03_metric_grid.yaml \\
        --set grid.backbone=vit grid.embedding_loss=supcon
"""
import argparse
from pathlib import Path
from typing import Any, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """override의 값으로 base를 덮어씀. dict끼리는 재귀 merge."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def as_list(value: Any) -> List:
    """설정값을 항상 리스트로 — 스칼라 1개도 조합 1개로 다루기 위한 정규화.

    `backbone: vit` 처럼 리스트를 벗겨 써도 문자열이 글자 단위로 순회되지 않는다.
    None / 빈 값은 빈 리스트.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """cfg["a"]["b"] = value 를 "a.b" 문자열로 수행. 중간 dict가 없으면 만든다."""
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        nxt = node.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            node[key] = nxt
        node = nxt
    node[keys[-1]] = value


def apply_overrides(cfg: dict, assignments: List[str]) -> dict:
    """["grid.backbone=vit", "train.epochs=1"] 형태의 덮어쓰기를 적용한다.

    값은 yaml 문법으로 해석되므로 리스트/숫자/불리언도 그대로 쓸 수 있다:
        grid.backbone=[vit,dinov2]   train.epochs=1   eval.visualize.enabled=false
    """
    for item in assignments or []:
        if "=" not in item:
            raise SystemExit(f"--set 은 key=value 형식이어야 합니다: {item!r}")
        dotted, raw = item.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        set_by_path(cfg, dotted.strip(), value)
        print(f"[config] override {dotted.strip()} = {value!r}", flush=True)
    return cfg


def load_config(config_path: str | Path) -> dict:
    """yaml 로드 + extends 상속 처리 + paths 절대경로 변환."""
    config_path = Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent_path = (config_path.parent / parent_name).resolve()
        parent_cfg = load_config(parent_path)
        cfg = _deep_merge(parent_cfg, cfg)

    # paths.* 상대경로 -> 프로젝트 루트 기준 절대경로
    if "paths" in cfg:
        for key, value in cfg["paths"].items():
            p = Path(value)
            if not p.is_absolute():
                cfg["paths"][key] = str((PROJECT_ROOT / p).resolve())

    return cfg


def load_config_from_cli(description: str = "") -> dict:
    """--config 필수, --protocol / --set 은 선택 (yaml 값을 덮어씀)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="yaml 설정 파일 경로")
    parser.add_argument("--protocol", default=None,
                        choices=["closed27_open15", "full42"],
                        help="yaml의 protocol을 덮어씀")
    parser.add_argument("--set", dest="overrides", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="yaml 항목 덮어쓰기 (예: grid.backbone=vit train.epochs=1)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.protocol:
        cfg["protocol"] = args.protocol
    apply_overrides(cfg, args.overrides)
    return cfg

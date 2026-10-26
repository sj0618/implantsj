#!/usr/bin/env python3
"""Inspect and execute the report recipes without importing the ML runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs/reproduction/reports.json"
TOKEN = re.compile(r"\{([a-zA-Z0-9_]+)\}")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_catalog(path=CATALOG):
    catalog = json.loads(Path(path).read_text())
    ids = [report["id"] for report in catalog["reports"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate report IDs")
    for report in catalog["reports"]:
        if not (ROOT / report["source"]).is_file():
            raise ValueError(f"Report missing: {report['source']}")
        for step in report["steps"]:
            if not (ROOT / (step["module"].replace(".", "/") + ".py")).is_file():
                raise ValueError(f"Module missing: {step['module']}")
            if not step.get("artifacts"):
                raise ValueError(f"Expected artifacts missing: {report['id']}/{step['id']}")
    return catalog


def make_plan(catalog, report_id, bindings, output):
    reports = [r for r in catalog["reports"] if report_id == "all" or r["id"] == report_id]
    if not reports:
        raise ValueError(f"Unknown report: {report_id}")
    plan = []
    for report in reports:
        values = {**bindings, "output": str(Path(output).resolve() / report["id"])}
        steps = []
        missing = set()
        generated = set()
        for step in report["steps"]:
            def render(value):
                for key in TOKEN.findall(value):
                    if key not in values and key not in generated:
                        missing.add(key)
                return TOKEN.sub(lambda m: str(values.get(m[1], m[0])), value)
            steps.append({**step, "argv": [sys.executable, "-m", step["module"], *map(render, step["args"])],
                          "artifacts": [render(p) for p in step["artifacts"]],
                          "exports": {key: render(value) for key, value in step.get("exports", {}).items()}})
            generated.update(step.get("exports", {}))
        plan.append({"id": report["id"], "source": report["source"],
                     "coverage": report["coverage"], "limitations": report["limitations"],
                     "blocked": report.get("blocked", []), "missing_inputs": sorted(missing), "steps": steps})
    return plan


def preflight(plan, bindings):
    problems = []
    for report in plan:
        problems.extend(f"{report['id']}: {reason}" for reason in report["blocked"])
        problems.extend(f"{report['id']}: input required: {key}" for key in report["missing_inputs"])
        for step in report["steps"]:
            for key in step.get("inputs", []):
                if key in bindings and not Path(bindings[key]).exists():
                    problems.append(f"{report['id']}: missing path for {key}: {bindings[key]}")
    return sorted(set(problems))


def execute(plan, bindings, output):
    problems = preflight(plan, bindings)
    if problems:
        raise ValueError("\n".join(problems))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    used = {key for report in plan for step in report["steps"] for key in step.get("inputs", [])}
    inputs = {key: {"path": bindings[key], "sha256": digest(bindings[key]) if Path(bindings[key]).is_file() else None}
              for key in sorted(used)}
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "pandas", "scipy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "catalog_sha256": digest(CATALOG), "inputs": inputs, "packages": packages,
              "plan": plan, "steps": [], "claim": "Execution only; historical metric parity is not established."}
    record["source_sha256"] = {str(p.relative_to(ROOT)): digest(p)
                               for folder in ("src", "tools", "pipelines", "two_stage")
                               for p in sorted((ROOT / folder).rglob("*.py"))}
    record_path = output / "run.json"
    def save():
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    save()
    try:
        for report in plan:
            report_dir = output / report["id"]
            report_dir.mkdir()
            generated = {}
            for step in report["steps"]:
                argv = [TOKEN.sub(lambda m: generated[m[1]], value) for value in step["argv"]]
                log = report_dir / (step["id"] + ".log")
                result = {"report": report["id"], "step": step["id"], "status": "running", "log": str(log)}
                record["steps"].append(result)
                save()
                print(f"Running {report['id']}/{step['id']}", flush=True)
                with log.open("w") as stream:
                    completed = subprocess.run(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                result["returncode"] = completed.returncode
                if completed.returncode:
                    raise RuntimeError(f"{report['id']}/{step['id']} failed; see {log}")
                artifacts = []
                for pattern in step["artifacts"]:
                    relative = Path(pattern).relative_to(output)
                    matches = sorted(p for p in output.glob(str(relative)) if p.is_file() and p.stat().st_size)
                    if not matches:
                        raise RuntimeError(f"Expected output missing: {pattern}")
                    artifacts.extend({"path": str(p.relative_to(output)), "sha256": digest(p)} for p in matches)
                result.update(status="executed", artifacts=artifacts)
                for key, pattern in step.get("exports", {}).items():
                    matches = sorted(output.glob(str(Path(pattern).relative_to(output))))
                    if len(matches) != 1 or not matches[0].is_file():
                        raise RuntimeError(f"Expected one generated input for {key}: {pattern}")
                    generated[key] = str(matches[0])
                result["argv"] = argv
                save()
        record["status"] = "executed"
    except BaseException as exc:
        record.update(status="failed", error=str(exc))
        if record["steps"]:
            record["steps"][-1]["status"] = "failed"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    return record_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "plan", "check", "run"))
    parser.add_argument("--report", default="all")
    parser.add_argument("--inputs", type=Path, help="JSON object mapping named inputs to local paths")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction")
    args = parser.parse_args()
    try:
        catalog = load_catalog()
        bindings = json.loads(args.inputs.read_text()) if args.inputs else {}
        if not isinstance(bindings, dict) or any(not isinstance(v, str) for v in bindings.values()):
            raise ValueError("Inputs must be a JSON object of path strings")
        unknown = set(bindings) - set(catalog["input_names"])
        if unknown:
            raise ValueError(f"Unknown or reserved input names: {sorted(unknown)}")
        bindings = {key: str(Path(value).expanduser().resolve()) for key, value in bindings.items()}
        plan = make_plan(catalog, args.report, bindings, args.output)
        if args.command == "list":
            for item in plan:
                print(f"{item['id']}\t{item['coverage']}\t{len(item['steps'])} steps")
        elif args.command == "plan":
            print(json.dumps(plan, ensure_ascii=False, indent=2))
        elif args.command == "check":
            problems = preflight(plan, bindings)
            print("\n".join(problems) if problems else "All required paths are present; data identity has not been verified.")
            return int(bool(problems))
        else:
            print(execute(plan, bindings, args.output))
    except (ValueError, OSError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

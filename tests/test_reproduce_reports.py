import ast
import json
from types import SimpleNamespace

import pytest

from tools import reproduce_reports as runner


def test_every_report_is_catalogued_and_cli_flags_exist():
    catalog = runner.load_catalog()
    sources = {r["source"] for r in catalog["reports"]}
    assert {str(p.relative_to(runner.ROOT)) for p in (runner.ROOT / "docs/reports").rglob("*.md")} <= sources
    for report in catalog["reports"]:
        assert report["limitations"]
        assert report["steps"] or report["blocked"]
        ids = [s["id"] for s in report["steps"]]
        assert len(ids) == len(set(ids))
        for step in report["steps"]:
            module = runner.ROOT / (step["module"].replace(".", "/") + ".py")
            tree = ast.parse(module.read_text())
            flags = {a.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                     for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)}
            assert {a for a in step["args"] if a.startswith("--")} <= flags


def minimal_plan(tmp_path):
    return [{"id": "example", "source": "README.md", "coverage": "protocol", "limitations": [],
             "blocked": [], "missing_inputs": [], "steps": [
                 {"id": "evaluate", "argv": ["python", "-m", "example", "space $(literal)"],
                  "inputs": [], "artifacts": [str(tmp_path / "run/example/results.csv")]}]}]


def test_missing_and_blocked_inputs_do_not_create_output(tmp_path):
    plan = minimal_plan(tmp_path)
    plan[0]["missing_inputs"] = ["manifest"]
    plan[0]["blocked"] = ["unknown original architecture"]
    with pytest.raises(ValueError, match="manifest"):
        runner.execute(plan, {}, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("returncode,write_artifact,status", [(0, True, "executed"), (0, False, "failed"), (7, False, "failed")])
def test_execution_requires_success_and_new_artifacts(tmp_path, monkeypatch, returncode, write_artifact, status):
    def run(argv, **kwargs):
        assert argv[-1] == "space $(literal)"
        assert "shell" not in kwargs
        if write_artifact:
            (tmp_path / "run/example/results.csv").write_text("metric,value\naccuracy,0.5\n")
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "test-commit\n")
    if status == "failed":
        with pytest.raises(RuntimeError):
            runner.execute(minimal_plan(tmp_path), {}, tmp_path / "run")
    else:
        runner.execute(minimal_plan(tmp_path), {}, tmp_path / "run")
    record = json.loads((tmp_path / "run/run.json").read_text())
    assert record["status"] == status
    assert record["steps"][0]["status"] == status
    with pytest.raises(FileExistsError):
        runner.execute(minimal_plan(tmp_path), {}, tmp_path / "run")


def test_training_outputs_are_resolved_without_external_checkpoint():
    catalog = runner.load_catalog()
    plan = runner.make_plan(catalog, "fresh_supcon7_to3", {}, "/tmp/unused-reproduction")
    assert "trained_checkpoint" not in plan[0]["missing_inputs"]
    assert plan[0]["steps"][0]["exports"]["trained_checkpoint"].endswith("/checkpoints/best.pt")
    assert "{trained_checkpoint}" in plan[0]["steps"][1]["argv"]


def test_external_paths_are_not_guessed():
    plan = runner.make_plan(runner.load_catalog(), "target5_episodic5", {}, "/tmp/unused-reproduction")
    assert set(plan[0]["missing_inputs"]) == {"supcon6_checkpoint", "target5_manifest", "source6_manifest"}


def test_generated_checkpoint_is_passed_to_next_step(tmp_path, monkeypatch):
    plan = minimal_plan(tmp_path)
    checkpoint = tmp_path / "run/example/train/only/checkpoints/best.pt"
    plan[0]["steps"][0].update(artifacts=[str(checkpoint)], exports={"trained_checkpoint": str(checkpoint)})
    plan[0]["steps"].append({"id": "second", "argv": ["python", "evaluate", "{trained_checkpoint}"],
                              "inputs": [], "artifacts": [str(tmp_path / "run/example/second.csv")]})
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test-checkpoint")
        else:
            assert argv[-1] == str(checkpoint)
            (tmp_path / "run/example/second.csv").write_text("metric,value\naccuracy,1\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "test-commit\n")
    runner.execute(plan, {}, tmp_path / "run")
    assert len(calls) == 2


def test_twostage_existing_manifest_is_preserved(tmp_path, monkeypatch):
    from pipelines.eval import evaluate_3label_twostage_metric_from_7label as module
    manifest = tmp_path / "split.csv"
    original = "path,label,split\na.png,Bego,train\nb.png,Bicon,valid\nc.png,ITI,test\n"
    manifest.write_text(original)
    def fail(*args, **kwargs):
        raise RuntimeError("checkpoint boundary reached")
    monkeypatch.setattr(module, "load_stage1_metric_model", fail)
    monkeypatch.setattr(module, "build_3label_manifest", lambda *a: pytest.fail("Original manifest must not be rebuilt"))
    monkeypatch.setattr("sys.argv", ["evaluate", "--manifest", str(manifest), "--use-existing-manifest",
                                   "--checkpoint", "missing.pt", "--output-root", str(tmp_path / "new"), "--allow-cpu"])
    with pytest.raises(RuntimeError, match="checkpoint boundary"):
        module.main()
    assert manifest.read_text() == original

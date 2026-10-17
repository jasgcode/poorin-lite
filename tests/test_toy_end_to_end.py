"""The toy pipeline through every workflow path, from a shell, in a repository of its own."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from repro.evaluation.evaluate import eval_results
from repro.execution.artifact import read_record
from repro.experiment.run_refs import collect_runs

TEMPLATE = Path(__file__).resolve().parents[1]
CALENDAR = "2026.10.17"
RUN_VERSION = re.compile(r"Run version: (\S+)")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A git repository holding the toy experiment, and an empty prefix."""
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    shutil.copy(TEMPLATE / "experiments" / "toy_regression.py", repo / "experiments" / "toy_regression.py")
    (repo / "experiments" / "__init__.py").write_text("")
    shutil.copy(TEMPLATE / ".gitignore", repo / ".gitignore")
    (repo / "docs" / "experiments").mkdir(parents=True)
    (repo / "docs" / "experiments" / "toy-lr.md").write_text(
        "---\ndate: 2026-10-17\nstatus: open\nsummary: Which learning rate fits the toy line?\n---\n"
    )
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run([*git, "add", "."], cwd=repo, check=True)
    subprocess.run([*git, "commit", "-q", "-m", "toy"], cwd=repo, check=True)
    prefix = tmp_path / "prefix"
    monkeypatch.setenv("REPRO_PREFIX", str(prefix))
    return repo, prefix


def _cli(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", *args], cwd=repo, env=os.environ.copy(), capture_output=True, text=True, timeout=120
    )


def _ok(result: subprocess.CompletedProcess) -> str:
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _records(prefix: Path) -> dict[str, dict]:
    return {
        f"{data['name']}@{data['version']}": data
        for data in (json.loads(p.read_text()) for p in prefix.rglob(".artifact.json"))
    }


def test_the_toy_pipeline_through_every_workflow_path(workspace):
    repo, prefix = workspace
    toy = "experiments.toy_regression"

    # The plan builds nothing.
    _ok(_cli(repo, toy, "--version", "dev"))
    assert _records(prefix) == {}

    # --run builds the data and the training run.
    out = _ok(_cli(repo, toy, "--version", "dev", "--run"))
    dev = RUN_VERSION.search(out).group(1)
    token = dev.split("-")[-2]
    assert any(name.endswith(f"toy/train@{dev}") for name in _records(prefix))

    # A calendar version builds once; a second run is served from cache.
    _ok(_cli(repo, toy, "--version", CALENDAR, "--run"))
    record = prefix / "toy" / "train" / CALENDAR / ".artifact.json"
    first = record.stat().st_mtime_ns
    _ok(_cli(repo, toy, "--version", CALENDAR, "--run"))
    assert record.stat().st_mtime_ns == first

    # --set names a new trial with a new recipe.
    _ok(_cli(repo, toy, "--version", CALENDAR, "--set", "lr=0.1", "--run"))
    trials = {label: data for label, data in _records(prefix).items() if label.startswith("toy/train/")}
    assert len(trials) == 1
    (trial,) = trials.values()
    assert trial["config"]["lr"] == 0.1
    base = _records(prefix)[f"toy/train@{CALENDAR}"]
    assert trial["fingerprint"] != base["fingerprint"]

    # A sweep builds one trial per value.
    _ok(_cli(repo, toy, "--version", "sweep-dev", "--sweep", "lr=[0.02, 0.03]", "--run"))
    swept = [r for r in collect_runs(str(prefix)) if "toy/train/" in r.label and "sweep" in r.label]
    assert len(swept) == 2

    # --resume continues the first dev run by its token.
    out = _ok(_cli(repo, toy, "--resume", token, "--run"))
    assert dev in out

    # runs list and runs show find the run by its token.
    assert f"toy/train@{dev}" in _ok(_cli(repo, "repro.experiment.runs", "list"))
    shown = _ok(_cli(repo, "repro.experiment.runs", "show", f"toy/train@{token}"))
    assert dev in shown

    # --experiment tags a run's records; runs list --experiment finds them.
    out = _ok(_cli(repo, toy, "--version", "dev", "--experiment", "toy-lr", "--run"))
    tagged = RUN_VERSION.search(out).group(1)
    listed = _ok(_cli(repo, "repro.experiment.runs", "list", "--experiment", "toy-lr"))
    assert sorted(line.split()[3].rsplit("/", 1)[-1] for line in listed.splitlines()) == [
        f"data@{tagged}",
        f"train@{tagged}",
    ]

    # evaluate scores every checkpoint of the run; eval_results reads the scores back.
    _ok(
        _cli(
            repo,
            "repro.evaluation.evaluate",
            f"toy/train@{token}",
            "--suite",
            "heldout_mse",
            "--steps",
            "all",
            "--version",
            "dev",
            "--run",
        )
    )
    label = next(r.label for r in collect_runs(str(prefix)) if r.label.endswith(f"toy/train@{dev}"))
    results = eval_results(str(prefix), run=label)
    assert sorted(r.step for r in results) == [50, 100, 150, 200]
    assert all(r.metrics["eval/mse"] < 1.0 for r in results)

    # A calendar version refuses uncommitted code.
    experiment = repo / "experiments" / "toy_regression.py"
    experiment.write_text(experiment.read_text() + "\n# an uncommitted change\n")
    refused = _cli(repo, toy, "--version", "2026.10.18", "--run")
    assert refused.returncode != 0
    assert read_record(str(prefix / "toy" / "train" / "2026.10.18")) is None

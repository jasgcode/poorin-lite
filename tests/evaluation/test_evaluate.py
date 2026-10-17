# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation handles over finished runs (repro.evaluation.evaluate)."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from click.testing import CliRunner

from repro.evaluation.contract import EvalResult, EvalRuntime, FinishedRun
from repro.evaluation.evaluate import Checkpoint, eval_results, evaluate, finished_run
from repro.evaluation.evaluate import main as evaluate_main
from repro.execution.artifact import Artifact, read_record, write_record
from repro.execution.lazy import OUT, apply, run
from repro.execution.step_status import STATUS_RUNNING, StatusFile
from repro.experiment.run_refs import RunRefError
from repro.experiment.runs import main as runs_main
from repro.provenance import username_segment

DEV = "accept-20261012-201822-60b9-dev"
OTHER_DEV = "accept-20261013-080000-c3d4-dev"


@dataclass(frozen=True)
class Score:
    """A toy suite: the loss is the step times ``scale``."""

    name: ClassVar[str] = "score"
    scale: float = 1.0

    def check(self, run: FinishedRun, runtime: EvalRuntime) -> None:
        pass

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        return EvalResult(
            run=run.label, step=step, suite=self.name, metrics={"eval/loss": self.scale * step}, examples=1
        )


@dataclass(frozen=True)
class Unoffered:
    name: ClassVar[str] = "unoffered"

    def check(self, run, runtime):
        raise AssertionError("not offered")

    def evaluate(self, run, step, output_path, runtime):
        raise AssertionError("not offered")


@dataclass(frozen=True)
class Stale:
    """A toy suite that cannot read the run's recipe."""

    name: ClassVar[str] = "stale"

    def check(self, run, runtime):
        raise ValueError(f"{run.label} records a stale field; drop it with --drop data.num_workers")

    def evaluate(self, run, step, output_path, runtime):
        raise AssertionError("refused when the plan is built")


class PlainCheckpoint(Artifact):
    """A checkpoint type that offers no suites, as a run's type does before it names its suites."""


class ToyCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {Score.name: Score}


class OtherCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {Score.name: Score}


class StaleCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {Stale.name: Stale}


def _train(out: str) -> None:
    for step in (1, 2):
        os.makedirs(os.path.join(out, "checkpoints", f"step-{step}"))


def train_toy(version: str = DEV, artifact_type: type[Artifact] = ToyCheckpoint) -> str:
    """A finished dev run of the toy template in the user's namespace; its full label."""
    handle = apply(f"users/{username_segment()}/toy/run", _train, version=version, artifact_type=artifact_type, out=OUT)
    run(handle)
    return handle.label


@pytest.fixture
def prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    return tmp_path


def test_an_evaluation_is_named_after_the_run_suite_and_step(prefix):
    train_toy()
    cell = evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Score(), version="2026.10.13")
    assert cell.label == f"eval/toy/run/{DEV}/score/step-2@2026.10.13"


def test_a_dev_evaluation_is_namespaced_and_two_run_versions_get_two_names(prefix):
    train_toy()
    train_toy(OTHER_DEV)
    version = "20261013-090000-e5f6-dev"
    first = evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Score(), version=version)
    second = evaluate(Checkpoint(finished_run("toy/run@c3d4"), 2), Score(), version=version)
    assert first.name == f"users/{username_segment()}/eval/toy/run/{DEV}/score/step-2"
    assert first.name != second.name


def test_the_suite_and_step_bear_identity_and_the_batch_size_does_not(prefix):
    train_toy()
    pinned = finished_run("toy/run@60b9")

    def fingerprint(step: int, suite: Score, batch_size: int | None = None) -> str:
        return evaluate(Checkpoint(pinned, step), suite, version="2026.10.13", batch_size=batch_size).fingerprint()

    base = fingerprint(2, Score())
    assert fingerprint(2, Score(scale=2.0)) != base
    assert fingerprint(1, Score()) != base
    assert fingerprint(2, Score(), batch_size=64) == base


def test_an_evaluation_builds_its_result_and_eval_results_finds_it(prefix):
    label = train_toy()
    (result,) = run(evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Score(scale=0.5), version="2026.10.13"))
    assert (result.run, result.step, result.metrics) == (label, 2, {"eval/loss": 1.0})
    assert [r.metrics for r in eval_results(str(prefix), run=label)] == [{"eval/loss": 1.0}]
    assert eval_results(str(prefix), suite="other") == []


def test_an_unfinished_run_is_not_a_finished_run(prefix):
    label = train_toy()
    StatusFile(str(prefix / label.replace("@", "/"))).write_status(STATUS_RUNNING)
    with pytest.raises(RunRefError, match="no finished run"):
        finished_run("toy/run@60b9")


def test_a_suite_the_checkpoint_type_does_not_offer_is_refused(prefix):
    train_toy()
    with pytest.raises(ValueError, match=r"offers the suites \['score'\]"):
        evaluate(Checkpoint(finished_run("toy/run@60b9"), 2), Unoffered(), version="2026.10.13")


CALENDAR = ["--version", "2026.10.13"]


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "checkout"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "model.py").write_text("v1\n")
    _git(path, "add", "model.py")
    _git(path, "commit", "-m", "Add the model")
    monkeypatch.chdir(path)
    return path


def _invoke(*args: str):
    return CliRunner().invoke(evaluate_main, list(args), catch_exceptions=False)


def test_the_plan_has_one_evaluation_per_permanent_checkpoint(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "all", *CALENDAR)
    assert result.exit_code == 0, result.output
    for step in (1, 2):
        assert f"eval/toy/run/{DEV}/score/step-{step}@2026.10.13" in result.output


def test_a_set_reaches_the_suite_and_names_the_evaluation(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "2", "--set", "suite.scale=3", *CALENDAR)
    assert result.exit_code == 0, result.output
    assert f"eval/toy/run/{DEV}/score/suite.scale=3/step-2@2026.10.13" in result.output


def test_a_set_outside_the_suite_is_refused(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "2", "--set", "step=1", *CALENDAR)
    assert result.exit_code != 0
    assert "has no field 'step'" in result.output


def test_a_step_without_a_checkpoint_is_refused_listing_the_steps(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "1,3", *CALENDAR)
    assert result.exit_code != 0
    assert "step [3]" in result.output and "[1, 2]" in result.output


def test_a_suite_the_run_does_not_offer_is_refused_listing_its_suites(prefix):
    train_toy()
    result = _invoke("toy/run@60b9", "--suite", "lm-eval", "--steps", "2", *CALENDAR)
    assert result.exit_code != 0
    assert "['score']" in result.output


def test_a_run_the_suite_cannot_evaluate_is_refused_before_anything_runs(prefix):
    train_toy(artifact_type=StaleCheckpoint)
    result = _invoke("toy/run@60b9", "--suite", "stale", "--steps", "all", *CALENDAR)
    assert result.exit_code != 0
    assert "--drop data.num_workers" in result.output
    assert "eval/toy/run" not in result.output
    assert eval_results(str(prefix)) == []


def test_an_unknown_or_unfinished_run_is_refused(prefix):
    result = _invoke("toy/run@60b9", "--suite", "score", "--steps", "2", *CALENDAR)
    assert result.exit_code != 0
    assert "no finished run toy/run@60b9" in result.output


def test_a_run_trained_before_evaluation_artifacts_is_refused_naming_the_migration(prefix):
    old = apply(f"users/{username_segment()}/toy/old", _train, version=DEV, artifact_type=PlainCheckpoint, out=OUT)
    run(old)
    result = _invoke("toy/old@60b9", "--suite", "score", "--steps", "2", *CALENDAR)
    assert result.exit_code != 0
    assert "runs migrate" in result.output


def test_migrate_names_the_checkpoint_type_and_drops_a_moved_field(prefix):
    old = apply(f"users/{username_segment()}/toy/old", _train, version=DEV, artifact_type=PlainCheckpoint, out=OUT)
    run(old)
    toy_type = f"{ToyCheckpoint.__module__}.ToyCheckpoint"
    result = CliRunner().invoke(runs_main, ["migrate", "toy/old@60b9", "--to", toy_type, "--drop", "out"])
    assert result.exit_code == 0, result.output
    record = read_record(old.path())
    assert record.result_type == toy_type
    assert "out" not in record.config
    planned = _invoke("toy/old@60b9", "--suite", "score", "--steps", "2", *CALENDAR)
    assert planned.exit_code == 0, planned.output


def _migrate(*args: str):
    return CliRunner().invoke(runs_main, ["migrate", *args])


def _train_old():
    old = apply(f"users/{username_segment()}/toy/old", _train, version=DEV, artifact_type=PlainCheckpoint, out=OUT)
    run(old)
    return old


def test_migrate_refuses_a_run_that_records_another_checkpoint_type(prefix):
    train_toy()
    result = _migrate("toy/run@60b9", "--to", f"{OtherCheckpoint.__module__}.OtherCheckpoint")
    assert result.exit_code != 0
    assert "already offers suites" in result.output
    assert read_record(finished_run("toy/run@60b9").path()).result_type.endswith(".ToyCheckpoint")


def test_migrate_refuses_to_drop_a_field_the_record_does_not_hold(prefix):
    old = _train_old()
    toy_type = f"{ToyCheckpoint.__module__}.ToyCheckpoint"
    result = _migrate("toy/old@60b9", "--to", toy_type, "--drop", "no.such.field")
    assert result.exit_code != 0
    assert "no config field 'no.such.field'" in result.output
    assert read_record(old.path()).result_type == f"{PlainCheckpoint.__module__}.PlainCheckpoint"


def test_migrate_refuses_a_checkpoint_type_that_cannot_be_imported(prefix):
    _train_old()
    result = _migrate("toy/old@60b9", "--to", "no.such.Module")
    assert result.exit_code != 0
    assert "'no.such.Module'" in result.output
    assert "Traceback" not in result.output


def test_migrate_refuses_a_checkpoint_type_without_a_module(prefix):
    _train_old()
    result = _migrate("toy/old@60b9", "--to", "Foo")
    assert result.exit_code != 0
    assert "'Foo'" in result.output
    assert "Traceback" not in result.output


def test_migrate_keeps_the_fingerprint_the_run_was_built_with(prefix):
    old = _train_old()
    before = read_record(old.path())
    result = _migrate("toy/old@60b9", "--to", f"{ToyCheckpoint.__module__}.ToyCheckpoint")
    assert result.exit_code == 0, result.output
    after = read_record(old.path())
    assert after.result_type != before.result_type
    assert (after.fingerprint, after.fingerprint_payload) == (before.fingerprint, before.fingerprint_payload)


def test_a_rerun_of_a_built_matrix_builds_nothing(checkout, prefix):
    train_toy()
    args = ("toy/run@60b9", "--suite", "score", "--steps", "all", *CALENDAR)
    built = _invoke(*args, "--run")
    assert built.exit_code == 0, built.output
    plan = _invoke(*args)
    lines = [line for line in plan.output.splitlines() if "eval/toy/run" in line]
    assert len(lines) == 2 and all(line.split()[0] == "built" for line in lines)


def test_migrate_accepts_a_run_whose_recorded_type_no_longer_imports(prefix):
    old = _train_old()
    write_record(read_record(old.path()).model_copy(update={"result_type": "gone.module.OldCheckpoint"}))
    result = _migrate("toy/old@60b9", "--to", f"{ToyCheckpoint.__module__}.ToyCheckpoint")
    assert result.exit_code == 0, result.output
    assert read_record(old.path()).result_type == f"{ToyCheckpoint.__module__}.ToyCheckpoint"

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaluation contract (repro.evaluation.contract)."""

from dataclasses import dataclass
from typing import ClassVar

import pytest

from repro.evaluation.contract import (
    EvalResult,
    FinishedRun,
    checkpoint_steps,
    offered_suites,
    recorded,
)
from repro.execution.artifact import Artifact, ArtifactRecord
from repro.execution.lazy import apply, run


@dataclass(frozen=True)
class Clip:
    history: int = 2


@dataclass(frozen=True)
class Score:
    name: ClassVar[str] = "score"

    def check(self, run, runtime):
        raise AssertionError("not called")

    def evaluate(self, run, step, output_path, runtime):
        raise AssertionError("not called")


class ScoredCheckpoint(Artifact):
    @classmethod
    def suites(cls):
        return {Score.name: Score}


def _score() -> EvalResult:
    return EvalResult(run="toy/run@2026.10.13", step=2, suite="score", metrics={"eval/loss": 0.5}, examples=4)


def test_an_eval_result_round_trips_through_its_record(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    (loaded,) = run(apply("eval/toy", _score, version="2026.10.13", artifact_type=EvalResult))
    assert (loaded.run, loaded.step, loaded.suite) == ("toy/run@2026.10.13", 2, "score")
    assert loaded.metrics == {"eval/loss": 0.5}
    assert (loaded.examples, loaded.samples_path) == (4, None)


def test_a_record_field_the_code_dropped_names_the_migration():
    record = ArtifactRecord(
        name="toy/run",
        version="2026.10.13",
        result_type="toy.ScoredCheckpoint",
        config={"data": {"__class__": "Clip", "history": 2, "num_workers": 16}},
    )
    finished = FinishedRun("toy/run@2026.10.13", "/runs/toy", record)
    with pytest.raises(ValueError, match=r"runs migrate toy/run@2026\.10\.13 .*--drop data\.num_workers$"):
        recorded(finished, "data", Clip)


def test_a_checkpoint_type_without_suites_is_refused_naming_the_migration():
    assert offered_suites("toy/run@2026.10.13", ScoredCheckpoint) == {"score": Score}
    with pytest.raises(ValueError, match="runs migrate"):
        offered_suites("toy/run@2026.10.13", Artifact)


def test_checkpoint_steps_lists_the_permanent_checkpoints_in_step_order(tmp_path):
    for name in ("step-1000", "step-500", "step-2000", "tmp"):
        (tmp_path / "checkpoints" / name).mkdir(parents=True)
    assert checkpoint_steps(str(tmp_path)) == [500, 1000, 2000]
    assert checkpoint_steps(str(tmp_path / "nothing")) == []

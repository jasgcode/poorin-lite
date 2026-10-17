# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweeps: grid fan-out, and selection as ordinary code over resolved outputs.

The toy trials here *write* their metrics to their own output, the way a real training run
does, so the selection test exercises the real fan-out/fan-in mechanics — run each trial, read
its recorded metrics back through its typed artifact, and reduce — without training anything.
There is no framework ``select``: selection is the reduction the experiment writes.
"""

import json
from pathlib import Path

import pytest

from repro.execution.artifact import Artifact
from repro.execution.lazy import ArtifactStep, resolve
from repro.experiment.cli import grid


class MetricsCheckpoint(Artifact):
    """A trial's output with its final metrics, as a training run records them."""

    def final_loss(self) -> float:
        return json.loads((Path(self.path) / "metrics.json").read_text())["eval/loss"]


def _write_metrics(config: dict) -> None:
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"eval/loss": config["loss"]}))


def _trial(learning_rate: float, weight_decay: float) -> ArtifactStep[MetricsCheckpoint]:
    """A toy trial that records ``loss = lr + wd`` to its output, like a real run."""
    return ArtifactStep(
        name=f"trials/lr{learning_rate}-wd{weight_decay}",
        version="2026.06.28",
        artifact_type=MetricsCheckpoint,
        run=_write_metrics,
        build_config=lambda ctx, lr=learning_rate, wd=weight_decay: {"out": ctx.output_path, "loss": lr + wd},
    )


def test_grid_is_the_cartesian_product():
    assert grid(learning_rate=[0.1, 0.2], weight_decay=[0.0, 0.5]) == [
        {"learning_rate": 0.1, "weight_decay": 0.0},
        {"learning_rate": 0.1, "weight_decay": 0.5},
        {"learning_rate": 0.2, "weight_decay": 0.0},
        {"learning_rate": 0.2, "weight_decay": 0.5},
    ]


def test_selection_is_user_code_over_resolved_metrics(tmp_path, monkeypatch):
    """The replacement for the old ``select``: run the sweep, read each trial's metrics through
    its typed artifact, and reduce — no framework verb involved."""
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))
    # losses: 0.1, 0.3 (lr.1), 0.3, 0.5 (lr.3) -> min is lr=0.1, wd=0.0
    trials = [_trial(**point) for point in grid(learning_rate=[0.1, 0.3], weight_decay=[0.0, 0.2])]

    scored = [(trial, resolve(trial).final_loss()) for trial in trials]
    best, best_loss = min(scored, key=lambda pair: pair[1])

    assert best_loss == pytest.approx(0.1)
    assert best.name == "trials/lr0.1-wd0.0"

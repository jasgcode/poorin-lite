"""A toy regression task: data from a known line, a linear model fit by SGD, and its evaluation suite.

Project code lives in ``src/project``; ``experiments/`` wires it into artifact steps. The
checkpoint type and its suites live here, beside the model, so a recorded ``result_type`` always
imports.
"""

import dataclasses
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from repro.evaluation.contract import EvalResult, EvalRuntime, FinishedRun, checkpoint_path, recorded
from repro.execution.artifact import Artifact

TRUE_WEIGHT = 3.0
TRUE_BIAS = 1.0
CHECKPOINT_FILE = "params.json"


@dataclass(frozen=True)
class LinearModel:
    """The model's recipe: where SGD starts."""

    init_weight: float = 0.0
    init_bias: float = 0.0


@dataclass(frozen=True)
class Params:
    weight: float
    bias: float


def sample_points(seed: int, count: int, noise: float) -> list[tuple[float, float]]:
    """``count`` points ``(x, 3x + 1 + noise)`` with ``x`` uniform in [-1, 1]."""
    rng = random.Random(seed)
    points = []
    for _ in range(count):
        x = rng.uniform(-1.0, 1.0)
        points.append((x, TRUE_WEIGHT * x + TRUE_BIAS + rng.gauss(0.0, noise)))
    return points


def mse(params: Params, points: list[tuple[float, float]]) -> float:
    return sum((params.weight * x + params.bias - y) ** 2 for x, y in points) / len(points)


def sgd_step(params: Params, x: float, y: float, lr: float) -> Params:
    error = params.weight * x + params.bias - y
    return Params(weight=params.weight - lr * 2 * error * x, bias=params.bias - lr * 2 * error)


def save_checkpoint(directory: str, params: Params) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / CHECKPOINT_FILE).write_text(json.dumps(dataclasses.asdict(params)))


def load_checkpoint(directory: str) -> Params:
    return Params(**json.loads((Path(directory) / CHECKPOINT_FILE).read_text()))


@dataclass(frozen=True)
class HeldoutMse:
    """Mean squared error on fresh points from the true line. The fields are the evaluation's recipe."""

    name: ClassVar[str] = "heldout_mse"
    seed: int = 1234
    num_examples: int = 200
    noise: float = 0.0

    def check(self, run: FinishedRun, runtime: EvalRuntime) -> None:
        # Rebuilding the model from the run's recorded recipe raises, naming the `runs migrate`
        # fix, when the current code no longer reads it.
        recorded(run, "model", LinearModel)

    def evaluate(self, run: FinishedRun, step: int, output_path: str, runtime: EvalRuntime) -> EvalResult:
        params = load_checkpoint(checkpoint_path(run, step))
        points = sample_points(self.seed, self.num_examples, self.noise)
        return EvalResult(
            run=run.label, step=step, suite=self.name, metrics={"eval/mse": mse(params, points)}, examples=len(points)
        )


class ToyCheckpoint(Artifact):
    """A toy training run's output: ``checkpoints/step-N/params.json`` for each saved step."""

    @classmethod
    def suites(cls):
        return {HeldoutMse.name: HeldoutMse}

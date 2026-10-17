"""The toy regression pipeline: data, then training with periodic checkpoints, scored by held-out MSE.

    uv run python -m experiments.toy_regression --version dev                  # print the plan
    uv run python -m experiments.toy_regression --version dev --run            # build it
    uv run python -m experiments.toy_regression --version dev --set lr=0.1 --run
    uv run python -m experiments.toy_regression --version dev --sweep "lr=[0.01, 0.1]" --run
    uv run python -m repro.evaluation.evaluate toy/train@TOKEN --suite heldout_mse --steps all --version dev --run

``--log-every`` shows how an experiment adds its own option: it changes only what the run logs, so
it is a runtime arg and stays out of the fingerprint. A value that changes results belongs in the
config, where ``--set`` reaches it.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import click

from project.toy import LinearModel, Params, ToyCheckpoint, mse, sample_points, save_checkpoint, sgd_step
from repro.accelerator import exclusive_accelerator
from repro.evaluation.contract import CHECKPOINT_STEP_PREFIX, DEFAULT_CHECKPOINTS_PATH
from repro.execution.artifact import Artifact
from repro.execution.build_context import resolve_version
from repro.execution.lazy import ArtifactStep, StepContext
from repro.experiment.cli import build_options
from repro.experiment.namespacing import user_namespaced_name
from repro.experiment.overrides import Overrides

logger = logging.getLogger(__name__)

DATA_NAME = "toy/data"
TRAIN_NAME = "toy/train"
TRAIN_FILE = "train.json"


@dataclass(frozen=True)
class DataConfig:
    output_path: str
    seed: int
    num_train: int
    noise: float


@dataclass(frozen=True)
class TrainConfig:
    output_path: str
    data: str
    model: LinearModel
    lr: float
    steps: int
    seed: int
    checkpoint_every: int
    log_every: int


def make_data(config: DataConfig) -> None:
    out = Path(config.output_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / TRAIN_FILE).write_text(json.dumps(sample_points(config.seed, config.num_train, config.noise)))


def train(config: TrainConfig) -> None:
    points = [(x, y) for x, y in json.loads((Path(config.data) / TRAIN_FILE).read_text())]
    rng = random.Random(config.seed)
    params = Params(config.model.init_weight, config.model.init_bias)
    with exclusive_accelerator(f"Training {config.output_path}"):
        for step in range(1, config.steps + 1):
            x, y = rng.choice(points)
            params = sgd_step(params, x, y, config.lr)
            if step % config.log_every == 0:
                logger.info("step %d train/mse %.5f", step, mse(params, points))
            if step % config.checkpoint_every == 0 or step == config.steps:
                directory = Path(config.output_path) / DEFAULT_CHECKPOINTS_PATH / f"{CHECKPOINT_STEP_PREFIX}{step}"
                save_checkpoint(str(directory), params)


def toy_pipeline(*, log_every: int, overrides: Overrides) -> ArtifactStep[ToyCheckpoint]:
    """The training run's handle; its data step is a dependency. Overrides apply to ``TrainConfig``."""
    data_version = resolve_version(DATA_NAME, None)
    data = ArtifactStep(
        name=user_namespaced_name(DATA_NAME, data_version),
        version=data_version,
        artifact_type=Artifact,
        run=make_data,
        build_config=lambda ctx: DataConfig(output_path=ctx.output_path, seed=0, num_train=256, noise=0.1),
    )
    train_version = resolve_version(TRAIN_NAME, None)

    def build_config(ctx: StepContext) -> TrainConfig:
        return overrides.apply(
            ctx,
            TrainConfig(
                output_path=ctx.output_path,
                data=ctx.artifact_path(data),
                model=LinearModel(),
                lr=0.05,
                steps=200,
                seed=0,
                checkpoint_every=50,
                log_every=ctx.runtime_arg("log_every"),
            ),
        )

    return ArtifactStep(
        name=user_namespaced_name(overrides.named(TRAIN_NAME), train_version),
        version=train_version,
        artifact_type=ToyCheckpoint,
        run=train,
        build_config=build_config,
        deps=(data,),
        runtime_args={"log_every": log_every},
    )


@click.command()
@click.option("--log-every", type=click.IntRange(min=1), default=50, help="Log the training MSE every N steps.")
@build_options
def main(log_every: int, overrides: Overrides) -> ArtifactStep[ToyCheckpoint]:
    return toy_pipeline(log_every=log_every, overrides=overrides)


if __name__ == "__main__":
    main()

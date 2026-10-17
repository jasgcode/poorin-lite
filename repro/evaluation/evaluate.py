# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluations as artifacts: handles over finished runs, and the command that runs a matrix.

    uv run python -m repro.evaluation.evaluate RUN@REF [RUN@REF ...] \\
        --suite NAME --steps 500,1000|all --version VERSION [--set suite.FIELD=VALUE] [--run]

An evaluation is an artifact step whose input is a finished run, pinned so it is never rebuilt, and
one of its permanent checkpoint steps; its recipe is a suite (:mod:`repro.evaluation.contract`).
Its name is ``eval/{run name}/{run version}/{suite}/step-{N}``, with the run's ``users/<you>/``
namespace stripped, and a dev evaluation is namespaced like a dev run. Its fingerprint is the run's
``name@version``, the step, and the suite's class and fields; the device batch size is a runtime
arg. Every evaluation step holds the accelerator lock.
"""

import re
from dataclasses import dataclass
from typing import Any, no_type_check

import click

from repro import settings
from repro.accelerator import exclusive_accelerator
from repro.evaluation.contract import (
    EvalResult,
    EvalRuntime,
    Suite,
    artifact_class,
    checkpoint_steps,
    finished_run_at,
    offered_suites,
)
from repro.execution.artifact import result_type_name
from repro.execution.build_context import resolve_version
from repro.execution.lazy import ArtifactStep, StepContext
from repro.experiment.cli import build_options
from repro.experiment.namespacing import user_namespaced_name
from repro.experiment.overrides import Overrides
from repro.experiment.run_refs import RunRefError, collect_runs, resolve_run

EVAL_PREFIX = "eval"
USER_NAMESPACE = re.compile(r"^users/[^/]+/")


@dataclass(frozen=True)
class Checkpoint:
    """One permanent checkpoint of a run: the run's handle and the step."""

    run: ArtifactStep
    step: int


@dataclass(frozen=True)
class EvalRecipe:
    """What ``--set`` reaches: ``suite.FIELD``. The run and the step are not overridable."""

    suite: Any


# The batch size holds a placeholder string while the fingerprint is computed, which the tests'
# beartype hook would reject.
@no_type_check
@dataclass(frozen=True)
class EvaluationConfig:
    run: str
    """The run's output directory."""
    step: int
    suite: Any
    output_path: str
    batch_size: int | None


def _never_built(config: Any) -> None:
    raise RuntimeError("a finished run is pinned and never built")


def finished_run(ref: str) -> ArtifactStep:
    """A pinned handle to the finished run ``ref`` (``NAME@REF``) names, typed by its record."""
    found = resolve_run(settings.prefix(), ref, finished_only=True)
    record = found.record
    if record is None:
        raise RunRefError(f"{found.label} is built but has no record, so its checkpoint type is unknown")
    return ArtifactStep(
        name=record.name,
        version=record.version,
        artifact_type=artifact_class(record.result_type),
        run=_never_built,
        build_config=lambda ctx: record.config or {},
        override_path=found.path,
    )


def run_evaluation(config: EvaluationConfig) -> EvalResult:
    """Score one checkpoint with one suite, holding the accelerator."""
    finished = finished_run_at(config.run)
    with exclusive_accelerator(f"Evaluation of {finished.label} at step {config.step}"):
        return config.suite.evaluate(finished, config.step, config.output_path, EvalRuntime(config.batch_size))


def evaluate(
    checkpoint: Checkpoint,
    suite: Suite,
    *,
    version: str | None = None,
    overrides: Overrides = Overrides(),
    batch_size: int | None = None,
) -> ArtifactStep[EvalResult]:
    """The evaluation of ``checkpoint`` by ``suite``.

    ``version`` defaults to the ambient build context's. ``overrides`` reach the suite's fields
    (``suite.clips``) and join the suite's segment of the name. ``batch_size`` is the device batch
    size for a suite whose result does not depend on it; it stays out of the fingerprint.
    """
    run = checkpoint.run
    suites = offered_suites(run.label, run.artifact_type)
    if not isinstance(suite, suites.get(suite.name, ())):
        raise ValueError(f"{run.label}'s checkpoint type offers the suites {sorted(suites)}, not {suite.name!r}")
    base = f"{EVAL_PREFIX}/{USER_NAMESPACE.sub('', run.name)}/{run.version}"
    step = f"step-{checkpoint.step}"
    version = resolve_version(f"{base}/{suite.name}/{step}", version)

    def build_config(ctx: StepContext) -> EvaluationConfig:
        recipe = overrides.apply(ctx, EvalRecipe(suite=suite))
        return EvaluationConfig(
            run=ctx.artifact_path(run),
            step=checkpoint.step,
            suite=recipe.suite,
            output_path=ctx.output_path,
            batch_size=ctx.runtime_arg("batch_size"),
        )

    return ArtifactStep(
        name=user_namespaced_name(f"{base}/{overrides.named(suite.name)}/{step}", version),
        version=version,
        artifact_type=EvalResult,
        run=run_evaluation,
        build_config=build_config,
        deps=(run,),
        runtime_args={"batch_size": batch_size},
    )


def eval_results(prefix: str, *, run: str | None = None, suite: str | None = None) -> list[EvalResult]:
    """Every evaluation recorded under ``prefix``, for one run (its full ``name@version``) or one suite."""
    wanted = result_type_name(EvalResult)
    results = [
        EvalResult.raw_load(r.path)
        for r in collect_runs(prefix)
        if r.record is not None and r.record.result_type == wanted
    ]
    return [r for r in results if (run is None or r.run == run) and (suite is None or r.suite == suite)]


STEPS_ALL = "all"


def _steps(text: str, label: str, available: list[int]) -> list[int]:
    if text == STEPS_ALL:
        if not available:
            raise click.UsageError(f"{label} has no permanent checkpoints")
        return available
    try:
        wanted = [int(part) for part in text.split(",")]
    except ValueError as e:
        raise click.BadParameter(f"{text!r} is not comma-separated steps or 'all'", param_hint="--steps") from e
    missing = [step for step in wanted if step not in available]
    if missing:
        raise click.UsageError(
            f"{label} has no permanent checkpoint at step {missing}; its checkpoints are at steps {available}"
        )
    return wanted


def evaluation_matrix(
    refs: tuple[str, ...], *, suite_name: str, steps: str, batch_size: int | None, overrides: Overrides
) -> list[ArtifactStep[EvalResult]]:
    """One evaluation per run, step, and the trial's suite settings; refuses before anything runs."""
    cells = []
    for ref in refs:
        try:
            run = finished_run(ref)
            suites = offered_suites(run.label, run.artifact_type)
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        if suite_name not in suites:
            raise click.UsageError(
                f"{run.label}'s checkpoint type offers the suites {sorted(suites)}, not {suite_name!r}"
            )
        suite = suites[suite_name]()
        finished = finished_run_at(run.path())
        try:
            suite.check(finished, EvalRuntime(batch_size))
        except ValueError as e:
            raise click.UsageError(str(e)) from e
        provenance = finished.record.provenance
        click.echo(f"{run.label} trained at {provenance.label if provenance is not None else 'unrecorded code'}")
        for step in _steps(steps, run.label, checkpoint_steps(run.path())):
            cells.append(evaluate(Checkpoint(run, step), suite, overrides=overrides, batch_size=batch_size))
    return cells


@click.command()
@click.argument("refs", nargs=-1, required=True, metavar="RUN@REF...")
@click.option("--suite", "suite_name", required=True, help="The suite, by the name the run's checkpoint type gives it.")
@click.option(
    "--steps", required=True, help="Checkpoint steps, comma-separated, or 'all' for every permanent checkpoint."
)
@click.option(
    "--batch-size",
    type=click.IntRange(min=1),
    default=None,
    help="Device batch size, for a suite whose result does not depend on it.",
)
@build_options
def main(
    refs: tuple[str, ...], suite_name: str, steps: str, batch_size: int | None, overrides: Overrides
) -> list[ArtifactStep[EvalResult]]:
    """Evaluate finished runs' checkpoints with one suite: print the plan, or build it with --run."""
    return evaluation_matrix(refs, suite_name=suite_name, steps=steps, batch_size=batch_size, overrides=overrides)


if __name__ == "__main__":
    main()

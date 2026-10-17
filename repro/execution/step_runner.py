# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step runner for StepSpec.

``StepRunner`` is a thin DAG scheduler that launches steps as they are yielded
from an iterable, as soon as their dependencies are satisfied. Before launching a
step it checks the step's recorded recipe and serves a finished output from cache;
:func:`run_step` then runs it under the output's lock and writes its status.
"""

import contextvars
import logging
import time
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import timedelta

from repro.execution.artifact import (
    FingerprintMismatchError,
    drift_message,
    recorded_recipe,
    write_started_recipe,
)
from repro.execution.fingerprint import describe_drift
from repro.execution.step_spec import StepSpec
from repro.execution.step_status import STATUS_FAILED, STATUS_SUCCESS, StepAlreadyDone, read_status, step_lock

logger = logging.getLogger(__name__)

# Steps built at once when a caller does not say. Trainings still run one at a time, under
# repro.accelerator.exclusive_accelerator.
DEFAULT_MAX_CONCURRENT = 8


def check_drift(step: StepSpec) -> None:
    """Raise :class:`FingerprintMismatchError` when ``step``'s output records another recipe, in a
    finished build's record or a started build's recipe. A pinned step points at existing data and
    is not compared. :func:`repro.execution.lazy.run` checks the whole graph before the runner
    starts; this covers code that calls the runner directly."""
    if step.pinned:
        return
    recorded = recorded_recipe(step.output_path)
    if recorded is not None and recorded.fingerprint != step.fingerprint:
        change = describe_drift(recorded.payload, step.fingerprint_payload)
        raise FingerprintMismatchError(
            drift_message(step.name, step.version, recorded.fingerprint, step.fingerprint, change)
        )


def step_is_built(step: StepSpec) -> bool:
    """True when the runner will serve ``step`` from cache rather than rebuild it.

    A step that never rebuilds (a non-``dev`` version, or a pinned step) whose output
    already has a ``SUCCESS`` status. This is the same condition the launcher uses to
    skip a step, evaluated here at scheduling time so an already-built node's
    dependencies are pruned from the schedule instead of rebuilt; the launcher
    re-checks authoritatively before serving the node. An unpinned ``dev`` artifact
    always rebuilds, so it is never pruned.
    """
    return not step.rebuilds and read_status(step.output_path) == STATUS_SUCCESS


def steps_to_build(steps: Iterable[StepSpec]) -> list[StepSpec]:
    """The steps a :meth:`StepRunner.run` of ``steps`` would launch, in schedule order.

    A built step is served from its cache, and the inputs behind it are pruned from the schedule
    (:func:`step_is_built`), so neither is listed.
    """
    seen: set[str] = set()
    pruned: set[str] = set()
    scheduled = [s for step in steps for s in _expand_unseen(step, seen, pruned)]
    return [s for s in scheduled if s.output_path not in pruned]


def _expand_unseen(step: StepSpec, seen: set[str], pruned: set[str]) -> list[StepSpec]:
    """Return ``step`` and its transitive deps (post-order), excluding any in ``seen``.

    ``seen`` is mutated in place to include every returned step, so subsequent
    calls skip nodes already scheduled by an earlier terminal. Cycles within
    this expansion raise ``ValueError`` — by DAG construction they shouldn't
    exist, but the check guards against silent hangs.

    When ``node``'s output is already cached (:func:`step_is_built`), the walk
    records its ``output_path`` in ``pruned`` and does not descend into its deps,
    so they are neither scheduled nor rebuilt. ``node`` itself is still returned (as
    a cache-only leaf the caller confirms).
    """
    in_stack: set[str] = set()
    ordered: list[StepSpec] = []

    def visit(s: StepSpec) -> None:
        path = s.output_path
        if path in seen:
            return
        if path in in_stack:
            raise ValueError(f"Cycle detected in step graph involving {s.label}")
        in_stack.add(path)
        if step_is_built(s):
            pruned.add(path)
        else:
            for dep in s.deps:
                visit(dep)
        in_stack.remove(path)
        seen.add(path)
        ordered.append(s)

    visit(step)
    return ordered


class StepRunner:
    """Runs ``StepSpec`` objects respecting their dependencies.

    Steps are launched eagerly as they are yielded from an iterable. Each step
    is launched as soon as its dependencies are satisfied. Already-succeeded
    steps (STATUS_SUCCESS on disk) are skipped automatically.
    """

    def run(self, steps: Iterable[StepSpec], *, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> None:
        """Eagerly run steps, launching each as soon as its deps are satisfied.

        For each step pulled from the iterable, its unseen transitive deps are
        scheduled in post-order before the step itself (deduped by
        ``output_path`` across the whole run). Already-succeeded deps
        (``STATUS_SUCCESS`` on disk) resolve via the cache check.
        Concurrency is bounded by the thread pool (``max_concurrent`` workers).
        """
        # Make step progress visible by default. Idempotent and non-clobbering:
        # skipped when the driver (or a wrapping app) already installed handlers.
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

        local_pool = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="repro-step-runner")

        # Keyed by output_path (guaranteed unique)
        completed: set[str] = set()
        failed: set[str] = set()
        running: dict[str, Future] = {}
        waiting: list[StepSpec] = []
        failures: list[Exception] = []
        # output_path → name@version, for human-readable logging
        path_to_name: dict[str, str] = {}
        # Outputs already cached: their deps are pruned from the schedule and they are
        # confirmed from cache rather than run.
        pruned: set[str] = set()

        def _display_name(output_path: str) -> str:
            return path_to_name.get(output_path, output_path)

        def _harvest(block: bool = False) -> None:
            """Move finished steps to completed/failed; with ``block``, wait for at least one."""
            if not running:
                return
            if block:
                wait(running.values(), return_when=FIRST_COMPLETED)
            for path in [p for p, future in running.items() if future.done()]:
                future = running.pop(path)
                step_name = _display_name(path)
                exc = future.exception()
                if exc is None:
                    completed.add(path)
                    continue
                logger.error("Step failed: %s", step_name, exc_info=exc)
                failed.add(path)
                wrapped = RuntimeError(f"Step failed: {step_name}")
                wrapped.__cause__ = exc
                failures.append(wrapped)

        def _flush_waiting() -> None:
            """Launch buffered steps whose deps are now met."""
            i = 0
            while i < len(waiting):
                step = waiting[i]
                path = step.output_path
                if any(d in failed for d in step.dep_paths):
                    waiting.pop(i)
                    failed.add(path)
                elif all(d in completed for d in step.dep_paths):
                    waiting.pop(i)
                    _do_launch(step)
                else:
                    i += 1

        def _do_launch(step: StepSpec) -> None:
            path = step.output_path
            handle = self._launch_step(step, local_pool=local_pool)
            if handle is not None:
                running[path] = handle
            else:
                completed.add(path)

        def _confirm_pruned(step: StepSpec) -> None:
            """Mark a pruned cache-only leaf complete without running it.

            Its dependencies were dropped from the schedule, so the step must not run.
            If its cached output is no longer a clean success, record a failure (which
            cascades to dependents) so a rerun rebuilds it and its now-unpruned inputs.
            """
            path = step.output_path
            check_drift(step)
            status = read_status(path)
            if status == STATUS_SUCCESS and not step.rebuilds:
                logger.info(f"Skip {step.label}: already succeeded (pruned subtree)")
                completed.add(path)
            else:
                failed.add(path)
                failures.append(
                    RuntimeError(
                        f"Step {step.label}: cached output disappeared after its inputs were "
                        f"pruned (status={status}); rerun to rebuild it and its inputs."
                    )
                )

        scheduled: set[str] = set()
        for raw_step in steps:
            for step in _expand_unseen(raw_step, scheduled, pruned):
                path_to_name[step.output_path] = step.label

                _harvest()
                _flush_waiting()

                path = step.output_path
                if path in pruned:
                    _confirm_pruned(step)
                elif any(d in failed for d in step.dep_paths):
                    failed.add(path)
                elif all(d in completed for d in step.dep_paths):
                    _do_launch(step)
                else:
                    waiting.append(step)

        # Drain remaining running and waiting steps
        while running or waiting:
            _harvest(block=True)
            _flush_waiting()
        local_pool.shutdown(wait=True)

        if failures:
            raise RuntimeError(f"{len(failures)} step(s) failed") from failures[0]

    def _launch_step(self, step: StepSpec, *, local_pool: ThreadPoolExecutor) -> Future | None:
        """Launch a single step. Returns None if skipped."""
        output_path = step.output_path
        step_name = step.label
        logger.info(f"Step = {step_name}\tFingerprint = {step.fingerprint}\tOutput_path = {output_path}")

        # Raises if the output records another recipe.
        check_drift(step)
        if read_status(output_path) == STATUS_SUCCESS and not step.rebuilds:
            logger.info(f"Skip {step_name}: already succeeded")
            return None
        # The drift check just passed, so this records the same recipe or the first one.
        write_started_recipe(output_path, step.fingerprint, step.fingerprint_payload)

        # Steps see the caller's context variables.
        ctx = contextvars.copy_context()
        return local_pool.submit(ctx.run, run_step, step)


def run_step(step: StepSpec) -> None:
    """Run ``step.fn`` under the output's lock, with status writes.

    The lock re-reads the status once acquired, so a step another worker finished meanwhile is not
    run again; a mutable (``dev``) step rebuilds over a finished output.
    """
    try:
        with step_lock(step.output_path, step.label, force_rerun=step.rebuilds) as status_file:
            try:
                t0 = time.monotonic()
                step.fn(step.output_path)
                elapsed = timedelta(seconds=time.monotonic() - t0)
                status_file.write_status(STATUS_SUCCESS)
                logger.info(f"Step {step.label} succeeded in {elapsed}")
            except Exception:
                status_file.write_status(STATUS_FAILED)
                raise
    except StepAlreadyDone:
        logger.info(f"Step {step.label} completed by another worker")

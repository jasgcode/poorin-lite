# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Click bootstrap for experiment ``main``s that defer their artifact versions.

An experiment's driver constructs a handle graph and then either prints its plan or builds it.
This module makes that a one-liner while wiring up the deferred-version machinery
(:mod:`repro.execution.build_context`):

- :func:`build_options` is a Click decorator that adds ``--version`` (the run-wide default),
  ``--version-of NAME=VERSION`` (one artifact's version, repeatable), ``--set PATH=VALUE`` (one
  config field, repeatable), ``--sweep PATH=[V1, ...]`` (one trial per value, repeatable; several
  form a grid), ``--accept-drift NAME`` (keep an output whose recipe changed, repeatable),
  ``--run`` (build vs. print the plan), ``--show-config`` (the plan plus each returned artifact's
  identity config), ``--max-concurrent``, ``--note TEXT`` (labels the run's code), and
  ``--allow-uncommitted`` (build calendar artifacts from a snapshot). It builds the wrapped
  function's handle graph **inside** a :func:`~repro.execution.build_context.build_context`
  assembled from those options, so any builder that deferred its version (omitted ``version=``)
  resolves against them. It calls the wrapped function once per trial with an ``overrides`` keyword
  (:class:`~repro.experiment.overrides.Overrides`), which the experiment applies to its config.
- :func:`experiment_main` wraps a ``build(*, overrides)`` function into a ready-to-run command.

``--version`` is required — there is no silent ``dev`` default, because a deferred *dataset* at a
mutable version rebuilds its (often multi-TB) cache on every run and, unlike a training checkpoint,
is not namespaced per user. To iterate, pass ``--version dev``: each invocation gets a fresh,
suffixed version, and ``--resume <version or token>`` continues an earlier one.

Before ``--run`` builds, the code is checked (:mod:`repro.experiment.code_check`): a calendar-
versioned artifact builds only from committed code, and a run that builds only ``dev`` artifacts
saves uncommitted changes as a snapshot commit under ``refs/repro/snapshots/``. The plan prints the
code's one-line label and what ``--run`` would do.

The printed plan gives each artifact's state at its output path (see
:mod:`repro.execution.cache_state`), so an artifact that resolved to a mutable version and will
rebuild shows as ``mutable``. An artifact whose output records another recipe shows as ``built from
another recipe`` with the changed fields, and ``--run`` refuses it unless ``--accept-drift`` names
it.

An experiment with its own options composes with :func:`build_options` by stacking it *below* the
experiment's own ``@click.option``s and ``@click.command()`` on top::

    @click.command()
    @click.option("--device", ...)
    @build_options
    def main(device, overrides):
        return build(device=device, overrides=overrides)   # version deferred to --version / --version-of

    if __name__ == "__main__":
        main()

Construct the whole handle graph inside ``build()``; a builder invoked lazily after it returns has
no active context and raises. Handles are frozen at construction, so reusing one across two
invocations keeps its first resolved version.
"""

import dataclasses
import functools
import itertools
import json
import secrets
import textwrap
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

import click
import yaml
from draccus.cfgparsing import parse_string

from repro import settings
from repro.execution.artifact import (
    DriftAcceptanceError,
    FingerprintMismatchError,
    is_mutable_version,
    validate_version,
)
from repro.execution.build_context import VersionCodex, build_context
from repro.execution.cache_state import HandleState, check_accept_drift, graph_states
from repro.execution.lazy import ArtifactStep, graph_handles, lower_all, run
from repro.execution.step_runner import DEFAULT_MAX_CONCURRENT, steps_to_build
from repro.experiment.code_check import SHOWN_HASH_LENGTH, CodeVerdict, check_code, code_line, plan_lines, refusal
from repro.experiment.overrides import Override, OverrideError, Overrides
from repro.experiment.run_refs import RunRefError, resume_version
from repro.provenance import Provenance, launched_as, save_snapshot

BuildResult = ArtifactStep | Mapping[str, ArtifactStep] | list[ArtifactStep] | tuple[ArtifactStep, ...]
BuildRunner = Callable[[list[ArtifactStep], int, frozenset[str]], None]


def _version_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Reject a malformed ``--version`` at parse time, as a Click error rather than a traceback."""
    if value is None:
        return None
    try:
        validate_version(value)
    except ValueError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e
    return value


def _resume_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
    """Accept a dev version, or the token of one (``60b9``): calendar versions need no resume."""
    if value is None:
        return None
    try:
        value = resume_version(settings.prefix(), value)
    except RunRefError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e
    value = _version_callback(ctx, param, value)
    if not is_mutable_version(value):
        raise click.BadParameter(
            f"{value!r} is a calendar version; --resume continues a '-dev' run", ctx=ctx, param=param
        )
    return value


def _version_of_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``NAME=VERSION`` into a validated ``{name: version}`` map, as Click errors."""
    versions: dict[str, str] = {}
    for item in value:
        name, sep, version = item.partition("=")
        if not sep or not name:
            raise click.BadParameter(f"{item!r} is not NAME=VERSION", ctx=ctx, param=param)
        if name in versions:
            raise click.BadParameter(f"duplicate --version-of for {name!r}", ctx=ctx, param=param)
        try:
            validate_version(version)
        except ValueError as e:
            raise click.BadParameter(f"--version-of {name!r}: {e}", ctx=ctx, param=param) from e
        versions[name] = version
    return versions


def _set_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> list[Override]:
    try:
        return [Override.parse(item) for item in value]
    except OverrideError as e:
        raise click.BadParameter(str(e), ctx=ctx, param=param) from e


def _reads_back_as_itself(text: str) -> bool:
    """Whether YAML reads ``text`` as the string ``text`` (``1e-3`` does; ``yes`` and ``a: b: c`` do not)."""
    try:
        return parse_string(text) == text
    except yaml.YAMLError:
        return False


def _sweep_callback(ctx: click.Context, param: click.Parameter, value: tuple[str, ...]) -> dict[str, list[str]]:
    """Parse repeated ``PATH=[V1, V2, ...]`` into ``{path: [value text, ...]}``.

    Each value text is decoded later like a ``--set`` value. A string element keeps its raw text
    only when YAML reads that text back as the same string, so ``1e-3`` stays text and decodes to
    0.001 for a float field. Every other element (a quoted string such as ``'yes'`` or ``'a: b'``,
    whose text YAML reads as another value or cannot read, a number, a mapping) is written back as
    JSON, which YAML reads back as the same value.
    """
    axes: dict[str, list[str]] = {}
    for item in value:
        path, sep, text = item.partition("=")
        try:
            values = parse_string(text) if sep else None
        except yaml.YAMLError as e:
            raise click.BadParameter(f"{item!r} is not YAML: {e}", ctx=ctx, param=param) from e
        if not isinstance(values, list) or not values:
            raise click.BadParameter(f"{item!r} is not PATH=[V1, V2, ...]", ctx=ctx, param=param)
        if path in axes:
            raise click.BadParameter(f"{path!r} is swept twice", ctx=ctx, param=param)
        axes[path] = [v if isinstance(v, str) and _reads_back_as_itself(v) else json.dumps(v) for v in values]
    return axes


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """The Cartesian product of named axes, as a list of parameter dicts, the last axis varying
    fastest: ``grid(lr=[1, 2], wd=[0])`` is ``[{"lr": 1, "wd": 0}, {"lr": 2, "wd": 0}]``."""
    keys = list(axes)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*axes.values())]


def run_version(version: str, *, now: datetime, token: str) -> str:
    """The version a run uses for ``--version``.

    A mutable version (``dev`` or ``<label>-dev``) gets a run suffix so each invocation writes to
    a fresh output path: ``dev`` becomes ``20261005-091500-a1b2-dev`` and ``probe-dev`` becomes
    ``probe-20261005-091500-a1b2-dev``. Without the suffix, a rerun lands on the previous run's
    ``{name}/dev`` path and a checkpointer resumes from its last checkpoint instead of
    training from scratch. A calendar version is returned unchanged. ``--resume`` reuses a
    suffixed version to continue that run.
    """
    if not is_mutable_version(version):
        return version
    label = version.removesuffix("dev").rstrip("-")
    stamped = f"{now:%Y%m%d-%H%M%S}-{token}"
    return f"{label}-{stamped}-dev" if label else f"{stamped}-dev"


def _as_handles(result: BuildResult) -> list[ArtifactStep]:
    """Normalize a ``build()`` return — one handle, a list/tuple, or a name→handle mapping."""
    if isinstance(result, ArtifactStep):
        return [result]
    if isinstance(result, Mapping):
        return list(result.values())
    if isinstance(result, list | tuple):
        return list(result)
    raise TypeError(
        f"build() must return an ArtifactStep, a list/tuple, or a mapping of them, not {type(result).__name__}"
    )


def _print_plan(states: list[HandleState], shown: list[HandleState], code: list[str]) -> None:
    """The prefix, the code the run would use, one line per artifact (state, fingerprint,
    ``name@version``) with the diff under a drifted one, then the identity config of each artifact
    in ``shown`` as YAML."""
    click.echo(f"Prefix: {settings.prefix()}")
    for line in code:
        click.echo(line)
    for state in states:
        click.echo(f"  {state.state.value:<26} {state.fingerprint}  {state.label}")
        if state.change:
            click.echo(textwrap.indent(state.change.strip(), "      "))
    for state in shown:
        config = json.loads(state.handle.fingerprint_payload())
        click.echo(f"\n# {state.label}\n{yaml.safe_dump(config, sort_keys=True).rstrip()}")


def _run_locally(handles: list[ArtifactStep], max_concurrent: int, accept_drift: frozenset[str]) -> None:
    run(*handles, max_concurrent=max_concurrent, accept_drift=accept_drift)


def build_options_with_runner(fn: Callable[..., BuildResult], runner: BuildRunner) -> Callable[..., None]:
    """Add shared build options and delegate ``--run`` to a domain-specific runner."""

    @click.option(
        "--version",
        "default_version",
        default=None,
        callback=_version_callback,
        help="Run-wide default version for artifacts that defer (omit version=). A calendar "
        "version YYYY.MM.DD, or 'dev' / '<label>-dev' for a fresh run with a unique suffix.",
    )
    @click.option(
        "--resume",
        "resume_version",
        default=None,
        callback=_resume_callback,
        metavar="VERSION",
        help="Continue an earlier dev run by its version, as printed when it started, or the 4-hex token in it.",
    )
    @click.option(
        "--version-of",
        "versions_of",
        multiple=True,
        metavar="NAME=VERSION",
        callback=_version_of_callback,
        help="The version of one artifact, by the name its builder passes. Repeatable.",
    )
    @click.option(
        "--set",
        "sets",
        multiple=True,
        metavar="PATH=VALUE",
        callback=_set_callback,
        help="Override one field of the experiment's config by dotted path; VALUE is YAML. Repeatable.",
    )
    @click.option(
        "--sweep",
        "sweeps",
        multiple=True,
        metavar="PATH=[V1, ...]",
        callback=_sweep_callback,
        help="One trial per value, as if --set PATH=Vi; several form a grid. Repeatable.",
    )
    @click.option(
        "--accept-drift",
        "accept_drift",
        multiple=True,
        metavar="NAME",
        help="Keep the output of the artifact NAME although its recipe changed, and record the change. Repeatable.",
    )
    @click.option("--run", "do_run", is_flag=True, help="Build the handles (default: print the plan).")
    @click.option(
        "--show-config",
        is_flag=True,
        help="Print the plan and each returned artifact's identity config as YAML; run nothing.",
    )
    @click.option(
        "--max-concurrent",
        type=click.IntRange(min=1),
        default=DEFAULT_MAX_CONCURRENT,
        show_default=True,
        help="Max steps built concurrently. Trainings run one at a time.",
    )
    @click.option(
        "--note",
        default=None,
        metavar="TEXT",
        help="What this run tries, in a few words. It labels the run's code in place of the commit subject.",
    )
    @click.option(
        "--allow-uncommitted",
        is_flag=True,
        help="Build calendar-versioned artifacts from uncommitted code, saved as a snapshot commit.",
    )
    @functools.wraps(fn)
    def wrapper(
        *args: Any,
        default_version: str | None,
        resume_version: str | None,
        versions_of: dict[str, str],
        sets: list[Override],
        sweeps: dict[str, list[str]],
        accept_drift: tuple[str, ...],
        do_run: bool,
        show_config: bool,
        max_concurrent: int,
        note: str | None,
        allow_uncommitted: bool,
        **kwargs: Any,
    ) -> None:
        if (default_version is None) == (resume_version is None):
            raise click.UsageError("pass exactly one of --version or --resume")
        if resume_version is not None:
            version = resume_version
        else:
            assert default_version is not None
            version = run_version(default_version, now=datetime.now(), token=secrets.token_hex(2))
        if is_mutable_version(version):
            click.echo(f"Run version: {version} (continue this run with --resume {version})")
        points = grid(**sweeps) if sweeps else [{}]
        codex = VersionCodex(default=version, by_name=versions_of)
        try:
            trials = [
                Overrides((*sets, *(Override.parse(f"{path}={text}") for path, text in point.items())))
                for point in points
            ]
            with build_context(codex):
                handles = [h for trial in trials for h in _as_handles(fn(*args, overrides=trial, **kwargs))]
            graph = graph_handles(handles)
            specs = lower_all(handles)  # applies every override for the checks below; reads nothing
        except OverrideError as e:
            raise click.UsageError(str(e)) from e
        unused = codex.unused_names()
        if unused:
            raise click.BadParameter(
                f"never applied to any artifact: {sorted(unused)}. The name is a typo, or that "
                "artifact hardcodes its version (an explicit version wins).",
                param_hint="--version-of",
            )
        unapplied = sorted({path for trial in trials for path in trial.unapplied()})
        if unapplied:
            raise click.BadParameter(
                f"never applied: {unapplied}. This experiment does not apply its overrides to its config.",
                param_hint="--set/--sweep",
            )
        if show_config and do_run:
            raise click.UsageError("--show-config prints the plan; it cannot be combined with --run")
        launch = dataclasses.replace(Provenance.capture(), note=note)
        check = check_code(launch, steps_to_build(specs), allow_uncommitted=allow_uncommitted)
        try:
            # A run reads every artifact's recorded recipe itself (lazy.run); only the plan and a
            # resume read them here.
            if not do_run or resume_version is not None:
                states = graph_states(graph)
                returned = {h.label for h in handles}
                if resume_version is not None:
                    empty = [s.label for s in states if s.label in returned and s.recorded_fingerprint is None]
                    if empty:
                        raise click.UsageError(
                            f"nothing to resume at {empty}: no run started there. "
                            "Check that the --set flags match the original run's."
                        )
                if not do_run:
                    check_accept_drift(states, accept_drift)  # as --run would, so a typo shows in the plan
                    shown = [s for s in states if s.label in returned] if show_config else []
                    _print_plan(states, shown, plan_lines(check))
                    return
            if check.verdict is CodeVerdict.REFUSED:
                raise click.ClickException(refusal(check))
            if check.verdict is CodeVerdict.SNAPSHOT:
                snapshot = save_snapshot(launch.note or launch.change_summary)
                launch = dataclasses.replace(launch, snapshot_commit=snapshot)
            click.echo(code_line(launch))
            with launched_as(launch):
                runner(handles, max_concurrent, frozenset(accept_drift))
            if launch.snapshot_commit is not None:
                click.echo(f"See changes: git diff {launch.base_commit} {launch.snapshot_commit[:SHOWN_HASH_LENGTH]}")
        except (FingerprintMismatchError, DriftAcceptanceError) as e:
            raise click.ClickException(str(e)) from e

    return wrapper


def build_options(fn: Callable[..., BuildResult]) -> Callable[..., None]:
    """Add the shared version, override, and run options to a command that returns the experiment's handle(s).

    ``fn`` returns one :class:`~repro.execution.lazy.ArtifactStep`, a list/tuple of them, or a
    name→handle mapping; it is called inside a :func:`~repro.execution.build_context.build_context`
    built from ``--version`` and ``--version-of``, once per trial, with an ``overrides`` keyword
    holding that trial's ``--set`` and ``--sweep`` values. Without ``--run`` the plan is printed, one
    line per artifact with its cache state; with it the handles are built. A ``--version-of`` naming
    an artifact that never deferred (a typo, or one that hardcodes its version) is rejected, and so
    is a ``--set`` or ``--sweep`` path the experiment never applied, so neither can slip a run under
    the wrong version or config.
    """

    return build_options_with_runner(fn, _run_locally)


def experiment_main(build: Callable[..., BuildResult]) -> Callable[..., None]:
    """Turn ``build`` into a ready-to-run experiment command with the shared options.

    ``build`` takes an ``overrides`` keyword, which it applies to its config.
    ``experiment_main(build)()`` is the whole ``if __name__ == "__main__":`` body for an experiment
    that takes no options of its own. For one that does, use :func:`build_options` directly (see the
    module docstring for the stacking order).
    """
    return click.command()(build_options(build))

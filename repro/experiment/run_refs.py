# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The runs under the prefix, and short references to one of them.

``NAME@REF`` names a run wherever one is typed (``evaluate``, ``runs show``):

- a full version names itself;
- a 4-hex token (``60b9``) names the dev run whose version it suffixes;
- ``latest`` names the newest dev run of ``NAME``;
- anything else is a label: ``accept`` names the newest ``accept-<stamp>-<token>-dev`` run.

"Newest" orders by the timestamp a dev version carries (:func:`repro.experiment.cli.run_version`).
``NAME`` matches a run's name with or without the user's ``users/<you>/`` namespace. References
resolve once, at the command line; names, fingerprints, and records hold only full versions.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath

from repro.execution.artifact import ArtifactRecord, read_record
from repro.execution.step_status import STATUS_FAILED, STATUS_FILENAME, STATUS_SUCCESS, read_status
from repro.provenance import username_segment

# Prefix directories that hold scratch space, not artifacts.
SCRATCH_DIRS = frozenset({"tmp"})
# A suffixed dev version: [label-]YYYYMMDD-HHMMSS-<token>-dev (repro.experiment.cli.run_version).
STAMPED_VERSION = re.compile(r"^(?:(?P<label>.+)-)?(?P<stamp>\d{8}-\d{6})-(?P<token>[0-9a-f]{4})-dev$")
TOKEN = re.compile(r"^[0-9a-f]{4}$")
LATEST = "latest"
# How many of a name's versions an unknown reference lists.
LISTED_VERSIONS = 10


class RunState(StrEnum):
    BUILT = "built"
    STARTED = "started"
    FAILED = "failed"


@dataclass(frozen=True)
class Run:
    """One output under the prefix."""

    label: str
    """``name@version``."""
    path: str
    state: RunState
    record: ArtifactRecord | None
    """``None`` until the build finishes."""
    when: datetime


def run_state(output_path: str) -> RunState:
    status = read_status(output_path)
    if status == STATUS_SUCCESS:
        return RunState.BUILT
    if status == STATUS_FAILED:
        return RunState.FAILED
    return RunState.STARTED


def _output_dirs(directory: Path, top: bool) -> Iterator[Path]:
    """Directories under ``directory`` that hold an output; an output holds no other output."""
    if (directory / STATUS_FILENAME).exists():
        yield directory
        return
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and not (top and entry.name in SCRATCH_DIRS):
            yield from _output_dirs(entry, top=False)


def collect_runs(prefix: str) -> list[Run]:
    """Every output under ``prefix``, newest first."""
    root = Path(prefix)
    if not root.exists():
        return []
    runs = []
    for directory in _output_dirs(root, top=True):
        relative = PurePosixPath(directory.relative_to(root).as_posix())
        path = f"{prefix.rstrip('/')}/{relative}"
        record = read_record(path)
        created = record.provenance.created_at if record is not None and record.provenance is not None else ""
        when = (
            datetime.fromisoformat(created)
            if created
            else datetime.fromtimestamp((directory / STATUS_FILENAME).stat().st_mtime)
        )
        runs.append(Run(f"{relative.parent}@{relative.name}", path, run_state(path), record, when))
    return sorted(runs, key=lambda r: r.when, reverse=True)


class RunRefError(ValueError):
    """A run reference that names no run, or several."""


@dataclass(frozen=True)
class StampedVersion:
    label: str | None
    stamp: str
    token: str


def stamped(version: str) -> StampedVersion | None:
    """The parts of a suffixed dev version, or ``None`` for any other version."""
    match = STAMPED_VERSION.match(version)
    return None if match is None else StampedVersion(match["label"], match["stamp"], match["token"])


def _split(label: str) -> tuple[str, str]:
    name, _, version = label.rpartition("@")
    return name, version


def _newest(runs: list[tuple[Run, StampedVersion]]) -> list[Run]:
    if not runs:
        return []
    return [max(runs, key=lambda pair: pair[1].stamp)[0]]


def resolve_run(prefix: str, ref: str, *, finished_only: bool) -> Run:
    """The run ``ref`` (``NAME@REF``) names under ``prefix``; only built runs when ``finished_only``."""
    name, sep, wanted = ref.rpartition("@")
    if not sep or not name or not wanted:
        raise RunRefError(f"{ref!r} is not NAME@REF")
    names = {name, f"users/{username_segment()}/{name}"}
    runs = [r for r in collect_runs(prefix) if _split(r.label)[0] in names]
    if finished_only:
        runs = [r for r in runs if r.state is RunState.BUILT]
    exact = [r for r in runs if _split(r.label)[1] == wanted]
    if exact:
        return exact[0]
    dev = [(r, s) for r in runs if (s := stamped(_split(r.label)[1])) is not None]
    if TOKEN.match(wanted):
        matches = [r for r, s in dev if s.token == wanted]
    elif wanted == LATEST:
        matches = _newest(dev)
    else:
        matches = _newest([(r, s) for r, s in dev if s.label == wanted])
    kind = "finished run" if finished_only else "run"
    if not matches:
        versions = sorted((_split(r.label)[1] for r in runs), reverse=True)[:LISTED_VERSIONS]
        raise RunRefError(f"no {kind} {ref}; {name} has the {kind}s {versions}")
    if len(matches) > 1:
        raise RunRefError(f"{ref} names several {kind}s, so pass the full version: {sorted(r.label for r in matches)}")
    return matches[0]


def resume_version(prefix: str, value: str) -> str:
    """The version ``--resume VALUE`` continues: ``VALUE`` itself, or the dev version whose token it is.

    ``--resume`` takes no name, so a token is matched against every dev run under ``prefix``; a
    sweep's trials share one version, so several runs may carry it.
    """
    if not TOKEN.match(value):
        return value
    versions = set()
    for run in collect_runs(prefix):
        version = _split(run.label)[1]
        parts = stamped(version)
        if parts is not None and parts.token == value:
            versions.add(version)
    if not versions:
        raise RunRefError(f"no dev run under {prefix} has the token {value}")
    if len(versions) > 1:
        raise RunRefError(f"the token {value} suffixes several versions, so pass one: {sorted(versions)}")
    return versions.pop()

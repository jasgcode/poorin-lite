# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""List the runs under the prefix, show the code behind one, and migrate one's record.

    uv run python -m repro.experiment.runs list [FILTER] [--experiment SLUG] [--limit N]
    uv run python -m repro.experiment.runs show NAME@REF | PATH
    uv run python -m repro.experiment.runs migrate NAME@REF --to MODULE.CLASS [--drop PATH ...]

``list`` and ``show`` read what the runner writes beside each output, the record (``.artifact.json``) and the
status file, so they need neither a tracker nor the experiment's code. Git, when the commit is in this
repository, adds what a record does not hold: how far its branch has moved since, a snapshot's line
counts, and the subject of a commit recorded before subjects were.
"""

import copy
import dataclasses
import shlex
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath

import click

from repro import settings
from repro.evaluation.contract import EvaluableCheckpoint, artifact_class, offered_suites
from repro.execution.artifact import ArtifactRecord, read_record, write_record
from repro.execution.step_status import STATUS_FILENAME
from repro.experiment.code_check import SHOWN_HASH_LENGTH
from repro.experiment.run_refs import Run, RunRefError, RunState, collect_runs, resolve_run, run_state
from repro.provenance import Provenance, read_git

DEFAULT_LIMIT = 30
# Continuation lines of `show` start under its values.
SHOW_INDENT = " " * 15
NO_CODE = "-"


# What `show` says in its Code line for an output whose record is missing, by the output's state.
NO_RECORD_REASONS = {
    RunState.STARTED: "not recorded yet: the record is written when the build finishes",
    RunState.FAILED: "not recorded: the build failed before writing its record",
    RunState.BUILT: "not recorded: the output has no record",
}
# ... and for a record written before provenance existed, or a legacy sidecar.
NO_PROVENANCE_REASON = "not recorded: the output predates code provenance"


def _commit(rev: str) -> str:
    """The full hash of the commit ``rev`` names when it is in this repository, else "".

    A record's commits come from a file, so every later Git call takes this hash, never the recorded
    value; ``--end-of-options`` stops a value that starts with "-" from being read as an option here.
    """
    return read_git(["rev-parse", "--verify", "--quiet", "--end-of-options", f"{rev}^{{commit}}"]).strip()


def _with_subject(provenance: Provenance) -> Provenance:
    """``provenance`` with its commit's subject looked up when the record predates subjects."""
    if provenance.commit_subject or not provenance.base_commit:
        return provenance
    full = _commit(provenance.base_commit)
    subject = read_git(["log", "-1", "--no-show-signature", "--format=%s", full]).strip() if full else ""
    return dataclasses.replace(provenance, commit_subject=subject)


def _code_label(run: Run) -> str:
    if run.record is None or run.record.provenance is None:
        return NO_CODE
    return _with_subject(run.record.provenance).label


def _time(iso: str) -> str:
    return f"{datetime.fromisoformat(iso):%Y-%m-%d %H:%M}" if iso else "unknown"


def _commit_lines(provenance: Provenance, full: str) -> list[str]:
    when = f", {_time(provenance.commit_time)}" if provenance.commit_time else ""
    branch = provenance.branch or "a detached HEAD"
    lines = [f"  Commit       {provenance.base_commit} on {branch}{when}"]
    if provenance.commit_subject:
        lines.append(f"{SHOW_INDENT}{provenance.commit_subject}")
    if not full:
        lines.append(f"{SHOW_INDENT}not in this repository")
        return lines
    if provenance.branch:
        moved = read_git(["rev-list", "--count", f"{full}..{provenance.branch}"]).strip()
        if moved == "0":
            lines.append(f"{SHOW_INDENT}{provenance.branch} has not moved since")
        elif moved:
            noun = "commit" if moved == "1" else "commits"
            lines.append(f"{SHOW_INDENT}{provenance.branch} has moved {moved} {noun} past it")
    return lines


def _line_counts(base: str, snapshot: str) -> dict[str, str]:
    """Per-path line counts of the snapshot against its base, both full hashes."""
    counts = {}
    for row in read_git(["diff", "--numstat", base, snapshot]).splitlines():
        added, removed, path = row.split("\t", 2)
        counts[path] = "binary" if added == "-" else f"+{added} -{removed}" if removed != "0" else f"+{added}"
    return counts


def _uncommitted_lines(provenance: Provenance, full: str) -> list[str]:
    snapshot = provenance.snapshot_commit
    kept = _commit(snapshot) if snapshot else ""
    count = len(provenance.uncommitted)
    files = f"{count} file{'' if count == 1 else 's'}" if count else "changes"
    if not snapshot:
        saved = "not saved"
    else:
        saved = f"saved as snapshot {snapshot[:SHOWN_HASH_LENGTH]}" + ("" if kept else " (not in this repository)")
    lines = [f"  Uncommitted  {files}, {saved}"]
    counts = _line_counts(full, kept) if full and kept else {}
    width = max((len(entry) - 3 for entry in provenance.uncommitted), default=0)
    for entry in provenance.uncommitted:
        status, path = entry[:2].strip(), entry[3:]
        lines.append(f"{SHOW_INDENT}  {status:<3} {path:<{width}}  {counts.get(path, '')}".rstrip())
    # The commands print the hashes Git verified, never the record's text.
    short = kept[:SHOWN_HASH_LENGTH]
    if full and kept:
        lines.append(f"  See changes  git diff {full[:SHOWN_HASH_LENGTH]} {short}")
    if kept:
        lines.append(f"  Restore      git worktree add ../run-{short} {short}")
    return lines


def describe(run: Run) -> list[str]:
    """Everything the record and Git say about one run's code."""
    lines = [f"{run.label}   {run.state}"]
    if run.record is None or run.record.provenance is None:
        reason = NO_RECORD_REASONS[run.state] if run.record is None else NO_PROVENANCE_REASON
        lines.append(f"  Code         {reason}")
        return lines
    provenance = _with_subject(run.record.provenance)
    lines.append(f"  Built        {_time(provenance.created_at)} by {provenance.built_by or 'unknown'}")
    if provenance.command_line:
        lines.append(f"  Command      {shlex.join(provenance.command_line)}")
    if provenance.note:
        lines.append(f"  Note         {provenance.note}")
    if not provenance.base_commit:
        lines.append("  Commit       none: not built from a Git checkout")
    else:
        full = _commit(provenance.base_commit)
        lines.extend(_commit_lines(provenance, full))
        if provenance.dirty:
            lines.extend(_uncommitted_lines(provenance, full))
    if run.record.deps:
        lines.append(f"  Inputs       {', '.join(run.record.deps)}")
    return lines


@click.group()
def main() -> None:
    """List the runs under the prefix, and show the code behind one."""


def _experiment_of(run: Run) -> str | None:
    """The experiment a finished run was launched for; ``None`` for one with no record or provenance yet."""
    if run.record is None or run.record.provenance is None:
        return None
    return run.record.provenance.experiment


@main.command("list")
@click.argument("name_filter", required=False, default="")
@click.option("--limit", type=click.IntRange(min=1), default=DEFAULT_LIMIT, show_default=True)
@click.option(
    "--experiment",
    default=None,
    metavar="SLUG",
    help="Keep only finished runs launched with --experiment SLUG.",
)
@click.option("--prefix", default=None, help="Read this prefix instead of REPRO_PREFIX.")
def list_runs(name_filter: str, limit: int, experiment: str | None, prefix: str | None) -> None:
    """One line per run, newest first; FILTER keeps runs whose name contains it."""
    runs = [
        r
        for r in collect_runs(prefix or settings.prefix())
        if name_filter in r.label and (experiment is None or _experiment_of(r) == experiment)
    ][:limit]
    width = max((len(r.label) for r in runs), default=0)
    for r in runs:
        click.echo(f"{r.when:%Y-%m-%d %H:%M}  {r.state:<7}  {r.label:<{width}}  {_code_label(r)}")


@main.command()
@click.argument("target")
@click.option("--prefix", default=None, help="Read this prefix instead of REPRO_PREFIX.")
def show(target: str, prefix: str | None) -> None:
    """Everything about one run's code.

    TARGET is NAME@REF (a version, a token, a label, or latest) or the output's path.
    """
    if "@" in target:
        try:
            found = resolve_run(prefix or settings.prefix(), target, finished_only=False)
        except RunRefError as e:
            raise click.ClickException(str(e)) from e
        click.echo("\n".join(describe(found)))
        return
    path = target.rstrip("/")
    record = read_record(path)
    if record is None and not (Path(path) / STATUS_FILENAME).exists():
        raise click.ClickException(f"no run at {path}")
    relative = PurePosixPath(path)
    label = f"{relative.parent.name}@{relative.name}"
    click.echo("\n".join(describe(Run(label, path, run_state(path), record, datetime.now()))))


def _offers_suites(result_type: str) -> bool:
    """Whether the recorded type imports and names suites; a renamed or suite-less type does not."""
    try:
        return issubclass(artifact_class(result_type), EvaluableCheckpoint)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def migrated(record: ArtifactRecord, checkpoint_type: str, dropped: Sequence[str]) -> ArtifactRecord:
    """``record`` naming ``checkpoint_type``, without the config fields at the dotted paths ``dropped``.

    The fingerprint and its payload stay as recorded: they describe what ran.
    """
    label = f"{record.name}@{record.version}"
    if record.result_type != checkpoint_type and _offers_suites(record.result_type):
        raise ValueError(
            f"{label} records {record.result_type}, which already offers suites; only a run whose recorded type "
            "offers none, or no longer imports, migrates to another type"
        )
    config = copy.deepcopy(record.config or {})
    for path in dropped:
        *parents, leaf = path.split(".")
        node = config
        for key in parents:
            node = node.get(key) if isinstance(node, dict) else None
        if not isinstance(node, dict) or leaf not in node:
            raise ValueError(f"{label} records no config field {path!r}")
        del node[leaf]
    return record.model_copy(update={"result_type": checkpoint_type, "config": config})


@main.command()
@click.argument("target")
@click.option("--to", "checkpoint_type", required=True, metavar="MODULE.CLASS", help="The template's checkpoint type.")
@click.option(
    "--drop",
    "dropped",
    multiple=True,
    metavar="PATH",
    help="A recorded config field the code moved out of the recipe. Repeatable.",
)
@click.option("--prefix", default=None, help="Read this prefix instead of REPRO_PREFIX.")
def migrate(target: str, checkpoint_type: str, dropped: tuple[str, ...], prefix: str | None) -> None:
    """Name the checkpoint type of a run whose recorded type offers no suites, once."""
    try:
        cls = artifact_class(checkpoint_type)
    except (ImportError, AttributeError, TypeError, ValueError) as e:
        raise click.ClickException(f"--to {checkpoint_type!r} is not an Artifact class: {e}") from e
    try:
        found = resolve_run(prefix or settings.prefix(), target, finished_only=True)
        offered_suites(found.label, cls)
        if found.record is None:
            raise ValueError(f"{found.label} has no record")
        write_record(migrated(found.record, checkpoint_type, dropped))
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"{found.label} now records {checkpoint_type}" + (f", without {', '.join(dropped)}" if dropped else ""))


if __name__ == "__main__":
    main()

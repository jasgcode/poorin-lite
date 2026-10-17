# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether ``--run`` may build from the working tree's code, and the lines that report it.

A calendar-versioned artifact is a result to cite, so it is built only from committed code. A run
that builds only ``dev`` artifacts may use uncommitted code, which the experiment command line saves
as a snapshot commit (:func:`repro.provenance.save_snapshot`). ``--allow-uncommitted`` lets a
calendar artifact build from a snapshot too.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from repro.execution.step_spec import StepSpec
from repro.provenance import Provenance

# The most uncommitted paths a refusal lists.
REFUSAL_PATHS = 10
# Aligns a continuation line under the plan's "Code:   " label.
PLAN_INDENT = " " * 8
# Hash characters a printed Git command uses: unambiguous in a repository of any size.
SHOWN_HASH_LENGTH = 12
# What to do about a refusal. The refused run may itself be a dev run whose calendar input would
# build, so neither suggests a dev version.
SNAPSHOT_ADVICE = "Commit the changes and run again, or pass --allow-uncommitted to build from a snapshot."
NO_CHECKOUT_ADVICE = "Run from a Git checkout, or pass --allow-uncommitted to build without recording the code."


class CodeVerdict(StrEnum):
    COMMITTED = "committed"
    """A clean checkout: the record names the commit."""
    CACHED = "cached"
    """Nothing would build: every artifact is served from its cache."""
    SNAPSHOT = "snapshot"
    """Uncommitted code may run, and is saved as a snapshot commit."""
    REFUSED = "refused"
    """Uncommitted or unknown code would build a calendar-versioned artifact."""
    UNRECORDED = "unrecorded"
    """No Git checkout: dev artifacts build, and no code is recorded."""


@dataclass(frozen=True)
class CodeCheck:
    provenance: Provenance
    calendar: tuple[str, ...]
    """``name@version`` of each calendar-versioned artifact the run would build."""
    verdict: CodeVerdict


def check_code(provenance: Provenance, builds: Sequence[StepSpec], *, allow_uncommitted: bool) -> CodeCheck:
    """Whether a run that launches ``builds`` (:func:`~repro.execution.step_runner.steps_to_build`)
    may build from the code ``provenance`` describes."""
    calendar = tuple(step.label for step in builds if not step.pinned and not step.rebuilds)
    checkout = bool(provenance.base_commit)
    if checkout and not provenance.dirty:
        verdict = CodeVerdict.COMMITTED
    elif not builds:
        verdict = CodeVerdict.CACHED
    elif calendar and not allow_uncommitted:
        verdict = CodeVerdict.REFUSED
    else:
        verdict = CodeVerdict.SNAPSHOT if checkout else CodeVerdict.UNRECORDED
    return CodeCheck(provenance, calendar, verdict)


def _artifacts(count: int) -> str:
    return f"{count} calendar-versioned artifact{'' if count == 1 else 's'}"


def _source(check: CodeCheck) -> str:
    """Where the refused build's code would come from."""
    return "from uncommitted code" if check.provenance.base_commit else "outside a Git checkout"


def code_line(provenance: Provenance) -> str:
    """The ``Code:`` line the plan and a run print."""
    return f"Code:   {provenance.label}"


def plan_lines(check: CodeCheck) -> list[str]:
    """The plan's ``Code:`` line and, for a tree that is not clean, what ``--run`` would do."""
    lines = [code_line(check.provenance)]
    if check.verdict is CodeVerdict.SNAPSHOT:
        lines.append(f"{PLAN_INDENT}--run saves a snapshot")
    elif check.verdict is CodeVerdict.UNRECORDED:
        lines.append(f"{PLAN_INDENT}not a Git checkout: --run records no code")
    elif check.verdict is CodeVerdict.REFUSED:
        lines.append(f"{PLAN_INDENT}--run refuses: {_artifacts(len(check.calendar))} would build {_source(check)}")
    return lines


def refusal(check: CodeCheck) -> str:
    """Why ``--run`` refuses, which artifacts it would have built, and what to do."""
    lines = [f"{_artifacts(len(check.calendar))} would build {_source(check)}:"]
    lines.extend(f"  {label}" for label in check.calendar)
    uncommitted = check.provenance.uncommitted
    if check.provenance.base_commit:
        shown = ", ".join(entry.strip() for entry in uncommitted[:REFUSAL_PATHS])
        more = f", and {len(uncommitted) - REFUSAL_PATHS} more" if len(uncommitted) > REFUSAL_PATHS else ""
        lines.append(f"Uncommitted ({len(uncommitted)}): {shown}{more}")
        lines.append(SNAPSHOT_ADVICE)
    else:
        lines.append("This is not a Git checkout, so the code cannot be recorded.")
        lines.append(NO_CHECKOUT_ADVICE)
    return "\n".join(lines)

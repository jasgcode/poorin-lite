# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether a run may build from the working tree's code (repro.experiment.code_check)."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from repro.execution.lazy import OUT, apply, lower_all, run
from repro.execution.step_runner import steps_to_build
from repro.experiment.code_check import CodeVerdict, check_code, plan_lines, refusal
from repro.provenance import Provenance

CLEAN = Provenance(tree_hash="t", base_commit="a0a406ff4", dirty=False, branch="main", built_by="u", commit_subject="s")
DIRTY = Provenance(
    tree_hash="t", base_commit="a0a406ff4", dirty=True, branch="main", built_by="u", uncommitted=(" M model.py",)
)
NO_CHECKOUT = Provenance(tree_hash="", base_commit="", dirty=False, branch=None, built_by="u")


def _write(out, **_inputs):
    os.makedirs(out, exist_ok=True)
    Path(out, "done").write_text("built")


def _builds(*handles):
    return steps_to_build(lower_all(list(handles)))


def _calendar():
    return apply("data/toy", _write, version="2026.07.16", out=OUT)


def _dev(data=None):
    return apply("models/toy", _write, version="dev", out=OUT, data=data)


@pytest.mark.parametrize(
    ("provenance", "handles", "allow", "verdict"),
    [
        (CLEAN, "calendar", False, CodeVerdict.COMMITTED),
        (DIRTY, "calendar", False, CodeVerdict.REFUSED),
        (DIRTY, "calendar", True, CodeVerdict.SNAPSHOT),
        (DIRTY, "dev", False, CodeVerdict.SNAPSHOT),
        (NO_CHECKOUT, "calendar", False, CodeVerdict.REFUSED),
        (NO_CHECKOUT, "dev", False, CodeVerdict.UNRECORDED),
    ],
)
def test_the_verdict_follows_the_tree_and_what_would_build(provenance, handles, allow, verdict):
    built = _builds(_calendar()) if handles == "calendar" else _builds(_dev())
    assert check_code(provenance, built, allow_uncommitted=allow).verdict is verdict


def test_a_dev_run_over_a_built_calendar_input_may_use_uncommitted_code():
    data = _calendar()
    run(data)
    check = check_code(DIRTY, _builds(_dev(data)), allow_uncommitted=False)
    assert (check.verdict, check.calendar) == (CodeVerdict.SNAPSHOT, ())


def test_a_run_that_builds_nothing_is_cached_and_never_refused():
    data = _calendar()
    run(data)
    assert check_code(DIRTY, _builds(data), allow_uncommitted=False).verdict is CodeVerdict.CACHED


def test_a_pinned_input_is_not_a_calendar_build(tmp_path):
    pinned = apply("data/pinned", _write, version="2026.07.16", out=OUT, pin=str(tmp_path))
    check = check_code(DIRTY, _builds(_dev(pinned)), allow_uncommitted=False)
    assert check.calendar == ()


def test_the_plan_predicts_what_run_would_do():
    snapshot = check_code(DIRTY, _builds(_dev()), allow_uncommitted=False)
    assert plan_lines(snapshot) == [f"Code:   {DIRTY.label}", "        --run saves a snapshot"]
    refused = check_code(DIRTY, _builds(_calendar()), allow_uncommitted=False)
    assert refused.calendar == ("data/toy@2026.07.16",)
    assert "--run refuses: 1 calendar-versioned artifact would build" in plan_lines(refused)[1]
    assert plan_lines(check_code(CLEAN, _builds(_dev()), allow_uncommitted=False)) == [f"Code:   {CLEAN.label}"]


def test_a_refusal_names_the_artifacts_and_the_uncommitted_paths():
    message = refusal(check_code(DIRTY, _builds(_calendar()), allow_uncommitted=False))
    assert "1 calendar-versioned artifact would build from uncommitted code:" in message
    assert "  data/toy@2026.07.16" in message
    assert "Uncommitted (1): M model.py" in message
    # The refused run may itself be a dev run whose calendar input would build, so the advice
    # never suggests switching to a dev version.
    assert message.endswith("Commit the changes and run again, or pass --allow-uncommitted to build from a snapshot.")


def test_a_refusal_lists_ten_paths_and_counts_the_rest():
    many = replace(DIRTY, uncommitted=tuple(f" M file{i}.py" for i in range(12)))
    lines = refusal(check_code(many, _builds(_calendar()), allow_uncommitted=False)).splitlines()
    shown = ", ".join(f"M file{i}.py" for i in range(10))
    assert f"Uncommitted (12): {shown}, and 2 more" in lines


def test_a_refusal_without_a_checkout_says_the_code_cannot_be_recorded():
    message = refusal(check_code(NO_CHECKOUT, _builds(_calendar()), allow_uncommitted=False))
    assert message.startswith("1 calendar-versioned artifact would build outside a Git checkout:")
    assert "This is not a Git checkout, so the code cannot be recorded." in message
    assert "Uncommitted" not in message
    assert message.endswith("Run from a Git checkout, or pass --allow-uncommitted to build without recording the code.")


def test_the_plan_predicts_a_refusal_without_a_checkout():
    check = check_code(NO_CHECKOUT, _builds(_calendar()), allow_uncommitted=False)
    assert (
        plan_lines(check)[1] == "        --run refuses: 1 calendar-versioned artifact would build outside a Git checkout"
    )


def test_the_plan_says_a_run_without_a_checkout_records_no_code():
    check = check_code(NO_CHECKOUT, _builds(_dev()), allow_uncommitted=False)
    assert plan_lines(check) == [f"Code:   {NO_CHECKOUT.label}", "        not a Git checkout: --run records no code"]

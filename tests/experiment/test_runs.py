# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Listing runs under the prefix and showing the code behind one (repro.experiment.runs)."""

import dataclasses
import os
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from repro.execution.artifact import ArtifactRecord, write_record
from repro.execution.lazy import OUT, apply, run
from repro.execution.step_status import STATUS_SUCCESS, StatusFile
from repro.experiment.runs import main
from repro.provenance import Provenance, launched_as, save_snapshot


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "checkout"
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "tester")
    (path / "model.py").write_text("v1\n")
    _git(path, "add", "model.py")
    _git(path, "commit", "-m", "Add the model")
    monkeypatch.chdir(path)
    return path


@pytest.fixture
def prefix(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "prefix"
    monkeypatch.setenv("REPRO_PREFIX", str(path))
    return path


def _write(out, **_inputs):
    os.makedirs(out, exist_ok=True)
    Path(out, "done").write_text("built")


def _fail(out):
    raise RuntimeError("boom")


def _build(name, provenance, version="2026.07.16", **inputs):
    handle = apply(name, _write, version=version, out=OUT, **inputs)
    with launched_as(provenance):
        run(handle)
    return handle


def _at(provenance, when):
    return dataclasses.replace(provenance, created_at=when)


def _labels(output: str) -> list[str]:
    """The ``name@version`` column of ``list``'s lines, top to bottom."""
    return [line.split()[3] for line in output.splitlines()]


def test_list_shows_runs_newest_first_with_their_code(checkout, prefix):
    clean = Provenance.capture()
    # Built d, a, c, b. Newest first is c, a, d, b: not alphabetical, not the build order, not its reverse.
    _build("data/d", _at(clean, "2026-10-11T09:15:00"))
    _build("data/a", _at(clean, "2026-10-11T09:30:00"))
    _build("data/c", _at(dataclasses.replace(clean, note="second try"), "2026-10-11T10:00:00"))
    _build("data/b", _at(clean, "2026-10-11T09:00:00"))
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0, result.output
    assert _labels(result.output) == [f"data/{name}@2026.07.16" for name in "cadb"]
    lines = result.output.splitlines()
    assert lines[0].startswith("2026-10-11 10:00  built") and lines[0].endswith("second try")
    assert lines[3].startswith("2026-10-11 09:00  built")
    assert lines[3].endswith(f"main@{clean.base_commit[:7]}  Add the model")


def test_list_filters_by_name(checkout, prefix):
    clean = Provenance.capture()
    for i in range(3):
        _build(f"data/toy{i}", _at(clean, f"2026-10-11T09:0{i}:00"))
    _build("models/toy", _at(clean, "2026-10-11T09:05:00"))
    result = CliRunner().invoke(main, ["list", "data/"])
    assert sorted(_labels(result.output)) == [f"data/toy{i}@2026.07.16" for i in range(3)]


def test_list_limit_keeps_the_newest_runs(checkout, prefix):
    clean = Provenance.capture()
    _build("data/toy0", _at(clean, "2026-10-11T09:00:00"))
    _build("data/toy1", _at(clean, "2026-10-11T09:02:00"))
    _build("data/toy2", _at(clean, "2026-10-11T09:01:00"))
    _build("models/toy", _at(clean, "2026-10-11T09:05:00"))
    result = CliRunner().invoke(main, ["list", "--limit", "2"])
    assert _labels(result.output) == ["models/toy@2026.07.16", "data/toy1@2026.07.16"]


def test_list_shows_a_failed_build_without_code(checkout, prefix):
    with pytest.raises(RuntimeError), launched_as(Provenance.capture()):
        run(apply("data/broken", _fail, version="2026.07.16", out=OUT))
    result = CliRunner().invoke(main, ["list"])
    assert "failed" in result.output and "data/broken@2026.07.16" in result.output
    assert result.output.rstrip().endswith("-")


def test_show_describes_a_run_from_a_snapshot(checkout, prefix):
    (checkout / "model.py").write_text("v2\n")
    launch = Provenance.capture()
    launch = dataclasses.replace(launch, note="try v2", snapshot_commit=save_snapshot("try v2"))
    _build("data/toy", launch, version="dev")
    _git(checkout, "commit", "-am", "Use v2")  # main moves one commit past the run
    result = CliRunner().invoke(main, ["show", "data/toy@dev"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "data/toy@dev   built" in out
    assert "Note         try v2" in out
    assert f"Commit       {launch.base_commit} on main" in out and "Add the model" in out
    assert "main has moved 1 commit past it" in out
    assert f"Uncommitted  1 file, saved as snapshot {launch.snapshot_commit[:12]}" in out
    assert "M   model.py" in out and "+1 -1" in out
    # The commands print hashes Git verified, never the record's text.
    base = _git(checkout, "rev-parse", launch.base_commit)
    snapshot = launch.snapshot_commit[:12]
    assert f"git diff {base[:12]} {snapshot}" in out
    assert f"git worktree add ../run-{snapshot} {snapshot}" in out


def test_show_reads_a_record_from_before_subjects_were_recorded(checkout, prefix):
    old = dataclasses.replace(Provenance.capture(), commit_subject="", commit_time="")
    _build("data/toy", old)
    result = CliRunner().invoke(main, ["show", str(prefix / "data" / "toy" / "2026.07.16")])
    assert result.exit_code == 0, result.output
    assert "Add the model" in result.output  # looked up with git log
    assert "main has not moved since" in result.output


def test_show_does_not_pass_a_recorded_commit_to_git_as_an_option(checkout, prefix, tmp_path):
    pwned = tmp_path / "pwned"
    crafted = dataclasses.replace(
        Provenance.capture(),
        dirty=True,
        base_commit=f"--output={pwned}",
        snapshot_commit=_git(checkout, "rev-parse", "HEAD"),
        uncommitted=("M  model.py",),
    )
    _build("data/toy", crafted)
    result = CliRunner().invoke(main, ["show", "data/toy@2026.07.16"])
    assert result.exit_code == 0, result.output
    assert not pwned.exists()


def test_show_says_a_failed_build_wrote_no_record(checkout, prefix):
    with pytest.raises(RuntimeError), launched_as(Provenance.capture()):
        run(apply("data/broken", _fail, version="2026.07.16", out=OUT))
    result = CliRunner().invoke(main, ["show", "data/broken@2026.07.16"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith("failed")
    assert "the build failed before writing its record" in result.output


def test_show_says_an_output_without_provenance_predates_it(checkout, prefix):
    output = prefix / "data" / "legacy" / "2026.01.01"
    write_record(ArtifactRecord(name="data/legacy", version="2026.01.01", output_path=str(output)))
    StatusFile(str(output)).write_status(STATUS_SUCCESS)
    result = CliRunner().invoke(main, ["show", "data/legacy@2026.01.01"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith("built")
    assert "the output predates code provenance" in result.output


def test_show_rejects_an_unknown_run(checkout, prefix):
    result = CliRunner().invoke(main, ["show", "data/nothing@2026.07.16"])
    assert result.exit_code != 0
    assert "no run data/nothing@2026.07.16" in result.output


def test_show_takes_a_short_reference(checkout, prefix):
    _build("data/thing", Provenance.capture(), version="probe-20261012-090000-a1b2-dev")
    result = CliRunner().invoke(main, ["show", "data/thing@a1b2"])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("data/thing@probe-20261012-090000-a1b2-dev")

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short references to runs (repro.experiment.run_refs)."""

from pathlib import Path

import pytest

from repro.execution.artifact import ArtifactRecord, write_record
from repro.execution.step_status import STATUS_RUNNING, STATUS_SUCCESS, StatusFile
from repro.experiment.run_refs import RunRefError, resolve_run, resume_version
from repro.provenance import username_segment

ACCEPT_OLD = "accept-20261011-210614-7f16-dev"
ACCEPT_NEW = "accept-20261012-201822-60b9-dev"
PROBE = "probe-20261012-090000-a1b2-dev"


@pytest.fixture
def prefix(tmp_path) -> Path:
    return tmp_path / "prefix"


def _run(prefix: Path, name: str, version: str, status: str = STATUS_SUCCESS) -> None:
    path = prefix / name / version
    path.mkdir(parents=True)
    write_record(ArtifactRecord(name=name, version=version, output_path=str(path)))
    StatusFile(str(path)).write_status(status)


def _dev(name: str) -> str:
    return f"users/{username_segment()}/{name}"


def _version(prefix: Path, ref: str, *, finished_only: bool = True) -> str:
    return resolve_run(str(prefix), ref, finished_only=finished_only).label.rpartition("@")[2]


def test_a_full_version_resolves_to_itself(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    _run(prefix, "episode/wam", "2026.10.13")
    assert _version(prefix, f"episode/wam@{ACCEPT_NEW}") == ACCEPT_NEW
    assert _version(prefix, "episode/wam@2026.10.13") == "2026.10.13"


def test_a_token_resolves_to_the_run_it_suffixes(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    assert _version(prefix, "episode/wam@60b9") == ACCEPT_NEW


def test_a_token_two_runs_share_is_refused_listing_both(prefix):
    twin = "probe-20261013-080000-60b9-dev"
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    _run(prefix, _dev("episode/wam"), twin)
    with pytest.raises(RunRefError) as refused:
        resolve_run(str(prefix), "episode/wam@60b9", finished_only=True)
    assert ACCEPT_NEW in str(refused.value) and twin in str(refused.value)


def test_a_label_resolves_to_its_latest_run_by_version_time(prefix):
    # Written newest first, so file times disagree with the versions' timestamps.
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), PROBE)
    assert _version(prefix, "episode/wam@accept") == ACCEPT_NEW


def test_latest_resolves_to_the_latest_dev_run(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), PROBE)
    _run(prefix, "episode/wam", "2026.10.13")
    assert _version(prefix, "episode/wam@latest") == PROBE


def test_an_unfinished_run_is_skipped_only_when_finished_runs_are_asked_for(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_OLD)
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW, status=STATUS_RUNNING)
    assert _version(prefix, "episode/wam@accept") == ACCEPT_OLD
    assert _version(prefix, "episode/wam@accept", finished_only=False) == ACCEPT_NEW


def test_an_unknown_reference_lists_the_versions_there_are(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW)
    with pytest.raises(RunRefError, match=ACCEPT_NEW):
        resolve_run(str(prefix), "episode/wam@ffff", finished_only=True)


def test_resume_takes_a_token_or_a_full_version(prefix):
    _run(prefix, _dev("episode/wam"), ACCEPT_NEW, status=STATUS_RUNNING)
    _run(prefix, _dev("episode/vla"), ACCEPT_NEW, status=STATUS_RUNNING)  # one sweep, one version
    assert resume_version(str(prefix), "60b9") == ACCEPT_NEW
    assert resume_version(str(prefix), PROBE) == PROBE
    with pytest.raises(RunRefError, match="no dev run"):
        resume_version(str(prefix), "ffff")

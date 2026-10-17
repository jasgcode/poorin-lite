# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What the plan and the drift check read at an artifact's output path."""

import pytest

from repro.execution.artifact import Artifact, FingerprintMismatchError
from repro.execution.cache_state import CacheState, HandleState, graph_states
from repro.execution.lazy import ArtifactStep, run

V = "2026.06.28"


class Toy(Artifact):
    payload: str


def _toy(version: str, payload: str, *, fail: bool = False) -> ArtifactStep[Toy]:
    def build(config):
        if fail:
            raise RuntimeError("the build crashed")
        return Toy(payload=config["payload"])

    return ArtifactStep(
        name="datasets/toy",
        version=version,
        artifact_type=Toy,
        run=build,
        build_config=lambda ctx: {"out": ctx.output_path, "payload": payload},
    )


def _state(handle: ArtifactStep) -> HandleState:
    (state,) = graph_states([handle])
    return state


@pytest.fixture(autouse=True)
def _prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_PREFIX", str(tmp_path))


def test_a_calendar_artifact_moves_from_not_built_to_built():
    assert _state(_toy(V, "a")).state is CacheState.NOT_BUILT
    run(_toy(V, "a"))
    assert _state(_toy(V, "a")).state is CacheState.BUILT


def test_a_changed_recipe_reads_as_drifted_with_the_field_diff():
    run(_toy(V, "a"))
    state = _state(_toy(V, "b"))
    assert state.state is CacheState.DRIFTED
    assert state.recorded_fingerprint == _toy(V, "a").fingerprint()
    assert "payload: 'a' -> 'b'" in state.change


def test_a_build_that_crashed_reads_as_started():
    with pytest.raises(RuntimeError):
        run(_toy(V, "a", fail=True))
    assert _state(_toy(V, "a")).state is CacheState.STARTED


def test_a_dev_version_reads_as_mutable():
    assert _state(_toy("dev", "a")).state is CacheState.MUTABLE


def test_a_pinned_artifact_is_not_checked(tmp_path):
    pinned = ArtifactStep(
        name="datasets/pinned",
        version=V,
        artifact_type=Toy,
        run=lambda config: None,
        build_config=lambda ctx: {},
        override_path=str(tmp_path / "elsewhere"),
    )
    assert _state(pinned).state is CacheState.PINNED


def test_handles_for_one_artifact_are_read_once():
    # Each trial of a sweep constructs the shared dataset; they are one artifact.
    run(_toy(V, "a"))
    states = graph_states([_toy(V, "a"), _toy(V, "a"), _toy("2026.06.29", "a")])
    assert [(s.label, s.state) for s in states] == [
        (f"datasets/toy@{V}", CacheState.BUILT),
        ("datasets/toy@2026.06.29", CacheState.NOT_BUILT),
    ]


def test_two_recipes_for_one_artifact_are_refused():
    with pytest.raises(FingerprintMismatchError, match=f"datasets/toy@{V} is built from two recipes"):
        graph_states([_toy(V, "a"), _toy(V, "b")])

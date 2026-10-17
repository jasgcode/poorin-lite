# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What exists at an artifact's output path: nothing, a started build, a finished one, or one
built from another recipe.

The plan that ``repro.experiment.cli`` prints and the drift check in
:func:`repro.execution.lazy.run` both read an artifact's state here, so they agree on what a run
will do. Handles are typed by a protocol so that ``lazy`` can import this module.
"""

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from repro.execution.artifact import (
    DriftAcceptanceError,
    FingerprintMismatchError,
    is_mutable_version,
    recorded_recipe,
)
from repro.execution.fingerprint import describe_drift, fingerprint_hash
from repro.execution.step_status import STATUS_SUCCESS, read_status


class Handle(Protocol):
    """The parts of an ``ArtifactStep`` the state check reads."""

    name: str
    version: str
    override_path: str | None

    @property
    def label(self) -> str: ...

    def path(self) -> str: ...

    def fingerprint_payload(self) -> str: ...


class CacheState(StrEnum):
    PINNED = "pinned"
    """Points at existing data through ``override_path``; not checked."""
    MUTABLE = "mutable"
    """A dev version: always builds, and a resumed one continues its checkpoints."""
    NOT_BUILT = "not built"
    STARTED = "started"
    """A calendar version whose build started but did not succeed: the build restarts."""
    BUILT = "built"
    DRIFTED = "built from another recipe"


@dataclass(frozen=True)
class HandleState:
    handle: Handle
    payload: str
    """The canonical config JSON ``fingerprint`` hashes."""
    fingerprint: str
    state: CacheState
    recorded_fingerprint: str | None
    change: str
    """The field-level diff from :func:`describe_drift`; empty unless ``DRIFTED``."""

    @property
    def label(self) -> str:
        return self.handle.label


def _handle_state(handle: Handle, payload: str, fingerprint: str) -> HandleState:
    """``handle``'s state, from the recipe and status recorded at its output path."""

    def state(cache_state: CacheState, recorded_fingerprint: str | None = None, change: str = "") -> HandleState:
        return HandleState(handle, payload, fingerprint, cache_state, recorded_fingerprint, change)

    if handle.override_path is not None:
        return state(CacheState.PINNED)
    path = handle.path()
    recorded = recorded_recipe(path)
    if recorded is not None and recorded.fingerprint != fingerprint:
        return state(CacheState.DRIFTED, recorded.fingerprint, describe_drift(recorded.payload, payload))
    recorded_fingerprint = recorded.fingerprint if recorded is not None else None
    if is_mutable_version(handle.version):
        return state(CacheState.MUTABLE, recorded_fingerprint)
    if read_status(path) == STATUS_SUCCESS:
        return state(CacheState.BUILT, recorded_fingerprint)
    if recorded is not None:
        return state(CacheState.STARTED, recorded_fingerprint)
    return state(CacheState.NOT_BUILT)


def graph_states(handles: Iterable[Handle]) -> list[HandleState]:
    """The state of each distinct ``name@version`` among ``handles``, a graph walk, in order.

    Several handles can build one output (each trial of a sweep constructs the shared dataset):
    handles that share a ``name@version`` and a recipe are one artifact, read once. Handles that
    share a ``name@version`` but not a recipe claim one output for two recipes, and raise
    :class:`FingerprintMismatchError` before anything is read.
    """
    first: dict[str, tuple[Handle, str, str]] = {}
    for handle in handles:
        payload = handle.fingerprint_payload()
        fingerprint = fingerprint_hash(payload)
        seen = first.setdefault(handle.label, (handle, payload, fingerprint))[2]
        if seen != fingerprint:
            raise FingerprintMismatchError(
                f"{handle.label} is built from two recipes in this graph, {seen} and {fingerprint}. "
                "Give each recipe its own name or version."
            )
    return [_handle_state(*entry) for entry in first.values()]


def check_accept_drift(states: Iterable[HandleState], accept_drift: Collection[str]) -> None:
    """Raise :class:`DriftAcceptanceError` if ``accept_drift`` names an artifact none of ``states``
    shows as drifted: a typo, or a stale flag."""
    drifted = {state.handle.name for state in states if state.state is CacheState.DRIFTED}
    unknown = sorted(set(accept_drift) - drifted)
    if unknown:
        raise DriftAcceptanceError(f"accept_drift names artifacts whose recipe did not change: {unknown}")

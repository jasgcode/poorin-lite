# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deferred artifact versions via an ambient build-time context.

Every :class:`~repro.execution.lazy.ArtifactStep` is addressed by an explicit ``name@version``.
Hardcoding a version on each builder call is tedious while iterating: a version is really a
*run-wide policy* ("build everything at ``2026.07.16``, but pin this one dataset") that today has
to be spelled out per artifact. This module lets a builder **defer** its version to a shared
policy that rides on an ambient context:

- A :class:`VersionCodex` maps an artifact ``name`` to a version — one ``default`` plus per-name
  versions in ``by_name``.
- :func:`build_context` makes a codex ambient (a :class:`~contextvars.ContextVar`) while an
  experiment's handles are *constructed*.
- :func:`resolve_version` is what a builder calls: an explicit ``version=`` always wins; otherwise
  the version comes from the ambient codex. A builder invoked with no explicit version and no
  active context is an error — a deferred version never silently defaults.

The version is resolved to a concrete string at *construction* time, before the handle is lowered.
Nothing about identity changes: the fingerprint, the ``{name}/{version}`` path, and the drift check
all see an ordinary version string, and resolving to the ``dev`` default reproduces today's
mutable, per-user-namespaced behavior. The whole handle graph a driver builds must be materialized
inside the ``with build_context(...)`` block — a builder (or a lazily-invoked factory) called after
the block exits has no context to draw from. Construction is expected to be synchronous; a
:class:`~contextvars.ContextVar` isolates concurrent async tasks and nests correctly, but does not
propagate into freshly spawned OS threads.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VersionCodex:
    """Resolves an artifact ``name`` to a version: a ``default`` plus per-name versions in ``by_name``.

    ``by_name`` is keyed by the ``name`` passed to the builder (before any per-user namespacing),
    matched exactly. Each version is validated where it is used, by the handle it versions.
    ``unused_names`` reports names that no builder ever asked for — typically a typo, or a name
    aimed at an artifact that hardcodes its version.
    """

    default: str
    by_name: Mapping[str, str] = field(default_factory=dict)
    # Names actually consulted via version_for, for the unused-name check. Instrumentation
    # only (compare=False, init=False); mutating the set on a frozen instance is allowed.
    _used: set[str] = field(default_factory=set, compare=False, repr=False, init=False)

    def version_for(self, name: str) -> str:
        """The version for artifact ``name``: its entry in ``by_name`` if there is one, else the default."""
        if name in self.by_name:
            self._used.add(name)
            return self.by_name[name]
        return self.default

    def unused_names(self) -> frozenset[str]:
        """Names in ``by_name`` never consulted since construction — a likely typo or a no-op."""
        return frozenset(self.by_name) - frozenset(self._used)


_ACTIVE: ContextVar["VersionCodex | None"] = ContextVar("repro_build_versions", default=None)


@contextmanager
def build_context(versions: VersionCodex) -> Iterator[VersionCodex]:
    """Make ``versions`` the ambient codex for the duration of the ``with`` block.

    Construct the experiment's handle graph inside the block so deferring builders can resolve
    their versions from it. Nests correctly and is restored on exit (including on exception).
    """
    token = _ACTIVE.set(versions)
    try:
        yield versions
    finally:
        _ACTIVE.reset(token)


def resolve_version(name: str, version: str | None) -> str:
    """The version for artifact ``name``: ``version`` if given, else the ambient codex's choice.

    An explicit ``version`` always wins (deferral is opt-in per call). With no explicit version and
    no active :func:`build_context`, this raises rather than guessing — a deferred version never
    silently defaults.
    """
    if version is not None:
        return version
    versions = _ACTIVE.get()
    if versions is None:
        raise ValueError(
            f"{name}: no version given and no active build_context. Pass version=..., or construct "
            "this handle inside build_context(VersionCodex(default=...))."
        )
    return versions.version_for(name)

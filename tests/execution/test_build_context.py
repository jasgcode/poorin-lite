# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for deferred artifact versions (``repro.execution.build_context``).

Covers the resolution contract (explicit wins, codex fallback, no-context error), context nesting
and restoration on exception, and the unused-name signal that guards against typos.
"""

import pytest

from repro.execution.build_context import VersionCodex, build_context, resolve_version


def _codex(default="2026.07.16", **by_name):
    return VersionCodex(default=default, by_name=by_name)


def test_version_for_prefers_the_named_version_then_default():
    codex = VersionCodex(default="2026.07.16", by_name={"tokenizer": "2026.06.28"})
    assert codex.version_for("tokenizer") == "2026.06.28"
    assert codex.version_for("anything-else") == "2026.07.16"


def test_resolve_version_explicit_wins_over_context():
    # An explicit version is not overridable by the codex: deferral is opt-in per call.
    with build_context(_codex(**{"data": "2026.06.28"})):
        assert resolve_version("data", "2026.01.01") == "2026.01.01"


def test_resolve_version_defers_to_codex_inside_context():
    with build_context(VersionCodex(default="dev", by_name={"pin-me": "2026.06.28"})):
        assert resolve_version("pin-me", None) == "2026.06.28"
        assert resolve_version("free", None) == "dev"


def test_resolve_version_without_context_raises():
    # The safety contract: a deferred version never silently defaults outside a build_context.
    with pytest.raises(ValueError, match="no active build_context"):
        resolve_version("orphan", None)


def test_build_context_nests_and_restores_on_exception():
    outer = _codex(default="2026.07.16")
    inner = _codex(default="2026.06.28")
    with build_context(outer):
        assert resolve_version("x", None) == "2026.07.16"
        with pytest.raises(RuntimeError):
            with build_context(inner):
                assert resolve_version("x", None) == "2026.06.28"
                raise RuntimeError("boom")
        # The inner context is torn down even though the block raised.
        assert resolve_version("x", None) == "2026.07.16"
    with pytest.raises(ValueError):
        resolve_version("x", None)


def test_unused_names_flags_names_never_consulted():
    codex = VersionCodex(default="dev", by_name={"used": "2026.06.28", "typo": "2026.06.28"})
    codex.version_for("used")
    codex.version_for("some-default-artifact")
    assert codex.unused_names() == frozenset({"typo"})

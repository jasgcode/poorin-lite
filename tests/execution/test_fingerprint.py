# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deterministic fingerprint encoder.

The fingerprint is only meaningful if the same config always serializes to the same
bytes, so these tests pin the canonicalization (timedeltas, paths, sets, classes,
enums) and the class each dataclass is recorded under. A type with no canonical form
raises; ``register_fingerprint`` teaches it an exact one.
"""

import dataclasses
from datetime import timedelta
from enum import Enum
from pathlib import Path

import pytest

from repro.execution.fingerprint import canonical_json, fingerprint_hash, register_fingerprint
from tests.choice_configs import AdamConfig, MuonConfig, SgdConfig


@dataclasses.dataclass(frozen=True)
class _Cfg:
    lr: float
    interval: timedelta = timedelta(minutes=10)


def test_canonical_json_is_deterministic_and_field_sensitive():
    same = canonical_json(_Cfg(lr=1e-3))
    assert same == canonical_json(_Cfg(lr=1e-3))
    assert same != canonical_json(_Cfg(lr=2e-3))
    assert fingerprint_hash(same) == fingerprint_hash(canonical_json(_Cfg(lr=1e-3)))
    assert fingerprint_hash(same) != fingerprint_hash(canonical_json(_Cfg(lr=2e-3)))


def test_timedelta_changes_register():
    assert canonical_json(_Cfg(lr=1.0, interval=timedelta(minutes=10))) != canonical_json(
        _Cfg(lr=1.0, interval=timedelta(minutes=20))
    )


def test_class_values_are_identified_by_qualified_name():
    assert canonical_json({"cls": Path}) != canonical_json({"cls": timedelta})
    assert "datetime.timedelta" in canonical_json({"cls": timedelta})


def test_path_is_serialized_by_string():
    assert canonical_json({"p": Path("a/b")}) == canonical_json({"p": "a/b"})


def test_sets_are_order_independent():
    assert canonical_json({"s": {3, 1, 2}}) == canonical_json({"s": {2, 3, 1}})
    assert canonical_json({"s": {1, 2}}) != canonical_json({"s": {1, 2, 3}})


class _Color(Enum):
    RED = "red"


def test_enum_uses_value():
    assert canonical_json({"c": _Color.RED}) == canonical_json({"c": "red"})


def test_register_fingerprint_gives_an_exact_form():
    class Special:
        def __init__(self, token: str) -> None:
            self._opaque = object()  # not deterministically serializable on its own
            self.token = token

    register_fingerprint(Special, lambda s: {"special": s.token})
    out = canonical_json({"s": Special("xyz")})
    assert "xyz" in out
    assert canonical_json({"s": Special("a")}) != canonical_json({"s": Special("b")})


def test_a_callable_is_rejected():
    with pytest.raises(TypeError, match="callable"):
        canonical_json({"fn": lambda x: x})


def test_a_default_repr_object_is_rejected():
    class Opaque:  # relies on object.__repr__ -> a memory address, not reproducible
        pass

    with pytest.raises(TypeError, match="default object repr"):
        canonical_json({"o": Opaque()})


def test_an_object_with_no_canonical_form_is_rejected():
    class Widget:
        def __init__(self, n: int) -> None:
            self.n = n

    with pytest.raises(TypeError, match="cannot fingerprint"):
        canonical_json({"w": Widget(1)})


@dataclasses.dataclass(frozen=True)
class _A:
    x: int = 1


@dataclasses.dataclass(frozen=True)
class _B:
    x: int = 1


def test_two_dataclasses_with_equal_fields_fingerprint_differently():
    assert canonical_json(_A()) != canonical_json(_B())


def test_a_registered_config_is_keyed_by_its_choice_name():
    assert '"__class__": "adam"' in canonical_json(AdamConfig())
    assert canonical_json(MuonConfig()) != canonical_json(SgdConfig())


def test_nested_dataclasses_carry_their_class_at_every_level():
    @dataclasses.dataclass(frozen=True)
    class Outer:
        inner: _A

    assert canonical_json(Outer(_A())).count("__class__") == 2

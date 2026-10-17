# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a config value back from the canonical form a record stores.

A record's ``config`` is the fingerprint's canonical JSON (:mod:`repro.execution.fingerprint`):
each dataclass carries its class under ``__class__``, as its draccus choice name when it is a
registered choice (``adam``, ``cached``) and its class name otherwise. :func:`decode_recorded`
walks the target type's fields, so a choice gets draccus's ``type`` key only where the declared
type is a choice base, and hands the result to ``draccus.decode``. Evaluating a run reads the run's
own recipe this way.
"""

import inspect
import types
import typing
from datetime import timedelta
from typing import Any, TypeVar

import draccus
from draccus.choice_types import CHOICE_TYPE_KEY, ChoiceRegistryBase

from repro.execution.fingerprint import CLASS_KEY, class_label

T = TypeVar("T")


def _decode_timedelta(value: Any) -> timedelta:
    """Seconds (what ``--set`` passes) or the fingerprint's ``{days, seconds, microseconds}`` (a recorded config)."""
    if isinstance(value, dict):
        return timedelta(**value)
    return timedelta(seconds=float(value))


def _encode_timedelta(value: timedelta) -> dict[str, int]:
    return {"days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}


# Draccus has no timedelta codec, and configs fingerprint timedeltas, so recorded configs and --set
# (whose overrides decode through this module) need one.
draccus.decode.register(timedelta, _decode_timedelta)
draccus.encode.register(timedelta, _encode_timedelta)


def field_types(cls: type) -> dict[str, Any]:
    """The evaluated annotation of each field of ``cls`` and its bases.

    ``typing.get_type_hints`` returns nothing for a class marked ``@no_type_check``, as grug's launch
    config is (so the tests' beartype hook accepts the placeholders a fingerprint puts in its fields).
    Decoding still needs the annotations, so this reads each class's own. Only an annotation written
    whole as a string is evaluated: ``Annotated`` is not unwrapped, and a string forward reference
    nested inside a generic (``list["Foo"]``) is not resolved.
    """
    found: dict[str, Any] = {}
    for base in reversed(cls.__mro__):
        found.update(inspect.get_annotations(base, eval_str=True))
    return found


def decode_recorded(cls: type[T], value: Any) -> T:
    """``value``, a config value as a record stores it, decoded as ``cls``.

    Raises ``draccus.utils.DecodingError`` when the value holds a field ``cls`` no longer has, and
    ``ValueError`` when a recorded class is neither the declared type nor one of its choices.
    """
    return draccus.decode(cls, _draccus_form(cls, value))


def _names(tp: Any, label: str) -> bool:
    """Whether the recorded class ``label`` is ``tp`` or one of its registered choices."""
    if not isinstance(tp, type):
        return False
    if class_label(tp) == label:
        return True
    return issubclass(tp, ChoiceRegistryBase) and label in tp.get_known_choices()


def _dataclass_form(tp: Any, value: dict[str, Any]) -> dict[str, Any]:
    label = value[CLASS_KEY]
    if not _names(tp, label):
        raise ValueError(f"the record holds a {label}, which is neither {tp} nor one of its choices")
    choice = class_label(tp) != label
    concrete = tp.get_choice_class(label) if choice else tp
    hints = field_types(concrete)
    # A key the class lacks passes through untouched, so draccus refuses it by name.
    form = {key: _draccus_form(hints.get(key, Any), item) for key, item in value.items() if key != CLASS_KEY}
    if choice:
        form[CHOICE_TYPE_KEY] = label
    return form


def _draccus_form(tp: Any, value: Any) -> Any:
    """``value`` with each ``__class__`` key replaced by what draccus reads at type ``tp``."""
    if typing.get_origin(tp) in (typing.Union, types.UnionType):
        if value is None:
            return None
        arms = [arm for arm in typing.get_args(tp) if arm is not type(None)]
        if isinstance(value, dict) and CLASS_KEY in value:
            matching = [arm for arm in arms if _names(arm, value[CLASS_KEY])]
            if not matching:
                raise ValueError(f"the record holds a {value[CLASS_KEY]}, which no arm of {tp} names")
            return _dataclass_form(matching[0], value)
        return _draccus_form(arms[0], value) if len(arms) == 1 else value
    if isinstance(value, dict) and CLASS_KEY in value:
        return _dataclass_form(tp, value)
    args = typing.get_args(tp)
    if isinstance(value, dict):
        return {key: _draccus_form(args[1] if len(args) == 2 else Any, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_draccus_form(args[0] if args else Any, item) for item in value]
    return value

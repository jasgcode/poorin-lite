# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic config fingerprinting for lazy artifacts.

A lazy artifact's identity is its explicit ``name@version``; its *fingerprint* is a
hash of the config its recipe builds, recorded so a drift check can tell a re-run of
the same recipe (a cache hit) from a changed recipe (which wants a version bump). For
that to mean anything the serialization must be **identical across processes**: the
same config must always produce the same bytes.

The fingerprint decides whether an existing output may be reused, so the encoder is total over
the types it knows and raises on any other: a value with no canonical form would make the same
config fingerprint differently between processes. :func:`register_fingerprint` teaches it a
canonical form for an identity-bearing custom type. A dataclass encodes as its fields plus a
``__class__`` key, so two configs that differ only in class fingerprint differently.

The encoder knows dataclasses, enums, timedeltas, paths, sets, classes, and draccus choice types.
A project that fingerprints arrays or tensors registers an encoder for them with
:func:`register_fingerprint`.
"""

import dataclasses
import functools
import hashlib
import inspect
import json
from collections.abc import Callable
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from draccus.choice_types import ChoiceRegistryBase

# The key a dataclass's class is recorded under in its canonical form.
CLASS_KEY = "__class__"

# Custom canonical encoders, keyed by type, consulted before the encoder raises.
_REGISTRY: dict[type, Callable[[Any], object]] = {}


def register_fingerprint(tp: type, encode: Callable[[Any], object]) -> None:
    """Teach the encoder a canonical form for an identity-bearing custom type.

    ``encode(value)`` must return a JSON-canonicalizable object (it is itself encoded).
    Registered converters are consulted before the encoder raises, including for
    subclasses of ``tp``.
    """
    _REGISTRY[tp] = encode


def class_label(cls: type) -> str:
    """A dataclass's class in a fingerprint: its draccus choice name when it is a registered choice
    (``adam``, ``grug_muon``), else its class name. The module is left out, so moving a class does
    not change fingerprints."""
    if issubclass(cls, ChoiceRegistryBase):
        for name, choice in cls.get_known_choices().items():
            if choice is cls:
                return name
    return cls.__qualname__


def _registered_encoder(tp: type) -> Callable[[Any], object] | None:
    for klass in tp.__mro__:
        if klass in _REGISTRY:
            return _REGISTRY[klass]
    return None


def _unfingerprintable(o: object, why: str) -> str:
    t = type(o)
    return (
        f"cannot fingerprint a config value of type {t.__module__}.{t.__qualname__}: {why}. "
        "A fingerprint must be identical across processes; make the value a dataclass, an Enum, or "
        "another canonical type, or register one via repro.execution.fingerprint.register_fingerprint."
    )


class _FingerprintEncoder(json.JSONEncoder):
    """Canonical JSON for config fingerprints.

    Canonical handlers run first, then registered converters; any other type raises.
    """

    def default(self, o):
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            # Shallow, so each nested dataclass reaches this method and records its own class.
            return {CLASS_KEY: class_label(type(o)), **{f.name: getattr(o, f.name) for f in dataclasses.fields(o)}}
        if isinstance(o, Enum):
            return o.value
        if isinstance(o, timedelta):
            return {"days": o.days, "seconds": o.seconds, "microseconds": o.microseconds}
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, set | frozenset):
            # A set's identity is its members, not their iteration order.
            return {"__set__": sorted(o, key=canonical_json)}
        if isinstance(o, type):
            # A class is identified by its fully-qualified name.
            return {"__type__": f"{o.__module__}.{o.__qualname__}"}

        encode = _registered_encoder(type(o))
        if encode is not None:
            return encode(o)

        if inspect.isroutine(o) or isinstance(o, functools.partial):
            raise TypeError(_unfingerprintable(o, "a callable has no stable identity"))
        if type(o).__repr__ is object.__repr__:
            raise TypeError(_unfingerprintable(o, "it uses the default object repr (a memory address)"))
        raise TypeError(_unfingerprintable(o, "no deterministic serialization is known"))


def canonical_json(config: object) -> str:
    """The canonical, deterministic JSON for ``config`` — the bytes a fingerprint hashes."""
    return json.dumps(config, sort_keys=True, cls=_FingerprintEncoder)


def fingerprint_hash(payload: str) -> str:
    """The short recipe fingerprint: an md5 prefix of a canonical payload."""
    return hashlib.md5(payload.encode()).hexdigest()[:8]


# Cap on how many changed config values a drift message spells out before summarizing the
# remainder, so a wholesale recipe change stays readable.
_MAX_DIFF_LINES = 20


def _diff_json(old: object, new: object, prefix: str = "") -> list[str]:
    """Dotted-path descriptions of where ``old`` and ``new`` (parsed JSON) differ."""
    if isinstance(old, dict) and isinstance(new, dict):
        changes: list[str] = []
        for key in sorted(set(old) | set(new)):
            sub = f"{prefix}.{key}" if prefix else key
            if key not in old:
                changes.append(f"{sub}: (added) {new[key]!r}")
            elif key not in new:
                changes.append(f"{sub}: {old[key]!r} (removed)")
            else:
                changes.extend(_diff_json(old[key], new[key], sub))
        return changes
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return [f"{prefix}: list of {len(old)} -> list of {len(new)}"]
        changes = []
        for i, (a, b) in enumerate(zip(old, new, strict=True)):
            changes.extend(_diff_json(a, b, f"{prefix}[{i}]"))
        return changes
    if old != new:
        return [f"{prefix or '(root)'}: {old!r} -> {new!r}"]
    return []


def describe_drift(old_payload: str | None, new_payload: str | None) -> str:
    """A field-level summary of how a recipe's config changed from the recorded one.

    Both arguments are canonical-JSON fingerprint payloads. Returns an empty string when either
    side is absent or the configs are identical; otherwise a capped, newline-prefixed listing of
    the changed dotted paths for a drift message.
    """
    if old_payload is None or new_payload is None:
        return ""
    changes = _diff_json(json.loads(old_payload), json.loads(new_payload))
    if not changes:
        return ""
    shown = changes[:_MAX_DIFF_LINES]
    body = "\n".join(f"  {c}" for c in shown)
    if len(changes) > len(shown):
        body += f"\n  …and {len(changes) - len(shown)} more"
    return f"\nChanged config values:\n{body}"

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed dotted-path overrides of an experiment's config, from ``--set`` and ``--sweep``.

``repro.experiment.cli`` parses the command line into one :class:`Overrides` per trial and passes
it to the experiment, which applies it inside ``build_config`` (so the fingerprint sees it), names
its artifact with :meth:`Overrides.named`, and derives its tracker run ID with
:meth:`Overrides.run_id`. A value is YAML decoded by draccus against the field's annotated type, so
union fields and draccus registries (``optimizer={type: muon}``) decode as they do in any draccus
config.
"""

import dataclasses
import difflib
import itertools
import re
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, TypeVar

import draccus
import yaml
from draccus.cfgparsing import parse_string
from draccus.utils import ParsingError

from repro.execution.fingerprint import canonical_json, fingerprint_hash
from repro.execution.lazy import StepContext
from repro.execution.recorded_config import field_types

T = TypeVar("T")


def _decode_timedelta(value: Any) -> timedelta:
    """Seconds (what ``--set`` passes) or the fingerprint's ``{days, seconds, microseconds}`` (a recorded config)."""
    if isinstance(value, dict):
        return timedelta(**value)
    return timedelta(seconds=float(value))


def _encode_timedelta(value: timedelta) -> dict[str, int]:
    return {"days": value.days, "seconds": value.seconds, "microseconds": value.microseconds}


# Draccus has no timedelta codec, and configs fingerprint timedeltas, so --set and recorded configs need one.
draccus.decode.register(timedelta, _decode_timedelta)
draccus.encode.register(timedelta, _encode_timedelta)

# A value that may appear in an artifact's name as written; any other value names the trial by digest.
_PLAIN_VALUE = re.compile(r"^[A-Za-z0-9_.+-]{1,40}$")
_MAX_READABLE_SUFFIX = 100


class OverrideError(ValueError):
    """A ``--set`` or ``--sweep`` value that cannot apply to the experiment's config."""


def _normalized(text: str) -> str:
    """``text`` in one spelling when it reads as a number or a boolean, so ``1e-3`` and ``0.001`` name
    the same artifact and ``010`` is named for the 8 it decodes to.

    YAML reads the value first, as the field will. YAML 1.1 reads ``1e-3`` (no dot) as a string,
    which Python's ``float`` then reads, as draccus does for a float field. A quoted string keeps
    its quotes: YAML strips them, so ``'010'`` reaches a ``str`` field as ``010``, not as 10. Any
    other value, or text YAML cannot read, keeps its text.
    """
    try:
        value = parse_string(text)
    except yaml.YAMLError:
        return text.strip()
    if isinstance(value, bool | int | float):
        return repr(value)
    # A string equal to its text was unquoted.
    if value == text.strip():
        for parse in (int, float):
            try:
                return repr(parse(value))
            except ValueError:
                continue
    return text.strip()


@dataclass(frozen=True)
class Override:
    """One ``PATH=VALUE``: a dotted path into the config and its YAML value as typed."""

    path: tuple[str, ...]
    text: str

    @classmethod
    def parse(cls, item: str) -> "Override":
        """Split ``PATH=VALUE`` at the first ``=``; the path is dotted identifiers."""
        key, sep, text = item.partition("=")
        path = tuple(key.split("."))
        if not sep or not all(segment.isidentifier() for segment in path):
            raise OverrideError(f"{item!r} is not PATH=VALUE with a dotted path of field names")
        return cls(path, text)

    @property
    def label(self) -> str:
        return f"{'.'.join(self.path)}={_normalized(self.text)}"


@dataclass(frozen=True)
class Overrides:
    """One trial's field overrides. ``apply`` runs inside ``build_config``, so the fingerprint sees them."""

    items: tuple[Override, ...] = ()
    # Paths apply() consumed, for the unapplied-override check. Instrumentation only: mutating the
    # set on a frozen instance is allowed (compare=False, init=False).
    _applied: set[tuple[str, ...]] = field(default_factory=set, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        # Sorted paths put a path before every path beneath it. Overlapping overrides would apply in
        # command-line order, while the trial's name and run ID sort them.
        paths = sorted(item.path for item in self.items)
        overlaps = [
            f"{'.'.join(outer)} and {'.'.join(inner)}"
            for outer, inner in itertools.combinations(paths, 2)
            if inner[: len(outer)] == outer
        ]
        if overlaps:
            raise OverrideError(f"one trial sets a field and the same field or one beneath it: {'; '.join(overlaps)}")

    def apply(self, ctx: StepContext, config: T) -> T:
        """``config`` with every override decoded into place."""
        for item in self.items:
            config = _replace_at(ctx, config, item, item.path)
            self._applied.add(item.path)
        return config

    def unapplied(self) -> list[str]:
        """Paths ``apply`` never consumed: the experiment ignored its overrides."""
        return sorted(".".join(item.path) for item in self.items if item.path not in self._applied)

    @property
    def digest(self) -> str:
        return fingerprint_hash(canonical_json(sorted(item.label for item in self.items)))

    def named(self, name: str) -> str:
        """``name``, or ``name/<overrides>``: readable when every value is plain, else ``set-<digest>``."""
        if not self.items:
            return name
        readable = ",".join(sorted(item.label for item in self.items))
        plain = all(_PLAIN_VALUE.match(_normalized(item.text)) for item in self.items)
        if plain and len(readable) <= _MAX_READABLE_SUFFIX:
            return f"{name}/{readable}"
        return f"{name}/set-{self.digest}"

    def run_id(self, base: str) -> str:
        """``base``, or ``base-<digest>``, so each trial logs to its own tracker run."""
        return base if not self.items else f"{base}-{self.digest}"


def _replace_at(ctx: StepContext, obj: Any, item: Override, path: tuple[str, ...]) -> Any:
    """``obj`` with ``item``'s value at ``path`` (the rest of ``item.path`` below ``obj``)."""
    full = ".".join(item.path)
    head, *rest = path
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        raise OverrideError(f"{full}: reaches a {type(obj).__name__}, which is not a dataclass with fields")
    names = [f.name for f in dataclasses.fields(obj)]
    if head not in names:
        close = difflib.get_close_matches(head, names, n=3)
        hint = f"; did you mean {', '.join(close)}?" if close else ""
        raise OverrideError(f"{full}: {type(obj).__name__} has no field {head!r}{hint}")
    current = getattr(obj, head)
    if ctx.is_context_value(current):
        raise OverrideError(
            f"{full}: its value comes from the step context ({current}), so it is not part of the recipe"
        )
    if rest:
        return dataclasses.replace(obj, **{head: _replace_at(ctx, current, item, tuple(rest))})
    field_type = field_types(type(obj))[head]
    try:
        value = draccus.decode(field_type, parse_string(item.text))
    except (yaml.YAMLError, ParsingError) as e:
        raise OverrideError(f"{full}={item.text}: {e}") from e
    return dataclasses.replace(obj, **{head: value})

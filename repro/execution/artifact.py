# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The realized artifact, its on-disk record, and the drift check.

An :class:`Artifact` is the produced, persisted output of a step; an ``ArtifactStep`` is the
inert handle that builds one. This module owns the
framework — one base type and the record around it — while concrete artifact types live with
their producers (``ToyCheckpoint`` in ``project.toy``):

- :class:`Artifact` — a directory with a record (provenance + an optional JSON payload) and a
  ``raw_load`` that reads it back. The default ``raw_load`` returns a handle into the path; a subclass
  that declares value fields round-trips them through the record's ``result`` automatically.
- :class:`ArtifactRecord` — the single descriptor written next to a step's output: its config,
  fingerprint, provenance, and (for a value artifact) its ``result``.
- ``read_record``/``write_record`` (the full record) and ``read_artifact``/``write_artifact``
  (the manual typed-payload API), two entry points over one serialization scheme.
- :func:`recorded_recipe` — the recipe an output records, which the drift check compares.
"""

import functools
import os
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Self, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from repro import settings
from repro.provenance import Provenance

M = TypeVar("M", bound=BaseModel)

# JSON-shaped value, used for the human-readable config and the value payload.
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

# The record file written next to every output, and the only record name read back. Dot-prefixed so
# the datakit ``normalize._discover_files`` walk and the tokenizer file filter — both skip dotfiles —
# never mistake it for training data and parse it as JSONL.
RECORD_FILENAME = ".artifact.json"

# The recipe a build records when it starts, before its record exists: a started build's
# fingerprint, so a rerun under another recipe is refused instead of resuming it. Kept apart from
# the record so an unfinished output never reads as built.
STARTED_FILENAME = ".artifact.started.json"

ARTIFACT_LOAD_CONTEXT_KEY = "artifact_load"


class FingerprintMismatchError(Exception):
    """An output's recorded recipe differs from the recipe now being built, or two handles in one
    graph build the same ``name@version`` from different recipes."""


class DriftAcceptanceError(ValueError):
    """``accept_drift`` named an artifact whose recipe did not change: a typo, or a stale flag."""


class ArtifactTypeMismatchError(Exception):
    """A served record's ``result_type`` differs from the requested handle's ``result_type``."""


def result_type_name(artifact_type: "type[Artifact]") -> str:
    """The canonical ``module.Qualname`` recorded as a value artifact's ``result_type``.

    The single source of truth for both sides of the round-trip: written into the record when a
    step succeeds and compared against on :meth:`Artifact.raw_load`.
    """
    return f"{artifact_type.__module__}.{artifact_type.__qualname__}"


class Artifact(BaseModel):
    """A produced, persisted artifact: a directory with a record and a ``raw_load``.

    The default ``raw_load`` is a data ref — it returns a handle into ``path`` whose ``.record``
    carries provenance and the run's config, pulling no weights/caches into the launcher. A
    subclass that declares value fields persists and reloads them through ``record.result`` with
    no override (see :meth:`result_payload`). Not frozen: ``raw_load`` sets ``path``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    path: str = ""

    @functools.cached_property
    def record(self) -> "ArtifactRecord | None":
        """The record sidecar at ``path`` (read once), or ``None`` if absent."""
        return read_record(self.path)

    def result_payload(self) -> dict | None:
        """What the record stores as ``result``: this artifact's *declared* value fields
        (every field but ``path``), or ``None`` for a pure data ref that declares none.

        Uses ``type(self).model_fields`` rather than ``model_dump()`` so ``extra="allow"`` extras
        (e.g. the cached ``record``) never leak into the payload. Override to persist something
        else.
        """
        keys = {name for name in type(self).model_fields if name != "path"}
        if not keys:
            return None
        return self.model_dump(mode="json", include=keys) or None

    @classmethod
    def raw_load(cls, source: str) -> Self:
        """The raw, unguarded loader: a handle into ``source`` whose value fields a subclass
        repopulates from ``record.result``. Prefer :func:`run`/:func:`resolve` (driver) or
        :meth:`StepContext.resolved` (inside a step) — reach for ``raw_load`` only to read an
        artifact you already know is built.

        Tolerant of a missing record by design: a pinned artifact resolves to its pre-existing
        data location (:meth:`ArtifactStep.path`) — real data, but no record there — and its
        resolved value is intentionally path-only, so a missing record yields a path handle
        rather than an error. A normal computed handle is never loaded before it is
        built (``run``/``resolve`` build first; ``StepContext.resolved`` reads only
        runner-materialized deps), so a missing record cannot arise through the supported paths.

        When a record *is* present it must agree: raises :class:`ArtifactTypeMismatchError` if it
        was written by a different artifact class (a value type that changed under a reused
        version).
        """
        rec = read_record(source)
        if rec is not None and rec.result_type and rec.result_type != result_type_name(cls):
            raise ArtifactTypeMismatchError(
                f"{source}: recorded result_type is {rec.result_type}, but loading as "
                f"{result_type_name(cls)}. The value type changed under a reused version — bump the version."
            )
        return cls(path=source, **((rec.result if rec is not None else None) or {}))


class ArtifactRecord(BaseModel):
    """The single descriptor written next to a step's output.

    All fields carry a default, so a minimal manual record (:func:`write_artifact`) loads
    without error; the lazy runner fills them all.
    """

    name: str = ""
    version: str = ""
    fingerprint: str = ""
    result_type: str = ""
    output_path: str = ""
    deps: list[str] = Field(default_factory=list)
    """Dependency identities as ``name@version`` strings."""
    config: dict[str, JSONValue] | None = None
    """The materialized config that ran (canonical-encoded), for humans and consumer metadata."""
    result: dict[str, JSONValue] | None = None
    """A value artifact's declared fields; ``None`` for a data artifact."""
    fingerprint_payload: str | None = None
    """The canonical config JSON the ``fingerprint`` hashes, kept for the drift diff."""
    accepted_from: list[str] = Field(default_factory=list)
    """Earlier fingerprints of this output whose changed recipe the owner accepted (``--accept-drift``)."""
    provenance: Provenance | None = None
    """Who/when/which-commit/which-argv produced this — ``None`` for a minimal manual write."""


# The one canonical CalVer form (``YYYY.MM.DD`` with an optional ``.N`` for two immutable
# revisions on the same day). Kept here, beside ``validate_version``, so callers agree on version
# identity without importing ``lazy``.
CALVER_RE = re.compile(r"^\d{4}\.\d{2}\.\d{2}(\.\d+)?$")


def is_mutable_version(version: str) -> bool:
    """A ``dev`` version is mutable: it rebuilds, unless the step is pinned, which is served as is. Its
    output still records its recipe, so the drift check still applies."""
    return version == "dev" or version.endswith("-dev")


def validate_path_segment(label: str, value: str) -> None:
    """A ``name``/``version`` is a single path segment: non-empty, no URL scheme, no ``..``, no
    leading/trailing slash. A malformed one is a caller bug, not a silent malformed path."""
    if not value:
        raise ValueError(f"{label} must be non-empty")
    if "://" in value or urlparse(value).scheme:
        raise ValueError(f"{label} {value!r} must not contain a URL scheme")
    if ".." in value:
        raise ValueError(f"{label} {value!r} must not contain '..'")
    if value.startswith("/") or value.endswith("/"):
        raise ValueError(f"{label} {value!r} must not start or end with '/'")


def validate_version(version: str) -> None:
    """Validate an artifact version string, raising :class:`ValueError` if malformed.

    A version is a path segment that is either a calendar version ``YYYY.MM.DD`` (optionally
    ``YYYY.MM.DD.N``) or a mutable ``dev``/``<label>-dev``. ``v1``-style tags are rejected: an
    artifact's version is the author's explicit statement of "when this recipe was frozen", not an
    opaque label.
    """
    validate_path_segment("version", version)
    if is_mutable_version(version):
        return
    if not CALVER_RE.match(version):
        raise ValueError(
            f"version {version!r} must be a calendar version YYYY.MM.DD (optionally YYYY.MM.DD.N) "
            "or a mutable 'dev'/'<label>-dev'"
        )


def resolve_path(prefix: str, path: str) -> str:
    """A relative ``path`` rooted at ``prefix``; an absolute path is used as-is."""
    return path if os.path.isabs(path) else os.path.join(prefix, path)


def artifact_path(prefix: str, name: str, version: str) -> str:
    """The output directory of ``name@version`` under ``prefix``: ``{prefix}/{name}/{version}``."""
    return os.path.join(prefix, name, version)


def _read_text(output_path: str, filename: str) -> str | None:
    path = Path(output_path) / filename
    return path.read_text() if path.exists() else None


def _write_text(output_path: str, filename: str, text: str) -> None:
    directory = Path(output_path)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(text)


def read_record(output_path: str) -> ArtifactRecord | None:
    """The record at ``{output_path}/.artifact.json``, else ``None``. A relative ``output_path`` is
    rooted at the prefix. A corrupt or partial file raises :class:`pydantic.ValidationError`."""
    text = _read_text(resolve_path(settings.prefix(), output_path), RECORD_FILENAME)
    return ArtifactRecord.model_validate_json(text) if text is not None else None


class StartedRecipe(BaseModel):
    """What :data:`STARTED_FILENAME` holds: the recipe of a build that started."""

    fingerprint: str
    fingerprint_payload: str


def write_started_recipe(output_path: str, fingerprint: str, payload: str) -> None:
    started = StartedRecipe(fingerprint=fingerprint, fingerprint_payload=payload)
    _write_text(output_path, STARTED_FILENAME, started.model_dump_json(indent=2))


@dataclass(frozen=True)
class RecordedRecipe:
    """The recipe an existing output was built, or started, from."""

    fingerprint: str
    payload: str | None
    """The canonical config JSON the fingerprint hashes, for the drift diff; ``None`` when not recorded."""
    finished: bool
    """Whether the record holds it; else the build only started."""


def recorded_recipe(output_path: str) -> RecordedRecipe | None:
    """The recipe recorded at ``output_path``: the record's when the build finished, else the one
    written when it started. ``None`` when neither records a fingerprint (nothing built, or a
    manual :func:`write_artifact`)."""
    record = read_record(output_path)
    if record is not None:
        return RecordedRecipe(record.fingerprint, record.fingerprint_payload, True) if record.fingerprint else None
    text = _read_text(output_path, STARTED_FILENAME)
    if text is None:
        return None
    started = StartedRecipe.model_validate_json(text)
    return RecordedRecipe(started.fingerprint, started.fingerprint_payload, False)


def accept_recipe(output_path: str, fingerprint: str, payload: str) -> None:
    """Record ``fingerprint`` as the recipe of the output at ``output_path``: the owner accepted a
    changed recipe. A finished build's record keeps the old fingerprint in ``accepted_from``; a
    started build's recipe is replaced, so the build resumes under the new recipe. The record is
    written back where it was read, with ``output_path`` updated, even when it names the prefix it
    was built under before the output was copied."""
    record = read_record(output_path)
    if record is None:
        if _read_text(output_path, STARTED_FILENAME) is None:
            raise ValueError(f"no recorded recipe at {output_path} to accept a change to")
        write_started_recipe(output_path, fingerprint, payload)
        return
    write_record(
        record.model_copy(
            update={
                "output_path": output_path,
                "fingerprint": fingerprint,
                "fingerprint_payload": payload,
                "accepted_from": [*record.accepted_from, record.fingerprint],
            }
        )
    )


def drift_message(name: str, version: str, recorded: str, now: str, change: str) -> str:
    """Why ``name@version`` cannot reuse its output, and the two ways forward."""
    return (
        f"{name}@{version}: the output records recipe {recorded}, but the recipe is now {now}.{change}\n"
        f"Change the version to build the new recipe, or pass --accept-drift {name} to keep the output."
    )


def write_record(record: ArtifactRecord) -> None:
    """Write ``record`` to ``{record.output_path}/.artifact.json``."""
    _write_text(record.output_path, RECORD_FILENAME, record.model_dump_json(indent=2))


def payload_json(value: object) -> JSONValue:
    """Canonical JSON encoding for a record payload: a ``BaseModel`` or dataclass to its dict,
    anything already JSON-shaped straight through."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value  # ty: ignore[invalid-return-type]


def read_artifact(output_path: str, schema: type[M]) -> M:
    """Load a typed payload: ``read_record(output_path).result`` validated as ``schema``.

    Raises :class:`FileNotFoundError` if no record carrying a ``result`` is present.
    """
    if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
        raise TypeError(f"schema must be a pydantic BaseModel subclass, got {schema!r}")
    record = read_record(output_path)
    if record is not None and record.result is not None:
        return schema.model_validate(record.result, context={ARTIFACT_LOAD_CONTEXT_KEY: True})
    raise FileNotFoundError(f"no artifact payload at {output_path}")


def write_artifact(value: object, output_path: str) -> None:
    """Write a minimal record carrying ``value`` as its ``result`` — the manual save API."""
    write_record(ArtifactRecord(output_path=output_path, result=payload_json(value)))

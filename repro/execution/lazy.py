# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact steps addressed by an explicit ``name@version``.

The model:

- An :class:`ArtifactStep` is a content-free handle to a versioned artifact:
  ``(name, version, artifact_type)`` plus how to build it — ``run``, ``build_config``, ``deps``.
  Constructing one runs nothing; :func:`run` / :func:`resolve` execute it.
- ``build_config(ctx)`` is a pure function of a :class:`StepContext`, which draws the line between
  identity and execution: values written as literals (model, hyperparameters, dep *versions*)
  bear identity, while values *pulled* from the context — ``ctx.output_path``,
  ``ctx.artifact_path(dep)``, ``ctx.prefix``, ``ctx.runtime_arg(key)`` — are
  where/how the step runs and never do.
- ``run(config)`` is the step function: a plain callable. It returns the produced
  :class:`~repro.execution.artifact.Artifact` (a data ref or a value), or ``None`` for a data
  step that writes its bytes to ``ctx.output_path``.
- Identity is the explicit ``{prefix}/{name}/{version}`` path — no content hash. The recipe
  *fingerprint* (the config built with dep versions in place of paths and placeholders for
  everything pulled from ``ctx``) is recorded beside the output, never in the path, and a changed
  recipe under an existing output is refused.

Existing data is referenced with a pin (``override_path``). :func:`run` builds handles and loads
their typed artifacts; :func:`resolve` does that for one. A custom step is
just a function that constructs and returns an ``ArtifactStep`` — there is no generic wrapper to
learn.
"""

import inspect
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Generic, TypeVar, cast

from repro import settings
from repro.execution.artifact import (
    Artifact,
    ArtifactRecord,
    FingerprintMismatchError,
    accept_recipe,
    artifact_path,
    drift_message,
    is_mutable_version,
    resolve_path,
    result_type_name,
    validate_path_segment,
    validate_version,
    write_record,
)
from repro.execution.build_context import resolve_version
from repro.execution.cache_state import CacheState, HandleState, check_accept_drift, graph_states
from repro.execution.fingerprint import canonical_json, fingerprint_hash
from repro.execution.step_runner import DEFAULT_MAX_CONCURRENT, StepRunner
from repro.execution.step_spec import StepSpec
from repro.provenance import launch_provenance

T = TypeVar("T", bound=Artifact)


@dataclass(frozen=True)
class StepContext:
    """What a step resolves its config against — and the dividing line for identity.

    Values written as literals in ``build_config`` (the model, hyperparameters, dep *versions*)
    bear identity and enter the fingerprint; values pulled from the context never do.
    :meth:`for_run` binds the live environment; :meth:`for_fingerprint` substitutes placeholders,
    which is what realizes the exclusion.

    Pull points: ``ctx.output_path`` (this step's output dir), ``ctx.artifact_path(dep)`` (a dep's
    resolved path), ``ctx.resolved(dep)`` (a dep's loaded, typed artifact — its value/metrics),
    ``ctx.prefix`` (the live storage prefix), and ``ctx.runtime_arg(key)`` (a step-declared runtime
    arg, e.g. a device or a logging interval).
    ``artifact_path`` and ``resolved`` accept only declared ``deps``: a handle the step did not
    list is a bug (the runner never materialized it), so both raise on one. ``is_fingerprint`` is
    ``True`` only under :meth:`for_fingerprint`, for a ``build_config`` that must emit an identity
    summary (rather than read records that do not exist yet).
    """

    output_path: str
    prefix: str
    is_fingerprint: bool
    _dep_ref: Callable[["ArtifactStep"], str]
    _runtime_args: Mapping[str, Any]
    _deps: tuple["ArtifactStep", ...] = ()
    _loaded: dict[int, Artifact] = field(default_factory=dict)

    def artifact_path(self, dep: "ArtifactStep") -> str:
        """The resolved output path of a dependency this step builds on."""
        self._require_declared(dep)
        return self._dep_ref(dep)

    def resolved(self, dep: "ArtifactStep[T]") -> T:
        """The loaded, typed artifact of a declared dependency — its value/metrics, not just a path.

        Reads the dep's record once and caches it, so resolving the same dep N times costs one
        store read. Loads the record sidecar (a data ref into the path), not the heavy payload, and
        never runs anything: the runner has materialized every declared dep before this step's
        function executes. Use ``.path`` on the result for the output dir.

        Raises :class:`ValueError` for an undeclared dep, and at fingerprint time (no records exist
        yet — branch on ``is_fingerprint`` and emit an identity summary instead).
        """
        if self.is_fingerprint:
            raise ValueError("ctx.resolved is unavailable at fingerprint time; branch on ctx.is_fingerprint")
        self._require_declared(dep)
        cached = self._loaded.get(id(dep))
        if cached is None:
            cached = dep.artifact_type.raw_load(self._dep_ref(dep))
            self._loaded[id(dep)] = cached
        return cast(T, cached)

    def _require_declared(self, dep: "ArtifactStep") -> None:
        if dep not in self._deps:
            raise ValueError(
                f"{dep.label} is not a declared dependency of this step; "
                "add it to deps=(...) so the runner materializes it before this step runs"
            )

    def runtime_arg(self, key: str) -> Any:
        """A step-declared runtime arg: its real value at run time, a ``<key>`` placeholder at
        fingerprint time. Pull execution choices through here so they reach the step but never
        bear on its identity."""
        try:
            return self._runtime_args[key]
        except KeyError:
            raise KeyError(
                f"runtime arg {key!r} is not declared in the step's runtime_args {sorted(self._runtime_args)}"
            ) from None

    def is_context_value(self, value: object) -> bool:
        """Whether ``value`` is what this fingerprint context put in place of a pulled value or a
        dependency reference, or a string built from a pulled value (``<output_path>/hf``): the
        parts of a config an override must not replace. Always ``False`` at run time, where the
        fingerprint pass has already refused them."""
        if not self.is_fingerprint or not isinstance(value, str):
            return False
        placeholders = (self.output_path, self.prefix, *self._runtime_args.values())
        if any(placeholder in value for placeholder in placeholders if placeholder):
            return True
        return value in {self._dep_ref(dep) for dep in self._deps}

    @staticmethod
    def for_run(
        output_path: str,
        prefix: str,
        *,
        runtime_args: Mapping[str, Any] | None = None,
        deps: Iterable["ArtifactStep"] = (),
    ) -> "StepContext":
        return StepContext(
            output_path=output_path,
            prefix=prefix,
            is_fingerprint=False,
            _dep_ref=lambda d: d.path(prefix),
            _runtime_args=runtime_args or {},
            _deps=tuple(deps),
        )

    @staticmethod
    def for_fingerprint(runtime_arg_keys: Iterable[str] = (), deps: Iterable["ArtifactStep"] = ()) -> "StepContext":
        return StepContext(
            output_path="<output_path>",
            prefix="<prefix>",
            is_fingerprint=True,
            _dep_ref=lambda d: d.label,
            _runtime_args={key: f"<{key}>" for key in runtime_arg_keys},
            _deps=tuple(deps),
        )


# eq=False gives handles object identity (hashable despite unhashable values in their config),
# so a handle can key a mixture dict. Value-equality lives in name@version and the fingerprint.
@dataclass(frozen=True, eq=False)
class ArtifactStep(Generic[T]):
    """A lazy, content-free handle to a versioned artifact and how to build it.

    ``run(build_config(ctx))`` produces the artifact: ``build_config`` assembles the config from
    the :class:`StepContext` (literals bear identity; pulled values do not), and ``run`` is the
    plain callable that does the work, returning the produced ``artifact_type`` (or ``None`` for a
    data step that writes to ``ctx.output_path``). ``deps`` are the handles this step builds on;
    pass the same handles ``build_config`` reads via ``ctx.artifact_path`` so they materialize
    first.
    """

    name: str
    version: str
    artifact_type: type[T]
    run: Callable[[Any], "Artifact | None"]
    build_config: Callable[[StepContext], Any]
    deps: tuple["ArtifactStep", ...] = ()
    runtime_args: Mapping[str, Any] = field(default_factory=dict)
    """Execution choices the config pulls via ``ctx.runtime_arg(key)``. Excluded from the
    fingerprint, so changing one never forks identity."""
    override_path: str | None = None
    """Pin the artifact to an existing location instead of ``{name}/{version}``. A relative
    pin is resolved against the prefix; an absolute one is used as-is. Writes no record."""

    def __post_init__(self) -> None:
        validate_path_segment("name", self.name)
        validate_version(self.version)
        if not is_mutable_version(self.version):
            for dep in self.deps:
                # A pin references existing data and never rebuilds, so a fixed parent may cite one.
                if is_mutable_version(dep.version) and dep.override_path is None:
                    raise ValueError(
                        f"{self.label} is a fixed version but depends on mutable "
                        f"{dep.label}; a mutable dep would rebuild while the fixed parent "
                        "stays cached. Make the parent dev, or pin the dep."
                    )

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    def fingerprint_payload(self) -> str:
        """The canonical config bytes this handle's fingerprint hashes: the config built with dep
        versions for paths and placeholders for pulled values."""
        config = self.build_config(StepContext.for_fingerprint(self.runtime_args.keys(), self.deps))
        return canonical_json(config)

    def fingerprint(self) -> str:
        """Stable id of *how* this artifact is built. Changes iff a config value or a dep
        version changes; independent of the storage prefix."""
        return fingerprint_hash(self.fingerprint_payload())

    def path(self, prefix: str | None = None) -> str:
        """The output directory: the pin's location, else ``{prefix}/{name}/{version}``."""
        resolved_prefix = prefix or settings.prefix()
        if self.override_path is not None:
            return resolve_path(resolved_prefix, self.override_path)
        return artifact_path(resolved_prefix, self.name, self.version)


def materialized_config(handle: "ArtifactStep", prefix: str) -> Any:
    """The concrete config a run would receive, for inspection/golden tests.

    Resolves ``ctx.output_path`` / ``ctx.artifact_path(dep)`` against ``prefix`` and passes the
    handle's real ``runtime_args`` (so a ``build_config`` that reads ``ctx.runtime_arg(...)`` does
    not ``KeyError``) without running anything.
    """
    return handle.build_config(
        StepContext.for_run(
            output_path=handle.path(prefix), prefix=prefix, runtime_args=handle.runtime_args, deps=handle.deps
        )
    )


def _lower(handle: "ArtifactStep", memo: dict[int, StepSpec]) -> StepSpec:
    """Lower a handle graph into a ``StepSpec`` graph.

    Each handle becomes a step at its output path, carrying its version, fingerprint, and pin, so
    the runner applies the drift check before serving a cached output (skipping a pin). The step fn
    writes a full :class:`ArtifactRecord` on success, unless pinned, with the launch's provenance
    (:func:`~repro.provenance.launch_provenance`). Pure transform of *structure*: it never
    inspects ``handle.run``.

    ``memo`` caches the spec of each handle by identity, so a handle several consumers share is
    lowered once. Without it a diamond re-walks the shared subgraph, and re-runs its
    ``build_config``, once per path through it.
    """
    cached = memo.get(id(handle))
    if cached is not None:
        return cached
    dep_specs = [_lower(dep, memo) for dep in handle.deps]
    payload = handle.fingerprint_payload()
    fingerprint = fingerprint_hash(payload)
    pinned = handle.override_path is not None

    def fn(output_path: str, _handle: "ArtifactStep" = handle) -> None:
        ctx = StepContext.for_run(
            output_path=output_path,
            prefix=settings.prefix(),
            runtime_args=_handle.runtime_args,
            deps=_handle.deps,
        )
        config = _handle.build_config(ctx)
        result = _handle.run(config)
        if pinned:
            return  # A pin references existing data; it writes no record.
        write_record(
            ArtifactRecord(
                name=_handle.name,
                version=_handle.version,
                fingerprint=fingerprint,
                result_type=result_type_name(_handle.artifact_type),
                output_path=output_path,
                deps=[dep.label for dep in _handle.deps],
                config=json.loads(canonical_json(config)),
                result=result.result_payload() if isinstance(result, Artifact) else None,
                fingerprint_payload=payload,
                provenance=launch_provenance(),
            )
        )

    spec = StepSpec(
        name=handle.name,
        version=handle.version,
        output_path=handle.path(),
        fingerprint=fingerprint,
        fingerprint_payload=payload,
        fn=fn,
        deps=dep_specs,
        pinned=pinned,
    )
    memo[id(handle)] = spec
    return spec


def _refuse_drift(states: list[HandleState], accept_drift: Collection[str]) -> None:
    """Refuse a run if any artifact in the graph records a different recipe, unless accepted.

    ``states`` covers every artifact, dependencies the runner would prune behind a built consumer
    included (:func:`~repro.execution.cache_state.graph_states`). An accepted artifact's recorded
    recipe is re-stamped, so the runner serves or resumes it.
    """
    check_accept_drift(states, accept_drift)
    drifted = [s for s in states if s.state is CacheState.DRIFTED]
    refused = [s for s in drifted if s.handle.name not in accept_drift]
    if refused:
        raise FingerprintMismatchError(
            "\n\n".join(
                drift_message(s.handle.name, s.handle.version, s.recorded_fingerprint or "", s.fingerprint, s.change)
                for s in refused
            )
        )
    for state in drifted:
        accept_recipe(state.handle.path(), state.fingerprint, state.payload)


def lower_all(handles: Sequence["ArtifactStep"]) -> list[StepSpec]:
    """Lower ``handles`` into one ``StepSpec`` graph, sharing common steps."""
    memo: dict[int, StepSpec] = {}
    return [_lower(h, memo) for h in handles]


def run(
    *handles: "ArtifactStep[T]",
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    accept_drift: Collection[str] = (),
) -> list[T]:
    """Build ``handles`` and return their loaded, typed artifacts, in argument order.

    The entry point for one or several handles: lowers the graph, builds every step (independent
    steps in parallel up to ``max_concurrent``), then loads each top-level handle's typed artifact
    from the record the runner just wrote — so the caller gets the resolved values directly, no
    separate read. A failed step from an earlier run is rerun.

    Before anything runs, every artifact in the graph is checked against the recipe its output
    records; ``accept_drift`` names artifacts whose changed recipe is accepted.

    Every artifact built records the launch's provenance: the one installed with
    :func:`~repro.provenance.launched_as`, as the experiment command line does with its snapshot
    and note, else one captured once per process.
    """
    _refuse_drift(graph_states(graph_handles(list(handles))), accept_drift)
    StepRunner().run(lower_all(handles), max_concurrent=max_concurrent)
    return [h.artifact_type.raw_load(h.path()) for h in handles]


def resolve(handle: "ArtifactStep[T]", *, max_concurrent: int = DEFAULT_MAX_CONCURRENT) -> T:
    """Build one ``handle`` and return its loaded, typed artifact — :func:`run` for a single handle.

    The driver-side "build this and give me the value" verb. To resolve several handles, prefer
    one :func:`run` call (which builds them in parallel and returns all their artifacts) over
    ``resolve``-ing in a loop.
    """
    return run(handle, max_concurrent=max_concurrent)[0]


def graph_handles(handles: list[ArtifactStep]) -> list[ArtifactStep]:
    """Every handle reachable from ``handles`` (deps included), deduped by identity, deps first."""
    seen: set[int] = set()
    order: list[ArtifactStep] = []

    def walk(handle: ArtifactStep) -> None:
        if id(handle) in seen:
            return
        seen.add(id(handle))
        for dep in handle.deps:
            walk(dep)
        order.append(handle)

    for handle in handles:
        walk(handle)
    return order


class _OutSentinel:
    """The :data:`OUT` sentinel for :func:`apply`: resolves to ``ctx.output_path`` at run time."""

    def __repr__(self) -> str:
        return "OUT"


OUT: Final = _OutSentinel()


def _collect_deps(inputs: Mapping[str, Any]) -> list["ArtifactStep"]:
    """The ``ArtifactStep`` handles in ``inputs`` (recursing list/tuple/dict), deduped by
    identity, in first-seen order."""
    deps: list[ArtifactStep] = []
    seen: set[int] = set()

    def walk(value: Any) -> None:
        if isinstance(value, ArtifactStep):
            if id(value) not in seen:
                seen.add(id(value))
                deps.append(value)
        elif isinstance(value, list | tuple):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)

    for value in inputs.values():
        walk(value)
    return deps


def _resolve_input(value: Any, ctx: StepContext) -> Any:
    """Resolve one ``apply`` input against ``ctx``: an ``ArtifactStep`` to its path, ``OUT`` to
    ``ctx.output_path``, recursing builtin containers; anything else is a literal."""
    if isinstance(value, ArtifactStep):
        return ctx.artifact_path(value)
    if value is OUT:
        return ctx.output_path
    if isinstance(value, list):
        return [_resolve_input(item, ctx) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_input(item, ctx) for item in value)
    if isinstance(value, dict):
        return {key: _resolve_input(item, ctx) for key, item in value.items()}
    return value


def apply(
    name: str,
    fn: Callable[..., Any],
    *,
    version: str | None = None,
    artifact_type: type[T] = Artifact,  # ty: ignore[invalid-parameter-default]
    pin: str | None = None,
    **inputs: Any,
) -> "ArtifactStep[T]":
    """A single-step builder that calls ``fn(**inputs)`` directly — for a step whose function
    takes keyword arguments rather than a config object.

    Each value in ``inputs`` is classified, recursing into ``list``/``tuple``/``dict``: an
    :class:`ArtifactStep` handle becomes a dep and resolves to ``ctx.artifact_path(handle)`` at run
    time (its ``name@version`` enters identity); the :data:`OUT` sentinel resolves to
    ``ctx.output_path``; anything else is a literal that bears identity. ``artifact_type`` selects the
    produced :class:`Artifact` type; ``pin`` references existing data.

    ``version`` defaults to the ambient :func:`~repro.execution.build_context.build_context`'s
    version for ``name`` when omitted (build inside :func:`~repro.execution.build_context.build_context`);
    an explicit value always wins.

    Raises :class:`TypeError` if ``fn``'s signature cannot bind the inputs.
    """
    version = resolve_version(name, version)
    try:
        inspect.signature(fn).bind(**inputs)
    except TypeError as e:
        fn_name = getattr(fn, "__name__", repr(fn))
        raise TypeError(f"apply({name!r}): {fn_name} cannot bind the given inputs: {e}") from e

    deps = _collect_deps(inputs)

    def build_config(ctx: StepContext) -> dict[str, Any]:
        return {key: _resolve_input(value, ctx) for key, value in inputs.items()}

    def step_fn(config: dict[str, Any], _fn: Callable[..., Any] = fn) -> Any:
        return _fn(**config)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=artifact_type,
        run=step_fn,
        build_config=build_config,
        deps=tuple(deps),
        override_path=pin,
    )

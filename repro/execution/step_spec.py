# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A lowered artifact: what the step runner schedules, serves from cache, and checks for drift.

:func:`repro.execution.lazy.lower_all` lowers each ``ArtifactStep`` handle into one ``StepSpec``.
The spec carries the artifact's identity as plain fields, so the runner needs nothing from the
lazy layer to decide whether a step is built, rebuilds, or records another recipe.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from repro.execution.artifact import is_mutable_version


@dataclass(frozen=True)
class StepSpec:
    name: str
    version: str
    output_path: str
    """The resolved output directory: ``{prefix}/{name}/{version}``, or the pin's location."""
    fingerprint: str
    fingerprint_payload: str
    """The canonical config JSON behind ``fingerprint``, recorded for the drift diff."""
    fn: Callable[[str], Any]
    """Builds the artifact at the output path it is given and writes its record."""
    deps: "list[StepSpec]" = dataclasses.field(default_factory=list)
    pinned: bool = False
    """Points at existing data (``pin=``): the drift check skips it."""

    @property
    def label(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def rebuilds(self) -> bool:
        """A ``dev`` version always rebuilds, over a finished output too; a pin never does."""
        return is_mutable_version(self.version) and not self.pinned

    @cached_property
    def dep_paths(self) -> list[str]:
        return [dep.output_path for dep in self.deps]

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reading a recorded config value back into its dataclass (repro.execution.recorded_config)."""

import json
import subprocess
import sys
from dataclasses import dataclass, field

import pytest
from draccus.utils import DecodingError

from repro.execution.fingerprint import canonical_json
from repro.execution.recorded_config import decode_recorded
from tests.choice_configs import AdamConfig, Component, DataConfig, HfSource, OptimizerConfig, UrlSource


@dataclass(frozen=True)
class Clip:
    history: int = 2
    horizon: int = 2


@dataclass(frozen=True)
class Recipe:
    clip: Clip
    optimizer: OptimizerConfig
    adam: AdamConfig
    label: str | None
    sizes: tuple[int, ...] = (1, 2)
    weights: dict[str, float] = field(default_factory=dict)


def _recorded(value: object) -> object:
    """``value`` as a record stores it."""
    return json.loads(canonical_json(value))


def test_a_recorded_config_decodes_to_the_dataclass_that_wrote_it():
    recipe = Recipe(
        clip=Clip(history=4),
        optimizer=AdamConfig(learning_rate=3e-3),
        adam=AdamConfig(),
        label=None,
        weights={"a": 0.5},
    )
    assert decode_recorded(Recipe, _recorded(recipe)) == recipe


def test_a_recorded_mixture_decodes_with_its_component_choices():
    data = DataConfig(
        components={
            "wikitext": Component(source=UrlSource(cache_dir="/c/wikitext", urls=("u",))),
            "fineweb": Component(source=HfSource(repo="hf/fineweb"), weight=0.5),
        },
    )
    assert _recorded(decode_recorded(DataConfig, _recorded(data))) == _recorded(data)


def test_a_field_the_class_no_longer_has_is_refused():
    recorded = _recorded(Clip())
    recorded["num_workers"] = 16
    with pytest.raises(DecodingError, match="num_workers"):
        decode_recorded(Clip, recorded)


def test_a_recorded_class_that_is_not_the_declared_one_is_refused():
    with pytest.raises(ValueError, match="adam"):
        decode_recorded(Clip, _recorded(AdamConfig()))


# Draccus codecs are registered process-wide, so the test decodes in an interpreter that imports
# nothing but the decoder: a codec registered elsewhere (by the launcher) cannot mask a missing one.
_DECODE_A_RECORDED_TIMEDELTA = """
import dataclasses, datetime, json
from repro.execution.fingerprint import canonical_json
from repro.execution.recorded_config import decode_recorded

@dataclasses.dataclass(frozen=True)
class Schedule:
    interval: datetime.timedelta

recorded = json.loads(canonical_json(Schedule(datetime.timedelta(minutes=10))))
assert decode_recorded(Schedule, recorded).interval == datetime.timedelta(minutes=10)
"""


def test_a_recorded_timedelta_decodes_without_the_launcher_imported():
    result = subprocess.run(
        [sys.executable, "-c", _DECODE_A_RECORDED_TIMEDELTA], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr

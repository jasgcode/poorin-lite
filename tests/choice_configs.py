"""Draccus choice types for tests: a registry field, as an optimizer or a data source is in a real config."""

from dataclasses import dataclass, field

import draccus


@dataclass(frozen=True)
class OptimizerConfig(draccus.ChoiceRegistry):
    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    @classmethod
    def default_choice_name(cls) -> str | None:
        return "adam"


@OptimizerConfig.register_subclass("adam")
@dataclass(frozen=True)
class AdamConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.95


@OptimizerConfig.register_subclass("muon")
@dataclass(frozen=True)
class MuonConfig(OptimizerConfig):
    momentum: float = 0.95


@OptimizerConfig.register_subclass("sgd")
@dataclass(frozen=True)
class SgdConfig(OptimizerConfig):
    momentum: float = 0.95


@dataclass(frozen=True)
class SourceConfig(draccus.ChoiceRegistry):
    cache_dir: str = ""


@SourceConfig.register_subclass("url")
@dataclass(frozen=True)
class UrlSource(SourceConfig):
    urls: tuple[str, ...] = ()


@SourceConfig.register_subclass("hf")
@dataclass(frozen=True)
class HfSource(SourceConfig):
    repo: str = ""


@dataclass(frozen=True)
class Component:
    source: SourceConfig
    weight: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    components: dict[str, Component] = field(default_factory=dict)
    shuffle: bool = True

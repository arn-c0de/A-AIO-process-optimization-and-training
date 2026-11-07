"""Deprecated (plan A4): typed view of four config sections that nothing consumed.

:func:`simple_sim.config.validate_config` no longer builds (and discards) a
:class:`SimulationConfig`; the same casts and error messages live in
:func:`simple_sim.config.core_section_values`. This module is kept as a thin shim for external
importers and warns on import; it will be removed with the next major cleanup.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from simple_sim.config import core_section_values

warnings.warn(
    "simple_sim.config_schema is deprecated; validation lives in simple_sim.config.validate_config "
    "(casts: simple_sim.config.core_section_values)",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    seed: int
    schema_version: int
    mode: str


@dataclass(frozen=True)
class RoiConfig:
    width_px: float
    height_px: float
    mm_per_px: float


@dataclass(frozen=True)
class TrainConfig:
    model: str
    epochs: int
    batch_size: int
    lr: float
    optimizer: str
    weight_decay: float
    pretrained: bool | None
    num_workers: int | None


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int
    num_workers: int | None


@dataclass(frozen=True)
class SimulationConfig:
    run: RunConfig
    roi: RoiConfig
    train: TrainConfig
    eval: EvalConfig


def parse_config_typed(cfg: dict[str, Any]) -> SimulationConfig:
    """Deprecated: :func:`simple_sim.config.core_section_values` wrapped in dataclasses."""
    v = core_section_values(cfg)
    return SimulationConfig(run=RunConfig(**v["run"]), roi=RoiConfig(**v["roi"]),
                            train=TrainConfig(**v["train"]), eval=EvalConfig(**v["eval"]))

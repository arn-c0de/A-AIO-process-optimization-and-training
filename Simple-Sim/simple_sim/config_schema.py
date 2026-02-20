"""Typed config schema helpers for runtime config dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


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
    pretrained: Optional[bool]
    num_workers: Optional[int]


@dataclass(frozen=True)
class EvalConfig:
    batch_size: int
    num_workers: Optional[int]


@dataclass(frozen=True)
class SimulationConfig:
    run: RunConfig
    roi: RoiConfig
    train: TrainConfig
    eval: EvalConfig



def parse_config_typed(cfg: Dict[str, Any]) -> SimulationConfig:
    """Parse minimal typed config view used for strict baseline checks."""
    try:
        run_raw = cfg["run"]
        roi_raw = cfg["roi"]
        train_raw = cfg["train"]
        eval_raw = cfg["eval"]
    except Exception as exc:
        raise ValueError(f"Missing required top-level section: {exc}") from exc

    run = RunConfig(
        run_id=str(run_raw.get("run_id", "")),
        seed=int(run_raw.get("seed", -1)),
        schema_version=int(run_raw.get("schema_version", -1)),
        mode=str(run_raw.get("mode", "")),
    )
    roi = RoiConfig(
        width_px=float(roi_raw.get("width_px", 0)),
        height_px=float(roi_raw.get("height_px", 0)),
        mm_per_px=float(roi_raw.get("mm_per_px", 0)),
    )
    train = TrainConfig(
        model=str(train_raw.get("model", "")),
        epochs=int(train_raw.get("epochs", 0)),
        batch_size=int(train_raw.get("batch_size", 0)),
        lr=float(train_raw.get("lr", 0.0)),
        optimizer=str(train_raw.get("optimizer", "")),
        weight_decay=float(train_raw.get("weight_decay", 0.0)),
        pretrained=train_raw.get("pretrained", None),
        num_workers=train_raw.get("num_workers", None),
    )
    eval_cfg = EvalConfig(
        batch_size=int(eval_raw.get("batch_size", 0)),
        num_workers=eval_raw.get("num_workers", None),
    )
    return SimulationConfig(run=run, roi=roi, train=train, eval=eval_cfg)

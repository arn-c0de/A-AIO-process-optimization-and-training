from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from simple_sim.io_utils import atomic_write_text
from simple_sim.io_utils import load_yaml as read_yaml_file


def range_mid(ranges: dict[str, Any], name: str, fallback: float) -> float:
    """Midpoint of the ``[lo, hi]`` range ``ranges[name]`` (``geometry_ranges`` of a profile); *fallback* when
    the entry is missing, not a two-element list or not numeric."""
    raw = ranges.get(name, [fallback, fallback])
    if isinstance(raw, list) and len(raw) == 2:
        try:
            return (float(raw[0]) + float(raw[1])) * 0.5
        except (TypeError, ValueError):
            return fallback
    return fallback


class YamlError(Exception):
    """Raised when YAML cannot be read/written."""


def load_yaml(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise YamlError(f"YAML file not found: {path}")
    try:
        data = read_yaml_file(path)
    except (OSError, yaml.YAMLError) as exc:
        raise YamlError(f"Failed to read YAML {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise YamlError(f"Expected mapping at top-level in {path}")
    return data


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        atomic_write_text(path, dump_yaml(data))
    except (OSError, yaml.YAMLError) as exc:
        raise YamlError(f"Failed to write YAML {path}: {exc}") from exc

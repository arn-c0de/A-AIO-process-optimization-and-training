from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class YamlError(Exception):
    """Raised when YAML cannot be read/written."""


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise YamlError(f"YAML file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        raise YamlError(f"Failed to read YAML {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise YamlError(f"Expected mapping at top-level in {path}")
    return data


def dump_yaml(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=False)


def save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = dump_yaml(data)
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        raise YamlError(f"Failed to write YAML {path}: {exc}") from exc

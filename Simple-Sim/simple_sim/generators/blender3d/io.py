"""I/O helpers for Blender 3D batch rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections.abc import Iterable


def write_jobs_jsonl(path: Path, jobs: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=True) + "\n")

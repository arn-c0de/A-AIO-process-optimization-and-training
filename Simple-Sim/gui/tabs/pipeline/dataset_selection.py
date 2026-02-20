"""Helpers for serializing dataset multi-selection values."""

from __future__ import annotations

import json
from pathlib import Path


def decode_dataset_paths_json(raw: str, *, sim_root: Path) -> list[Path]:
    try:
        obj = json.loads(raw) if raw else []
    except Exception:
        return []
    if not isinstance(obj, list):
        return []
    out: list[Path] = []
    for item in obj:
        if isinstance(item, str) and item.strip():
            p = Path(item)
            if not p.is_absolute():
                p = (sim_root / p).resolve()
            out.append(p)
    return out


def encode_dataset_paths_json(paths: list[Path], *, sim_root: Path) -> str:
    uniq: list[str] = []
    seen: set[str] = set()
    for p in paths:
        rp = Path(p)
        if not rp.is_absolute():
            rp = (sim_root / rp).resolve()
        try:
            s = str(rp.resolve().relative_to(sim_root.resolve()))
        except Exception:
            s = str(rp)
        if s and s not in seen:
            seen.add(s)
            uniq.append(s)
    return json.dumps(uniq, ensure_ascii=True)

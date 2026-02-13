"""Helpers for "multi-model" bundles.

A bundle is a directory containing per-profile checkpoints, allowing a single "model" selection
to work across multiple component profiles without mixing weights.

Convention:
- A bundle is any directory (often ending with `.bundle`) that contains checkpoints named by profile.
- We store/update `bundle.json` for discoverability, but scripts can operate without it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


BUNDLE_META_NAME = "bundle.json"


def is_bundle_dir(p: Path) -> bool:
    p = Path(p)
    if not p.exists() or not p.is_dir():
        return False
    if (p / BUNDLE_META_NAME).exists():
        return True
    # Heuristic: allow `.bundle` dirs even before metadata is written.
    return p.name.endswith(".bundle")


def _safe_profile_filename(profile_id: str) -> str:
    # Keep common characters used in profile IDs; replace everything else.
    # Example: "chip_0603_resistor@1" stays intact.
    s = (profile_id or "").strip()
    if not s:
        s = "unknown_profile"
    s = re.sub(r"[^A-Za-z0-9_.@-]+", "_", s)
    return s


def bundle_checkpoint_path(bundle_dir: Path, profile_id: str, *, kind: str = "best") -> Path:
    """Return the checkpoint path inside a bundle for the given profile.

    kind:
      - "best": <profile>.pt
      - "last": <profile>_last.pt
    """
    bundle_dir = Path(bundle_dir)
    base = _safe_profile_filename(profile_id)
    if kind == "last":
        return bundle_dir / f"{base}_last.pt"
    return bundle_dir / f"{base}.pt"


@dataclass(frozen=True)
class BundleMeta:
    bundle_version: int
    created_at: str
    updated_at: str
    checkpoints: Dict[str, str]  # profile_id -> filename


def read_bundle_meta(bundle_dir: Path) -> Optional[BundleMeta]:
    p = Path(bundle_dir) / BUNDLE_META_NAME
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    cps = obj.get("checkpoints")
    if not isinstance(cps, dict):
        cps = {}
    try:
        return BundleMeta(
            bundle_version=int(obj.get("bundle_version") or 1),
            created_at=str(obj.get("created_at") or ""),
            updated_at=str(obj.get("updated_at") or ""),
            checkpoints={str(k): str(v) for k, v in cps.items()},
        )
    except Exception:
        return None


def upsert_bundle_meta(bundle_dir: Path, profile_id: str, ckpt_path: Path) -> None:
    """Update `bundle.json` to include profile_id -> checkpoint filename."""
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    meta = read_bundle_meta(bundle_dir)
    if meta is None:
        checkpoints: Dict[str, str] = {}
        created_at = now
    else:
        checkpoints = dict(meta.checkpoints)
        created_at = meta.created_at or now

    try:
        rel = str(Path(ckpt_path).resolve().relative_to(bundle_dir.resolve()))
    except Exception:
        rel = Path(ckpt_path).name

    checkpoints[str(profile_id)] = rel
    out = {
        "bundle_version": 1,
        "created_at": created_at,
        "updated_at": now,
        "checkpoints": checkpoints,
    }
    (bundle_dir / BUNDLE_META_NAME).write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


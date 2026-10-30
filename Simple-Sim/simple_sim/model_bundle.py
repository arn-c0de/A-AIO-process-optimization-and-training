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
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


BUNDLE_META_NAME = "bundle.json"


def is_bundle_dir(p: Path) -> bool:
    """Return True if *p* looks like a model bundle directory.

    A directory qualifies if it contains ``bundle.json`` or its name ends
    with ``.bundle`` (allowing detection before metadata is first written).

    Args:
        p: Path to inspect

    Returns:
        True if *p* is an existing bundle directory, False otherwise
    """
    p = Path(p)
    if not p.exists() or not p.is_dir():
        return False
    if (p / BUNDLE_META_NAME).exists():
        return True
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
    checkpoints: dict[str, str]  # profile_id -> filename


def read_bundle_meta(bundle_dir: Path) -> BundleMeta | None:
    """Read and parse ``bundle.json`` from *bundle_dir*.

    Args:
        bundle_dir: Bundle directory containing ``bundle.json``

    Returns:
        Parsed :class:`BundleMeta`, or ``None`` if the file is absent or invalid
    """
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


def _checkpoint_profile_id(ckpt_path: Path) -> str | None:
    """Best-effort read of ``component_profile.profile_id`` from a checkpoint."""
    try:
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        pid = (ckpt.get("component_profile") or {}).get("profile_id")
        return str(pid) if pid else None
    except Exception:
        return None


def scan_bundle_checkpoints(bundle_dir: Path) -> dict[str, str]:
    """Rebuild a profile_id -> filename index from the ``*.pt`` files in *bundle_dir*.

    ``*_last.pt`` files are skipped. The profile ID is read from the checkpoint
    metadata when possible, otherwise the filename stem is used.
    """
    checkpoints: dict[str, str] = {}
    for pt in sorted(Path(bundle_dir).glob("*.pt")):
        if pt.stem.endswith("_last"):
            continue
        pid = _checkpoint_profile_id(pt) or pt.stem
        checkpoints.setdefault(pid, pt.name)
    return checkpoints


def _quarantine_corrupt_meta(meta_path: Path) -> Path | None:
    """Move an unreadable ``bundle.json`` aside so it is not silently overwritten."""
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup = meta_path.with_name(f"{meta_path.name}.corrupt-{stamp}")
    try:
        meta_path.replace(backup)
    except OSError:
        return None
    return backup


def upsert_bundle_meta(bundle_dir: Path, profile_id: str, ckpt_path: Path) -> None:
    """Insert or update a profile entry in ``bundle.json``.

    Creates the bundle directory and ``bundle.json`` if they do not yet exist.

    Args:
        bundle_dir: Target bundle directory
        profile_id: Profile identifier to register
        ckpt_path: Absolute or relative path to the checkpoint file
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    meta_path = bundle_dir / BUNDLE_META_NAME
    meta = read_bundle_meta(bundle_dir)
    checkpoints: dict[str, str]
    if meta is None and meta_path.exists():
        # Unparseable index: keep a backup and rebuild from the checkpoints on disk
        # instead of rewriting the index with only the current profile.
        backup = _quarantine_corrupt_meta(meta_path)
        checkpoints = scan_bundle_checkpoints(bundle_dir)
        warnings.warn(
            f"Corrupt {BUNDLE_META_NAME} in {bundle_dir} "
            f"(backed up to {backup.name if backup else 'n/a'}); "
            f"rebuilt index from {len(checkpoints)} checkpoint(s).",
            stacklevel=2,
        )
        created_at = now
    elif meta is None:
        checkpoints = {}
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


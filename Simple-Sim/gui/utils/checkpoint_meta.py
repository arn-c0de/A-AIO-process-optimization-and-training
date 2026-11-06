"""Checkpoint metadata helpers shared by GUI tabs (tk-free).

torch is imported lazily (via ``simple_sim.models``) so the pure helpers here can be
used and tested without loading torch.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any


def read_checkpoint_dict(path: Path) -> dict[str, Any]:
    """Load a ``.pt`` checkpoint on CPU; non-dict payloads yield ``{}``. Load errors propagate."""
    from simple_sim.models import load_checkpoint

    ckpt = load_checkpoint(path)
    return ckpt if isinstance(ckpt, dict) else {}


def checkpoint_profile(ckpt: dict[str, Any] | None) -> tuple[Any, Any]:
    """``(profile_id, profile_hash)`` of ``ckpt["component_profile"]`` or ``(None, None)``."""
    comp = (ckpt or {}).get("component_profile")
    return (comp.get("profile_id"), comp.get("profile_hash")) if isinstance(comp, dict) else (None, None)


def short_profile_hash(profile_hash: str) -> str:
    """First 12 chars of a profile hash, dropping an ``algo:`` prefix (``sha256:abc...`` -> ``abc...``)."""
    return profile_hash.split(":")[1][:12] if ":" in profile_hash else profile_hash[:12]


def profile_label_suffix(ckpt: dict[str, Any]) -> str:
    """Selection-label suffix describing a checkpoint's component profile."""
    profile_data = ckpt.get("component_profile")
    if not profile_data:
        return " | Profile: ⚠ No profile (legacy)"
    profile_id = profile_data.get("profile_id", "unknown")
    profile_hash = profile_data.get("profile_hash", "")
    return f" | Profile: {profile_id} ({short_profile_hash(profile_hash)}...)"


def checkpoint_class_names(model_path: Path) -> list[str]:
    """Class names stored in a checkpoint (first ``*.pt`` by name for a bundle dir); ``[]`` on any error."""
    try:
        p = Path(model_path)
        if p.is_dir():
            candidates = sorted(p.glob("*.pt"), key=lambda x: x.name)
            if not candidates:
                return []
            p = candidates[0]
        classes = read_checkpoint_dict(p).get("class_names")
        if isinstance(classes, list):
            return [str(c) for c in classes]
    except Exception:
        pass
    return []


class CheckpointMetaCache:
    """Thread-safe cache of ``{component_profile, class_names[, error]}`` keyed by (path, mtime_ns, size)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _key(p: Path) -> tuple[str, int, int] | None:
        try:
            rp = Path(p).resolve()
            st = rp.stat()
            return (str(rp), int(st.st_mtime_ns), int(st.st_size))
        except (OSError, RuntimeError):
            return None

    def cached(self, p: Path) -> dict[str, Any] | None:
        """Cached metadata, ``None`` if not cached (or the file changed); ``{"error": "missing"}`` if unreadable."""
        if (key := self._key(p)) is None:
            return {"error": "missing"}
        with self._lock:
            return self._cache.get(key)

    def read(self, p: Path) -> dict[str, Any]:
        """Metadata for *p*, loading the checkpoint when uncached (slow)."""
        if (key := self._key(p)) is None:
            return {"error": "missing"}
        with self._lock:
            if (hit := self._cache.get(key)) is not None:
                return hit
        meta: dict[str, Any]
        try:
            ckpt = read_checkpoint_dict(Path(key[0]))
            comp = ckpt.get("component_profile")
            names = ckpt.get("class_names")
            meta = {
                "component_profile": dict(comp) if isinstance(comp, dict) else None,
                "class_names": list(names) if isinstance(names, (list, tuple)) else None,
            }
            del ckpt
        except Exception as e:
            meta = {"component_profile": None, "class_names": None, "error": str(e)}
        with self._lock:
            for k in [k for k in self._cache if k[0] == key[0] and k != key]:
                self._cache.pop(k, None)  # drop stale versions
            self._cache[key] = meta
        return meta

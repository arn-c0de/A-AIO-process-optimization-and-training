"""Small, dependency-free file/format helpers shared by scripts, tools and the GUI.

Everything here is pure stdlib so it can be imported anywhere (no torch/cv2/tk).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically (write ``<name>.tmp`` then rename)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    """Serialize *obj* as JSON and write it to *path* atomically."""
    atomic_write_text(path, json.dumps(obj, indent=indent, ensure_ascii=True) + "\n")


def load_json(path: Path) -> Any:
    """Parse the JSON file at *path* (UTF-8); errors propagate (see :func:`read_json` for a lenient read)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_yaml(path: Path) -> Any:
    """Parse the YAML file at *path* (UTF-8, ``yaml.safe_load``); ``None`` for an empty file.

    Errors propagate (``FileNotFoundError``, ``yaml.YAMLError``). PyYAML is imported lazily, so
    ``io_utils`` stays importable without it.
    """
    import yaml

    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dict rows as JSONL atomically (stream to ``<name>.tmp``, then rename)."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    tmp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    """Return parsed JSON from *path*, or *default* if missing/unreadable/invalid."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append dict rows to a JSONL file (one JSON object per line)."""
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


#: Prefix of the content hashes stored in manifests and checkpoints (``sha256:<hex>``).
SHA256_PREFIX = "sha256:"


def sha256_tag(digest: Any) -> str:
    """``sha256:<hex>`` for a finished ``hashlib.sha256()`` object (manifest / profile hash format)."""
    return f"{SHA256_PREFIX}{digest.hexdigest()}"


def now_iso(*, utc: bool = False, timespec: str = "auto") -> str:
    """Current time as ISO 8601: naive local time (manifests, reports), or UTC with a ``Z`` suffix."""
    if utc:
        return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec=timespec) + "Z"
    return datetime.now().isoformat(timespec=timespec)


def now_tag(*, micro: bool = True) -> str:
    """Filesystem-safe local timestamp tag ``YYYYmmdd_HHMMSS`` (plus ``_ffffff`` with *micro*)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f" if micro else "%Y%m%d_%H%M%S")


def unique_path(p: Path) -> Path:
    """Return *p* if it does not exist, else ``<stem>_NNN<suffix>`` that does not exist."""
    p = Path(p)
    if not p.exists():
        return p
    for i in range(1, 10000):
        cand = p.parent / f"{p.stem}_{i:03d}{p.suffix}"
        if not cand.exists():
            return cand
    return p.parent / f"{p.stem}_{time.time_ns()}{p.suffix}"


def format_bytes(n: int, decimals: int = 2) -> str:
    """Human-readable byte count (``1.50 MB``); negative values show as ``0 B``."""
    try:
        v = float(max(0, int(n)))
    except (TypeError, ValueError):
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while v >= 1024.0 and u < len(units) - 1:
        v /= 1024.0
        u += 1
    if u == 0:
        return f"{int(v)} {units[u]}"
    return f"{v:.{decimals}f} {units[u]}"


def path_size_bytes(p: Path) -> int:
    """On-disk size of a file, or the recursive size of a directory (symlinks skipped)."""
    p = Path(p)
    try:
        if p.is_file():
            return int(p.stat().st_size)
        if not p.is_dir():
            return 0
    except OSError:
        return 0
    return _tree_size(str(p))


def _tree_size(top: str) -> int:
    """Sum of ``st_size`` below *top*, the walk of ``os.walk(top)`` (no symlinked files or dirs,
    unreadable entries skipped) with one ``scandir`` entry per file instead of two ``Path`` stats."""
    total = 0
    stack = [top]
    while stack:
        try:
            it = os.scandir(stack.pop())
        except OSError:
            continue
        with it:
            while True:
                try:
                    entry = next(it)
                except StopIteration:
                    break
                except OSError:
                    break
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    is_dir = False
                try:
                    if entry.is_symlink():
                        continue
                    if is_dir:
                        stack.append(entry.path)
                    else:
                        total += int(entry.stat(follow_symlinks=False).st_size)
                except OSError:
                    continue
    return total


def rel_to(root: Path, p: Path) -> str:
    """Path of *p* relative to *root* as string, or the absolute path if outside."""
    try:
        return str(Path(p).resolve().relative_to(Path(root).resolve()))
    except (OSError, ValueError):
        return str(p)


def parse_iso_ts(s: Any) -> float:
    """Parse an ISO timestamp to epoch seconds; ``0.0`` if unparseable."""
    try:
        return datetime.fromisoformat(str(s)).timestamp()
    except (TypeError, ValueError):
        return 0.0

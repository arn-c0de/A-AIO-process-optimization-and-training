"""Well-known Simple-Sim output locations shared by GUI tabs (tk-free)."""

from __future__ import annotations

from pathlib import Path


def model_root(sim_root: Path) -> Path:
    """Root directory for model checkpoints (``outputs/models``)."""
    return Path(sim_root) / "outputs" / "models"


def newest_file(directory: Path, pattern: str, *, min_mtime: float | None = None) -> Path | None:
    """Most recently modified file in *directory* matching *pattern* (optionally ``mtime >= min_mtime``)."""
    try:
        cand = [(p.stat().st_mtime, p) for p in Path(directory).glob(pattern)]
    except OSError:
        return None
    if min_mtime is not None:
        cand = [t for t in cand if t[0] >= min_mtime]
    return sorted(cand, key=lambda t: t[0])[-1][1] if cand else None


def list_model_paths(model_root: Path, *, include_bundles: bool = True) -> list[Path]:
    """Checkpoint candidates under ``model_root``, deduplicated by resolved path.

    Searches ``*.pt`` in the root and ``imports/``, ``**/*.pt`` in ``versions/``
    and, with ``include_bundles``, bundle directories (``*.bundle`` and
    ``bundles/*``) next to each of those. Creates ``model_root`` if missing.
    """
    root = Path(model_root)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    cand: list[Path] = []
    for search in (root, root / "versions", root / "imports"):
        if not search.is_dir():
            continue
        cand.extend(sorted(search.glob("**/*.pt" if search.name == "versions" else "*.pt")))
        if include_bundles:
            cand.extend(p for p in sorted(search.glob("*.bundle")) if p.is_dir())
            bundles = search / "bundles"
            if bundles.is_dir():
                cand.extend(p for p in sorted(bundles.iterdir()) if p.is_dir())
    uniq: dict[str, Path] = {}
    for p in cand:
        try:
            if p.is_file() or p.is_dir():
                uniq[str(p.resolve())] = p
        except Exception:
            continue
    return list(uniq.values())

"""Model list grouping shared by the weights and merge tabs (Tk-free)."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

FAVORITES = "Favorites"
SNAPSHOTS = "Snapshots"
IMPORTS = "Imports"
BUNDLES = "Bundles"
UNCATEGORIZED = "Uncategorized"


def available_groups(custom_groups: Iterable[str], assignments: Mapping[str, str], *, bundles: bool = False) -> list[str]:
    """Group names in display order: built-in, custom, assigned-only, then ``Uncategorized``."""
    names = [FAVORITES, SNAPSHOTS, IMPORTS] + ([BUNDLES] if bundles else [])
    names.extend(g for g in custom_groups if g and g not in names)
    names.extend(v for v in sorted(set(assignments.values())) if v and v not in names)
    if UNCATEGORIZED not in names:
        names.append(UNCATEGORIZED)
    return names


def group_for_model(path: Path, rel: str, *, model_root: Path, assignments: Mapping[str, str], groups: Iterable[str],
                    favorite: bool, bundled: bool = False) -> str:
    """Group of the model at *path* (*rel*: its key in *assignments*).

    An explicit assignment to a still existing group wins, then favourites, bundles and the
    ``versions/`` / ``imports/`` folders below *model_root*.
    """
    assigned = assignments.get(rel)
    if assigned is not None and assigned in groups:
        return assigned
    if favorite:
        return FAVORITES
    if bundled:
        return BUNDLES
    resolved = str(path.resolve())
    root = model_root.resolve()
    if resolved.startswith(str((root / "versions").resolve()) + os.sep):
        return SNAPSHOTS
    if resolved.startswith(str((root / "imports").resolve()) + os.sep):
        return IMPORTS
    return UNCATEGORIZED

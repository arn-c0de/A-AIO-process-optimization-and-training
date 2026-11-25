"""Model list grouping shared by the weights and merge tabs (Tk-free)."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from simple_sim.io_utils import atomic_write_json, read_json

FAVORITES = "Favorites"
SNAPSHOTS = "Snapshots"
IMPORTS = "Imports"
BUNDLES = "Bundles"
UNCATEGORIZED = "Uncategorized"

# Group state is shared by the weights and merge tabs (settings store + favorites.json in the model root).
KEY_CUSTOM_GROUPS = "weights.custom_groups"
KEY_GROUP_ASSIGNMENTS = "weights.group_assignments"
FAVORITES_FILE = "favorites.json"


def load_group_state(store: Any) -> tuple[list[str], dict[str, str]]:
    """``(custom group names, {model rel path: group})`` from the settings store (empty without one)."""
    if store is None:
        return [], {}
    custom = store.get_json(KEY_CUSTOM_GROUPS, want=list, fallback=[])
    assignments = store.get_json(KEY_GROUP_ASSIGNMENTS, want=dict, fallback={})
    return ([str(k) for k in custom if isinstance(k, str) and k],
            {str(k): str(v) for k, v in assignments.items() if k and v})


def save_group_state(store: Any, custom_groups: list[str], assignments: Mapping[str, str]) -> None:
    """Write the group state to *store* (the caller schedules the save)."""
    store.set_json(KEY_GROUP_ASSIGNMENTS, dict(assignments))
    store.set_json(KEY_CUSTOM_GROUPS, list(custom_groups))


def load_path_items(p: Path, list_key: str) -> dict[str, dict[str, Any]]:
    """``{path: item}`` from a ``{list_key: [{"path": ...}, ...]}`` JSON file (``{}`` if missing/invalid)."""
    obj = read_json(p)
    items = obj.get(list_key) if isinstance(obj, dict) else None
    out: dict[str, dict[str, Any]] = {}
    for it in items if isinstance(items, list) else []:
        if isinstance(it, dict) and (rp := str(it.get("path") or "").strip()):
            out[rp] = it
    return out


def save_path_items(p: Path, list_key: str, items: Mapping[str, dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(p, {list_key: sorted(items.values(), key=lambda it: str(it.get("path") or ""))})


def load_favorites(model_root: Path) -> dict[str, dict[str, Any]]:
    return load_path_items(Path(model_root) / FAVORITES_FILE, "favorites")


def save_favorites(model_root: Path, favorites: Mapping[str, dict[str, Any]]) -> None:
    save_path_items(Path(model_root) / FAVORITES_FILE, "favorites", favorites)


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

"""Shared dataset category/archive metadata for GUI tabs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from gui.utils.settings_store import SettingsStore

DEFAULT_CATEGORY = "Uncategorized"
KEY_CUSTOM_CATEGORIES = "datasets.custom_categories"
KEY_CATEGORY_ASSIGNMENTS = "datasets.category_assignments"
KEY_ARCHIVED_PATHS = "datasets.archived_paths"


def _safe_json_load(raw: object, *, want: type, fallback):
    if not isinstance(raw, str):
        return fallback
    try:
        obj = json.loads(raw)
    except Exception:
        return fallback
    return obj if isinstance(obj, want) else fallback


def dataset_relative_key(sim_root: Path, dataset_dir: Path) -> str:
    try:
        return str(dataset_dir.resolve().relative_to(sim_root.resolve()))
    except Exception:
        return str(dataset_dir.resolve())


def dataset_display_name(dataset_dir: Path, *, runs_root: Path, versions_root: Path) -> str:
    try:
        rp = dataset_dir.resolve()
        if str(rp).startswith(str(runs_root.resolve()) + os.sep):
            return rp.name
        if str(rp).startswith(str(versions_root.resolve()) + os.sep):
            return f"{rp.parent.name}:{rp.name}"
    except Exception:
        pass
    return dataset_dir.name


class DatasetCatalog:
    """In-memory view over dataset category + archive metadata in SettingsStore."""

    def __init__(self, sim_root: Path, store: Optional[SettingsStore]) -> None:
        self.sim_root = sim_root
        self.store = store
        self.custom_categories: List[str] = []
        self.category_assignments: Dict[str, str] = {}
        self.archived_paths: Set[str] = set()
        self.reload()

    def reload(self) -> None:
        if self.store is None:
            self.custom_categories = []
            self.category_assignments = {}
            self.archived_paths = set()
            return
        custom = _safe_json_load(self.store.get(KEY_CUSTOM_CATEGORIES, "[]"), want=list, fallback=[])
        assign = _safe_json_load(self.store.get(KEY_CATEGORY_ASSIGNMENTS, "{}"), want=dict, fallback={})
        archived = _safe_json_load(self.store.get(KEY_ARCHIVED_PATHS, "[]"), want=list, fallback=[])

        self.custom_categories = [str(it).strip() for it in custom if str(it).strip() and str(it).strip() != DEFAULT_CATEGORY]
        self.category_assignments = {
            str(k): str(v).strip()
            for k, v in assign.items()
            if str(k).strip() and str(v).strip()
        }
        self.archived_paths = {str(it).strip() for it in archived if str(it).strip()}

    def save(self, widget=None) -> None:
        if self.store is None:
            return
        self.store.set(KEY_CUSTOM_CATEGORIES, json.dumps(sorted(set(self.custom_categories)), ensure_ascii=True))
        self.store.set(KEY_CATEGORY_ASSIGNMENTS, json.dumps(self.category_assignments, ensure_ascii=True))
        self.store.set(KEY_ARCHIVED_PATHS, json.dumps(sorted(self.archived_paths), ensure_ascii=True))
        if widget is not None:
            self.store.schedule_save(widget)

    def all_categories(self) -> List[str]:
        cats = [DEFAULT_CATEGORY]
        cats.extend(sorted({c for c in self.custom_categories if c and c != DEFAULT_CATEGORY}))
        return cats

    def category_for(self, dataset_dir: Path) -> str:
        return self.category_assignments.get(dataset_relative_key(self.sim_root, dataset_dir), DEFAULT_CATEGORY)

    def is_archived(self, dataset_dir: Path) -> bool:
        return dataset_relative_key(self.sim_root, dataset_dir) in self.archived_paths

    def set_category(self, dataset_dirs: List[Path], category: str) -> None:
        category = str(category).strip() or DEFAULT_CATEGORY
        if category != DEFAULT_CATEGORY and category not in self.custom_categories:
            self.custom_categories.append(category)
        for ds in dataset_dirs:
            self.category_assignments[dataset_relative_key(self.sim_root, ds)] = category

    def set_archived(self, dataset_dirs: List[Path], archived: bool) -> None:
        for ds in dataset_dirs:
            key = dataset_relative_key(self.sim_root, ds)
            if archived:
                self.archived_paths.add(key)
            else:
                self.archived_paths.discard(key)

    def rename_category(self, old_name: str, new_name: str) -> bool:
        old = str(old_name).strip()
        new = str(new_name).strip()
        if not old or not new or old == DEFAULT_CATEGORY:
            return False
        if old == new:
            return True
        if old in self.custom_categories:
            self.custom_categories = [new if c == old else c for c in self.custom_categories]
        for key, val in list(self.category_assignments.items()):
            if val == old:
                self.category_assignments[key] = new
        return True

    def delete_category(self, name: str) -> bool:
        cat = str(name).strip()
        if not cat or cat == DEFAULT_CATEGORY:
            return False
        self.custom_categories = [c for c in self.custom_categories if c != cat]
        for key, val in list(self.category_assignments.items()):
            if val == cat:
                self.category_assignments[key] = DEFAULT_CATEGORY
        return True

    def prune_unknown(self, existing_dataset_dirs: List[Path]) -> None:
        valid = {dataset_relative_key(self.sim_root, p) for p in existing_dataset_dirs}
        self.archived_paths = {k for k in self.archived_paths if k in valid}
        self.category_assignments = {k: v for k, v in self.category_assignments.items() if k in valid}


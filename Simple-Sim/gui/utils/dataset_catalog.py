"""Shared dataset category/archive metadata for GUI tabs."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from gui.utils.settings_store import SettingsStore

DEFAULT_CATEGORY = "Uncategorized"
KEY_CUSTOM_CATEGORIES = "datasets.custom_categories"
KEY_CATEGORY_ASSIGNMENTS = "datasets.category_assignments"
KEY_ARCHIVED_PATHS = "datasets.archived_paths"


def dataset_relative_key(sim_root: Path, dataset_dir: Path) -> str:
    try:
        return str(dataset_dir.resolve().relative_to(sim_root.resolve()))
    except Exception:
        return str(dataset_dir.resolve())


def _strictly_under(path: Path, root: Path) -> bool:
    """True when resolved *path* lies below resolved *root* (not *root* itself)."""
    rp, rr = path.resolve(), root.resolve()
    return rp != rr and rp.is_relative_to(rr)


def dataset_display_name(dataset_dir: Path, *, runs_root: Path, versions_root: Path) -> str:
    try:
        rp = dataset_dir.resolve()
        if _strictly_under(rp, runs_root):
            return rp.name
        if _strictly_under(rp, versions_root):
            return f"{rp.parent.name}:{rp.name}"
    except Exception:
        pass
    return dataset_dir.name


def sim_data_roots(sim_root: Path, *, create: bool = True) -> tuple[Path, Path]:
    """Return ``(runs_root, versions_root)`` under ``outputs/sim_data`` (created by default)."""
    sim_data = Path(sim_root) / "outputs" / "sim_data"
    runs = sim_data / "runs"
    versions = sim_data / "versions"
    if create:
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)
    return runs, versions


def list_dataset_dirs(sim_root: Path) -> list[Path]:
    """Unsorted dataset directories: ``runs/*`` followed by snapshot ``versions/*/*``."""
    runs, versions = sim_data_roots(sim_root)
    cand: list[Path] = [p for p in runs.iterdir() if p.is_dir()]
    cand.extend([p for p in versions.glob("*/*") if p.is_dir()])
    return cand


def dataset_kind(sim_root: Path, dataset_dir: Path) -> str | None:
    """``"run"`` (under ``outputs/sim_data/runs``), ``"snapshot"`` (under ``.../versions``), else ``None``."""
    runs, versions = sim_data_roots(sim_root, create=False)
    try:
        if _strictly_under(dataset_dir, runs):
            return "run"
        if _strictly_under(dataset_dir, versions):
            return "snapshot"
    except Exception:
        pass
    return None


def is_snapshot_dataset(sim_root: Path, dataset_dir: Path) -> bool:
    """True when ``dataset_dir`` lives under ``outputs/sim_data/versions``."""
    return dataset_kind(sim_root, dataset_dir) == "snapshot"


def dataset_sort_key(sim_root: Path, dataset_dir: Path) -> tuple:
    """Runs first (by name), then snapshots newest-first."""
    if is_snapshot_dataset(sim_root, dataset_dir):
        try:
            mtime = dataset_dir.stat().st_mtime
        except Exception:
            mtime = 0.0
        return (1, -mtime, dataset_dir.name)
    return (0, dataset_dir.name)


def sorted_dataset_dirs(sim_root: Path) -> list[Path]:
    """``list_dataset_dirs`` sorted with ``dataset_sort_key``."""
    return sorted(list_dataset_dirs(sim_root), key=lambda p: dataset_sort_key(sim_root, p))


def _unique_labels(labeled: Iterable[tuple[str, Path]]) -> dict[str, Path]:
    """Suffix duplicate labels as ``name (2)``, ``name (3)``..."""
    out: dict[str, Path] = {}
    seen: dict[str, int] = {}
    for base, p in labeled:
        n = seen.get(base, 0) + 1
        seen[base] = n
        out[base if n == 1 else f"{base} ({n})"] = p
    return out


def unique_dataset_labels(dataset_dirs: Iterable[Path], *, runs_root: Path, versions_root: Path) -> dict[str, Path]:
    """Map display labels to dataset dirs, suffixing duplicates as ``name (2)``, ``name (3)``..."""
    return _unique_labels(
        (dataset_display_name(p, runs_root=runs_root, versions_root=versions_root), p) for p in dataset_dirs
    )


def sorted_dataset_labels(sim_root: Path) -> dict[str, Path]:
    """``unique_dataset_labels`` of every dataset, ordered like ``sorted_dataset_dirs``."""
    runs, versions = sim_data_roots(sim_root)
    return unique_dataset_labels(sorted_dataset_dirs(sim_root), runs_root=runs, versions_root=versions)


def same_path(a: Path, b: Path) -> bool:
    """``a`` and ``b`` name the same file system entry (resolved; plain comparison if that fails)."""
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return str(a) == str(b)


def label_for_dataset(labels: dict[str, Path], dataset_dir: Path | None, *, alt_label: str | None = None) -> str | None:
    """The label in *labels* whose path is *dataset_dir* (or that equals *alt_label*), else ``None``."""
    if dataset_dir is None:
        return None
    for label, p in labels.items():
        if same_path(p, dataset_dir) or (alt_label is not None and label == alt_label):
            return label
    return None


def catalog_labels(sim_root: Path, catalog: DatasetCatalog, *, save_widget=None) -> dict[str, Path]:
    """Unique ``[Category] name`` labels -> dataset dir for every non-archived dataset.

    Reloads ``catalog``, prunes entries for datasets that no longer exist and
    saves it (debounced via ``save_widget`` when given). Order follows
    ``sorted_dataset_dirs``.
    """
    catalog.reload()
    cand = sorted_dataset_dirs(sim_root)
    catalog.prune_unknown(cand)
    catalog.save(save_widget)
    runs, versions = sim_data_roots(sim_root, create=False)
    return _unique_labels(
        (catalog.display_label(p, runs_root=runs, versions_root=versions), p) for p in cand if not catalog.is_archived(p)
    )


class DatasetCatalog:
    """In-memory view over dataset category + archive metadata in SettingsStore."""

    def __init__(self, sim_root: Path, store: SettingsStore | None) -> None:
        self.sim_root = sim_root
        self.store = store
        self.custom_categories: list[str] = []
        self.category_assignments: dict[str, str] = {}
        self.archived_paths: set[str] = set()
        self.reload()

    def reload(self) -> None:
        if self.store is None:
            self.custom_categories = []
            self.category_assignments = {}
            self.archived_paths = set()
            return
        custom = self.store.get_json(KEY_CUSTOM_CATEGORIES, want=list, fallback=[])
        assign = self.store.get_json(KEY_CATEGORY_ASSIGNMENTS, want=dict, fallback={})
        archived = self.store.get_json(KEY_ARCHIVED_PATHS, want=list, fallback=[])

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
        self.store.set_json(KEY_CUSTOM_CATEGORIES, sorted(set(self.custom_categories)))
        self.store.set_json(KEY_CATEGORY_ASSIGNMENTS, self.category_assignments)
        self.store.set_json(KEY_ARCHIVED_PATHS, sorted(self.archived_paths))
        if widget is not None:
            self.store.schedule_save(widget)

    def all_categories(self) -> list[str]:
        cats = [DEFAULT_CATEGORY]
        cats.extend(sorted({c for c in self.custom_categories if c and c != DEFAULT_CATEGORY}))
        return cats

    def category_for(self, dataset_dir: Path) -> str:
        return self.category_assignments.get(dataset_relative_key(self.sim_root, dataset_dir), DEFAULT_CATEGORY)

    def display_label(self, dataset_dir: Path, *, runs_root: Path, versions_root: Path) -> str:
        """Display name prefixed with ``[category] `` unless the dataset is uncategorized."""
        base = dataset_display_name(dataset_dir, runs_root=runs_root, versions_root=versions_root)
        category = self.category_for(dataset_dir)
        return f"[{category}] {base}" if category and category != DEFAULT_CATEGORY else base

    def is_archived(self, dataset_dir: Path) -> bool:
        return dataset_relative_key(self.sim_root, dataset_dir) in self.archived_paths

    def set_category(self, dataset_dirs: list[Path], category: str) -> None:
        category = str(category).strip() or DEFAULT_CATEGORY
        if category != DEFAULT_CATEGORY and category not in self.custom_categories:
            self.custom_categories.append(category)
        for ds in dataset_dirs:
            self.category_assignments[dataset_relative_key(self.sim_root, ds)] = category

    def set_archived(self, dataset_dirs: list[Path], archived: bool) -> None:
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

    def prune_unknown(self, existing_dataset_dirs: list[Path]) -> None:
        valid = {dataset_relative_key(self.sim_root, p) for p in existing_dataset_dirs}
        self.archived_paths = {k for k in self.archived_paths if k in valid}
        self.category_assignments = {k: v for k, v in self.category_assignments.items() if k in valid}


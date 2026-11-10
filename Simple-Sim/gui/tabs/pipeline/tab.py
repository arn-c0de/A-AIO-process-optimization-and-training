"""Pipeline Control Tab - Enhanced version of the original monitor."""

from __future__ import annotations
import logging
import threading
from tkinter import ttk
from pathlib import Path
from typing import Any


from gui.core.tasks import DispatchQueue
from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.dataset_catalog import DatasetCatalog

from simple_sim.config import DEFAULT_COMPONENT_PROFILE
from simple_sim.file_cache import FileCache
from simple_sim.schema import MetaRow

from .ui import PipelineUI
from .logic import PipelineLogic
from .settings_mixin import PipelineSettingsMixin
from .run_mixin import PipelineRunMixin
from .monitor_mixin import PipelineMonitorMixin
from .selection_mixin import PipelineSelectionMixin, load_dataset_view
from .dataset_actions_mixin import PipelineDatasetActionsMixin
from gui.utils.tk_safe import quiet

__all__ = ["PipelineControlTab", "load_dataset_view"]  # load_dataset_view: re-exported for callers of this module

_LOG = logging.getLogger(__name__)


class PipelineControlTab(PipelineSettingsMixin, PipelineRunMixin, PipelineMonitorMixin, PipelineSelectionMixin, PipelineDatasetActionsMixin, BaseTab):
    """Tab 1: Pipeline control with enhanced monitoring."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        # PipelineLogic only put()s into these; DispatchQueue drains them on the Tk thread
        # through the app dispatcher (bound in build_ui) instead of an after(120) poll.
        self.log_q: DispatchQueue = DispatchQueue()
        self.event_q: DispatchQueue = DispatchQueue()
        self.event_log_path = self.sim_root / "outputs" / "live" / "events.jsonl"

        self.ui = PipelineUI(self, self.frame)
        self.logic = PipelineLogic(sim_root, self.log_q, self.event_q, self.event_log_path)
        self.logic.ui_call = self.post  # thread-safe hop onto the Tk loop

        self._thumb_refs = []  # keep PhotoImage references
        self._recent_imgs: list[str] = []
        self._dataset_dirs: list[Path] = []
        self._dataset_labels: list[str] = []
        self._dataset_by_label: dict[str, Path] = {}
        self._label_dict: dict[str, str] = {}  # Map sample_id to class label
        self._image_to_sample: dict[str, str] = {}  # Map image_path to sample_id
        self._meta_by_sample: dict[str, MetaRow] = {}
        self._meta_by_image_rel: dict[str, MetaRow] = {}
        
        self._last_profile_id: str = DEFAULT_COMPONENT_PROFILE
        # Temp configs written under outputs/live for the current run; removed when the run ends.
        self._temp_cfg_paths: list[Path] = []

        self._precise_settings: dict[str, Any] = {
            "total": 100,
            "per_class": False,
            "class_names": ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"],
            "classes": {},
            "multi_profiles": [],
            "multi_totals": {},
            "multi_classes": {},
        }
        self._precise_popup_ref: Any = None  # PrecisePopup instance while open

        self._last_stats_ts: float = 0.0
        self._last_gpu_ts: float = 0.0
        self._last_ds_size_ts: float = 0.0
        self._prev_proc_running: bool = False
        self._dataset_size_job_id: int = 0
        self._milestone_thresholds = [1000, 5000, 10000]
        self._last_thumb_refresh_ts: float = 0.0
        self._thumb_refresh_min_interval_s: float = 0.35
        self._thumb_refresh_pending: bool = False
        self._thumb_after_id: str | None = None
        self._thumb_gen: int = 0  # bumps per thumbnail job; stale worker results are dropped
        self._thumb_busy: bool = False
        self._thumb_rerun: bool = False
        self._ds_load_gen: int = 0  # bumps per dataset selection; stale index loads are dropped
        self._status_after_id: str | None = None
        # Stat-first cache of per-dataset manifest facts (profile id/hash) for the model dropdown.
        self._manifest_cache = FileCache(max_entries=128)

        self._model_paths: list[Path] = []
        self._model_resolved: dict[Path, Path] = {}  # path -> resolve(), filled by _refresh_models
        self._all_model_combo_values: list[str] = []
        self._model_profile_cache: dict[str, tuple[str | None, str | None]] = {}
        self._model_meta_gen: int = 0  # bumps per _refresh_models so stale background loads are dropped
        self._compat_after_id: str | None = None
        self._compat_loading: set[str] = set()

        self._profile_paths: list[Path] = []
        self._dataset_milestones: dict[str, int] = {}
        self._filter_popup_extras: dict[str, Any] = {}
        self._dataset_catalog = DatasetCatalog(sim_root, state.settings_store)
        self._ui_built: bool = False
        
    def build_ui(self) -> None:
        if self._ui_built:
            return
        self._ui_built = True
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self._ensure_initial_placeholder_dataset()
        self._refresh_datasets()
        self._refresh_models()
        self._refresh_profile_models()
        self._refresh_profiles()
        # Re-scan configs/profiles/ whenever the list opens: profiles added while the GUI runs show up.
        self.ui.profile_combo.configure(postcommand=lambda: self.ui.set_profile_combo_values(
            self.logic.get_profile_paths(self.ui.var_render_backend.get().strip() or "opencv_2d")))
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self._apply_profile_model_lock()
        self.ui.var_model.trace("w", lambda *args: self._schedule_profile_compat_check())
        self.bind_queue(self.log_q, self._on_log_batch, max_items=200)
        self.bind_queue(self.event_q, self._on_event_batch, max_items=120)
        self._tick_status()

    def _ensure_initial_placeholder_dataset(self) -> None:
        runs = self.sim_root / "outputs" / "sim_data" / "runs"
        versions = self.sim_root / "outputs" / "sim_data" / "versions"
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)

        has_runs = any(p.is_dir() for p in runs.iterdir())
        has_versions = any(p.is_dir() for p in versions.glob("*/*"))
        if has_runs or has_versions:
            return

        placeholder = (runs / "run_placeholder_empty").resolve()
        profile_id = DEFAULT_COMPONENT_PROFILE
        cfg_info = self.logic.config_for_profile(profile_id, want_backend="opencv_2d")
        if not cfg_info:
            cfg_info = self.logic.config_for_profile(profile_id, want_backend=None)

        cfg_path: Path | None = Path(cfg_info["path"]) if cfg_info and cfg_info.get("path") else None
        if cfg_path is None or not cfg_path.exists():
            fallback = self.sim_root / "configs" / "run_0001.yaml"
            if fallback.exists():
                cfg_path = fallback
            else:
                cfg_path = next((p for p in sorted((self.sim_root / "configs").glob("*.yaml")) if p.is_file()), None)

        if cfg_path is None or not cfg_path.exists():
            _LOG.warning("No config found to create initial placeholder dataset.")
            return

        try:
            if self.logic.ensure_dataset_skeleton(profile_id, cfg_path, placeholder, lambda _msg: None):
                self._append_log(f"[startup] created placeholder dataset: {placeholder.name}\n")
        except Exception as exc:
            _LOG.warning("Failed to create initial placeholder dataset: %s", exc)

    def _append_log(self, text: str) -> None:
        # Tk widgets must only be touched on the main thread.
        if threading.current_thread() is not threading.main_thread():
            self.log_q.put(text)
            return
        with quiet():
            self.ui.append_log(text)

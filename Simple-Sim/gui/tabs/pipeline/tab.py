"""Pipeline Control Tab - Enhanced version of the original monitor."""

from __future__ import annotations
import json
import logging
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Any
import shutil
import shlex

import yaml

from gui import dialogs
from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.components.overlay_renderer import draw_defect_overlay
from gui.components.filter_popup import open_filter_popup
from gui.components.precise_popup import open_precise_popup
from gui.utils.settings_store import SettingsStore, restore_persisted_vars, wire_vars_autosave
from gui.utils.dataset_ops import delete_samples, list_dataset_samples, move_samples, DatasetSampleInfo
from gui.utils.checkpoint_meta import short_profile_hash
from gui.utils.dataset_catalog import DatasetCatalog, catalog_labels, dataset_display_name, sim_data_roots
from gui.utils.filter_profile_store import (
    extract_filter_popup_extras,
    float_or_default,
    formatted_filter_settings,
    image_filters_payload,
    load_filter_popup_extras_from_profiles_json,
    normalized_filter_settings,
)
from gui.utils.pipeline_helpers import (
    DATASET_DERIVED_KEYS,
    PERSISTED_FIELD_MAP,
    apply_precise_counts,
    chain_commands,
    config_class_names,
    continue_train_argv,
    decode_dataset_paths_json,
    encode_dataset_paths_json,
    eval_argv,
    mixed_profile_config,
    multi_train_argvs,
    parse_profile_ids,
    parse_run_count,
    parse_snap_every,
    parse_snap_keep,
    precise_counts,
    precise_counts_for_profile,
    profile_compat_status,
    profile_model_paths,
    profile_run_targets,
    resolve_under,
    sample_color_from_total,
    samples_text,
    summarize_labels,
    summarize_profile_ids,
    train_argv,
    unique_timestamped_dir,
    validate_argv,
    write_temp_yaml,
)

from simple_sim.config import DEFAULT_COMPONENT_PROFILE
from simple_sim.io_utils import append_jsonl, rel_to
from simple_sim.schema import read_jsonl, LabelRow, MetaRow
from simple_sim.model_bundle import bundle_checkpoint_path

from .ui import PipelineUI
from .logic import PipelineLogic

_LOG = logging.getLogger(__name__)

class PipelineControlTab(BaseTab):
    """Tab 1: Pipeline control with enhanced monitoring."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        self.log_q: queue.Queue[str] = queue.Queue()
        self.event_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self.event_log_path = self.sim_root / "outputs" / "live" / "events.jsonl"

        self.ui = PipelineUI(self, self.frame)
        self.logic = PipelineLogic(sim_root, self.log_q, self.event_q, self.event_log_path)
        self.logic.ui_call = lambda fn: self.frame.after(0, fn)

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
        self._dataset_size_q: queue.Queue[tuple[int, int, str]] = queue.Queue()
        self._milestone_thresholds = [1000, 5000, 10000]
        self._last_thumb_refresh_ts: float = 0.0
        self._thumb_refresh_min_interval_s: float = 0.35
        self._thumb_refresh_pending: bool = False

        self._model_paths: list[Path] = []
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
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self._apply_profile_model_lock()
        self.ui.var_model.trace("w", lambda *args: self._schedule_profile_compat_check())
        self._tick_ui()

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

    def _store(self) -> SettingsStore | None:
        return self.state.settings_store

    def _append_log(self, text: str) -> None:
        # Tk widgets must only be touched on the main thread.
        if threading.current_thread() is not threading.main_thread():
            self.log_q.put(text)
            return
        try:
            self.ui.append_log(text)
        except Exception:
            pass

    def _confirm_pipeline_start(
        self,
        *,
        task: str,
        out_dir: str,
        run_mode: str,
        run_count: int,
        dataset_mode: str,
        profile_ids: list[str],
        multi_enabled: bool,
        multi_mode: str,
        image_filters: dict[str, Any],
    ) -> bool:
        task_label = {
            "full": "Full Pipeline",
            "generate_only": "Generate Only",
            "generate_mixed": "Generate Mixed",
        }.get(task, task)
        run_label = "continuous" if run_count == -1 else str(run_count)
        profile_label = ", ".join(profile_ids[:5]) if profile_ids else "-"
        if len(profile_ids) > 5:
            profile_label += f" (+{len(profile_ids) - 5})"
        mode_raw = str(image_filters.get("filter_mode", "custom") or "custom").strip().lower()
        filter_mode = mode_raw if mode_raw in {"custom", "realism"} else "custom"
        realism_enabled = bool(image_filters.get("realism_enabled", False))
        realism_constraints = bool(image_filters.get("realism_constraints_enabled", True))
        realism_profile = str(image_filters.get("realism_profile_id", "profile_industrial_cam"))

        filter_toggles = [k for k, v in image_filters.items() if isinstance(k, str) and k.startswith("enable_") and k != "enable" and bool(v)]
        filter_details = (
            f"filter_mode={filter_mode}\n"
            f"enable={bool(image_filters.get('enable', True))}\n"
            f"realism_enabled={realism_enabled}\n"
            f"realism_constraints_enabled={realism_constraints}\n"
            f"realism_profile_id={realism_profile}\n"
            f"cardinal_rotation_90={bool(image_filters.get('cardinal_rotation_90', True))}\n"
            f"enabled_filter_keys={', '.join(sorted(filter_toggles)) if filter_toggles else '-'}"
        )

        precise_summary = "off"
        precise_details = "Precise mode is not active."
        if run_mode == "precise":
            s = self._precise_settings
            per_class = bool(s.get("per_class", False))
            multi_profiles = list(s.get("multi_profiles") or [])
            if multi_profiles:
                precise_summary = f"on | multi={len(multi_profiles)} | per_class={'yes' if per_class else 'no'}"
            else:
                precise_summary = f"on | single profile | per_class={'yes' if per_class else 'no'}"
            precise_details = (
                f"total={s.get('total', 100)}\n"
                f"per_class={per_class}\n"
                f"class_names={', '.join(s.get('class_names') or []) or '-'}\n"
                f"classes={json.dumps(s.get('classes') or {}, ensure_ascii=True)}\n"
                f"multi_profiles={json.dumps(multi_profiles, ensure_ascii=True)}\n"
                f"multi_totals={json.dumps(s.get('multi_totals') or {}, ensure_ascii=True)}\n"
                f"multi_classes={json.dumps(s.get('multi_classes') or {}, ensure_ascii=True)}"
            )

        model_target = self.ui.var_model.get().strip() if task != "generate_only" else "(n/a for generate only)"

        top = tk.Toplevel(self.frame.winfo_toplevel())
        top.title("Start pipeline")
        top.transient(self.frame.winfo_toplevel())
        top.resizable(True, True)
        top.minsize(740, 460)
        top.geometry("860x620")

        result = {"ok": False}

        outer = ttk.Frame(top, padding=10)
        outer.pack(fill="both", expand=True)

        header = (
            f"Task: {task_label}\n"
            f"Run mode: {run_mode} ({run_label})\n"
            f"Dataset mode: {dataset_mode}\n"
            f"Output: {out_dir}\n"
            f"Profiles: {profile_label}\n"
            f"Multi-profile: {'on' if multi_enabled else 'off'} ({multi_mode})\n"
            f"Model target: {model_target}\n"
            f"Filter mode: {filter_mode} (realism_enabled={realism_enabled})\n"
            f"Precise: {precise_summary}"
        )
        ttk.Label(outer, text=header, justify="left", anchor="w").pack(fill="x", pady=(0, 10))

        sep = ttk.Separator(outer, orient="horizontal")
        sep.pack(fill="x", pady=(0, 8))

        details_wrap = ttk.Frame(outer)
        details_wrap.pack(fill="both", expand=True)

        def _add_collapsible(parent: ttk.Frame, title: str, text: str) -> None:
            block = ttk.Frame(parent)
            block.pack(fill="x", pady=(0, 6))
            state = {"open": False}
            body = ttk.Frame(block)
            btn = ttk.Button(block, text=f"\u25b8 {title}")
            btn.pack(anchor="w")

            height = min(14, max(3, len(text.splitlines())))
            txt = tk.Text(body, height=height, wrap="word")
            txt.insert("1.0", text)
            txt.configure(state="disabled")
            txt.pack(fill="x", expand=True)

            def _toggle() -> None:
                if state["open"]:
                    body.pack_forget()
                    btn.configure(text=f"\u25b8 {title}")
                    state["open"] = False
                else:
                    body.pack(fill="x", pady=(4, 0))
                    btn.configure(text=f"▼ {title}")
                    state["open"] = True

            btn.configure(command=_toggle)

        _add_collapsible(details_wrap, "Filter Details", filter_details)
        _add_collapsible(details_wrap, "Precise Details", precise_details)
        if task != "generate_only":
            _add_collapsible(details_wrap, "Model Details", f"model_path={self.ui.var_model.get().strip()}")

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(8, 0))

        def _on_cancel() -> None:
            result["ok"] = False
            top.destroy()

        def _on_start() -> None:
            result["ok"] = True
            top.destroy()

        ttk.Button(btns, text="Cancel", width=12, command=_on_cancel).pack(side="right")
        ttk.Button(btns, text="Start", width=12, command=_on_start).pack(side="right", padx=(0, 8))

        top.bind("<Escape>", lambda _e: _on_cancel())
        top.protocol("WM_DELETE_WINDOW", _on_cancel)
        top.grab_set()
        top.wait_window()
        return bool(result["ok"])

    def _confirm_overwrite_existing_new_dataset(self, out_dir: str) -> bool:
        """Double confirmation before overwriting an existing dataset in Create New mode."""
        out_path = self.logic._resolve_out_dir(out_dir)
        if not out_path.exists():
            return True
        if not self.ui.ask_yes_no(
            "Output already exists",
            "Dataset Mode is 'Create New', but the output folder already exists.\n\n"
            f"{out_path}\n\n"
            "Continuing can overwrite existing dataset contents.\n\nContinue?",
        ):
            return False
        return self.ui.ask_yes_no(
            "Confirm overwrite",
            "Final confirmation:\n\n"
            f"Overwrite existing dataset at:\n{out_path}\n\n"
            "Proceed?",
        )

    def _ask_multi_dataset_train_strategy(self, dss: list[Path]) -> str | None:
        msg = (
            f"{len(dss)} datasets selected.\n\n"
            "Yes: Merge and keep merged dataset\n"
            "No: Merge into temporary dataset and auto-cleanup\n"
            "Cancel: Abort"
        )
        choice = messagebox.askyesnocancel("Multi-dataset training", msg)
        if choice is None:
            return None
        return "merge_keep" if choice else "merge_temp"

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None: return

        bindings = self._iter_setting_bindings()
        restore_persisted_vars(st, bindings, skip_empty_strings=True)
        try:
            raw = str(st.get("pipeline.precise_settings_json", "") or "").strip()
            if raw:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    for k, v in loaded.items():
                        if k in self._precise_settings:
                            self._precise_settings[k] = v
        except Exception:
            pass

        try: self._refresh_datasets(); self._on_dataset_multi_toggle(); self._on_dataset_selected()
        except Exception: pass
        # _on_dataset_selected() -> _apply_dataset_settings() derives these from the dataset;
        # put the persisted values back on top.
        restore_persisted_vars(st, [b for b in bindings if b[0] in DATASET_DERIVED_KEYS], skip_empty_strings=True)
        try: self._refresh_models()
        except Exception: pass
        try: self._load_filter_popup_extras_from_profiles(st)
        except Exception: self._filter_popup_extras = {}

    def _load_filter_popup_extras_from_profiles(self, st: SettingsStore) -> None:
        raw_profiles = str(st.get("pipeline.filter_profiles_json", "") or "").strip()
        raw_active = str(st.get("pipeline.filter_profile_active", "") or "").strip() or "Default"
        self._filter_popup_extras = load_filter_popup_extras_from_profiles_json(raw_profiles, raw_active)

    def _wire_settings_autosave(self) -> None:
        wire_vars_autosave(self._store(), self._iter_setting_bindings(), self.frame)

    def _iter_setting_bindings(self) -> list[tuple[str, Any]]:
        return [(key, getattr(self.ui, attr_name)) for key, attr_name in PERSISTED_FIELD_MAP]

    def _filter_var_value(self, key: str) -> Any:
        """Current value of the UI variable ``var_filter_<key>``."""
        return getattr(self.ui, f"var_filter_{key}").get()

    def _current_filter_settings_dict(self) -> dict[str, Any]:
        data = formatted_filter_settings(self._filter_var_value)
        data.update(self._filter_popup_extras)
        return data

    def _apply_filter_settings_dict(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        self._filter_popup_extras = extract_filter_popup_extras(data)
        for key, value in normalized_filter_settings(data.get).items():
            getattr(self.ui, f"var_filter_{key}").set(value)

    def _load_filter_profiles(self) -> tuple[dict[str, dict[str, Any]], str]:
        st = self._store()
        profiles: dict[str, dict[str, Any]] = {}
        active = "Default"
        current_live: dict[str, Any] = {}
        if st is not None:
            raw_profiles = str(st.get("pipeline.filter_profiles_json", "") or "").strip()
            raw_active = str(st.get("pipeline.filter_profile_active", "") or "").strip()
            if raw_active:
                active = raw_active
            if raw_profiles:
                try:
                    obj = json.loads(raw_profiles)
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            if isinstance(k, str) and isinstance(v, dict):
                                profiles[k] = v
                except Exception:
                    pass
            current_live = normalized_filter_settings(lambda k, d: st.get(f"pipeline.filter.{k}", d))
            current_live.update(self._filter_popup_extras)
        if not profiles:
            profiles = {"Default": current_live or self._current_filter_settings_dict()}
            active = "Default"
        if active not in profiles:
            active = sorted(profiles.keys())[0]
        return profiles, active

    def _save_filter_profiles(self, profiles: dict[str, dict[str, Any]], active: str) -> None:
        st = self._store()
        if st is None:
            return
        st.set("pipeline.filter_profiles_json", json.dumps(profiles, ensure_ascii=True))
        st.set("pipeline.filter_profile_active", str(active))
        st.schedule_save(self.frame)

    def _image_filters_payload(self) -> dict[str, Any]:
        return image_filters_payload(self._filter_var_value, self._filter_popup_extras)

    # ------------------------------------------------------------------
    # Precise mode helpers
    # ------------------------------------------------------------------

    def _open_precise_popup(self) -> None:
        """Open the precise mode settings popup (non-modal)."""
        # If already open, just bring it to front
        if self._precise_popup_ref is not None and self._precise_popup_ref.is_open():
            try:
                self._precise_popup_ref._win.lift()
                self._precise_popup_ref._win.focus_set()
            except Exception:
                pass
            return

        multi_enabled = bool(self.ui.var_profiles_multi.get())
        self._precise_settings["multi_profiles"] = (
            self._profiles_multi_list(available=self._profiles_multi_available_values())
            if multi_enabled else []
        )
        self._refresh_precise_classes()

        def on_apply(updated: dict[str, Any]) -> None:
            self._precise_settings.update(updated)
            self._save_precise_settings()

        self._precise_popup_ref = open_precise_popup(
            parent=self.frame,
            settings=self._precise_settings,
            on_apply=on_apply,
            get_class_names_func=self._get_precise_class_names_from_config,
        )

    def _get_precise_class_names_from_config(self) -> list[str]:
        cfg_str = self.ui.var_config.get().strip()
        return config_class_names(resolve_under(self.sim_root, cfg_str) if cfg_str else None)

    def _refresh_precise_classes(self) -> None:
        """Sync class names and multi-profile list into _precise_settings; push to open popup."""
        names = self._get_precise_class_names_from_config()
        self._precise_settings["class_names"] = names
        default = int(self._precise_settings.get("total", 100))
        for cls in names:
            self._precise_settings["classes"].setdefault(cls, default)

        multi_enabled = bool(self.ui.var_profiles_multi.get())
        self._precise_settings["multi_profiles"] = (
            self._profiles_multi_list(available=self._profiles_multi_available_values())
            if multi_enabled else []
        )

        # Live-update the popup if it is currently open
        if self._precise_popup_ref is not None and self._precise_popup_ref.is_open():
            try:
                self._precise_popup_ref.refresh(self._precise_settings)
            except Exception:
                pass

    def _save_precise_settings(self) -> None:
        st = self._store()
        if st is not None:
            st.set("pipeline.precise_settings_json", json.dumps(self._precise_settings, ensure_ascii=True))
            st.schedule_save(self.frame)

    def _get_precise_counts(self) -> dict[str, Any] | None:
        """Return sample-count overrides for precise mode, or None on validation error."""
        counts, err = precise_counts(self._precise_settings)
        if err:
            self.ui.show_messagebox("error", "Error", err)
        return counts

    def _write_precise_config(self, cfg_path: Path, counts: dict[str, Any]) -> Path:
        """Write a temp config YAML with the classes section overridden by counts."""
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        return self._write_temp_cfg(apply_precise_counts(data, counts), prefix="precise_")

    def _write_temp_cfg(self, data: Any, *, prefix: str) -> Path:
        """Write a temp config under outputs/live; it is removed when the current run ends."""
        path = write_temp_yaml(data, self.sim_root / "outputs" / "live", prefix=prefix)
        self._temp_cfg_paths.append(path)
        return path

    # ------------------------------------------------------------------

    def _open_image_filters_popup(self) -> None:
        """Open shared image filter popup."""
        profiles, active_profile = self._load_filter_profiles()

        def on_close(updated_profiles: dict[str, dict[str, Any]], updated_active: str) -> None:
            self._save_filter_profiles(updated_profiles, updated_active)

        open_filter_popup(
            parent=self.frame,
            profiles=profiles,
            active_profile=active_profile,
            on_close=on_close,
            get_current_values=self._current_filter_settings_dict,
            set_current_values=self._apply_filter_settings_dict,
            float_or_default=float_or_default,
            show_messagebox=self.ui.show_messagebox,
            ask_string=self.ui.ask_string,
            ask_yes_no=self.ui.ask_yes_no,
            sim_root=str(self.sim_root),
        )

    def _refresh_profile_models(self) -> None:
        paths = self.logic.get_profile_classifier_models()
        values = [""] + [str(p.resolve().relative_to(self.sim_root.resolve())) if p.is_relative_to(self.sim_root) else str(p) for p in paths]
        self.ui.set_profile_model_combo_values(values)
        if (cur := self.ui.var_profile_model.get().strip()) and cur in values: return
        if not cur: self.ui.var_profile_model.set("")

    def _apply_profile_model_lock(self) -> None:
        locked = bool(self.ui.var_profile_model_lock.get())
        if locked and not self.ui.var_profile_model.get().strip():
            self.ui.var_profile_model_lock.set(False)
            return
        self.ui.set_profile_model_combo_state("disabled" if locked else "readonly")

    def _build_profile_model(self) -> None:
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first.")
            return

        preset = (self.ui.var_profile_build_preset.get().strip() or "quick").lower()
        if preset not in ("quick", "full"): preset = "quick"

        cfg, out_ds, out_model = profile_model_paths(self.sim_root, preset, time.strftime("%Y%m%d_%H%M%S"))
        if not cfg.exists():
            self.ui.show_messagebox("error", "Missing config", f"Config not found:\n{cfg}")
            return

        rel_model = rel_to(self.sim_root, out_model)
        self.ui.var_profile_model.set(rel_model)
        self.ui.var_profile_model_lock.set(True)
        self._apply_profile_model_lock()
        
        def run_simple_cmd_callback(cmd_args: list[str], extra_env: dict[str, str] | None = None) -> None:
            self._run_simple_cmd(cmd_args, extra_env)

        self.logic.build_profile_model(preset, str(cfg), str(out_ds), str(out_model), self._append_log, run_simple_cmd_callback)
        self.ui.var_profile_model.set(rel_model)

    def start_pipeline(self) -> None: self._start_pipeline(task="full")
    def start_pipeline_generate_only(self) -> None: self._start_pipeline(task="generate_only")

    def _rel_to_sim_root(self, p: Path) -> str:
        return rel_to(self.sim_root, p)

    def _build_run_specs(
        self,
        *,
        task: str,
        run_count: int,
        dataset_mode: str,
        multi_enabled: bool,
        multi_mode: str,
        profile_ids: list[str],
        render_backend: str,
        out_dir: str,
    ) -> tuple[list[dict[str, str]], str, Any, str] | None:
        run_specs: list[dict[str, str]] = []
        effective_task = task
        run_complete_callback = None

        if multi_enabled and multi_mode == "mixed":
            if task != "generate_only":
                self.ui.show_messagebox("error", "Error", "Multi-profile mode 'mixed' is supported only for Generate Only.")
                return None
            if len(profile_ids) < 2:
                self.ui.show_messagebox("error", "Error", "Mixed dataset mode requires at least 2 selected profiles.")
                return None
            if run_count != 1:
                self.ui.show_messagebox("error", "Error", "Mixed dataset generation currently supports only a single run.")
                return None

            resolved_configs: list[tuple[str, Path, str, str]] = []
            backends_used: set[str] = set()
            for pid in profile_ids:
                cfg_info = self.logic.config_for_profile(pid, want_backend=render_backend or None)
                if not cfg_info:
                    cfg_info = self.logic.config_for_profile(pid, want_backend=None)
                if not cfg_info:
                    self.ui.show_messagebox("error", "Error", f"No matching config found for profile:\n{pid}\n\nPick or create a config with run.component_profile={pid}.")
                    return None
                cfg_path = Path(cfg_info["path"])
                run_id = str(cfg_info.get("run_id") or pid)
                try:
                    cfg_data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                except Exception:
                    cfg_data = {}
                backend = str((cfg_data.get("render") or {}).get("backend") or "opencv_2d").strip() or "opencv_2d"
                resolved_configs.append((pid, cfg_path, run_id, backend))
                backends_used.add(backend)

            if backends_used == {"opencv_2d"}:
                cfg_str = self.ui.var_config.get().strip()
                cfg_path = resolve_under(self.sim_root, cfg_str) if cfg_str else None
                if not cfg_path or not cfg_path.exists():
                    self.ui.show_messagebox("error", "Error", f"Config file not found:\n{cfg_str}")
                    return None
                try:
                    base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                except Exception as e:
                    self.ui.show_messagebox("error", "Error", f"Failed to read config:\n{cfg_path}\n\n{e}")
                    return None

                try:
                    tmp_cfg_path = self._write_temp_cfg(mixed_profile_config(base_cfg, profile_ids), prefix="run_profile_cls_selected_")
                except Exception as e:
                    self.ui.show_messagebox("error", "Error", f"Failed to write temporary config under outputs/live:\n\n{e}")
                    return None

                run_specs.append({
                    "profile_id": "multi", "config": self._rel_to_sim_root(tmp_cfg_path), "out_dir": out_dir,
                    "model_path": self.ui.var_model.get().strip(),
                })
                effective_task = "generate_mixed"
            else:
                final_out = self.logic._resolve_out_dir(out_dir)
                replace_existing_final_out = False
                if final_out.exists() and dataset_mode == "extend":
                    replace_existing_final_out = True
                    self._append_log(f"[mixed] reusing selected output path: {final_out}\n")
                elif final_out.exists():
                    final_out = unique_timestamped_dir(final_out, time.strftime("%Y%m%d_%H%M%S"))
                    out_dir = str(final_out)
                    self.ui.var_out.set(rel_to(self.sim_root, final_out))
                    self._append_log(f"[mixed] output exists, using new dataset path: {final_out}\n")

                live_dir = (self.sim_root / "outputs" / "live")
                live_dir.mkdir(parents=True, exist_ok=True)
                tmp_root = live_dir / f"mixed_profile_gen_{int(time.time())}_{os.getpid()}"
                try:
                    tmp_root.mkdir(parents=True, exist_ok=False)
                except Exception as e:
                    self.ui.show_messagebox("error", "Error", f"Failed to create temporary mixed-generation directory:\n{tmp_root}\n\n{e}")
                    return None

                self._append_log(f"[mixed] Cross-backend profiles detected ({', '.join(sorted(backends_used))}). Generating per profile and auto-merging.\n")
                for idx, (pid, cfg_path, _run_id, backend) in enumerate(resolved_configs):
                    tmp_out = tmp_root / f"{idx:02d}_{self.logic.slugify_name(pid)}"
                    run_specs.append({
                        "profile_id": pid,
                        "config": self._rel_to_sim_root(cfg_path),
                        "out_dir": str(tmp_out),
                        "model_path": self.ui.var_model.get().strip(),
                    })
                    self._append_log(f"[mixed] {pid} -> backend={backend} tmp={tmp_out}\n")
                effective_task = "generate_only"

                def _merge_after_run(_run_i: int, specs: list[dict[str, str]], log_callback) -> bool:
                    ok = True
                    src_dirs: list[Path] = []
                    try:
                        src_dirs = [Path(spec["out_dir"]).resolve() for spec in specs]
                        merge_sources = list(src_dirs)
                        merge_target = final_out
                        tmp_target: Path | None = None

                        if replace_existing_final_out and final_out.exists():
                            if dataset_mode == "extend":
                                merge_sources = [final_out] + merge_sources
                                tmp_target = final_out.parent / f"{final_out.name}.__merge_tmp_{int(time.time())}_{os.getpid()}"
                                if tmp_target.exists():
                                    shutil.rmtree(tmp_target, ignore_errors=True)
                                merge_target = tmp_target
                                log_callback(f"[mixed] extend mode: including existing dataset as merge source: {final_out}\n")
                            else:
                                shutil.rmtree(final_out, ignore_errors=True)
                                log_callback(f"[mixed] removed previous dataset at target: {final_out}\n")

                        log_callback(f"[mixed] merging {len(merge_sources)} source datasets into: {merge_target}\n")
                        self.logic.merge_datasets_for_training(merge_sources, merge_target, log_callback, log_callback)
                        if tmp_target is not None and merge_target == tmp_target:
                            if final_out.exists():
                                shutil.rmtree(final_out, ignore_errors=True)
                            tmp_target.rename(final_out)
                            merge_target = final_out
                            log_callback(f"[mixed] extend mode: replaced target dataset with merged result: {final_out}\n")
                        log_callback(f"[mixed] merged dataset ready: {final_out}\n")
                    except Exception as e:
                        log_callback(f"[mixed] merge failed: {e}\n")
                        ok = False
                    if not ok:
                        # Keep freshly generated samples so a failed merge can be retried manually.
                        log_callback(f"[mixed] temp generation data kept after failed merge: {tmp_root}\n")
                    else:
                        try:
                            for p in src_dirs:
                                if p.exists():
                                    shutil.rmtree(p, ignore_errors=True)
                            if tmp_root.exists():
                                shutil.rmtree(tmp_root, ignore_errors=True)
                            log_callback(f"[mixed] cleaned temp generation data: {tmp_root}\n")
                        except Exception as e:
                            log_callback(f"[mixed] temp cleanup warning: {e}\n")
                    return ok

                run_complete_callback = _merge_after_run
        else:
            base_out = Path(out_dir)
            base_model = Path(self.ui.var_model.get().strip())
            for pid in profile_ids:
                cfg_info = self.logic.config_for_profile(pid, want_backend=render_backend or None)
                if not cfg_info:
                    self.ui.show_messagebox("error", "Error", f"No matching config found for profile:\n{pid}\n\nPick or create a config with run.component_profile={pid}.")
                    return None
                cfg_path = Path(cfg_info["path"])
                cfg_rel = self._rel_to_sim_root(cfg_path)
                run_id = str(cfg_info.get("run_id") or pid)
                out_i, model_i = profile_run_targets(
                    base_out, base_model, run_id, multi=len(profile_ids) > 1, models_root=self.sim_root / "outputs" / "models"
                )

                run_specs.append({
                    "profile_id": pid, "config": cfg_rel, "out_dir": str(out_i), "model_path": str(model_i),
                })

        return run_specs, effective_task, run_complete_callback, out_dir

    def _apply_precise_counts_to_run_specs(self, run_specs: list[dict[str, str]]) -> list[dict[str, str]] | None:
        counts = self._get_precise_counts()
        if counts is None:
            return None
        patched_specs: list[dict[str, str]] = []
        for spec in run_specs:
            cfg_path = resolve_under(self.sim_root, spec["config"])
            spec_counts = precise_counts_for_profile(counts, spec.get("profile_id", ""))
            try:
                patched_path = self._write_precise_config(cfg_path, spec_counts)
                patched_specs.append({**spec, "config": self._rel_to_sim_root(patched_path)})
            except Exception as e:
                self.ui.show_messagebox("error", "Error", f"Failed to write precise config:\n\n{e}")
                return None
        return patched_specs
    
    def _start_pipeline(self, *, task: str) -> None:
        if self.logic.is_busy(): return
        
        run_mode, dataset_mode = self.ui.var_run_mode.get(), self.ui.var_dataset_mode.get()
        out_dir = ""
        if dataset_mode == "extend":
            if not (selected_ds := self._selected_dataset_dir()):
                self.ui.show_messagebox("error", "Error", "No dataset selected to extend.\n\nSelect a dataset or use 'Create New' mode.")
                return
            try:
                if str(selected_ds.resolve()).startswith(str((self.sim_root / "outputs" / "sim_data" / "versions").resolve()) + os.sep):
                    self.ui.show_messagebox("error", "Error", "Cannot extend/overwrite a dataset snapshot under outputs/sim_data/versions.\n\nSelect a dataset under outputs/sim_data/runs instead.")
                    return
            except Exception: pass
            if not self.ui.ask_yes_no("Extend dataset?", f"Extend Existing will APPEND new samples into the selected dataset folder.\n\nDataset:\n{selected_ds}\n\nContinue?"): return
            out_dir = str(selected_ds)
            self._append_log(f"\n=== Extending selected dataset: {out_dir} ===\n")
        else:
            out_dir = self.ui.var_out.get().strip()
            self._append_log(f"\n=== Creating new dataset: {out_dir} ===\n")

        try: run_count = parse_run_count(run_mode, self.ui.var_run_count.get())
        except ValueError:
            self.ui.show_messagebox("error", "Error", "Invalid run count. Please enter a positive integer.")
            return

        self._append_log(f"Run mode: {run_mode}" + (f" ({run_count}x)" if run_count > 0 else " (continuous)") + "\n\n")

        render_backend = self.ui.var_render_backend.get().strip() or ""
        
        multi_enabled = bool(self.ui.var_profiles_multi.get())
        multi_mode = self.ui.var_profiles_multi_mode.get().strip() or "separate"
        
        profile_ids = []
        if multi_enabled:
            profile_ids = self._profiles_multi_list(available=self._profiles_multi_available_values())
            if not profile_ids:
                self.ui.show_messagebox("error", "Error", "Multi-profile mode is enabled but no profiles are selected.\n\nClick Pick and select at least 1 profile.")
                return
        else:
            if p := self.ui.var_profile.get().strip(): profile_ids = [p]

        try:
            if profile_ids: self._autoselect_config_for_profile(profile_ids[0], prefer_quiet=False)
        except Exception: pass

        built_specs = self._build_run_specs(
            task=task,
            run_count=run_count,
            dataset_mode=dataset_mode,
            multi_enabled=multi_enabled,
            multi_mode=multi_mode,
            profile_ids=profile_ids,
            render_backend=render_backend,
            out_dir=out_dir,
        )
        if built_specs is None:
            return
        run_specs, effective_task, run_complete_callback, out_dir = built_specs

        if dataset_mode == "new" and not self._confirm_overwrite_existing_new_dataset(out_dir):
            return
        
        # Precise mode: patch each spec's config with user-specified sample counts
        if run_mode == "precise":
            patched_specs = self._apply_precise_counts_to_run_specs(run_specs)
            if patched_specs is None:
                return
            run_specs = patched_specs

        image_filters = self._image_filters_payload()

        if not self._confirm_pipeline_start(
            task=effective_task, out_dir=out_dir, run_mode=run_mode, run_count=run_count,
            dataset_mode=dataset_mode, profile_ids=profile_ids, multi_enabled=multi_enabled, multi_mode=multi_mode,
            image_filters=image_filters,
        ): return

        # Mixed cross-backend generation runs through temporary per-profile output dirs.
        # Those dirs are always newly created and must not use --extend.
        runtime_dataset_mode = dataset_mode
        if (
            dataset_mode == "extend"
            and multi_enabled
            and multi_mode == "mixed"
            and effective_task == "generate_only"
        ):
            runtime_dataset_mode = "new"
            self._append_log("[mixed] using internal dataset mode 'new' for temporary generation dirs; extend is applied during final merge.\n")

        self.ui.set_run_buttons_state(True)
        self.logic.stop_evt.clear()

        status_vars = {
            "phase": self.ui.var_phase, "run_progress": self.ui.var_run_progress,
            "epoch": self.ui.var_epoch, "ips": self.ui.var_ips, "last": self.ui.var_last,
            "dataset_dir": self.state.dataset_dir,
            "snap_every": self._snap_every_n(), "snap_keep": self._snap_keep_n(), "autosnap": self.ui.var_autosnap.get()
        }
        self.logic.start_pipeline(
            run_specs=run_specs, run_count=run_count, dataset_mode=runtime_dataset_mode, task=effective_task,
            image_filters=image_filters,
            event_callback=self._handle_event, log_callback=self._append_log,
            ui_update_callback=lambda: self.frame.after(0, self._after_pipeline_run_ui_update),
            ui_reset_callback=lambda: self.frame.after(0, self._reset_ui_on_pipeline_end),
            status_vars=status_vars,
            run_complete_callback=run_complete_callback,
        )

    def _after_pipeline_run_ui_update(self) -> None:
        self._refresh_datasets()
        if self.ui.var_dataset_mode.get() != "extend":
            out_path = self.logic._resolve_out_dir(self.ui.var_out.get())
            runs_base = (self.sim_root / "outputs" / "sim_data" / "runs").resolve()
            try:
                out_path.resolve().relative_to(runs_base)
                self.ui.var_dataset.set(out_path.name)
                self._on_dataset_selected()
            except Exception: pass
        else:
            if self.state.dataset_dir: self._start_dataset_size_calc(self.state.dataset_dir)

    def _reset_ui_on_pipeline_end(self) -> None:
        for p in self._temp_cfg_paths:
            try: p.unlink(missing_ok=True)
            except Exception: pass
        self._temp_cfg_paths.clear()
        self.ui.set_run_buttons_state(False)
        self.ui.var_phase.set("phase: idle")
        self.ui.var_run_progress.set("run: -")
        self.logic.current_run_iteration = 0
        self.logic.total_run_count = 0

    def stop_pipeline(self) -> None:
        self.logic.stop_pipeline()

    def _run_simple_cmd(self, args: list[str], extra_env: dict[str, str] | None = None, *, cleanup_dir: Path | None = None) -> bool:
        # cleanup_dir (or a pending logic._temp_merge_cleanup_dir) is removed when this process exits with rc=0.
        cleanup_dir = cleanup_dir or self.logic._temp_merge_cleanup_dir
        self.logic._temp_merge_cleanup_dir = None
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first.")
            if cleanup_dir is not None: self.logic.cleanup_temp_merge_dir(cleanup_dir, 0, self._append_log)
            return False

        self.ui.set_run_buttons_state(True)
        self.logic.stop_evt.clear()
        
        env = os.environ.copy()
        env["EVENT_LOG"] = str(self.event_log_path)
        env["SIMPLE_SIM_EVENT_LOG"] = str(self.event_log_path)
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env: env.update(extra_env)

        # A single element is a pre-built shell command line; multiple elements are argv and get quoted
        # so dataset/model paths containing spaces survive bash -lc.
        shell_cmd = args[0] if len(args) == 1 else shlex.join(args)
        cmd = ["bash", "-lc", shell_cmd]

        def on_exit(rc: int | None) -> None:  # reader thread; logging goes through log_q
            if cleanup_dir is not None: self.logic.cleanup_temp_merge_dir(cleanup_dir, rc, self._append_log)

        try: self.logic.launch_process(cmd, env, self._append_log, self._handle_event, on_exit=on_exit)
        except Exception as e:
            self._append_log(f"\nError starting command: {e}\n"); self.ui.set_run_buttons_state(False)
            if cleanup_dir is not None: self.logic.cleanup_temp_merge_dir(cleanup_dir, 0, self._append_log)
            return False
        self.ui.var_phase.set("running cmd")
        return True

    def _handle_event(self, evt: dict[str, Any]) -> None:
        # Event tailing runs in a worker thread; marshal to the Tk thread.
        if threading.current_thread() is not threading.main_thread():
            self.event_q.put(evt)
            return
        et = evt.get("event")
        if et == "gen_start":
            self.ui.var_phase.set("phase: generating")
            if out_dir := evt.get("output_dir"): self.state.dataset_dir = Path(out_dir)
        elif et == "gen_progress":
            self.ui.var_phase.set("phase: generating")
            self.ui.var_ips.set(f"img/s: {evt.get('samp_per_s', '-')}")
            if last_img := evt.get("last_image_path"): self._push_image(last_img)
        elif et == "gen_done":
            self.ui.var_phase.set("phase: generated")
            if out_dir := evt.get("output_dir"): self.state.dataset_dir = Path(out_dir)
            if last_img := evt.get("last_image_path"): self._push_image(last_img)
        elif et == "train_start":
            self.ui.var_phase.set("phase: training")
            if ds := evt.get("dataset_dir"): self.state.dataset_dir = Path(ds)
        elif et == "epoch_start":
            self.ui.var_phase.set("phase: training")
            self.ui.var_epoch.set(f"epoch: {evt.get('epoch', '-')}/{evt.get('epochs', '-')}")
        elif et == "train_batch":
            self.ui.var_phase.set("phase: training")
            if ips := evt.get("img_per_s"): self.ui.var_ips.set(f"img/s: {ips:.1f}")
            if lid := evt.get("last_id"): self.ui.var_last.set(f"last: {lid}")
            if limg := evt.get("last_image_path"): self._push_image(limg)
        elif et == "eval_start":
            self.ui.var_phase.set("phase: evaluating")
            if ds := evt.get("dataset_dir"): self.state.dataset_dir = Path(ds)
        elif et == "eval_batch":
            self.ui.var_phase.set("phase: evaluating")
            if ips := evt.get("img_per_s"): self.ui.var_ips.set(f"img/s: {ips:.1f}")
            if limg := evt.get("last_image_path"): self._push_image(limg)
        elif et == "eval_done":
            self.ui.var_phase.set("phase: done")
        elif et == "validate_start":
            self.ui.var_phase.set("phase: validating")
        elif et == "validate_done":
            ok = evt.get("ok")
            self.ui.var_phase.set("phase: validated_ok" if ok else "phase: validated_fail")

    def _push_image(self, rel_path: str) -> None:
        if not self.state.dataset_dir: return
        img_path = self.state.dataset_dir / rel_path
        if not img_path.exists(): return
        name = str(img_path)
        if self._recent_imgs and self._recent_imgs[-1] == name: return
        self._recent_imgs.append(name)
        self._recent_imgs = self._recent_imgs[-12:]
        # Rendering 12 thumbnails can be expensive; schedule refresh on UI tick.
        self._thumb_refresh_pending = True

    def _maybe_refresh_thumbnails(self) -> None:
        if not self._thumb_refresh_pending:
            return
        now = time.time()
        if now - self._last_thumb_refresh_ts < self._thumb_refresh_min_interval_s:
            return
        self._last_thumb_refresh_ts = now
        self._thumb_refresh_pending = False
        self._refresh_thumbnails()

    def _refresh_thumbnails(self) -> None:
        self.ui.refresh_thumbnails(self._recent_imgs, self.state.dataset_dir, self._label_dict,
                                   self._image_to_sample, self._meta_by_sample, self._meta_by_image_rel, draw_defect_overlay)

    def _tick_ui(self) -> None:
        # Keep UI responsive even under heavy log/event throughput.
        log_batch: list[str] = []
        for _ in range(200):
            try:
                log_batch.append(self.log_q.get_nowait())
            except queue.Empty:
                break
        if log_batch:
            self._append_log("".join(log_batch))

        for _ in range(120):
            try:
                self._handle_event(self.event_q.get_nowait())
            except queue.Empty:
                break

        try:
            while True:
                job_id, size_b, ds_name = self._dataset_size_q.get_nowait()
                if job_id == self._dataset_size_job_id:
                    self.ui.var_ds_size.set(f"DS: {self.ui.fmt_bytes(size_b)} ({ds_name})")
        except queue.Empty:
            pass
        
        if not self.logic.is_busy():
            if self.ui.btn_start["state"] == "disabled" or self.ui.btn_start_generate["state"] == "disabled": self.ui.set_run_buttons_state(False)

        self._maybe_refresh_thumbnails()
        self._maybe_update_stats()
        self.frame.after(120, self._tick_ui)

    def _maybe_update_stats(self) -> None:
        now = time.time()
        if now - self._last_stats_ts < 1.0: return
        self._last_stats_ts = now

        proc_running = self.logic.is_process_running()
        if self._prev_proc_running and not proc_running:
            if self.state.dataset_dir: self._start_dataset_size_calc(self.state.dataset_dir)
            self._last_ds_size_ts = now
        self._prev_proc_running = proc_running

        if proc_running and (now - self._last_ds_size_ts) >= 5.0:
            if self.state.dataset_dir: self._start_dataset_size_calc(self.state.dataset_dir)
            self._last_ds_size_ts = now

    def _snap_every_n(self) -> int:
        return parse_snap_every(str(self.ui.var_snap_every.get()))

    def _snap_keep_n(self) -> int:
        return parse_snap_keep(str(self.ui.var_snap_keep.get()))

    def _start_dataset_size_calc(self, ds: Path) -> None:
        self._dataset_size_job_id += 1
        job_id = self._dataset_size_job_id
        self.ui.var_ds_size.set("DS: calculating...")
        
        def update_ui(size_b: int) -> None:
            self._dataset_size_q.put((job_id, size_b, ds.name))

        self.logic.start_dataset_size_calc(ds, update_ui)

    def _refresh_dataset_stats(self) -> None:
        ds = self.state.dataset_dir
        if not ds:
            self.ui.set_dataset_samples_info("Samples: -", "black")
            self.ui.var_ds_size.set("DS: -")
            return

        manifest_path = ds / "dataset_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8") as f: manifest = json.load(f)
                stats = manifest.get("dataset_stats", {})
                splits = stats.get("splits", {})
                self._set_dataset_samples_from_manifest(ds, stats.get("total_samples"), splits.get("train"), splits.get("val"), splits.get("test"))
            except Exception as exc:
                _LOG.warning("Failed to refresh dataset stats: %s", exc)
                self._set_dataset_samples_info("Samples: manifest error", "")
        else: self._set_dataset_samples_info("Samples: manifest missing", "")

        self._start_dataset_size_calc(ds)

    def _set_dataset_samples_from_manifest(self, ds: Path, total: int | None, train: int | None, val: int | None, test: int | None) -> None:
        if total is None: return self._set_dataset_samples_info("Samples: manifest missing", "")
        self._set_dataset_samples_info(samples_text(total, train, val, test), sample_color_from_total(total))
        self._check_milestone(ds, total)

    def _set_dataset_samples_info(self, text: str, color: str) -> None:
        self.ui.set_dataset_samples_info(text, color)

    def _check_milestone(self, ds: Path, total: int) -> None:
        name = ds.name
        current = self._dataset_milestones.get(name, 0)
        for threshold in self._milestone_thresholds:
            if total >= threshold and threshold > current:
                self._dataset_milestones[name] = threshold
                self._log_dataset_milestone(ds, threshold, total)
                break

    def _log_dataset_milestone(self, ds: Path, threshold: int, total: int) -> None:
        event = {
            "ts": time.time(), "event": "dataset_milestone", "dataset": ds.name,
            "threshold": threshold, "samples": total,
        }
        try:
            self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(self.event_log_path, [event])
        except Exception as e: _LOG.warning("Failed to log milestone: %s", e)

    def _apply_dataset_settings(self, ds: Path) -> None:
        if not ds: return
        if (config_path := ds / "config.yaml").exists(): self.ui.var_config.set(str(config_path))
        if profile_id := self.logic.get_dataset_profile_id(ds): self._set_profile_value(profile_id)
        
        self.ui.var_name.set(ds.name)
        self.ui.var_out.set(str(ds.resolve()))
        
        model_path = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        try: self.ui.var_model.set(str(model_path.resolve()))
        except Exception: self.ui.var_model.set(str(model_path))
        if not bool(self.ui.var_profiles_multi.get()):
            self.ui.var_dataset_mode.set("extend")
        self._update_model_dropdown_for_dataset()

    def _set_profile_value(self, profile_id: str) -> None:
        self.ui.var_profile.set(profile_id)
        self._last_profile_id = profile_id

    def _refresh_datasets(self) -> None:
        self._dataset_by_label = catalog_labels(self.sim_root, self._dataset_catalog, save_widget=self.frame)
        self._dataset_labels = list(self._dataset_by_label)
        self._dataset_dirs = list(self._dataset_by_label.values())

        self.ui.set_dataset_combo_values(self._dataset_labels)
        self._sync_multi_selection_after_dataset_refresh()

        if (cur := self.ui.var_dataset.get().strip()) and cur in self._dataset_by_label: return
        if self.state.dataset_dir:
            runs, versions = sim_data_roots(self.sim_root, create=False)
            want = self._dataset_catalog.display_label(self.state.dataset_dir, runs_root=runs, versions_root=versions)
            for label, p in self._dataset_by_label.items():
                if p == self.state.dataset_dir or label == want: self.ui.var_dataset.set(label); return
        if self._dataset_labels: self.ui.var_dataset.set(self._dataset_labels[0]); self._on_dataset_selected()

    def _dataset_multi_tooltip_text(self) -> str:
        labels = self._selected_dataset_labels()
        if not labels: return "(no datasets selected)"
        if len(labels) <= 12: return "\n".join(labels)
        return "\n".join(labels[:12]) + f"\n... (+{len(labels) - 12} more)"

    def _on_dataset_multi_toggle(self) -> None:
        enabled = bool(self.ui.var_dataset_multi.get())

        if enabled:
            paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
            if not paths and (ds := self._selected_dataset_dir_single()) is not None:
                self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json([ds]))

        self.ui.toggle_dataset_multi_widgets(enabled)
        self._sync_multi_summary()
        
        if enabled:
            if ds0 := self._selected_dataset_dir():
                if label0 := self._dataset_label_for_path(ds0): self.ui.var_dataset.set(label0)
        self._on_dataset_selected()

    def _open_dataset_multi_picker(self) -> None:
        if not self._dataset_labels:
            self.ui.show_messagebox("info", "Datasets", "No datasets found. Click Refresh after generating datasets.")
            return

        selected_paths = {str(p.resolve()) for p in self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())}
        if not selected_paths and (ds := self._selected_dataset_dir_single()) is not None:
            selected_paths.add(str(ds.resolve()))
        preselected = [
            label for label in self._dataset_labels
            if (p := self._dataset_by_label.get(label)) is not None and str(p.resolve()) in selected_paths
        ]
        picked = dialogs.pick_from_list(
            self.frame, "Select Datasets", self._dataset_labels, multi=True, selected=preselected,
            prompt="Select one or more datasets. One model is trained on all of them.", size=(640, 480),
        )
        if picked is None:
            return
        paths: list[Path] = [self._dataset_by_label[label] for label in picked if label in self._dataset_by_label]
        if not paths and (ds := self._selected_dataset_dir_single()) is not None:
            paths = [ds]

        self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json(paths))
        self._sync_multi_summary()
        if paths:
            if label0 := self._dataset_label_for_path(paths[0]):
                self.ui.var_dataset.set(label0)
        self._on_dataset_selected()

    def _decode_dataset_paths_json(self, s: str) -> list[Path]:
        return decode_dataset_paths_json(s, self.sim_root)

    def _encode_dataset_paths_json(self, paths: list[Path]) -> str:
        return encode_dataset_paths_json(paths, self.sim_root)

    def _dataset_label_for_path(self, p: Path) -> str | None:
        rp = p.resolve()
        for label, pp in self._dataset_by_label.items():
            if pp.resolve() == rp: return label
        for label, pp in self._dataset_by_label.items():
            if pp.name == p.name: return label
        return None

    def _selected_dataset_labels(self) -> list[str]:
        return [label for ds in self._selected_dataset_dirs() if (label := self._dataset_label_for_path(ds))]

    def _sync_multi_summary(self) -> None:
        self.ui.var_dataset_multi_summary.set(summarize_labels(self._selected_dataset_labels()))

    def _sync_multi_selection_after_dataset_refresh(self) -> None:
        if not hasattr(self.ui, "var_dataset_multi_paths"): return
        paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
        if not paths: self._sync_multi_summary(); return

        valid = {str(p.resolve()) for p in self._dataset_dirs}
        filtered = [p for p in paths if str(p.resolve()) in valid]

        if filtered != paths: self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json(filtered))
        if bool(self.ui.var_dataset_multi.get()) and filtered:
            if label0 := self._dataset_label_for_path(filtered[0]): self.ui.var_dataset.set(label0)
        self._sync_multi_summary()

    def _display_for_dataset(self, p: Path, *, runs: Path, versions: Path) -> str:
        return dataset_display_name(p, runs_root=runs, versions_root=versions)

    @staticmethod
    def _profile_from_meta(meta: dict[str, Any] | None) -> tuple[str | None, str | None]:
        comp = (meta or {}).get("component_profile")
        return (comp.get("profile_id"), comp.get("profile_hash")) if isinstance(comp, dict) else (None, None)

    def _refresh_models(self) -> None:
        paths = self.logic.get_all_model_paths()
        self._model_paths = paths
        self._model_profile_cache.clear()
        self._model_meta_gen += 1
        gen, uncached = self._model_meta_gen, []
        for p in self._model_paths:
            key = str(p.resolve())
            if p.is_dir(): self._model_profile_cache[key] = ("multi", None); continue
            if (meta := self.logic.cached_checkpoint_meta(p)) is None: uncached.append(p); continue
            self._model_profile_cache[key] = self._profile_from_meta(meta)

        if uncached:
            # torch.load of every checkpoint is slow: do it off the Tk thread and re-filter when done.
            def worker() -> None:
                res = {str(p.resolve()): self._profile_from_meta(self.logic.read_checkpoint_meta(p)) for p in uncached}
                def apply() -> None:
                    if gen != self._model_meta_gen: return
                    self._model_profile_cache.update(res)
                    self._update_model_dropdown_for_dataset(); self._schedule_profile_compat_check()
                try: self.frame.after(0, apply)
                except Exception: pass
            threading.Thread(target=worker, daemon=True).start()

        values = [str(p.resolve().relative_to(self.sim_root.resolve())) if p.is_relative_to(self.sim_root) else str(p) for p in paths]
        self._all_model_combo_values = values
        self._update_model_dropdown_for_dataset()

    def _update_model_dropdown_for_dataset(self) -> None:
        ds = self.state.dataset_dir
        values = list(self._all_model_combo_values)
        if ds:
            profile_id = self.logic.get_dataset_profile_id(ds)
            profile_hash = self.logic.get_dataset_profile_hash(ds)
            if profile_id:
                exact, id_only, multi, legacy = [], [], [], []
                for p in self._model_paths:
                    rel_path = str(p.resolve())
                    pid, phash = self._model_profile_cache.get(rel_path, (None, None))
                    if pid == "multi" or (isinstance(pid, str) and pid.lower().startswith("multi")):
                        try: multi.append(str(p.resolve().relative_to(self.sim_root.resolve())))
                        except Exception: multi.append(str(p))
                        continue
                    if pid is None:
                        try: legacy.append(str(p.resolve().relative_to(self.sim_root.resolve())))
                        except Exception: legacy.append(str(p))
                        continue
                    if pid != profile_id: continue
                    if profile_hash and phash and phash == profile_hash: exact.append(str(p.relative_to(self.sim_root)))
                    else: id_only.append(str(p.relative_to(self.sim_root)))
                values = exact + id_only + multi + legacy
        
        self.ui.set_model_combo_values(values)
        if values:
            if self.ui.var_model.get().strip() not in values: self.ui.var_model.set(values[0])
        else: self.ui.var_model.set("")

    def _add_new_model(self) -> None:
        dss = self._selected_dataset_dirs()
        default_name = dss[0].name if dss else ""
        is_multi = bool(self.ui.var_dataset_multi.get()) and len(dss) > 1
        if is_multi and default_name and "multitrained" not in default_name.lower(): default_name = f"{default_name}_MultiTrained"
        
        name = self.ui.ask_string("New Model", "Model name (without .pt):", initialvalue=default_name)
        if not (name := (name or "").strip()): return
        if is_multi and "multitrained" not in name.lower(): name = f"{name}_MultiTrained"
        if "/" in name or "\\" in name:
            self.ui.show_messagebox("error", "Error", "Name must not contain path separators.")
            return

        suffix = ".bundle" if self.ui.var_model_bundle.get() else ".pt"
        model_path = self.sim_root / "outputs" / "models" / f"{name}{suffix}"
        rel = str(model_path.resolve().relative_to(self.sim_root.resolve()))

        if rel not in self._all_model_combo_values: self._all_model_combo_values.insert(0, rel)
        if rel not in (current_values := list(self.ui.model_combo["values"])):
            current_values.insert(0, rel)
            self.ui.set_model_combo_values(current_values)
        self.ui.var_model.set(rel)
        self._append_log(f"[model] New model target: {rel}\n")

    def _refresh_profiles(self) -> None:
        self.ui.set_profile_combo_values(self.logic.get_profile_paths(self.ui.var_render_backend.get().strip() or "opencv_2d"))

        cur = self.ui.var_profile.get().strip()
        if not cur or cur not in self.ui.profile_combo["values"]:
            if DEFAULT_COMPONENT_PROFILE in self.ui.profile_combo["values"]: self.ui.var_profile.set(DEFAULT_COMPONENT_PROFILE)
            elif self.ui.profile_combo["values"]: self.ui.var_profile.set(self.ui.profile_combo["values"][0])

        if bool(self.ui.var_profiles_multi.get()): self._set_profiles_multi(self._profiles_multi_list(available=self._profiles_multi_available_values()))
        self._on_profile_selected()

    def _profiles_multi_available_values(self) -> list[str]:
        mode = self.ui.var_profiles_multi_mode.get().strip() or "separate"
        if mode == "mixed":
            all_profiles = set(self.logic.get_profile_paths("opencv_2d")) | set(self.logic.get_profile_paths("blender_3d"))
            return sorted(all_profiles)
        return list(self.ui.profile_combo["values"] or [])

    def _profiles_multi_list(self, *, available: list[str] | None = None) -> list[str]:
        available_set = set(available or list(self.ui.profile_combo["values"] or []))
        return parse_profile_ids(self.ui.var_profiles_multi_json.get(), available_set)

    def _set_profiles_multi(self, pids: list[str]) -> None:
        pids = [str(x).strip() for x in (pids or []) if str(x).strip()]
        try: self.ui.var_profiles_multi_json.set(json.dumps(pids, ensure_ascii=True))
        except Exception: pass
        self.ui.var_profiles_multi_summary.set(summarize_profile_ids(pids))

    def _on_profiles_multi_toggle(self) -> None:
        enabled = bool(self.ui.var_profiles_multi.get())
        self.ui.set_profiles_multi_state(enabled)
        if enabled:
            self.ui.var_dataset_mode.set("new")
            cur = self._profiles_multi_list(available=self._profiles_multi_available_values())
            if not cur and (pid := self.ui.var_profile.get().strip()): cur = [pid]
            self._set_profiles_multi(cur)
        try: self._refresh_precise_classes()
        except Exception: pass

    def _pick_profiles_multi(self) -> None:
        values = self._profiles_multi_available_values()
        if not values:
            self.ui.show_messagebox("info", "Profiles", "No profiles available.")
            return
        cur = set(self._profiles_multi_list(available=values))
        sel = dialogs.pick_from_list(
            self.frame, "Select Profiles", values, multi=True, selected=[v for v in values if v in cur],
            prompt="Select one or more component profiles to generate.",
        )
        if sel is None:
            return
        self._set_profiles_multi(sel)
        if sel:
            self.ui.var_profile.set(sel[0])
        try:
            self._refresh_precise_classes()
        except Exception:
            pass

    def _on_render_backend_changed(self, _event: object | None = None) -> None:
        try:
            self._refresh_profiles()
            if pid := self.ui.var_profile.get().strip(): self._autoselect_config_for_profile(pid, prefer_quiet=True)
        except Exception: pass

    def _on_profile_selected(self, _event: object | None = None) -> None:
        if not (profile_id := self.ui.var_profile.get().strip()): return

        try: self._autoselect_config_for_profile(profile_id, prefer_quiet=True)
        except Exception: pass

        # Do not open profile-switch/create popups automatically.
        # Keep profile selection passive and only update state.
        self._last_profile_id = profile_id

    def _select_dataset(self, ds: Path) -> None:
        self._refresh_datasets()
        runs, versions = self.sim_root / "outputs" / "sim_data" / "runs", self.sim_root / "outputs" / "sim_data" / "versions"
        label = self._display_for_dataset(ds, runs=runs, versions=versions)
        for key, value in self._dataset_by_label.items():
            if value == ds or key == label: self.ui.var_dataset.set(key); self._on_dataset_selected(); return
        self.state.dataset_dir = ds
        self.ui.var_dataset.set(label)
        self._on_dataset_selected()

    def _autoselect_config_for_profile(self, profile_id: str, *, prefer_quiet: bool) -> None:
        want_backend = (self.ui.var_render_backend.get().strip() or None)
        cfg_info = self.logic.config_for_profile(profile_id, want_backend=want_backend)
        if not cfg_info and want_backend: cfg_info = self.logic.config_for_profile(profile_id, want_backend=None)
        if not cfg_info: return

        cfg_path: Path = cfg_info["path"]
        try: rel = str(cfg_path.resolve().relative_to(self.sim_root.resolve()))
        except Exception: rel = str(cfg_path)

        if (cur := self.ui.var_config.get().strip()) == rel: return

        if prefer_quiet and cur and want_backend:
            try:
                cur_path = (self.sim_root / cur) if not Path(cur).is_absolute() else Path(cur)
                if cur_path.exists():
                    data = yaml.safe_load(cur_path.read_text(encoding="utf-8"))
                    cur_backend = ((data.get("render") or {}).get("backend") or "").strip()
                    cur_profile = ((data.get("run") or {}).get("component_profile") or "")
                    if cur_backend == want_backend and str(cur_profile).strip() == str(profile_id).strip(): return
            except Exception: return

        self.ui.var_config.set(rel)
        try:
            self._refresh_precise_classes()
        except Exception:
            pass

    def _show_profile_info(self) -> None:
        profile_id = self.ui.var_profile.get()
        if not profile_id: self.ui.show_messagebox("info", "Profile Info", "No profile selected"); return

        profiles_dir = self.sim_root / "configs" / "profiles"
        profile_path = profiles_dir / f"{profile_id}.yaml"
        if not profile_path.exists(): self.ui.show_messagebox("error", "Error", f"Profile file not found:\n{profile_path}"); return

        try:
            with open(profile_path) as f: profile_data = yaml.safe_load(f)
            info_text = f"Profile ID: {profile_id}\nPath: {profile_path}\n\n" + "="*60 + "\n\n" + yaml.dump(profile_data, default_flow_style=False, sort_keys=False)
            dialogs.show_text(self.frame, f"Profile Info: {profile_id}", info_text, size=(640, 520))

        except Exception as e: self.ui.show_messagebox("error", "Error", f"Failed to load profile:\n{str(e)}")

    def _schedule_profile_compat_check(self, delay_ms: int = 300) -> None:
        # Debounce: var_model is written repeatedly while typing/refreshing; check only after edits settle.
        if self._compat_after_id is not None:
            try: self.frame.after_cancel(self._compat_after_id)
            except Exception: pass
        self._compat_after_id = self.frame.after(delay_ms, self._run_scheduled_profile_compat_check)

    def _run_scheduled_profile_compat_check(self) -> None:
        self._compat_after_id = None
        self._check_profile_compatibility()

    def _check_profile_compatibility(self) -> None:
        try:
            if not (ds := self._selected_dataset_dir()): self.ui.var_profile_compat.set(""); return
            manifest_path = ds / "dataset_manifest.json"
            if not manifest_path.exists(): self.ui.var_profile_compat.set(""); return

            with open(manifest_path) as f: manifest = json.load(f)
            mver = int(manifest.get('manifest_version', 1) or 1)
            ds_profile_id = (manifest.get('component_profile', {}).get('profile_id') if mver == 1 else "multi")
            ds_profile_hash = (manifest.get('component_profile', {}).get('profile_hash') if mver == 1 else "multi")

            if not ds_profile_id: self.ui.var_profile_compat.set(""); return

            if not (model_path_str := self.ui.var_model.get().strip()): self.ui.var_profile_compat.set(""); return
            model_path = self.logic._resolve_model_path(model_path_str)
            if not model_path.exists(): self.ui.var_profile_compat.set(""); return
            
            if model_path.is_dir():
                if not (cand := bundle_checkpoint_path(model_path, ds_profile_id, kind="best")).exists():
                    self.ui.var_profile_compat.set(f"⚠ Multi-model: no weights for {ds_profile_id}"); self.ui.set_compat_color("orange"); return
                model_path = cand

            if (meta := self.logic.cached_checkpoint_meta(model_path)) is None:
                # Uncached: load metadata in the background, then re-run this check on the Tk thread.
                self.ui.var_profile_compat.set("… checking model profile"); self.ui.set_compat_color("gray")
                if (key := str(model_path.resolve())) not in self._compat_loading:
                    self._compat_loading.add(key)
                    def worker() -> None:
                        try: self.logic.read_checkpoint_meta(model_path)
                        finally:
                            def done() -> None: self._compat_loading.discard(key); self._schedule_profile_compat_check(0)
                            try: self.frame.after(0, done)
                            except Exception: pass
                    threading.Thread(target=worker, daemon=True).start()
                return
            if meta.get("error"): self.ui.var_profile_compat.set(""); return
            text, color = profile_compat_status(ds_profile_id, ds_profile_hash, meta.get("component_profile"))
            self.ui.var_profile_compat.set(text); self.ui.set_compat_color(color)

        except Exception as e: _LOG.warning("Profile compatibility check failed: %s", e); self.ui.var_profile_compat.set("")

    def _show_dataset_profile_info(self) -> None:
        if not (ds := self._selected_dataset_dir()): self.ui.show_messagebox("info", "Dataset Profile", "No dataset selected"); return
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists(): self.ui.show_messagebox("error", "No Manifest", f"Dataset has no manifest (legacy dataset).\n\nUse the backfill tool to add a manifest:\n  .venv/bin/python tools/backfill_manifest.py --data {ds}"); return

        try:
            with open(manifest_path) as f: manifest = json.load(f)
            profile_id, profile_hash, profile_path_str = manifest.get('component_profile', {}).get('profile_id', 'unknown'), manifest.get('component_profile', {}).get('profile_hash', ''), manifest.get('component_profile', {}).get('profile_path', '')
            profiles_dir = self.sim_root / "configs" / "profiles"
            profile_path = profiles_dir / f"{profile_id}.yaml"

            info_text = f"Dataset: {ds.name}\nProfile ID: {profile_id}\nProfile Hash: {profile_hash}\nProfile Path: {profile_path_str}\n\n" + "="*60 + "\nMANIFEST METADATA\n" + "="*60 + "\n\n" + json.dumps(manifest, indent=2)
            if profile_path.exists(): info_text += "\n\n" + "="*60 + "\nPROFILE DETAILS\n" + "="*60 + "\n\n" + yaml.dump(yaml.safe_load(profile_path.read_text(encoding="utf-8")), default_flow_style=False, sort_keys=False)
            else: info_text += f"\n\n⚠ Profile file not found at: {profile_path}"
            dialogs.show_text(self.frame, f"Dataset Profile: {ds.name}", info_text, size=(680, 560))

        except Exception as e: self.ui.show_messagebox("error", "Error", f"Failed to load dataset profile info:\n{str(e)}")

    def _apply_name_to_out_and_model(self) -> None:
        base = self.logic.slugify_name(self.ui.var_name.get())
        name = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}" if self.ui.var_name_ts.get() else base
        out_rel, model_rel = f"outputs/sim_data/runs/{name}", f"outputs/models/{name}.{'bundle' if self.ui.var_model_bundle.get() else 'pt'}"
        self.ui.var_out.set(out_rel); self.ui.var_model.set(model_rel)
        self._refresh_models()
        self._append_log(f"[naming] out={out_rel} model={model_rel}\n")

    def _selected_dataset_dir(self) -> Path | None:
        dss = self._selected_dataset_dirs(); return dss[0] if dss else None

    def _selected_dataset_dir_single(self) -> Path | None:
        label = self.ui.var_dataset.get().strip(); return self._dataset_by_label.get(label)

    def _selected_dataset_dirs(self) -> list[Path]:
        if self.ui.var_dataset_multi.get():
            paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
            return [p for p in paths if p.exists() and p.is_dir() and not self._dataset_catalog.is_archived(p)]
        return [ds] if (ds := self._selected_dataset_dir_single()) is not None else []

    def refresh(self) -> None:
        if not self.initialized:
            return
        self._refresh_datasets()

    def _on_dataset_selected(self, _evt: object | None = None) -> None:
        ds = self._selected_dataset_dir()
        if not ds: self.ui.var_dataset_profile.set("Profile: -"); self.ui.var_ds_samples.set("Samples: -"); return
        
        self.state.dataset_dir, self._recent_imgs = ds, []
        self._label_dict, self._image_to_sample, self._meta_by_sample, self._meta_by_image_rel = {}, {}, {}, {}
        self._start_dataset_size_calc(ds)

        try:
            if (meta_path := ds / "meta.jsonl").exists():
                meta_rows = read_jsonl(meta_path, MetaRow)
                self._image_to_sample = {row.image_path: row.id for row in meta_rows}
                self._meta_by_sample = {row.id: row for row in meta_rows}
                self._meta_by_image_rel = {row.image_path: row for row in meta_rows}
            if (labels_path := ds / "labels.jsonl").exists():
                self._label_dict = {row.id: row.class_name for row in read_jsonl(labels_path, LabelRow)}
        except Exception as e: _LOG.warning("Failed to load metadata/labels: %s", e)

        try:
            if (manifest_path := ds / "dataset_manifest.json").exists():
                with open(manifest_path) as f: manifest = json.load(f)
                profile_id = manifest.get('component_profile', {}).get('profile_id', 'unknown')
                profile_hash = manifest.get('component_profile', {}).get('profile_hash', '')
                stats = manifest.get("dataset_stats", {}); total_samples = stats.get("total_samples")
                splits = stats.get("splits", {}); train, val, test = splits.get("train"), splits.get("val"), splits.get("test")
                self.ui.var_dataset_profile.set(f"Profile: {profile_id} ({short_profile_hash(profile_hash)}...)")
                self._set_dataset_samples_from_manifest(ds, total_samples, train, val, test)
            else:
                self.ui.var_dataset_profile.set("Profile: ⚠ No manifest (legacy dataset)")
                self._set_dataset_samples_info("Samples: legacy dataset", "black")
        except Exception as e:
            self.ui.var_dataset_profile.set("Profile: ⚠ Error loading manifest"); self._set_dataset_samples_info("Samples: error", "")
            _LOG.warning("Failed to load dataset manifest: %s", e)

        self._schedule_profile_compat_check(); self._apply_dataset_settings(ds)

        if (img_dir := ds / "images").exists():
            for p in sorted(img_dir.glob("*.png"))[-12:]: self._recent_imgs.append(str(p))
            self._refresh_thumbnails()

        try: self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception: pass
        
        try: self._refresh_models()
        except Exception: pass

    def _delete_dataset(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before deleting datasets."); return
        if not dialogs.confirm(self.frame, "Delete Dataset", f"Permanently delete the dataset folder and all its images?\n\n{ds}", danger=True): return
        try: shutil.rmtree(ds); self._append_log(f"\n[deleted dataset {ds}]\n")
        except Exception as e: self.ui.show_messagebox("error", "Delete failed", str(e))
        self._refresh_datasets()

    def _delete_images_from_dataset(self) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            self.ui.show_messagebox("info", "Delete images", "No dataset selected.")
            return
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "Stop the running process before editing datasets.")
            return

        sample_ids = self._open_sample_selector(ds, title=f"Delete images from {ds.name}", action_text="Delete")
        if not sample_ids:
            return

        if not self.ui.ask_yes_no(
            "Delete images",
            f"Delete {len(sample_ids)} sample(s) from dataset '{ds.name}'?",
        ):
            return

        try:
            summary = delete_samples(ds, sample_ids)
            self._append_log(f"[dataset edit] deleted {summary.affected} sample(s) from {ds}\n")
            self._refresh_datasets()
            if label := self._dataset_label_for_path(ds):
                self.ui.var_dataset.set(label)
            self._on_dataset_selected()
            try:
                self.parent.event_generate("<<DatasetChanged>>", when="tail")
            except Exception:
                pass
        except Exception as exc:
            self.ui.show_messagebox("error", "Delete images failed", str(exc))

    def _move_images_between_datasets(self) -> None:
        src_ds = self._selected_dataset_dir()
        if not src_ds:
            self.ui.show_messagebox("info", "Move images", "No source dataset selected.")
            return
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "Stop the running process before editing datasets.")
            return

        target = self._open_target_dataset_selector(src_ds)
        if target is None:
            return

        sample_ids = self._open_sample_selector(src_ds, title=f"Move images from {src_ds.name}", action_text="Move")
        if not sample_ids:
            return

        if not self.ui.ask_yes_no(
            "Move images",
            f"Move {len(sample_ids)} sample(s)\nfrom '{src_ds.name}'\nto '{target.name}'?",
        ):
            return

        try:
            summary = move_samples(src_ds, target, sample_ids, enforce_profile_match=True)
            self._append_log(
                f"[dataset edit] moved {summary.affected} sample(s): {src_ds} -> {target}\n"
            )
            self._refresh_datasets()
            if label := self._dataset_label_for_path(src_ds):
                self.ui.var_dataset.set(label)
            self._on_dataset_selected()
            try:
                self.parent.event_generate("<<DatasetChanged>>", when="tail")
            except Exception:
                pass
        except Exception as exc:
            self.ui.show_messagebox("error", "Move images failed", str(exc))

    def _open_sample_selector(self, dataset_dir: Path, *, title: str, action_text: str) -> list[str]:
        try:
            samples = list_dataset_samples(dataset_dir)
        except Exception as exc:
            self.ui.show_messagebox("error", "Dataset read failed", str(exc))
            return []

        if not samples:
            self.ui.show_messagebox("info", "Dataset is empty", f"No samples found in '{dataset_dir.name}'.")
            return []

        samples_sorted = sorted(samples, key=lambda s: s.sample_id)
        ids = [row.sample_id for row in samples_sorted]
        picked = dialogs.pick_from_list(
            self.frame, title, ids, multi=True, labels=[self._sample_list_label(r) for r in samples_sorted],
            prompt=f"Dataset: {dataset_dir.name}. Select one or more samples.", size=(860, 520), ok_text=action_text,
        )
        return list(picked or [])

    def _open_target_dataset_selector(self, source_dataset: Path) -> Path | None:
        targets = [p for p in self._dataset_dirs if p.resolve() != source_dataset.resolve()]
        if not targets:
            self.ui.show_messagebox("info", "Move Images", "No target dataset available.")
            return None
        targets_sorted = sorted(targets, key=lambda p: self._dataset_label_for_path(p) or p.name)
        keys = [str(p) for p in targets_sorted]
        picked = dialogs.pick_from_list(
            self.frame, "Select Target Dataset", keys,
            labels=[self._dataset_label_for_path(p) or p.name for p in targets_sorted],
            prompt=f"Source dataset: {source_dataset.name}. Choose the dataset the images move to.",
            size=(620, 420), ok_text="Use Target",
        )
        if not picked:
            return None
        return targets_sorted[keys.index(picked[0])]

    def _sample_list_label(self, sample: DatasetSampleInfo) -> str:
        return f"{sample.sample_id} | class={sample.class_name} | split={sample.split} | image={Path(sample.image_path).name}"

    def _create_new_dataset(self) -> None:
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before creating datasets."); return

        initial = self.logic.slugify_name(self.ui.var_name.get())
        if not (new_name := self.ui.ask_string("Create dataset", "New dataset folder name (will be created under outputs/sim_data/runs):", initialvalue=initial)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name: self.ui.show_messagebox("error", "Error", "Name must not contain path separators."); return

        out_dir = (self.sim_root / "outputs" / "sim_data" / "runs" / new_name).resolve()
        if out_dir.exists():
            if self.ui.ask_yes_no("Dataset exists", f"Dataset already exists:\n\n{out_dir}\n\nSelect it?"): self._select_dataset(out_dir); return

        if not (config_candidate := Path(self.ui.var_config.get())).exists(): self.ui.show_messagebox("error", "Missing config", f"Config not found:\n\n{config_candidate}\n\nPick a valid Config first, then retry."); return
        if not (profile_id := self.ui.var_profile.get().strip()): self.ui.show_messagebox("error", "Missing profile", "No profile selected."); return

        try:
            if self.logic.ensure_dataset_skeleton(profile_id, config_candidate, out_dir, self._append_log):
                self.ui.var_dataset_mode.set("new"); self.ui.var_out.set(str(out_dir)); self.ui.var_name.set(new_name)
                model_path = (self.sim_root / "outputs" / "models" / f"{new_name}.pt").resolve()
                self.ui.var_model.set(str(model_path)); self._refresh_models()
                self._refresh_datasets(); self._select_dataset(out_dir)
                self._append_log(f"[create dataset] {out_dir}\n")
        except Exception as e: self.ui.show_messagebox("error", "Create failed", str(e))

    def _snapshot_dataset_selected(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before snapshotting datasets."); return

        versions_root = self.sim_root / "outputs" / "sim_data" / "versions"
        versions_root.mkdir(parents=True, exist_ok=True)
        group, tag = ds.name, time.strftime("%Y%m%d_%H%M%S")
        dst = versions_root / group / f"{group}_{tag}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): self.ui.show_messagebox("error", "Error", f"Snapshot destination already exists:\n{dst}"); return
        if not self.ui.ask_yes_no("Snapshot dataset", f"Create snapshot?\n\nFrom:\n{ds}\n\nTo:\n{dst}"): return
        self._append_log(f"\n[snapshot dataset] {ds} -> {dst}\n")

        def link_or_copy_file(src: Path, dstp: Path) -> None:
            dstp.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(src, dstp)
            except Exception: shutil.copy2(src, dstp)

        def worker() -> None:
            try:
                for dirpath, _dirnames, filenames in os.walk(ds):
                    rel = Path(dirpath).relative_to(ds)
                    for fn in filenames: link_or_copy_file(Path(dirpath) / fn, dst / rel / fn)
                self.log_q.put(f"[snapshot dataset] done: {dst}\n")
            except Exception as e: self.log_q.put(f"[snapshot dataset] failed: {e}\n"); shutil.rmtree(dst, ignore_errors=True)
            finally: self.frame.after(0, self._refresh_datasets)
        threading.Thread(target=worker, daemon=True).start()

    def _rename_dataset_selected(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before renaming datasets."); return

        sim_data = (self.sim_root / "outputs" / "sim_data").resolve()
        runs = (sim_data / "runs").resolve()
        versions = (sim_data / "versions").resolve()
        rp = ds.resolve()
        kind, old_name, parent = "", rp.name, rp.parent
        if str(rp).startswith(str(runs) + os.sep): kind = "run"
        elif str(rp).startswith(str(versions) + os.sep): kind = "snapshot"
        else: self.ui.show_messagebox("error", "Error", f"Can only rename datasets under:\n{runs}\n{versions}\n\nSelected:\n{ds}"); return

        if not (new_name := self.ui.ask_string("Rename dataset", f"New folder name for this {kind} dataset:", initialvalue=old_name)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name: self.ui.show_messagebox("error", "Error", "Name must not contain path separators."); return
        
        dst = parent / new_name
        if dst.exists(): self.ui.show_messagebox("error", "Error", f"Target already exists:\n{dst}"); return
        if not self.ui.ask_yes_no("Rename dataset", f"Rename?\n\nFrom:\n{rp}\n\nTo:\n{dst}"): return

        try: rp.rename(dst)
        except Exception as e: self.ui.show_messagebox("error", "Error", f"Rename failed:\n{e}"); return

        self.state.dataset_dir = dst; self._refresh_datasets()
        try:
            label = self._display_for_dataset(dst, runs=runs, versions=versions)
            for k, v in self._dataset_by_label.items():
                if v == dst or k == label: self.ui.var_dataset.set(k); break
        except Exception: pass
        self._on_dataset_selected()

        cur_model_str = self.ui.var_model.get().strip(); cur_resolved_model_path = str(self.logic._resolve_model_path(cur_model_str))
        old_default_model_path = str((self.sim_root / "outputs" / "models" / f"{old_name}.pt").resolve())
        if cur_resolved_model_path == old_default_model_path: self.ui.var_model.set(str(self.sim_root / "outputs" / "models" / f"{new_name}.pt"))
        
        cur_out_str = self.ui.var_out.get().strip()
        try:
            cur_resolved_out_path = str(self.logic._resolve_out_dir(cur_out_str))
            old_out_path = str((self.sim_root / "outputs" / "sim_data" / "runs" / old_name).resolve())
            if cur_resolved_out_path == old_out_path: self.ui.var_out.set(str(self.sim_root / "outputs" / "sim_data" / "runs" / new_name))
        except Exception: pass

        self._rename_models_for_dataset(old_name, new_name)
        self._append_log(f"[rename dataset] {rp} -> {dst}\n")
        try: self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception: pass

    def _rename_models_for_dataset(self, old_name: str, new_name: str) -> None:
        models_root = self.sim_root / "outputs" / "models"
        for suffix in ["", "_last"]:
            old, new = models_root / f"{old_name}{suffix}.pt", models_root / f"{new_name}{suffix}.pt"
            if old.exists():
                try: old.rename(new)
                except Exception as e: _LOG.warning("Failed to rename model %s -> %s: %s", old, new, e)
            old_meta, new_meta = models_root / f"{old_name}{suffix}.pt.meta.json", models_root / f"{new_name}{suffix}.pt.meta.json"
            if old_meta.exists():
                try: old_meta.rename(new_meta)
                except Exception as e: _LOG.warning("Failed to rename meta %s -> %s: %s", old_meta, new_meta, e)

        versions_root = models_root / "versions"
        old_dir, new_dir = versions_root / old_name, versions_root / new_name
        if old_dir.exists():
            try: old_dir.rename(new_dir)
            except Exception as e: _LOG.warning("Failed to rename versioned models %s -> %s: %s", old_dir, new_dir, e)

    def _validate_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        if len(dss) == 1: self._run_simple_cmd(validate_argv(dss[0])); return

        self._append_log(f"\n=== Validate {len(dss)} datasets ===\n")
        self._run_simple_cmd([chain_commands([validate_argv(ds) for ds in dss])])

    def _train_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        if len(dss) > 1:
            if not (strat := self._ask_multi_dataset_train_strategy(dss)): return
            if strat in ("merge_keep", "merge_temp"):
                if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first."); return
                
                ts = time.strftime("%Y%m%d_%H%M%S")
                name = ""
                if strat == "merge_keep":
                    default = f"{dss[0].name}_merged_{len(dss)}"
                    if not (name := self.ui.ask_string("Merge datasets", "Output dataset folder name (will be created under outputs/sim_data/runs):", initialvalue=default)): return
                    if not (name := name.strip()) or "/" in name or "\\" in name: self.ui.show_messagebox("error", "Error", "Invalid dataset name."); return
                else: name = f"__tmp_merge_{ts}"
                
                out_ds = (self.sim_root / "outputs" / "sim_data" / "runs" / name).resolve()
                if out_ds.exists(): self.ui.show_messagebox("error", "Error", f"Merge output already exists:\n{out_ds}"); return

                model_s = self.ui.var_model.get().strip()
                model_out = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{name}.pt").resolve()
                delete_after = (strat == "merge_temp")

                self._append_log(f"\n=== Merge {len(dss)} datasets -> {out_ds.name} ===\n")
                for ds in dss: self._append_log(f"  - {ds}\n")
                self._append_log("\n")

                self.logic.loop_running = True; self.logic.stop_evt.clear()  # keep Busy guards closed until training starts
                def worker() -> None:
                    try:
                        self.log_q.put(f"[merge] building merged dataset: {out_ds}\n")
                        self.logic.merge_datasets_for_training(dss, out_ds, self.log_q.put, self.log_q.put)
                        self.log_q.put(f"[merge] done: {out_ds}\n")

                        def kick_off_train() -> None:
                            self.logic.loop_running = False
                            if self.logic.stop_evt.is_set():
                                self._append_log("[merge] stopped by user; training not started\n")
                                if delete_after: shutil.rmtree(out_ds, ignore_errors=True)
                                self._refresh_datasets(); return
                            try:
                                if strat == "merge_keep": self._select_dataset(out_ds)
                                else: self._refresh_datasets()
                            except Exception: pass

                            cmd = chain_commands([train_argv(out_ds, model_out)])
                            self._run_simple_cmd([cmd], cleanup_dir=out_ds if delete_after else None)

                        self.frame.after(0, kick_off_train)
                    except Exception as e:
                        self.log_q.put(f"[merge] failed: {e}\n")
                        if delete_after: shutil.rmtree(out_ds, ignore_errors=True)
                        def _merge_failed_ui() -> None: self.logic.loop_running = False
                        self.frame.after(0, _merge_failed_ui)
                self.ui.set_run_buttons_state(True)
                try: threading.Thread(target=worker, daemon=True).start()
                except Exception: self.logic.loop_running = False; raise
                return

        model_s = self.ui.var_model.get().strip()
        model_out = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()

        if len(dss) == 1: self._run_simple_cmd(train_argv(dss[0], model_out)); return

        allow_mixed_profiles = False
        if self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_out):
            if not self.ui.ask_yes_no("Allow mixed-profile training into a single model?", "You selected datasets with different component profiles.\n\nRecommended: train into a multi-model bundle (.bundle) so each profile gets its own checkpoint.\n\nIf you continue anyway, the training will resume across different profiles by passing --allow-profile-mismatch (warm-start). This is not guaranteed to work and can degrade accuracy.\n\nContinue?"): return
            allow_mixed_profiles = True

        extra_s = self.ui.var_continue_epochs.get().strip() or "10"
        try: extra = int(extra_s)
        except Exception: self.ui.show_messagebox("error", "Error", "+epochs must be an integer."); return
        if extra < 0: self.ui.show_messagebox("error", "Error", "+epochs must be >= 0."); return

        out_mode = self.ui.var_continue_out_mode.get().strip() or "best"
        if out_mode not in ("last", "best"): out_mode = "best"

        argvs = multi_train_argvs(
            dss, model_out, [self.logic.get_dataset_profile_id(ds) for ds in dss],
            is_bundle=self.logic._is_model_bundle_target(model_out), extra_epochs=extra, out_mode=out_mode,
            allow_profile_mismatch=allow_mixed_profiles,
        )

        self._append_log(f"\n=== Multi-dataset train: {len(dss)} datasets ===\n")
        for ds in dss: self._append_log(f"  - {ds}\n")
        self._append_log(f"out: {model_out}\n+epochs per subsequent dataset: {extra} (out-mode={out_mode})\n\n")
        self._run_simple_cmd([chain_commands(argvs)])

    def _eval_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        
        model_s = self.ui.var_model.get().strip()
        model_in = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_in.exists(): self.ui.show_messagebox("error", "Missing model", f"Model not found:\n{model_in}\n\nRun Train first."); return
        if len(dss) == 1: self._run_simple_cmd(eval_argv(dss[0], model_in)); return

        allow_mixed_profiles = False
        if self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_in):
            if not self.ui.ask_yes_no("Allow mixed-profile eval on a single model?", "You selected datasets with different component profiles.\n\nRecommended: evaluate a multi-model bundle (.bundle) so each dataset uses its matching checkpoint.\n\nIf you continue anyway, eval will pass --allow-profile-mismatch. Results may be meaningless.\n\nContinue?"): return
            allow_mixed_profiles = True

        self._append_log(f"\n=== Eval on {len(dss)} datasets ===\n")
        self._run_simple_cmd([chain_commands([eval_argv(ds, model_in, allow_profile_mismatch=allow_mixed_profiles) for ds in dss])])

    def _continue_train_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return

        model_s = self.ui.var_model.get().strip()
        model_path = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_path.exists(): self.ui.show_messagebox("error", "Missing model", f"Model not found to resume from:\n{model_path}\n\nRun Train first."); return

        extra_s = self.ui.var_continue_epochs.get().strip() or "10"
        try: extra = int(extra_s)
        except Exception: self.ui.show_messagebox("error", "Error", "Extra epochs must be an integer."); return
        if extra < 1: self.ui.show_messagebox("error", "Error", "Extra epochs must be >= 1."); return

        out_mode = self.ui.var_continue_out_mode.get().strip() or "last"
        if out_mode not in ("last", "best"): out_mode = "last"

        allow_mixed_profiles = False
        if len(dss) > 1 and self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_path):
            if not self.ui.ask_yes_no("Allow mixed-profile resume into a single model?", "You selected datasets with different component profiles.\n\nRecommended: resume into a multi-model bundle (.bundle).\n\nIf you continue anyway, the resume will pass --allow-profile-mismatch. This is not guaranteed to work.\n\nContinue?"): return
            allow_mixed_profiles = True

        if len(dss) > 1 and self.logic._is_model_bundle_target(model_path):
            missing: list[str] = []
            for ds in dss:
                if not (pid := self.logic.get_dataset_profile_id(ds)): missing.append(f"{ds} (missing profile metadata)"); continue
                try:
                    if not bundle_checkpoint_path(model_path, pid, kind="best").exists(): missing.append(f"{pid} (no checkpoint in bundle)")
                except Exception: missing.append(f"{pid} (bundle lookup failed)")
            if missing: self.ui.show_messagebox("error", "Missing bundle checkpoints", "The selected model bundle does not have checkpoints for all selected datasets.\n\n" + "\n".join(missing)); return

        if len(dss) == 1:
            ds = dss[0]
            self._append_log(f"\n=== Continue training: +{extra} epochs (out={out_mode}) ===\n" f"data:  {ds}\n" f"model: {model_path}\n\n")
            self._run_simple_cmd(continue_train_argv(ds, model_path, extra_epochs=extra, out_mode=out_mode))
            return

        argvs = [
            continue_train_argv(ds, model_path, extra_epochs=extra, out_mode=out_mode, allow_profile_mismatch=allow_mixed_profiles)
            for ds in dss
        ]

        self._append_log(f"\n=== Continue training on {len(dss)} datasets: +{extra} epochs each (out={out_mode}) ===\n" f"model: {model_path}\n")
        for ds in dss: self._append_log(f"data:  {ds}\n")
        self._append_log("\n")

        self._run_simple_cmd([chain_commands(argvs)])

"""Pipeline Control Tab - Enhanced version of the original monitor."""

from __future__ import annotations
import json
import os
import queue
import threading
import time
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import shutil
from datetime import datetime
import re
import shlex
import tempfile

import torch
from PIL import Image, ImageTk
import cv2
import yaml

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.tooltip import ToolTip
from gui.components.overlay_renderer import draw_defect_overlay
from gui.components.filter_popup import open_filter_popup
from gui.utils.settings_store import SettingsStore

from simple_sim.schema import read_jsonl, LabelRow, MetaRow
from simple_sim.model_bundle import bundle_checkpoint_path
from simple_sim.schema import write_jsonl
from simple_sim.manifest import read_dataset_manifest, write_dataset_manifest, write_multi_profile_manifest

from .ui import PipelineUI
from .logic import PipelineLogic


class PipelineControlTab(BaseTab):
    """Tab 1: Pipeline control with enhanced monitoring."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)

        self.log_q: queue.Queue[str] = queue.Queue()
        self.event_q: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.event_log_path = self.sim_root / "outputs" / "live" / "events.jsonl"

        self.ui = PipelineUI(self, self.frame)
        self.logic = PipelineLogic(sim_root, self.log_q, self.event_q, self.event_log_path)

        self._thumb_refs = []  # keep PhotoImage references
        self._recent_imgs: list[str] = []
        self._dataset_dirs: list[Path] = []
        self._dataset_labels: list[str] = []
        self._dataset_by_label: dict[str, Path] = {}
        self._label_dict: dict[str, str] = {}  # Map sample_id to class label
        self._image_to_sample: dict[str, str] = {}  # Map image_path to sample_id
        self._meta_by_sample: dict[str, MetaRow] = {}
        self._meta_by_image_rel: dict[str, MetaRow] = {}
        
        self._last_profile_id: str = "chip_0603_resistor@1"
        self._suspend_profile_event: bool = False

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
        self._model_profile_cache: dict[str, tuple[Optional[str], Optional[str]]] = {}

        self._profile_paths: list[Path] = []
        self._dataset_milestones: dict[str, int] = {}
        
    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self._refresh_datasets()
        self._refresh_models()
        self._refresh_profile_models()
        self._refresh_profiles()
        self._load_persisted_settings()
        self._wire_settings_autosave()
        self._apply_profile_model_lock()
        self.ui.var_model.trace("w", lambda *args: self._check_profile_compatibility())
        self._tick_ui()

    def _store(self) -> Optional[SettingsStore]:
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
        image_filters: Dict[str, Any],
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

        msg = (
            f"Task: {task_label}\n"
            f"Run mode: {run_mode} ({run_label})\n"
            f"Dataset mode: {dataset_mode}\n"
            f"Output: {out_dir}\n"
            f"Profiles: {profile_label}\n"
            f"Multi-profile: {'on' if multi_enabled else 'off'} ({multi_mode})\n\n"
            f"90° rotation: {'on' if bool(image_filters.get('cardinal_rotation_90', True)) else 'off'}\n"
            f"Filter button: {'on' if bool(image_filters.get('enable', True)) else 'off'}\n\n"
            "Start now?"
        )
        return self.ui.ask_yes_no("Start pipeline", msg)

    def _ask_multi_dataset_train_strategy(self, dss: list[Path]) -> Optional[str]:
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

        for key, var in [
            ("pipeline.run_mode", self.ui.var_run_mode), ("pipeline.run_count", self.ui.var_run_count),
            ("pipeline.dataset_mode", self.ui.var_dataset_mode), ("pipeline.config", self.ui.var_config),
            ("pipeline.out", self.ui.var_out), ("pipeline.model", self.ui.var_model),
            ("pipeline.autosnap", self.ui.var_autosnap), ("pipeline.snap_every", self.ui.var_snap_every),
            ("pipeline.snap_keep", self.ui.var_snap_keep), ("pipeline.continue_epochs", self.ui.var_continue_epochs),
            ("pipeline.continue_out_mode", self.ui.var_continue_out_mode), ("pipeline.name", self.ui.var_name),
            ("pipeline.name_ts", self.ui.var_name_ts), ("pipeline.dataset_selection", self.ui.var_dataset),
            ("pipeline.dataset_multi_enabled", self.ui.var_dataset_multi), ("pipeline.dataset_multi_paths", self.ui.var_dataset_multi_paths),
            ("pipeline.profile_model", self.ui.var_profile_model), ("pipeline.profile_model_locked", self.ui.var_profile_model_lock),
            ("pipeline.profile_build_preset", self.ui.var_profile_build_preset), ("pipeline.render_backend", self.ui.var_render_backend),
            ("pipeline.filter.cardinal_rotation_90", self.ui.var_filter_cardinal_rotation_90),
            ("pipeline.filter.enable_rotation", self.ui.var_filter_enable_rotation),
            ("pipeline.filter.enable_blur", self.ui.var_filter_enable_blur),
            ("pipeline.filter.enable_grain", self.ui.var_filter_enable_grain),
            ("pipeline.filter.enable_brightness", self.ui.var_filter_enable_brightness),
            ("pipeline.filter.enable_contrast", self.ui.var_filter_enable_contrast),
            ("pipeline.filter.rotation_strength", self.ui.var_filter_rotation_strength),
            ("pipeline.filter.blur_strength", self.ui.var_filter_blur_strength),
            ("pipeline.filter.grain_strength", self.ui.var_filter_grain_strength),
            ("pipeline.filter.brightness_strength", self.ui.var_filter_brightness_strength),
            ("pipeline.filter.contrast_strength", self.ui.var_filter_contrast_strength),
            ("pipeline.filter.enable_perspective", self.ui.var_filter_enable_perspective),
            ("pipeline.filter.perspective_strength", self.ui.var_filter_perspective_strength),
            ("pipeline.filter.enable_motion_blur", self.ui.var_filter_enable_motion_blur),
            ("pipeline.filter.motion_blur_strength", self.ui.var_filter_motion_blur_strength),
            ("pipeline.filter.enable_saturation", self.ui.var_filter_enable_saturation),
            ("pipeline.filter.saturation_factor", self.ui.var_filter_saturation_factor),
            ("pipeline.filter.enable_hue_shift", self.ui.var_filter_enable_hue_shift),
            ("pipeline.filter.hue_shift_deg", self.ui.var_filter_hue_shift_deg),
            ("pipeline.filter.enable_shadow", self.ui.var_filter_enable_shadow),
            ("pipeline.filter.shadow_strength", self.ui.var_filter_shadow_strength),
            ("pipeline.filter.enable_reflection", self.ui.var_filter_enable_reflection),
            ("pipeline.filter.reflection_strength", self.ui.var_filter_reflection_strength),
            ("pipeline.filter.enable_vignetting", self.ui.var_filter_enable_vignetting),
            ("pipeline.filter.vignetting_strength", self.ui.var_filter_vignetting_strength),
            ("pipeline.filter.enable_chromatic_aberration", self.ui.var_filter_enable_chromatic_aberration),
            ("pipeline.filter.chromatic_strength", self.ui.var_filter_chromatic_strength),
            ("pipeline.filter.enable_jpeg_compression", self.ui.var_filter_enable_jpeg_compression),
            ("pipeline.filter.jpeg_quality", self.ui.var_filter_jpeg_quality),
            ("pipeline.filter.enable_color_temperature", self.ui.var_filter_enable_color_temperature),
            ("pipeline.filter.color_temperature_kelvin", self.ui.var_filter_color_temperature_kelvin),
            ("pipeline.filter.enable_lens_distortion", self.ui.var_filter_enable_lens_distortion),
            ("pipeline.filter.distortion_k1", self.ui.var_filter_distortion_k1),
            ("pipeline.filter.enable_dust", self.ui.var_filter_enable_dust),
            ("pipeline.filter.dust_density", self.ui.var_filter_dust_density),
            ("pipeline.filter.enable_sharpen", self.ui.var_filter_enable_sharpen),
            ("pipeline.filter.sharpen_strength", self.ui.var_filter_sharpen_strength),
        ]:
            v = st.get(key, None)
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            try:
                var.set(v)
            except Exception:
                pass

        try: self._refresh_datasets(); self._on_dataset_multi_toggle(); self._on_dataset_selected()
        except Exception: pass
        try: self._refresh_models()
        except Exception: pass

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None: return

        for var, key in [
            (self.ui.var_run_mode, "pipeline.run_mode"), (self.ui.var_run_count, "pipeline.run_count"),
            (self.ui.var_dataset_mode, "pipeline.dataset_mode"), (self.ui.var_config, "pipeline.config"),
            (self.ui.var_out, "pipeline.out"), (self.ui.var_model, "pipeline.model"),
            (self.ui.var_autosnap, "pipeline.autosnap"), (self.ui.var_snap_every, "pipeline.snap_every"),
            (self.ui.var_snap_keep, "pipeline.snap_keep"), (self.ui.var_continue_epochs, "pipeline.continue_epochs"),
            (self.ui.var_continue_out_mode, "pipeline.continue_out_mode"), (self.ui.var_name, "pipeline.name"),
            (self.ui.var_name_ts, "pipeline.name_ts"), (self.ui.var_dataset, "pipeline.dataset_selection"),
            (self.ui.var_dataset_multi, "pipeline.dataset_multi_enabled"), (self.ui.var_dataset_multi_paths, "pipeline.dataset_multi_paths"),
            (self.ui.var_profile_model, "pipeline.profile_model"), (self.ui.var_profile_model_lock, "pipeline.profile_model_locked"),
            (self.ui.var_profile_build_preset, "pipeline.profile_build_preset"), (self.ui.var_render_backend, "pipeline.render_backend"),
            (self.ui.var_filter_cardinal_rotation_90, "pipeline.filter.cardinal_rotation_90"),
            (self.ui.var_filter_enable_rotation, "pipeline.filter.enable_rotation"),
            (self.ui.var_filter_enable_blur, "pipeline.filter.enable_blur"),
            (self.ui.var_filter_enable_grain, "pipeline.filter.enable_grain"),
            (self.ui.var_filter_enable_brightness, "pipeline.filter.enable_brightness"),
            (self.ui.var_filter_enable_contrast, "pipeline.filter.enable_contrast"),
            (self.ui.var_filter_rotation_strength, "pipeline.filter.rotation_strength"),
            (self.ui.var_filter_blur_strength, "pipeline.filter.blur_strength"),
            (self.ui.var_filter_grain_strength, "pipeline.filter.grain_strength"),
            (self.ui.var_filter_brightness_strength, "pipeline.filter.brightness_strength"),
            (self.ui.var_filter_contrast_strength, "pipeline.filter.contrast_strength"),
            (self.ui.var_filter_enable_perspective, "pipeline.filter.enable_perspective"),
            (self.ui.var_filter_perspective_strength, "pipeline.filter.perspective_strength"),
            (self.ui.var_filter_enable_motion_blur, "pipeline.filter.enable_motion_blur"),
            (self.ui.var_filter_motion_blur_strength, "pipeline.filter.motion_blur_strength"),
            (self.ui.var_filter_enable_saturation, "pipeline.filter.enable_saturation"),
            (self.ui.var_filter_saturation_factor, "pipeline.filter.saturation_factor"),
            (self.ui.var_filter_enable_hue_shift, "pipeline.filter.enable_hue_shift"),
            (self.ui.var_filter_hue_shift_deg, "pipeline.filter.hue_shift_deg"),
            (self.ui.var_filter_enable_shadow, "pipeline.filter.enable_shadow"),
            (self.ui.var_filter_shadow_strength, "pipeline.filter.shadow_strength"),
            (self.ui.var_filter_enable_reflection, "pipeline.filter.enable_reflection"),
            (self.ui.var_filter_reflection_strength, "pipeline.filter.reflection_strength"),
            (self.ui.var_filter_enable_vignetting, "pipeline.filter.enable_vignetting"),
            (self.ui.var_filter_vignetting_strength, "pipeline.filter.vignetting_strength"),
            (self.ui.var_filter_enable_chromatic_aberration, "pipeline.filter.enable_chromatic_aberration"),
            (self.ui.var_filter_chromatic_strength, "pipeline.filter.chromatic_strength"),
            (self.ui.var_filter_enable_jpeg_compression, "pipeline.filter.enable_jpeg_compression"),
            (self.ui.var_filter_jpeg_quality, "pipeline.filter.jpeg_quality"),
            (self.ui.var_filter_enable_color_temperature, "pipeline.filter.enable_color_temperature"),
            (self.ui.var_filter_color_temperature_kelvin, "pipeline.filter.color_temperature_kelvin"),
            (self.ui.var_filter_enable_lens_distortion, "pipeline.filter.enable_lens_distortion"),
            (self.ui.var_filter_distortion_k1, "pipeline.filter.distortion_k1"),
            (self.ui.var_filter_enable_dust, "pipeline.filter.enable_dust"),
            (self.ui.var_filter_dust_density, "pipeline.filter.dust_density"),
            (self.ui.var_filter_enable_sharpen, "pipeline.filter.enable_sharpen"),
            (self.ui.var_filter_sharpen_strength, "pipeline.filter.sharpen_strength"),
        ]:
            var.trace_add("write", lambda *a, v=var, k=key: (st.set(k, v.get()), st.schedule_save(self.frame)))

    def _float_or_default(self, value: str, *, default: float) -> float:
        try:
            return float(str(value).strip())
        except Exception:
            return float(default)

    def _current_filter_settings_dict(self) -> Dict[str, Any]:
        return {
            # Existing filters
            "cardinal_rotation_90": bool(self.ui.var_filter_cardinal_rotation_90.get()),
            "enable_rotation": bool(self.ui.var_filter_enable_rotation.get()),
            "enable_blur": bool(self.ui.var_filter_enable_blur.get()),
            "enable_grain": bool(self.ui.var_filter_enable_grain.get()),
            "enable_brightness": bool(self.ui.var_filter_enable_brightness.get()),
            "enable_contrast": bool(self.ui.var_filter_enable_contrast.get()),
            "rotation_strength": f"{self._float_or_default(self.ui.var_filter_rotation_strength.get(), default=1.0):.2f}",
            "blur_strength": f"{self._float_or_default(self.ui.var_filter_blur_strength.get(), default=1.0):.2f}",
            "grain_strength": f"{self._float_or_default(self.ui.var_filter_grain_strength.get(), default=1.0):.2f}",
            "brightness_strength": f"{self._float_or_default(self.ui.var_filter_brightness_strength.get(), default=1.0):.2f}",
            "contrast_strength": f"{self._float_or_default(self.ui.var_filter_contrast_strength.get(), default=1.0):.2f}",
            # High priority new filters
            "enable_perspective": bool(self.ui.var_filter_enable_perspective.get()),
            "perspective_strength": f"{self._float_or_default(self.ui.var_filter_perspective_strength.get(), default=1.0):.2f}",
            "enable_motion_blur": bool(self.ui.var_filter_enable_motion_blur.get()),
            "motion_blur_strength": f"{self._float_or_default(self.ui.var_filter_motion_blur_strength.get(), default=1.0):.2f}",
            "enable_saturation": bool(self.ui.var_filter_enable_saturation.get()),
            "saturation_factor": f"{self._float_or_default(self.ui.var_filter_saturation_factor.get(), default=1.0):.2f}",
            "enable_hue_shift": bool(self.ui.var_filter_enable_hue_shift.get()),
            "hue_shift_deg": f"{self._float_or_default(self.ui.var_filter_hue_shift_deg.get(), default=0.0):.2f}",
            "enable_shadow": bool(self.ui.var_filter_enable_shadow.get()),
            "shadow_strength": f"{self._float_or_default(self.ui.var_filter_shadow_strength.get(), default=0.3):.2f}",
            "enable_reflection": bool(self.ui.var_filter_enable_reflection.get()),
            "reflection_strength": f"{self._float_or_default(self.ui.var_filter_reflection_strength.get(), default=0.5):.2f}",
            # Medium/Low priority new filters
            "enable_vignetting": bool(self.ui.var_filter_enable_vignetting.get()),
            "vignetting_strength": f"{self._float_or_default(self.ui.var_filter_vignetting_strength.get(), default=1.0):.2f}",
            "enable_chromatic_aberration": bool(self.ui.var_filter_enable_chromatic_aberration.get()),
            "chromatic_strength": f"{self._float_or_default(self.ui.var_filter_chromatic_strength.get(), default=1.0):.2f}",
            "enable_jpeg_compression": bool(self.ui.var_filter_enable_jpeg_compression.get()),
            "jpeg_quality": f"{int(self._float_or_default(self.ui.var_filter_jpeg_quality.get(), default=85))}",
            "enable_color_temperature": bool(self.ui.var_filter_enable_color_temperature.get()),
            "color_temperature_kelvin": f"{int(self._float_or_default(self.ui.var_filter_color_temperature_kelvin.get(), default=5500))}",
            "enable_lens_distortion": bool(self.ui.var_filter_enable_lens_distortion.get()),
            "distortion_k1": f"{self._float_or_default(self.ui.var_filter_distortion_k1.get(), default=0.0):.2f}",
            "enable_dust": bool(self.ui.var_filter_enable_dust.get()),
            "dust_density": f"{self._float_or_default(self.ui.var_filter_dust_density.get(), default=0.3):.2f}",
            "enable_sharpen": bool(self.ui.var_filter_enable_sharpen.get()),
            "sharpen_strength": f"{self._float_or_default(self.ui.var_filter_sharpen_strength.get(), default=1.0):.2f}",
        }

    def _apply_filter_settings_dict(self, data: Dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        # Existing filters
        self.ui.var_filter_cardinal_rotation_90.set(bool(data.get("cardinal_rotation_90", True)))
        self.ui.var_filter_enable_rotation.set(bool(data.get("enable_rotation", True)))
        self.ui.var_filter_enable_blur.set(bool(data.get("enable_blur", True)))
        self.ui.var_filter_enable_grain.set(bool(data.get("enable_grain", True)))
        self.ui.var_filter_enable_brightness.set(bool(data.get("enable_brightness", True)))
        self.ui.var_filter_enable_contrast.set(bool(data.get("enable_contrast", True)))
        self.ui.var_filter_rotation_strength.set(str(data.get("rotation_strength", "1.00")))
        self.ui.var_filter_blur_strength.set(str(data.get("blur_strength", "1.00")))
        self.ui.var_filter_grain_strength.set(str(data.get("grain_strength", "1.00")))
        self.ui.var_filter_brightness_strength.set(str(data.get("brightness_strength", "1.00")))
        self.ui.var_filter_contrast_strength.set(str(data.get("contrast_strength", "1.00")))
        # High priority new filters (enabled by default)
        self.ui.var_filter_enable_perspective.set(bool(data.get("enable_perspective", True)))
        self.ui.var_filter_perspective_strength.set(str(data.get("perspective_strength", "1.00")))
        self.ui.var_filter_enable_motion_blur.set(bool(data.get("enable_motion_blur", True)))
        self.ui.var_filter_motion_blur_strength.set(str(data.get("motion_blur_strength", "1.00")))
        self.ui.var_filter_enable_saturation.set(bool(data.get("enable_saturation", True)))
        self.ui.var_filter_saturation_factor.set(str(data.get("saturation_factor", "1.00")))
        self.ui.var_filter_enable_hue_shift.set(bool(data.get("enable_hue_shift", True)))
        self.ui.var_filter_hue_shift_deg.set(str(data.get("hue_shift_deg", "0.00")))
        self.ui.var_filter_enable_shadow.set(bool(data.get("enable_shadow", True)))
        self.ui.var_filter_shadow_strength.set(str(data.get("shadow_strength", "0.30")))
        self.ui.var_filter_enable_reflection.set(bool(data.get("enable_reflection", True)))
        self.ui.var_filter_reflection_strength.set(str(data.get("reflection_strength", "0.50")))
        # Medium/Low priority new filters (disabled by default)
        self.ui.var_filter_enable_vignetting.set(bool(data.get("enable_vignetting", False)))
        self.ui.var_filter_vignetting_strength.set(str(data.get("vignetting_strength", "1.00")))
        self.ui.var_filter_enable_chromatic_aberration.set(bool(data.get("enable_chromatic_aberration", False)))
        self.ui.var_filter_chromatic_strength.set(str(data.get("chromatic_strength", "1.00")))
        self.ui.var_filter_enable_jpeg_compression.set(bool(data.get("enable_jpeg_compression", False)))
        self.ui.var_filter_jpeg_quality.set(str(data.get("jpeg_quality", "85")))
        self.ui.var_filter_enable_color_temperature.set(bool(data.get("enable_color_temperature", False)))
        self.ui.var_filter_color_temperature_kelvin.set(str(data.get("color_temperature_kelvin", "5500")))
        self.ui.var_filter_enable_lens_distortion.set(bool(data.get("enable_lens_distortion", False)))
        self.ui.var_filter_distortion_k1.set(str(data.get("distortion_k1", "0.00")))
        self.ui.var_filter_enable_dust.set(bool(data.get("enable_dust", False)))
        self.ui.var_filter_dust_density.set(str(data.get("dust_density", "0.30")))
        self.ui.var_filter_enable_sharpen.set(bool(data.get("enable_sharpen", False)))
        self.ui.var_filter_sharpen_strength.set(str(data.get("sharpen_strength", "1.00")))

    def _load_filter_profiles(self) -> tuple[Dict[str, Dict[str, Any]], str]:
        st = self._store()
        profiles: Dict[str, Dict[str, Any]] = {}
        active = "Default"
        current_live: Dict[str, Any] = {}
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
            current_live = {
                # Existing filters
                "cardinal_rotation_90": bool(st.get("pipeline.filter.cardinal_rotation_90", True)),
                "enable_rotation": bool(st.get("pipeline.filter.enable_rotation", True)),
                "enable_blur": bool(st.get("pipeline.filter.enable_blur", True)),
                "enable_grain": bool(st.get("pipeline.filter.enable_grain", True)),
                "enable_brightness": bool(st.get("pipeline.filter.enable_brightness", True)),
                "enable_contrast": bool(st.get("pipeline.filter.enable_contrast", True)),
                "rotation_strength": str(st.get("pipeline.filter.rotation_strength", "1.00")),
                "blur_strength": str(st.get("pipeline.filter.blur_strength", "1.00")),
                "grain_strength": str(st.get("pipeline.filter.grain_strength", "1.00")),
                "brightness_strength": str(st.get("pipeline.filter.brightness_strength", "1.00")),
                "contrast_strength": str(st.get("pipeline.filter.contrast_strength", "1.00")),
                # High priority new filters
                "enable_perspective": bool(st.get("pipeline.filter.enable_perspective", True)),
                "perspective_strength": str(st.get("pipeline.filter.perspective_strength", "1.00")),
                "enable_motion_blur": bool(st.get("pipeline.filter.enable_motion_blur", True)),
                "motion_blur_strength": str(st.get("pipeline.filter.motion_blur_strength", "1.00")),
                "enable_saturation": bool(st.get("pipeline.filter.enable_saturation", True)),
                "saturation_factor": str(st.get("pipeline.filter.saturation_factor", "1.00")),
                "enable_hue_shift": bool(st.get("pipeline.filter.enable_hue_shift", True)),
                "hue_shift_deg": str(st.get("pipeline.filter.hue_shift_deg", "0.00")),
                "enable_shadow": bool(st.get("pipeline.filter.enable_shadow", True)),
                "shadow_strength": str(st.get("pipeline.filter.shadow_strength", "0.30")),
                "enable_reflection": bool(st.get("pipeline.filter.enable_reflection", True)),
                "reflection_strength": str(st.get("pipeline.filter.reflection_strength", "0.50")),
                # Medium/Low priority new filters
                "enable_vignetting": bool(st.get("pipeline.filter.enable_vignetting", False)),
                "vignetting_strength": str(st.get("pipeline.filter.vignetting_strength", "1.00")),
                "enable_chromatic_aberration": bool(st.get("pipeline.filter.enable_chromatic_aberration", False)),
                "chromatic_strength": str(st.get("pipeline.filter.chromatic_strength", "1.00")),
                "enable_jpeg_compression": bool(st.get("pipeline.filter.enable_jpeg_compression", False)),
                "jpeg_quality": str(st.get("pipeline.filter.jpeg_quality", "85")),
                "enable_color_temperature": bool(st.get("pipeline.filter.enable_color_temperature", False)),
                "color_temperature_kelvin": str(st.get("pipeline.filter.color_temperature_kelvin", "5500")),
                "enable_lens_distortion": bool(st.get("pipeline.filter.enable_lens_distortion", False)),
                "distortion_k1": str(st.get("pipeline.filter.distortion_k1", "0.00")),
                "enable_dust": bool(st.get("pipeline.filter.enable_dust", False)),
                "dust_density": str(st.get("pipeline.filter.dust_density", "0.30")),
                "enable_sharpen": bool(st.get("pipeline.filter.enable_sharpen", False)),
                "sharpen_strength": str(st.get("pipeline.filter.sharpen_strength", "1.00")),
            }
        if not profiles:
            profiles = {"Default": current_live or self._current_filter_settings_dict()}
            active = "Default"
        if active not in profiles:
            active = sorted(profiles.keys())[0]
        return profiles, active

    def _save_filter_profiles(self, profiles: Dict[str, Dict[str, Any]], active: str) -> None:
        st = self._store()
        if st is None:
            return
        st.set("pipeline.filter_profiles_json", json.dumps(profiles, ensure_ascii=True))
        st.set("pipeline.filter_profile_active", str(active))
        st.schedule_save(self.frame)

    def _image_filters_payload(self) -> Dict[str, Any]:
        has_any_filter = any([
            bool(self.ui.var_filter_cardinal_rotation_90.get()),
            bool(self.ui.var_filter_enable_rotation.get()),
            bool(self.ui.var_filter_enable_blur.get()),
            bool(self.ui.var_filter_enable_grain.get()),
            bool(self.ui.var_filter_enable_brightness.get()),
            bool(self.ui.var_filter_enable_contrast.get()),
            bool(self.ui.var_filter_enable_perspective.get()),
            bool(self.ui.var_filter_enable_motion_blur.get()),
            bool(self.ui.var_filter_enable_saturation.get()),
            bool(self.ui.var_filter_enable_hue_shift.get()),
            bool(self.ui.var_filter_enable_shadow.get()),
            bool(self.ui.var_filter_enable_reflection.get()),
            bool(self.ui.var_filter_enable_vignetting.get()),
            bool(self.ui.var_filter_enable_chromatic_aberration.get()),
            bool(self.ui.var_filter_enable_jpeg_compression.get()),
            bool(self.ui.var_filter_enable_color_temperature.get()),
            bool(self.ui.var_filter_enable_lens_distortion.get()),
            bool(self.ui.var_filter_enable_dust.get()),
            bool(self.ui.var_filter_enable_sharpen.get()),
        ])
        return {
            "enable": has_any_filter,
            # Existing filters
            "cardinal_rotation_90": bool(self.ui.var_filter_cardinal_rotation_90.get()),
            "enable_rotation": bool(self.ui.var_filter_enable_rotation.get()),
            "enable_blur": bool(self.ui.var_filter_enable_blur.get()),
            "enable_grain": bool(self.ui.var_filter_enable_grain.get()),
            "enable_brightness": bool(self.ui.var_filter_enable_brightness.get()),
            "enable_contrast": bool(self.ui.var_filter_enable_contrast.get()),
            "rotation_strength": self._float_or_default(self.ui.var_filter_rotation_strength.get(), default=1.0),
            "blur_strength": self._float_or_default(self.ui.var_filter_blur_strength.get(), default=1.0),
            "grain_strength": self._float_or_default(self.ui.var_filter_grain_strength.get(), default=1.0),
            "brightness_strength": self._float_or_default(self.ui.var_filter_brightness_strength.get(), default=1.0),
            "contrast_strength": self._float_or_default(self.ui.var_filter_contrast_strength.get(), default=1.0),
            # High priority new filters
            "enable_perspective": bool(self.ui.var_filter_enable_perspective.get()),
            "perspective_strength": self._float_or_default(self.ui.var_filter_perspective_strength.get(), default=1.0),
            "perspective_angle_x": self._float_or_default(self.ui.var_filter_perspective_angle_x.get(), default=0.0),
            "perspective_angle_y": self._float_or_default(self.ui.var_filter_perspective_angle_y.get(), default=0.0),
            "enable_motion_blur": bool(self.ui.var_filter_enable_motion_blur.get()),
            "motion_blur_strength": self._float_or_default(self.ui.var_filter_motion_blur_strength.get(), default=1.0),
            "motion_blur_angle": self._float_or_default(self.ui.var_filter_motion_blur_angle.get(), default=0.0),
            "enable_saturation": bool(self.ui.var_filter_enable_saturation.get()),
            "saturation_factor": self._float_or_default(self.ui.var_filter_saturation_factor.get(), default=1.0),
            "enable_hue_shift": bool(self.ui.var_filter_enable_hue_shift.get()),
            "hue_shift_deg": self._float_or_default(self.ui.var_filter_hue_shift_deg.get(), default=0.0),
            "enable_shadow": bool(self.ui.var_filter_enable_shadow.get()),
            "shadow_strength": self._float_or_default(self.ui.var_filter_shadow_strength.get(), default=0.3),
            "shadow_size": self._float_or_default(self.ui.var_filter_shadow_size.get(), default=0.2),
            "enable_reflection": bool(self.ui.var_filter_enable_reflection.get()),
            "reflection_strength": self._float_or_default(self.ui.var_filter_reflection_strength.get(), default=0.5),
            "reflection_size": self._float_or_default(self.ui.var_filter_reflection_size.get(), default=0.15),
            # Medium/Low priority new filters
            "enable_vignetting": bool(self.ui.var_filter_enable_vignetting.get()),
            "vignetting_strength": self._float_or_default(self.ui.var_filter_vignetting_strength.get(), default=1.0),
            "enable_chromatic_aberration": bool(self.ui.var_filter_enable_chromatic_aberration.get()),
            "chromatic_strength": self._float_or_default(self.ui.var_filter_chromatic_strength.get(), default=1.0),
            "enable_jpeg_compression": bool(self.ui.var_filter_enable_jpeg_compression.get()),
            "jpeg_quality": int(self._float_or_default(self.ui.var_filter_jpeg_quality.get(), default=85)),
            "enable_color_temperature": bool(self.ui.var_filter_enable_color_temperature.get()),
            "color_temperature_kelvin": int(self._float_or_default(self.ui.var_filter_color_temperature_kelvin.get(), default=5500)),
            "enable_lens_distortion": bool(self.ui.var_filter_enable_lens_distortion.get()),
            "distortion_k1": self._float_or_default(self.ui.var_filter_distortion_k1.get(), default=0.0),
            "distortion_k2": self._float_or_default(self.ui.var_filter_distortion_k2.get(), default=0.0),
            "enable_dust": bool(self.ui.var_filter_enable_dust.get()),
            "dust_density": self._float_or_default(self.ui.var_filter_dust_density.get(), default=0.3),
            "dust_size": self._float_or_default(self.ui.var_filter_dust_size.get(), default=2.0),
            "enable_sharpen": bool(self.ui.var_filter_enable_sharpen.get()),
            "sharpen_strength": self._float_or_default(self.ui.var_filter_sharpen_strength.get(), default=1.0),
        }

    def _open_image_filters_popup(self) -> None:
        """Open shared image filter popup."""
        profiles, active_profile = self._load_filter_profiles()

        def on_close(updated_profiles: Dict[str, Dict[str, Any]], updated_active: str) -> None:
            self._save_filter_profiles(updated_profiles, updated_active)

        open_filter_popup(
            parent=self.frame,
            profiles=profiles,
            active_profile=active_profile,
            on_close=on_close,
            get_current_values=self._current_filter_settings_dict,
            set_current_values=self._apply_filter_settings_dict,
            float_or_default=self._float_or_default,
            show_messagebox=self.ui.show_messagebox,
            ask_string=self.ui.ask_string,
            ask_yes_no=self.ui.ask_yes_no,
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
        if self.logic.is_process_running():
            self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first.")
            return

        preset = (self.ui.var_profile_build_preset.get().strip() or "quick").lower()
        if preset not in ("quick", "full"): preset = "quick"

        cfg = self.sim_root / "configs" / ("run_profile_cls.yaml" if preset == "full" else "run_profile_cls_quick.yaml")
        out_ds = self.sim_root / "outputs" / "sim_data" / "runs" / ("run_profile_cls" if preset == "full" else "run_profile_cls_quick")
        out_model = self.sim_root / "outputs" / "models" / ("profile_classifier_v1.pt" if preset == "full" else "profile_classifier_quick.pt")

        if not cfg.exists():
            self.ui.show_messagebox("error", "Missing config", f"Config not found:\\n{cfg}")
            return
        
        tag = time.strftime("%Y%m%d_%H%M%S")
        if out_ds.exists(): out_ds = out_ds.parent / f"{out_ds.name}_{tag}"
        if out_model.exists(): out_model = out_model.with_name(f"{out_model.stem}_{tag}{out_model.suffix}")

        try:
            rel_model = str(out_model.resolve().relative_to(self.sim_root.resolve()))
        except Exception: rel_model = str(out_model)
        self.ui.var_profile_model.set(rel_model)
        self.ui.var_profile_model_lock.set(True)
        self._apply_profile_model_lock()
        
        def run_simple_cmd_callback(cmd_args: List[str], extra_env: Optional[Dict[str, str]] = None) -> None:
            self._run_simple_cmd(cmd_args, extra_env)

        self.logic.build_profile_model(preset, str(cfg), str(out_ds), str(out_model), self._append_log, run_simple_cmd_callback)
        self.ui.var_profile_model.set(rel_model)

    def start_pipeline(self) -> None: self._start_pipeline(task="full")
    def start_pipeline_generate_only(self) -> None: self._start_pipeline(task="generate_only")
    
    def _start_pipeline(self, *, task: str) -> None:
        if self.logic.is_process_running(): return
        
        run_mode, dataset_mode = self.ui.var_run_mode.get(), self.ui.var_dataset_mode.get()
        out_dir = ""
        if dataset_mode == "extend":
            if not (selected_ds := self._selected_dataset_dir()):
                self.ui.show_messagebox("error", "Error", "No dataset selected to extend.\\n\\nSelect a dataset or use 'Create New' mode.")
                return
            try:
                if str((self.sim_root / "outputs" / "sim_data" / "versions").resolve()).startswith(str(selected_ds.resolve()) + os.sep):
                    self.ui.show_messagebox("error", "Error", "Cannot extend/overwrite a dataset snapshot under outputs/sim_data/versions.\\n\\nSelect a dataset under outputs/sim_data/runs instead.")
                    return
            except Exception: pass
            if not self.ui.ask_yes_no("Extend dataset?", f"Extend Existing will APPEND new samples into the selected dataset folder.\\n\\nDataset:\\n{selected_ds}\\n\\nContinue?"): return
            out_dir = str(selected_ds)
            self._append_log(f"\n=== Extending selected dataset: {out_dir} ===\n")
        else:
            out_dir = self.ui.var_out.get().strip()
            self._append_log(f"\n=== Creating new dataset: {out_dir} ===\n")

        run_count = 1
        if run_mode == "multiple":
            try:
                if (rc := int(self.ui.var_run_count.get())) < 1: raise ValueError()
                run_count = rc
            except Exception:
                self.ui.show_messagebox("error", "Error", "Invalid run count. Please enter a positive integer.")
                return
        elif run_mode == "continuous": run_count = -1

        self._append_log(f"Run mode: {run_mode}" + (f" ({run_count}x)" if run_count > 0 else " (continuous)") + "\n\n")

        render_backend = self.ui.var_render_backend.get().strip() or ""
        
        multi_enabled = bool(self.ui.var_profiles_multi.get())
        multi_mode = self.ui.var_profiles_multi_mode.get().strip() or "separate"
        
        profile_ids = []
        if multi_enabled:
            profile_ids = self._profiles_multi_list(available=list(self.ui.profile_combo["values"]))
            if not profile_ids:
                self.ui.show_messagebox("error", "Error", "Multi-profile mode is enabled but no profiles are selected.\\n\\nClick Pick and select at least 1 profile.")
                return
            if dataset_mode == "extend":
                self.ui.show_messagebox("error", "Error", "Multi-profile generation does not support Extend Existing.\\n\\nSwitch Dataset mode to 'Create New'.")
                return
        else:
            if p := self.ui.var_profile.get().strip(): profile_ids = [p]

        try:
            if profile_ids: self._autoselect_config_for_profile(profile_ids[0], prefer_quiet=False)
        except Exception: pass

        run_specs: list[dict[str, str]] = []
        effective_task = task

        def _rel_to_sim_root(p: Path) -> str:
            try: return str(p.resolve().relative_to(self.sim_root.resolve()))
            except Exception: return str(p)

        if multi_enabled and multi_mode == "mixed":
            if task != "generate_only":
                self.ui.show_messagebox("error", "Error", "Multi-profile mode 'mixed' is supported only for Generate Only.")
                return
            if render_backend != "opencv_2d":
                self.ui.show_messagebox("error", "Error", "Multi-profile mode 'mixed' currently supports only Render backend opencv_2d.")
                return
            if len(profile_ids) < 2:
                self.ui.show_messagebox("error", "Error", "Mixed dataset mode requires at least 2 selected profiles.")
                return

            cfg_str = self.ui.var_config.get().strip()
            cfg_path = (self.sim_root / cfg_str) if cfg_str and not Path(cfg_str).is_absolute() else Path(cfg_str) if cfg_str else None
            if not cfg_path or not cfg_path.exists():
                self.ui.show_messagebox("error", "Error", f"Config file not found:\\n{cfg_str}")
                return
            try: base_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception as e:
                self.ui.show_messagebox("error", "Error", f"Failed to read config:\\n{cfg_path}\\n\\n{e}")
                return

            run_block = dict(base_cfg.get("run") or {})
            run_block.pop("component_profile", None)
            run_block["mode"] = "profile_classifier"
            run_block["schema_version"] = int(run_block.get("schema_version", 2) or 2)
            run_block["component_profiles"] = list(profile_ids)
            if not run_block.get("run_id"): run_block["run_id"] = "run_profile_cls_selected"
            base_cfg["run"] = run_block
            base_cfg.setdefault("render", {})["backend"] = "opencv_2d"

            live_dir = (self.sim_root / "outputs" / "live")
            live_dir.mkdir(parents=True, exist_ok=True)
            try:
                with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", prefix="run_profile_cls_selected_", dir=str(live_dir), delete=False) as tf:
                    yaml.safe_dump(base_cfg, tf, sort_keys=False)
                    tmp_cfg_path = Path(tf.name)
            except Exception as e:
                self.ui.show_messagebox("error", "Error", f"Failed to write temporary config under outputs/live:\\n\\n{e}")
                return

            run_specs.append({
                "profile_id": "multi", "config": _rel_to_sim_root(tmp_cfg_path), "out_dir": out_dir,
                "model_path": self.ui.var_model.get().strip(),
            })
            effective_task = "generate_mixed"
        else:
            base_out = Path(out_dir)
            base_model = Path(self.ui.var_model.get().strip())
            for i, pid in enumerate(profile_ids):
                cfg_info = self.logic.config_for_profile(pid, want_backend=render_backend or None)
                if not cfg_info:
                    self.ui.show_messagebox("error", "Error", f"No matching config found for profile:\\n{pid}\\n\\nPick or create a config with run.component_profile={pid}.")
                    return
                cfg_path = Path(cfg_info["path"])
                cfg_rel = _rel_to_sim_root(cfg_path)
                run_id = str(cfg_info.get("run_id") or pid)

                out_i = base_out.with_name(run_id) if len(profile_ids) > 1 else base_out
                if len(profile_ids) > 1:
                    if base_out.name in {"runs", "versions"}: out_i = base_out / run_id
                    elif base_out.parent.name in {"runs", "versions"}: out_i = base_out.parent / run_id

                # Keep the user-selected model path for a single-profile run.
                # Only auto-derive per-profile model names when running multiple profiles.
                if len(profile_ids) > 1:
                    model_i = (self.sim_root / "outputs" / "models" / f"{run_id}.pt")
                    if base_model.suffix == ".pt":
                        if base_model.parent.name == "models": model_i = base_model.parent / f"{run_id}.pt"
                        else: model_i = base_model.with_name(f"{run_id}.pt")
                else:
                    model_i = base_model if str(base_model).strip() else (self.sim_root / "outputs" / "models" / f"{run_id}.pt")

                run_specs.append({
                    "profile_id": pid, "config": cfg_rel, "out_dir": str(out_i), "model_path": str(model_i),
                })
        
        image_filters = self._image_filters_payload()

        if not self._confirm_pipeline_start(
            task=effective_task, out_dir=out_dir, run_mode=run_mode, run_count=run_count,
            dataset_mode=dataset_mode, profile_ids=profile_ids, multi_enabled=multi_enabled, multi_mode=multi_mode,
            image_filters=image_filters,
        ): return

        self.ui.set_run_buttons_state(True)
        self.logic.stop_evt.clear()

        status_vars = {
            "phase": self.ui.var_phase, "run_progress": self.ui.var_run_progress,
            "epoch": self.ui.var_epoch, "ips": self.ui.var_ips, "last": self.ui.var_last,
            "dataset_dir": self.state.dataset_dir,
            "snap_every": self._snap_every_n(), "snap_keep": self._snap_keep_n(), "autosnap": self.ui.var_autosnap.get()
        }
        self.logic.start_pipeline(
            run_specs=run_specs, run_count=run_count, dataset_mode=dataset_mode, task=effective_task,
            image_filters=image_filters,
            event_callback=self._handle_event, log_callback=self._append_log,
            ui_update_callback=lambda: self.frame.after(0, self._after_pipeline_run_ui_update),
            ui_reset_callback=lambda: self.frame.after(0, self._reset_ui_on_pipeline_end),
            status_vars=status_vars
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
        self.ui.set_run_buttons_state(False)
        self.ui.var_phase.set("phase: idle")
        self.ui.var_run_progress.set("run: -")
        self.logic.current_run_iteration = 0
        self.logic.total_run_count = 0

    def stop_pipeline(self) -> None:
        self.logic.stop_pipeline()

    def _run_simple_cmd(self, args: list[str], extra_env: Optional[Dict[str, str]] = None) -> None:
        if self.logic.is_process_running():
            self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first.")
            return

        self.ui.set_run_buttons_state(True)
        self.logic.stop_evt.clear()
        
        env = os.environ.copy()
        env["EVENT_LOG"] = str(self.event_log_path)
        env["SIMPLE_SIM_EVENT_LOG"] = str(self.event_log_path)
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env: env.update(extra_env)

        cmd = ["bash", "-lc", " ".join(args)]

        self.logic.proc = subprocess.Popen(cmd, cwd=str(self.sim_root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
        p = self.logic.proc
        
        threading.Thread(target=self.logic._read_process_output_thread, args=(p, self._append_log), daemon=True).start()
        threading.Thread(target=self.logic._tail_events_thread, args=(self._handle_event,), daemon=True).start()
        self.ui.var_phase.set("running cmd")

    def _handle_event(self, evt: Dict[str, Any]) -> None:
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
        
        if not self.logic.is_process_running():
            if self.ui.btn_start["state"] == "disabled" or self.ui.btn_start_generate["state"] == "disabled": self.ui.set_run_buttons_state(False)

        self._maybe_refresh_thumbnails()
        self._maybe_update_stats()
        self.frame.after(120, self._tick_ui)

    def _maybe_update_stats(self) -> None:
        now = time.time()
        if now - self._last_stats_ts < 1.0: return
        self._last_stats_ts = now

        cpu_pct, ram_info, gpu_pct = self.logic.get_system_stats()
        self.ui.update_stats_bar(cpu_pct, ram_info, gpu_pct)

        proc_running = self.logic.is_process_running()
        if self._prev_proc_running and not proc_running:
            if self.state.dataset_dir: self._start_dataset_size_calc(self.state.dataset_dir)
            self._last_ds_size_ts = now
        self._prev_proc_running = proc_running

        if proc_running and (now - self._last_ds_size_ts) >= 5.0:
            if self.state.dataset_dir: self._start_dataset_size_calc(self.state.dataset_dir)
            self._last_ds_size_ts = now

    def _snap_every_n(self) -> int:
        try:
            if (s := self.ui.var_snap_every.get().strip()): n = int(s)
            else: n = 1
        except Exception: n = 5
        return max(1, n)

    def _snap_keep_n(self) -> int:
        try:
            if (s := self.ui.var_snap_keep.get().strip()): k = int(s)
            else: k = 0
        except Exception: k = 30
        return max(0, k)

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
                with open(manifest_path, "r", encoding="utf-8") as f: manifest = json.load(f)
                stats = manifest.get("dataset_stats", {})
                splits = stats.get("splits", {})
                self._set_dataset_samples_from_manifest(ds, stats.get("total_samples"), splits.get("train"), splits.get("val"), splits.get("test"))
            except Exception as exc:
                print(f"Failed to refresh dataset stats: {exc}")
                self._set_dataset_samples_info("Samples: manifest error", "")
        else: self._set_dataset_samples_info("Samples: manifest missing", "")

        self._start_dataset_size_calc(ds)

    def _set_dataset_samples_from_manifest(self, ds: Path, total: Optional[int], train: Optional[int], val: Optional[int], test: Optional[int]) -> None:
        if total is None: return self._set_dataset_samples_info("Samples: manifest missing", "")
        parts = [f"Samples: {total}"]
        if train is not None and val is not None and test is not None: parts.append(f"(t{train}/v{val}/s{test})")
        desc, color = " ".join(parts), self._sample_color_from_total(total)
        self._set_dataset_samples_info(desc, color)
        self._check_milestone(ds, total)

    def _set_dataset_samples_info(self, text: str, color: str) -> None:
        self.ui.set_dataset_samples_info(text, color)

    def _sample_color_from_total(self, total: int) -> str:
        if total >= 10000: return "green"
        if total >= 5000: return "blue"
        if total >= 1000: return "orange"
        return "black"

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
            with open(self.event_log_path, "a", encoding="utf-8") as f: f.write(json.dumps(event) + "\n")
        except Exception as e: print(f"Failed to log milestone: {e}")

    def _apply_dataset_settings(self, ds: Path) -> None:
        if not ds: return
        if (config_path := ds / "config.yaml").exists(): self.ui.var_config.set(str(config_path))
        if profile_id := self.logic.get_dataset_profile_id(ds): self._set_profile_value(profile_id)
        
        self.ui.var_name.set(ds.name)
        self.ui.var_out.set(str(ds.resolve()))
        
        model_path = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        try: self.ui.var_model.set(str(model_path.resolve()))
        except Exception: self.ui.var_model.set(str(model_path))
        self.ui.var_dataset_mode.set("extend")
        self._update_model_dropdown_for_dataset()

    def _set_profile_value(self, profile_id: str) -> None:
        self._suspend_profile_event = True
        self.ui.var_profile.set(profile_id)
        self._last_profile_id = profile_id

    def _refresh_datasets(self) -> None:
        sim_data, runs, versions = self.sim_root / "outputs" / "sim_data", self.sim_root / "outputs" / "sim_data" / "runs", self.sim_root / "outputs" / "sim_data" / "versions"
        runs.mkdir(parents=True, exist_ok=True); versions.mkdir(parents=True, exist_ok=True)

        cand: list[Path] = [p for p in runs.iterdir() if p.is_dir()]
        cand.extend([p for p in versions.glob("*/*") if p.is_dir()])
        cand.sort(key=lambda p: (1, -p.stat().st_mtime, p.name) if str(p.resolve()).startswith(str(versions.resolve()) + os.sep) else (0, p.name))

        self._dataset_dirs, self._dataset_labels, self._dataset_by_label = cand, [], {}
        seen: dict[str, int] = {}
        for p in cand:
            base_label = self._display_for_dataset(p, runs=runs, versions=versions)
            n = seen.get(base_label, 0) + 1; seen[base_label] = n
            label = base_label if n == 1 else f"{base_label} ({n})"
            self._dataset_labels.append(label); self._dataset_by_label[label] = p

        self.ui.set_dataset_combo_values(self._dataset_labels)
        self._sync_multi_selection_after_dataset_refresh()

        if (cur := self.ui.var_dataset.get().strip()) and cur in self._dataset_by_label: return
        if self.state.dataset_dir:
            want = self._display_for_dataset(self.state.dataset_dir, runs=runs, versions=versions)
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

        self.ui.toggle_dataset_multi_widgets(enabled, self._dataset_multi_tooltip_text)
        self._sync_multi_summary()
        
        if enabled:
            if ds0 := self._selected_dataset_dir():
                if label0 := self._dataset_label_for_path(ds0): self.ui.var_dataset.set(label0)
        self._on_dataset_selected()

    def _open_dataset_multi_picker(self) -> None:
        if not self._dataset_labels:
            self.ui.show_messagebox("info", "Datasets", "No datasets found. Click Refresh after generating datasets.")
            return

        top = tk.Toplevel(self.frame.winfo_toplevel()); top.title("Select datasets"); top.transient(self.frame.winfo_toplevel()); top.grab_set()
        frm = ttk.Frame(top, padding=10); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select one or more datasets (Ctrl/Shift for multi-select):").pack(anchor="w")
        lb = tk.Listbox(frm, selectmode="extended", height=min(18, max(6, len(self._dataset_labels)))); lb.pack(fill="both", expand=True, pady=(6, 8))
        for s in self._dataset_labels: lb.insert("end", s)

        selected_paths = set(str(p.resolve()) if not p.is_absolute() else str(p) for p in self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get()))
        if not selected_paths and (ds := self._selected_dataset_dir_single()) is not None: selected_paths.add(str(ds.resolve()))

        for i, label in enumerate(self._dataset_labels):
            if (p := self._dataset_by_label.get(label)) is not None and str(p.resolve()) in selected_paths: lb.selection_set(i)
        
        btns = ttk.Frame(frm); btns.pack(fill="x")
        ttk.Button(btns, text="Select all", command=lambda: lb.selection_set(0, "end")).pack(side="left")
        ttk.Button(btns, text="Clear", command=lambda: lb.selection_clear(0, "end")).pack(side="left", padx=(8, 0))
        ttk.Button(btns, text="Cancel", command=top.destroy).pack(side="right")
        ttk.Button(btns, text="OK", command=lambda: self._on_multi_picker_ok(lb, top)).pack(side="right", padx=(0, 8))
        lb.bind("<Double-Button-1>", lambda _e: self._on_multi_picker_ok(lb, top))
        top.minsize(640, 320)

    def _on_multi_picker_ok(self, listbox: tk.Listbox, dialog: tk.Toplevel) -> None:
        idxs = list(listbox.curselection())
        paths: list[Path] = [self._dataset_by_label[self._dataset_labels[int(i)]] for i in idxs]
        if not paths and (ds := self._selected_dataset_dir_single()) is not None: paths = [ds]
        
        self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json(paths))
        self._sync_multi_summary()

        if paths:
            if label0 := self._dataset_label_for_path(paths[0]): self.ui.var_dataset.set(label0)
        self._on_dataset_selected()
        dialog.destroy()

    def _decode_dataset_paths_json(self, s: str) -> list[Path]:
        try: arr = json.loads(s or "[]")
        except Exception: return []
        if not isinstance(arr, list): return []
        
        out: list[Path] = []
        for it in arr:
            if not isinstance(it, str): continue
            p = Path(it)
            if not p.is_absolute(): p = (self.sim_root / p).resolve()
            out.append(p)
        
        seen: set[str] = set()
        return [p for p in out if not (key := str(p.resolve())) in seen and not seen.add(key)]

    def _encode_dataset_paths_json(self, paths: list[Path]) -> str:
        return json.dumps([str(p.resolve().relative_to(self.sim_root.resolve())) if not p.is_absolute() else str(p) for p in paths])

    def _dataset_label_for_path(self, p: Path) -> Optional[str]:
        rp = p.resolve()
        for label, pp in self._dataset_by_label.items():
            if pp.resolve() == rp: return label
        for label, pp in self._dataset_by_label.items():
            if pp.name == p.name: return label
        return None

    def _selected_dataset_labels(self) -> list[str]:
        return [label for ds in self._selected_dataset_dirs() if (label := self._dataset_label_for_path(ds))]

    def _sync_multi_summary(self) -> None:
        labels = self._selected_dataset_labels()
        if not labels: self.ui.var_dataset_multi_summary.set("(none)")
        elif len(labels) == 1: self.ui.var_dataset_multi_summary.set(labels[0])
        else: self.ui.var_dataset_multi_summary.set(f"{labels[0]} (+{len(labels) - 1})")

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
        try:
            rp = p.resolve()
            if str(rp).startswith(str(runs.resolve()) + os.sep): return rp.name
            if str(rp).startswith(str(versions.resolve()) + os.sep): return f"{rp.parent.name}:{rp.name}"
        except Exception: pass
        return p.name

    def _refresh_models(self) -> None:
        paths = self.logic.get_all_model_paths()
        self._model_paths = paths
        self._model_profile_cache.clear()
        for p in self._model_paths:
            self._model_profile_cache[str(p.resolve())] = self.logic.load_model_profile(p)

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
            if (current := self.ui.var_model.get().strip()) not in values: self.ui.var_model.set(values[0])
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
            if "chip_0603_resistor@1" in self.ui.profile_combo["values"]: self.ui.var_profile.set("chip_0603_resistor@1")
            elif self.ui.profile_combo["values"]: self.ui.var_profile.set(self.ui.profile_combo["values"][0])

        if bool(self.ui.var_profiles_multi.get()): self._set_profiles_multi(self._profiles_multi_list(available=list(self.ui.profile_combo["values"])))
        self._on_profile_selected()

    def _profiles_multi_list(self, *, available: Optional[list[str]] = None) -> list[str]:
        available_set = set(available or list(self.ui.profile_combo["values"] or []))
        raw = (self.ui.var_profiles_multi_json.get().strip() or "[]")
        items: list[str] = []
        try:
            if isinstance(v := json.loads(raw), list): items = [x.strip() for x in v if isinstance(x, str) and x.strip()]
        except Exception: items = [x.strip() for x in raw.split(",") if x.strip()]
        
        seen = set()
        return [pid for pid in items if (pid in available_set or not available_set) and not pid in seen and not seen.add(pid)]

    def _set_profiles_multi(self, pids: list[str]) -> None:
        pids = [str(x).strip() for x in (pids or []) if str(x).strip()]
        try: self.ui.var_profiles_multi_json.set(json.dumps(pids, ensure_ascii=True))
        except Exception: pass
        if not pids: self.ui.var_profiles_multi_summary.set("")
        elif len(pids) <= 3: self.ui.var_profiles_multi_summary.set(", ".join(pids))
        else: self.ui.var_profiles_multi_summary.set(", ".join(pids[:3]) + f" (+{len(pids) - 3})")

    def _on_profiles_multi_toggle(self) -> None:
        enabled = bool(self.ui.var_profiles_multi.get())
        self.ui.set_profiles_multi_state(enabled)
        if enabled:
            cur = self._profiles_multi_list(available=list(self.ui.profile_combo["values"]))
            if not cur and (pid := self.ui.var_profile.get().strip()): cur = [pid]
            self._set_profiles_multi(cur)

    def _pick_profiles_multi(self) -> None:
        values = list(self.ui.profile_combo["values"])
        if not values: self.ui.show_messagebox("info", "Profiles", "No profiles available."); return

        cur = set(self._profiles_multi_list(available=values))
        win = tk.Toplevel(self.frame); win.title("Select Profiles"); win.geometry("520x420")
        frm = ttk.Frame(win, padding=10); frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select one or more profiles to generate:").pack(anchor="w")
        lb = tk.Listbox(frm, selectmode="extended", height=16); lb.pack(fill="both", expand=True, pady=(8, 0))
        sb = ttk.Scrollbar(frm, orient="vertical", command=lb.yview); sb.pack(side="left", fill="y", pady=(8, 0))
        lb.configure(yscrollcommand=sb.set)
        for i, pid in enumerate(values): lb.insert("end", pid);
        for i, pid in enumerate(values):
            if pid in cur: lb.selection_set(i)

        btns = ttk.Frame(frm); btns.pack(fill="x", pady=(10, 0))
        ttk.Button(btns, text="Select All", command=lambda: lb.selection_set(0, "end")).pack(side="left")
        ttk.Button(btns, text="OK", command=lambda: self._on_pick_profiles_multi_ok(lb, win)).pack(side="right")
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 8))
        win.minsize(560, 260)

    def _on_pick_profiles_multi_ok(self, listbox: tk.Listbox, dialog: tk.Toplevel) -> None:
        idxs = list(listbox.curselection())
        values = list(self.ui.profile_combo["values"])
        sel = [values[i] for i in idxs if 0 <= i < len(values)]
        self._set_profiles_multi(sel)
        if sel: self.ui.var_profile.set(sel[0])
        dialog.destroy()

    def _on_render_backend_changed(self, _event: Optional[object] = None) -> None:
        try:
            self._refresh_profiles()
            if pid := self.ui.var_profile.get().strip(): self._autoselect_config_for_profile(pid, prefer_quiet=True)
        except Exception: pass

    def _on_profile_selected(self, _event: Optional[object] = None) -> None:
        if self._suspend_profile_event: self._suspend_profile_event = False; return
        if not (profile_id := self.ui.var_profile.get().strip()): return

        try: self._autoselect_config_for_profile(profile_id, prefer_quiet=True)
        except Exception: pass

        if (ds := self._selected_dataset_dir()) and self.logic.get_dataset_profile_id(ds) == profile_id:
            self._last_profile_id = profile_id
            return

        if (candidate := self._find_dataset_for_profile(profile_id)):
            if self.ui.ask_yes_no("Switch dataset", f"A dataset for profile '{profile_id}' already exists:\\n{candidate}\\n\\nSwitch to it so profiles stay separated?"):
                self._select_dataset(candidate); self._last_profile_id = profile_id; return
            self._revert_profile_selection(); return

        if self.ui.ask_yes_no("Create dataset", f"No dataset currently matches profile '{profile_id}'.\\nWould you like to prepare a new out path for this profile?"):
            self._prepare_dataset_for_profile(profile_id, self.logic.config_for_profile(profile_id, want_backend=(self.ui.var_render_backend.get().strip() or None)))
            self._last_profile_id = profile_id; return

        self._revert_profile_selection()

    def _find_dataset_for_profile(self, profile_id: str) -> Optional[Path]:
        runs_root = self.sim_root / "outputs" / "sim_data" / "runs"
        if not runs_root.exists(): return None
        for ds in sorted(runs_root.iterdir()):
            if not ds.is_dir(): continue
            if self.logic.get_dataset_profile_id(ds) == profile_id: return ds
        return None

    def _select_dataset(self, ds: Path) -> None:
        self._refresh_datasets()
        runs, versions = self.sim_root / "outputs" / "sim_data" / "runs", self.sim_root / "outputs" / "sim_data" / "versions"
        label = self._display_for_dataset(ds, runs=runs, versions=versions)
        for key, value in self._dataset_by_label.items():
            if value == ds or key == label: self.ui.var_dataset.set(key); self._on_dataset_selected(); return
        self.state.dataset_dir = ds
        self.ui.var_dataset.set(label)
        self._on_dataset_selected()

    def _revert_profile_selection(self) -> None:
        self._suspend_profile_event = True
        self.ui.var_profile.set(self._last_profile_id or "chip_0603_resistor@1")

    def _prepare_dataset_for_profile(self, profile_id: str, cfg_info: Optional[Dict[str, Any]]) -> None:
        cfg_path: Optional[Path] = None
        if cfg_info:
            cfg_path = cfg_info["path"]
            self.ui.var_config.set(str(cfg_path))
            suggested_run = cfg_info.get("run_id")
        else: suggested_run = self.logic.slugify_name(profile_id)
        
        self.ui.var_dataset_mode.set("new")
        if suggested_run:
            suggested_out = self.sim_root / "outputs" / "sim_data" / "runs" / suggested_run
            self.ui.var_out.set(str(suggested_out))
            self.ui.var_name.set(suggested_run)
            suggested_model = self.sim_root / "outputs" / "models" / f"{suggested_run}.pt"
            self.ui.var_model.set(str(suggested_model))
        self._append_log(f"[profile] prepared dataset for profile {profile_id}\n")
        
        if cfg_info:
            self.ui.show_messagebox("info", "Ready", f"Config '{cfg_path.name}' assigned.\\nOut directory: {self.ui.var_out.get()}\\nRun the pipeline to generate the matching dataset for this profile.")
        else:
            self.ui.show_messagebox("info", "Ready", "No config automatically detected for this profile.\\nPick or create a config that sets `run.component_profile` to " f"'{profile_id}', then run the pipeline to create the dataset.")

        config_candidate = cfg_path if cfg_path else Path(self.ui.var_config.get())
        out_dir = Path(self.ui.var_out.get())
        try:
            if self.logic.ensure_dataset_skeleton(profile_id, config_candidate, out_dir, self._append_log):
                self._refresh_datasets()
                self._select_dataset(out_dir)
        except Exception as exc: print(f"Failed to create dataset placeholder: {exc}")

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

    def _show_profile_info(self) -> None:
        profile_id = self.ui.var_profile.get()
        if not profile_id: self.ui.show_messagebox("info", "Profile Info", "No profile selected"); return

        profiles_dir = self.sim_root / "configs" / "profiles"
        profile_path = profiles_dir / f"{profile_id}.yaml"
        if not profile_path.exists(): self.ui.show_messagebox("error", "Error", f"Profile file not found:\\n{profile_path}"); return

        try:
            with open(profile_path, 'r') as f: profile_data = yaml.safe_load(f)
            dialog = tk.Toplevel(self.frame); dialog.title(f"Profile Info: {profile_id}"); dialog.geometry("600x500")
            text_frame = ttk.Frame(dialog); text_frame.pack(fill="both", expand=True, padx=10, pady=10)
            scrollbar = ttk.Scrollbar(text_frame); scrollbar.pack(side="right", fill="y")
            text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10)); text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)

            info_text = f"Profile ID: {profile_id}\\nPath: {profile_path}\\n\\n" + "="*60 + "\\n\\n" + yaml.dump(profile_data, default_flow_style=False, sort_keys=False)
            text.insert("1.0", info_text); text.config(state="disabled")
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

        except Exception as e: self.ui.show_messagebox("error", "Error", f"Failed to load profile:\\n{str(e)}")

    def _check_profile_compatibility(self) -> None:
        try:
            if not (ds := self._selected_dataset_dir()): self.ui.var_profile_compat.set(""); return
            manifest_path = ds / "dataset_manifest.json"
            if not manifest_path.exists(): self.ui.var_profile_compat.set(""); return

            with open(manifest_path, 'r') as f: manifest = json.load(f)
            mver = int(manifest.get('manifest_version', 1) or 1)
            ds_profile_id = (manifest.get('component_profile', {}).get('profile_id') if mver == 1 else "multi")
            ds_profile_hash = (manifest.get('component_profile', {}).get('profile_hash') if mver == 1 else "multi")

            if not ds_profile_id: self.ui.var_profile_compat.set(""); return

            if not (model_path_str := self.ui.var_model.get().strip()): self.ui.var_profile_compat.set(""); return
            model_path = self.logic._resolve_model_path(model_path_str)
            if not model_path.exists(): self.ui.var_profile_compat.set(""); return
            
            if model_path.is_dir():
                if not (cand := bundle_checkpoint_path(model_path, ds_profile_id, kind="best")).exists():
                    self.ui.var_profile_compat.set(f"⚠ Multi-model: no weights for {ds_profile_id}"); self.ui.lbl_profile_compat.configure(foreground="orange"); return
                model_path = cand

            checkpoint = torch.load(model_path, map_location='cpu')
            if not (model_profile := checkpoint.get('component_profile')):
                self.ui.var_profile_compat.set("⚠ Model has no profile (legacy)"); self.ui.lbl_profile_compat.configure(foreground="orange"); return

            model_profile_id, model_profile_hash = model_profile.get('profile_id'), model_profile.get('profile_hash')

            if isinstance(model_profile_id, str) and model_profile_id.lower().startswith("multi"):
                self.ui.var_profile_compat.set("✓ Compatible (multi-model)"); self.ui.lbl_profile_compat.configure(foreground="green")
            elif model_profile_id != ds_profile_id:
                self.ui.var_profile_compat.set(f"✗ INCOMPATIBLE: Model={model_profile_id}, Dataset={ds_profile_id}"); self.ui.lbl_profile_compat.configure(foreground="red")
            elif model_profile_hash != ds_profile_hash:
                self.ui.var_profile_compat.set(f"⚠ Profile hash mismatch (same ID, different version)"); self.ui.lbl_profile_compat.configure(foreground="orange")
            else:
                self.ui.var_profile_compat.set(f"✓ Compatible: {ds_profile_id}"); self.ui.lbl_profile_compat.configure(foreground="green")

        except Exception as e: print(f"Profile compatibility check failed: {e}"); self.ui.var_profile_compat.set("")

    def _show_dataset_profile_info(self) -> None:
        if not (ds := self._selected_dataset_dir()): self.ui.show_messagebox("info", "Dataset Profile", "No dataset selected"); return
        manifest_path = ds / "dataset_manifest.json"
        if not manifest_path.exists(): self.ui.show_messagebox("error", "No Manifest", f"Dataset has no manifest (legacy dataset).\\n\\nUse the backfill tool to add a manifest:\\n  .venv/bin/python tools/backfill_manifest.py --data {ds}"); return

        try:
            with open(manifest_path, 'r') as f: manifest = json.load(f)
            profile_id, profile_hash, profile_path_str = manifest.get('component_profile', {}).get('profile_id', 'unknown'), manifest.get('component_profile', {}).get('profile_hash', ''), manifest.get('component_profile', {}).get('profile_path', '')
            profiles_dir = self.sim_root / "configs" / "profiles"
            profile_path = profiles_dir / f"{profile_id}.yaml"

            dialog = tk.Toplevel(self.frame); dialog.title(f"Dataset Profile: {ds.name}"); dialog.geometry("650x550")
            text_frame = ttk.Frame(dialog); text_frame.pack(fill="both", expand=True, padx=10, pady=10)
            scrollbar = ttk.Scrollbar(text_frame); scrollbar.pack(side="right", fill="y")
            text = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Courier", 10)); text.pack(side="left", fill="both", expand=True)
            scrollbar.config(command=text.yview)

            info_text = f"Dataset: {ds.name}\\nProfile ID: {profile_id}\\nProfile Hash: {profile_hash}\\nProfile Path: {profile_path_str}\\n\\n" + "="*60 + "\\nMANIFEST METADATA\\n" + "="*60 + "\\n\\n" + json.dumps(manifest, indent=2)
            if profile_path.exists(): info_text += "\\n\\n" + "="*60 + "\\nPROFILE DETAILS\\n" + "="*60 + "\\n\\n" + yaml.dump(yaml.safe_load(profile_path.read_text(encoding="utf-8")), default_flow_style=False, sort_keys=False)
            else: info_text += f"\\n\\n⚠ Profile file not found at: {profile_path}"
            text.insert("1.0", info_text); text.config(state="disabled")
            ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 10))

        except Exception as e: self.ui.show_messagebox("error", "Error", f"Failed to load dataset profile info:\\n{str(e)}")

    def _apply_name_to_out_and_model(self) -> None:
        base = self.logic.slugify_name(self.ui.var_name.get())
        name = f"{base}_{time.strftime('%Y%m%d_%H%M%S')}" if self.ui.var_name_ts.get() else base
        out_rel, model_rel = f"outputs/sim_data/runs/{name}", f"outputs/models/{name}.{'bundle' if self.ui.var_model_bundle.get() else 'pt'}"
        self.ui.var_out.set(out_rel); self.ui.var_model.set(model_rel)
        self._refresh_models()
        self._append_log(f"[naming] out={out_rel} model={model_rel}\n")

    def _selected_dataset_dir(self) -> Optional[Path]:
        dss = self._selected_dataset_dirs(); return dss[0] if dss else None

    def _selected_dataset_dir_single(self) -> Optional[Path]:
        label = self.ui.var_dataset.get().strip(); return self._dataset_by_label.get(label)

    def _selected_dataset_dirs(self) -> list[Path]:
        if self.ui.var_dataset_multi.get():
            paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
            return [p for p in paths if p.exists() and p.is_dir()]
        return [ds] if (ds := self._selected_dataset_dir_single()) is not None else []

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
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
        except Exception as e: print(f"Failed to load metadata/labels: {e}")

        try:
            if (manifest_path := ds / "dataset_manifest.json").exists():
                with open(manifest_path, 'r') as f: manifest = json.load(f)
                profile_id = manifest.get('component_profile', {}).get('profile_id', 'unknown')
                profile_hash = manifest.get('component_profile', {}).get('profile_hash', '')
                stats = manifest.get("dataset_stats", {}); total_samples = stats.get("total_samples")
                splits = stats.get("splits", {}); train, val, test = splits.get("train"), splits.get("val"), splits.get("test")
                hash_short = profile_hash.split(':')[1][:12] if ':' in profile_hash else profile_hash[:12]
                self.ui.var_dataset_profile.set(f"Profile: {profile_id} ({hash_short}...)")
                self._set_dataset_samples_from_manifest(ds, total_samples, train, val, test)
            else:
                self.ui.var_dataset_profile.set("Profile: ⚠ No manifest (legacy dataset)")
                self._set_dataset_samples_info("Samples: legacy dataset", "black")
        except Exception as e:
            self.ui.var_dataset_profile.set(f"Profile: ⚠ Error loading manifest"); self._set_dataset_samples_info("Samples: error", "")
            print(f"Failed to load dataset manifest: {e}")

        self._check_profile_compatibility(); self._apply_dataset_settings(ds)

        if (img_dir := ds / "images").exists():
            for p in sorted(img_dir.glob("*.png"))[-12:]: self._recent_imgs.append(str(p))
            self._refresh_thumbnails()

        try: self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception: pass
        
        try:
            if (prev_ds := self.state.dataset_dir) is not None:
                cur_model = self.ui.var_model.get().strip()
                prev_default = str((self.sim_root / "outputs" / "models" / f"{prev_ds.name}.pt").resolve())
                cur_resolved = str(self.logic._resolve_model_path(cur_model))
                if cur_resolved == prev_default: self.ui.var_model.set(str(self.sim_root / "outputs" / "models" / f"{ds.name}.pt"))
            self._refresh_models()
        except Exception: pass

    def _delete_dataset(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_process_running(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before deleting datasets."); return
        if not self.ui.ask_yes_no("Delete dataset", f"Delete dataset folder?\\n\\n{ds}"): return
        try: shutil.rmtree(ds); self._append_log(f"\n[deleted dataset {ds}]\n")
        except Exception as e: self.ui.show_messagebox("error", "Delete failed", str(e))
        self._refresh_datasets()

    def _create_new_dataset(self) -> None:
        if self.logic.is_process_running(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before creating datasets."); return

        initial = self.logic.slugify_name(self.ui.var_name.get())
        if not (new_name := self.ui.ask_string("Create dataset", "New dataset folder name (will be created under outputs/sim_data/runs):", initialvalue=initial)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name: self.ui.show_messagebox("error", "Error", "Name must not contain path separators."); return

        out_dir = (self.sim_root / "outputs" / "sim_data" / "runs" / new_name).resolve()
        if out_dir.exists():
            if self.ui.ask_yes_no("Dataset exists", f"Dataset already exists:\\n\\n{out_dir}\\n\\nSelect it?"): self._select_dataset(out_dir); return

        if not (config_candidate := Path(self.ui.var_config.get())).exists(): self.ui.show_messagebox("error", "Missing config", f"Config not found:\\n\\n{config_candidate}\\n\\nPick a valid Config first, then retry."); return
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
        if self.logic.is_process_running(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before snapshotting datasets."); return

        sim_data, versions_root = self.sim_root / "outputs" / "sim_data", self.sim_root / "outputs" / "sim_data" / "versions"
        versions_root.mkdir(parents=True, exist_ok=True)
        group, tag, dst = ds.name, time.strftime("%Y%m%d_%H%M%S"), versions_root / group / f"{group}_{tag}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): self.ui.show_messagebox("error", "Error", f"Snapshot destination already exists:\\n{dst}"); return
        if not self.ui.ask_yes_no("Snapshot dataset", f"Create snapshot?\\n\\nFrom:\\n{ds}\\n\\nTo:\\n{dst}"): return
        self._append_log(f"\n[snapshot dataset] {ds} -> {dst}\n")

        def link_or_copy_file(src: Path, dstp: Path) -> None:
            dstp.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(src, dstp)
            except Exception: shutil.copy2(src, dstp)

        def worker() -> None:
            try:
                for dirpath, dirnames, filenames in os.walk(ds):
                    rel = Path(dirpath).relative_to(ds)
                    for fn in filenames: link_or_copy_file(Path(dirpath) / fn, dst / rel / fn)
                self.log_q.put(f"[snapshot dataset] done: {dst}\n")
            except Exception as e: self.log_q.put(f"[snapshot dataset] failed: {e}\n"); shutil.rmtree(dst)
            finally: self.frame.after(0, self._refresh_datasets)
        threading.Thread(target=worker, daemon=True).start()

    def _rename_dataset_selected(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_process_running(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before renaming datasets."); return

        sim_data = (self.sim_root / "outputs" / "sim_data").resolve()
        runs = (sim_data / "runs").resolve()
        versions = (sim_data / "versions").resolve()
        rp = ds.resolve()
        kind, old_name, parent = "", rp.name, rp.parent
        if str(rp).startswith(str(runs) + os.sep): kind = "run"
        elif str(rp).startswith(str(versions) + os.sep): kind = "snapshot"
        else: self.ui.show_messagebox("error", "Error", f"Can only rename datasets under:\\n{runs}\\n{versions}\\n\\nSelected:\\n{ds}"); return

        if not (new_name := self.ui.ask_string("Rename dataset", f"New folder name for this {kind} dataset:", initialvalue=old_name)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name: self.ui.show_messagebox("error", "Error", "Name must not contain path separators."); return
        
        dst = parent / new_name
        if dst.exists(): self.ui.show_messagebox("error", "Error", f"Target already exists:\\n{dst}"); return
        if not self.ui.ask_yes_no("Rename dataset", f"Rename?\\n\\nFrom:\\n{rp}\\n\\nTo:\\n{dst}"): return

        try: rp.rename(dst)
        except Exception as e: self.ui.show_messagebox("error", "Error", f"Rename failed:\\n{e}"); return

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
                except Exception as e: print(f"Failed to rename model {old} -> {new}: {e}")
            old_meta, new_meta = models_root / f"{old_name}{suffix}.pt.meta.json", models_root / f"{new_name}{suffix}.pt.meta.json"
            if old_meta.exists():
                try: old_meta.rename(new_meta)
                except Exception as e: print(f"Failed to rename meta {old_meta} -> {new_meta}: {e}")

        versions_root = models_root / "versions"
        old_dir, new_dir = versions_root / old_name, versions_root / new_name
        if old_dir.exists():
            try: old_dir.rename(new_dir)
            except Exception as e: print(f"Failed to rename versioned models {old_dir} -> {new_dir}: {e}")

    def _validate_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        if len(dss) == 1: self._run_simple_cmd(["./.venv/bin/python", "tools/validate_dataset.py", "--data", str(dss[0])]); return

        parts: list[str] = []
        for ds in dss: parts.append(" ".join(shlex.quote(x) for x in ["./.venv/bin/python", "tools/validate_dataset.py", "--data", str(ds)]))
        self._append_log(f"\n=== Validate {len(dss)} datasets ===\n")
        self._run_simple_cmd([" && ".join(parts)])

    def _train_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        if len(dss) > 1:
            if not (strat := self._ask_multi_dataset_train_strategy(dss)): return
            if strat in ("merge_keep", "merge_temp"):
                if self.logic.is_process_running(): self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first."); return
                
                ts = time.strftime("%Y%m%d_%H%M%S")
                name = ""
                if strat == "merge_keep":
                    default = f"{dss[0].name}_merged_{len(dss)}"
                    if not (name := self.ui.ask_string("Merge datasets", "Output dataset folder name (will be created under outputs/sim_data/runs):", initialvalue=default)): return
                    if not (name := name.strip()) or "/" in name or "\\" in name: self.ui.show_messagebox("error", "Error", "Invalid dataset name."); return
                else: name = f"__tmp_merge_{ts}"
                
                out_ds = (self.sim_root / "outputs" / "sim_data" / "runs" / name).resolve()
                if out_ds.exists(): self.ui.show_messagebox("error", "Error", f"Merge output already exists:\\n{out_ds}"); return

                model_s = self.ui.var_model.get().strip()
                model_out = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{name}.pt").resolve()
                delete_after = (strat == "merge_temp")

                self._append_log(f"\n=== Merge {len(dss)} datasets -> {out_ds.name} ===\n")
                for ds in dss: self._append_log(f"  - {ds}\n")
                self._append_log("\n")

                def worker() -> None:
                    try:
                        self.log_q.put(f"[merge] building merged dataset: {out_ds}\n")
                        self.logic.merge_datasets_for_training(dss, out_ds, self.log_q.put, self.log_q.put)
                        self.log_q.put(f"[merge] done: {out_ds}\n")

                        def kick_off_train() -> None:
                            try:
                                if strat == "merge_keep": self._select_dataset(out_ds)
                                else: self._refresh_datasets()
                            except Exception: pass

                            argv = ["./.venv/bin/python", "scripts/train.py", "--data", str(out_ds), "--out", str(model_out)]
                            cmd = " ".join(shlex.quote(x) for x in argv)
                            if delete_after: self.logic._temp_merge_cleanup_dir = out_ds
                            self._run_simple_cmd([cmd])

                        self.frame.after(0, kick_off_train)
                    except Exception as e:
                        self.log_q.put(f"[merge] failed: {e}\n")
                        if delete_after:
                            try: shutil.rmtree(out_ds)
                            except Exception: pass
                threading.Thread(target=worker, daemon=True).start(); return

        model_s = self.ui.var_model.get().strip()
        model_out = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()

        if len(dss) == 1: self._run_simple_cmd(["./.venv/bin/python", "scripts/train.py", "--data", str(dss[0]), "--out", str(model_out)]); return

        allow_mixed_profiles = False
        if self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_out):
            if not self.ui.ask_yes_no("Allow mixed-profile training into a single model?", "You selected datasets with different component profiles.\\n\\nRecommended: train into a multi-model bundle (.bundle) so each profile gets its own checkpoint.\\n\\nIf you continue anyway, the training will resume across different profiles by passing --allow-profile-mismatch (warm-start). This is not guaranteed to work and can degrade accuracy.\\n\\nContinue?"): return
            allow_mixed_profiles = True

        extra_s = self.ui.var_continue_epochs.get().strip() or "10"
        try: extra = int(extra_s)
        except Exception: self.ui.show_messagebox("error", "Error", "+epochs must be an integer."); return
        if extra < 0: self.ui.show_messagebox("error", "Error", "+epochs must be >= 0."); return

        out_mode = self.ui.var_continue_out_mode.get().strip() or "best"
        if out_mode not in ("last", "best"): out_mode = "best"

        parts: list[str] = []
        is_bundle = self.logic._is_model_bundle_target(model_out)
        seen_pids: set[str] = set()
        for i, ds in enumerate(dss):
            pid = self.logic.get_dataset_profile_id(ds) or f"__no_profile__:{ds}"
            argv: list[str] = ["./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_out)]
            if is_bundle:
                if pid in seen_pids: argv.extend(["--resume", str(model_out), "--extra-epochs", str(extra), "--out-mode", out_mode])
                else: seen_pids.add(pid)
            elif i > 0:
                argv.extend(["--resume", str(model_out), "--extra-epochs", str(extra), "--out-mode", out_mode])
                if allow_mixed_profiles and self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not is_bundle: argv.append("--allow-profile-mismatch")
            parts.append(" ".join(shlex.quote(x) for x in argv))

        self._append_log(f"\n=== Multi-dataset train: {len(dss)} datasets ===\n")
        for ds in dss: self._append_log(f"  - {ds}\n")
        self._append_log(f"out: {model_out}\n+epochs per subsequent dataset: {extra} (out-mode={out_mode})\n\n")
        self._run_simple_cmd([" && ".join(parts)])

    def _eval_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        
        model_s = self.ui.var_model.get().strip()
        model_in = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_in.exists(): self.ui.show_messagebox("error", "Missing model", f"Model not found:\\n{model_in}\\n\\nRun Train first."); return
        if len(dss) == 1: self._run_simple_cmd(["./.venv/bin/python", "scripts/eval.py", "--data", str(dss[0]), "--model", str(model_in)]); return

        allow_mixed_profiles = False
        if self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_in):
            if not self.ui.ask_yes_no("Allow mixed-profile eval on a single model?", "You selected datasets with different component profiles.\\n\\nRecommended: evaluate a multi-model bundle (.bundle) so each dataset uses its matching checkpoint.\\n\\nIf you continue anyway, eval will pass --allow-profile-mismatch. Results may be meaningless.\\n\\nContinue?"): return
            allow_mixed_profiles = True

        parts: list[str] = []
        for ds in dss:
            argv = ["./.venv/bin/python", "scripts/eval.py", "--data", str(ds), "--model", str(model_in)]
            if allow_mixed_profiles and self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_in): argv.append("--allow-profile-mismatch")
            parts.append(" ".join(shlex.quote(x) for x in argv))
        self._append_log(f"\n=== Eval on {len(dss)} datasets ===\n")
        self._run_simple_cmd([" && ".join(parts)])

    def _continue_train_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return

        model_s = self.ui.var_model.get().strip()
        model_path = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_path.exists(): self.ui.show_messagebox("error", "Missing model", f"Model not found to resume from:\\n{model_path}\\n\\nRun Train first."); return

        extra_s = self.ui.var_continue_epochs.get().strip() or "10"
        try: extra = int(extra_s)
        except Exception: self.ui.show_messagebox("error", "Error", "Extra epochs must be an integer."); return
        if extra < 1: self.ui.show_messagebox("error", "Error", "Extra epochs must be >= 1."); return

        out_mode = self.ui.var_continue_out_mode.get().strip() or "last"
        if out_mode not in ("last", "best"): out_mode = "last"

        allow_mixed_profiles = False
        if len(dss) > 1 and self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_path):
            if not self.ui.ask_yes_no("Allow mixed-profile resume into a single model?", "You selected datasets with different component profiles.\\n\\nRecommended: resume into a multi-model bundle (.bundle).\\n\\nIf you continue anyway, the resume will pass --allow-profile-mismatch. This is not guaranteed to work.\\n\\nContinue?"): return
            allow_mixed_profiles = True

        if len(dss) > 1 and self.logic._is_model_bundle_target(model_path):
            missing: list[str] = []
            for ds in dss:
                if not (pid := self.logic.get_dataset_profile_id(ds)): missing.append(f"{ds} (missing profile metadata)"); continue
                try:
                    if not (ckpt := bundle_checkpoint_path(model_path, pid, kind="best")).exists(): missing.append(f"{pid} (no checkpoint in bundle)")
                except Exception: missing.append(f"{pid} (bundle lookup failed)")
            if missing: self.ui.show_messagebox("error", "Missing bundle checkpoints", "The selected model bundle does not have checkpoints for all selected datasets.\\n\\n" + "\\n".join(missing)); return

        if len(dss) == 1:
            ds = dss[0]
            self._append_log(f"\n=== Continue training: +{extra} epochs (out={out_mode}) ===\n" f"data:  {ds}\n" f"model: {model_path}\n\n")
            self._run_simple_cmd([
                "./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_path),
                "--resume", str(model_path), "--extra-epochs", str(extra), "--out-mode", out_mode,
            ])
            return

        parts: list[str] = []
        for ds in dss:
            argv = ["./.venv/bin/python", "scripts/train.py", "--data", str(ds), "--out", str(model_path), "--resume", str(model_path)]
            if allow_mixed_profiles and self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_path): argv.append("--allow-profile-mismatch")
            argv.extend(["--extra-epochs", str(extra), "--out-mode", out_mode])
            parts.append(" ".join(shlex.quote(x) for x in argv))

        self._append_log(f"\n=== Continue training on {len(dss)} datasets: +{extra} epochs each (out={out_mode}) ===\n" f"model: {model_path}\n")
        for ds in dss: self._append_log(f"data:  {ds}\n")
        self._append_log("\n")

        self._run_simple_cmd([" && ".join(parts)])

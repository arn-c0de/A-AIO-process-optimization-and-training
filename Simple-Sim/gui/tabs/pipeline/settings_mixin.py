"""Persisted settings, image-filter profiles and precise-mode counts of the pipeline tab.

Mixin of :class:`gui.tabs.pipeline.tab.PipelineControlTab`; it relies on the attributes that class sets up
(``ui``, ``logic``, ``state``, ``sim_root`` ...).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any


from gui.components.filter_popup import open_filter_popup
from gui.components.precise_popup import open_precise_popup
from gui.utils.settings_store import SettingsStore, restore_persisted_vars, wire_vars_autosave
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
    config_class_names,
    precise_counts,
    resolve_under,
    write_temp_yaml,
)

from simple_sim.io_utils import load_yaml
import contextlib


class PipelineSettingsMixin:
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
                        # multi_profiles follows the (not persisted) multi-profile checkbox, see
                        # _refresh_precise_classes; a stale list would drive single-profile counts.
                        if k in self._precise_settings and k != "multi_profiles":
                            self._precise_settings[k] = v
        except Exception:
            pass

        try: self._refresh_datasets(); self._on_dataset_multi_toggle(); self._on_dataset_selected()
        except Exception: pass
        # _on_dataset_selected() -> _apply_dataset_settings() derives these from the dataset;
        # put the persisted values back on top.
        restore_persisted_vars(st, [b for b in bindings if b[0] in DATASET_DERIVED_KEYS], skip_empty_strings=True)
        with contextlib.suppress(Exception): self._refresh_models()
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
            with contextlib.suppress(Exception):
                self._precise_popup_ref.refresh(self._precise_settings)

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
        data = load_yaml(cfg_path) or {}
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

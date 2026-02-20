"""Controller helpers for filter popup profile/state operations."""

from __future__ import annotations

import tkinter as tk
from typing import Any, Dict, Optional

from .models import FilterPopupPayload


def refresh_profile_list(popup: Any, select_name: Optional[str] = None) -> None:
    names = sorted(popup.profiles.keys(), key=lambda s: s.lower())
    popup.lb_profiles.delete(0, "end")
    for n in names:
        popup.lb_profiles.insert("end", n)
    target = select_name or popup.active_profile
    if target in names:
        idx = names.index(target)
        popup.lb_profiles.selection_clear(0, "end")
        popup.lb_profiles.selection_set(idx)
        popup.lb_profiles.activate(idx)
    if hasattr(popup, "_populate_left_filter_profiles"):
        popup._populate_left_filter_profiles()


def selected_profile_name(popup: Any) -> Optional[str]:
    sel = popup.lb_profiles.curselection()
    if not sel:
        return None
    try:
        return str(popup.lb_profiles.get(sel[0]))
    except Exception:
        return None


def sync_to_external_vars(popup: Any) -> None:
    raw_values: Dict[str, Any] = {}
    for key, var in popup.filter_vars.items():
        if isinstance(var, tk.BooleanVar):
            raw_values[key] = var.get()
        elif isinstance(var, tk.DoubleVar):
            raw_values[key] = f"{var.get():.2f}"
        elif isinstance(var, tk.StringVar):
            raw_values[key] = var.get()
    payload = FilterPopupPayload.from_mapping(raw_values)
    popup.set_current_values(payload.to_mapping())


def persist_active_profile(popup: Any) -> None:
    if popup.active_profile in popup.profiles:
        sync_to_external_vars(popup)
        popup.profiles[popup.active_profile] = popup.get_current_values()


def update_internal_vars(popup: Any, data: Dict[str, Any]) -> None:
    payload = FilterPopupPayload.from_mapping(data)
    normalized = payload.to_mapping()
    for key, var in popup.filter_vars.items():
        if key not in normalized:
            continue
        value = normalized[key]
        if isinstance(var, tk.BooleanVar):
            if isinstance(value, str):
                var.set(value.strip().lower() in {"1", "true", "yes", "on"})
            else:
                var.set(bool(value))
        elif isinstance(var, (tk.DoubleVar, tk.StringVar)):
            try:
                if isinstance(var, tk.DoubleVar):
                    var.set(float(value))
                else:
                    var.set(str(value))
            except (ValueError, TypeError):
                pass


def load_profile_from_selection(popup: Any, _evt: Optional[tk.Event] = None) -> None:
    name = selected_profile_name(popup)
    if not name:
        return
    persist_active_profile(popup)
    data = popup.profiles.get(name)
    if isinstance(data, dict):
        popup.set_current_values(data)
        popup.active_profile = name
        update_internal_vars(popup, data)
        persist_active_profile(popup)


def new_profile(popup: Any) -> None:
    name = popup.ask_string("New Filter Profile", "Profile name:", "")
    if not name:
        return
    name = name.strip()
    if not name:
        return
    if name in popup.profiles:
        popup.show_messagebox("warning", "Exists", f"Profile '{name}' already exists.")
        return
    sync_to_external_vars(popup)
    popup.profiles[name] = popup.get_current_values()
    popup.active_profile = name
    refresh_profile_list(popup, select_name=name)


def save_profile(popup: Any) -> None:
    name = selected_profile_name(popup)
    if not name:
        popup.show_messagebox("warning", "No selection", "Select a profile first.")
        return
    sync_to_external_vars(popup)
    popup.profiles[name] = popup.get_current_values()
    popup.active_profile = name


def rename_profile(popup: Any) -> None:
    old = selected_profile_name(popup)
    if not old:
        popup.show_messagebox("warning", "No selection", "Select a profile first.")
        return
    new = popup.ask_string("Rename Filter Profile", "New name:", old)
    if not new:
        return
    new = new.strip()
    if not new or new == old:
        return
    if new in popup.profiles:
        popup.show_messagebox("warning", "Exists", f"Profile '{new}' already exists.")
        return
    popup.profiles[new] = popup.profiles.pop(old)
    if popup.active_profile == old:
        popup.active_profile = new
    persist_active_profile(popup)
    refresh_profile_list(popup, select_name=new)


def delete_profile(popup: Any) -> None:
    name = selected_profile_name(popup)
    if not name:
        popup.show_messagebox("warning", "No selection", "Select a profile first.")
        return
    if len(popup.profiles) <= 1:
        popup.show_messagebox("warning", "Blocked", "At least one profile must remain.")
        return
    if not popup.ask_yes_no("Delete Filter Profile", f"Delete profile '{name}'?"):
        return
    popup.profiles.pop(name, None)
    if popup.active_profile == name:
        popup.active_profile = sorted(popup.profiles.keys(), key=lambda s: s.lower())[0]
        data = popup.profiles.get(popup.active_profile, {})
        popup.set_current_values(data)
        update_internal_vars(popup, data)
    persist_active_profile(popup)
    refresh_profile_list(popup, select_name=popup.active_profile)


def reset_defaults(popup: Any) -> None:
    defaults = popup._get_default_values()
    popup.set_current_values(defaults)
    update_internal_vars(popup, defaults)
    persist_active_profile(popup)


def reset_to_one(popup: Any) -> None:
    name = selected_profile_name(popup)
    if not name:
        popup.show_messagebox("warning", "No selection", "Select a profile first.")
        return
    reset_vals = popup._get_default_values()
    popup.profiles[name] = reset_vals
    popup.set_current_values(reset_vals)
    update_internal_vars(popup, reset_vals)
    persist_active_profile(popup)
    refresh_profile_list(popup, select_name=name)


def set_clean_filters(popup: Any) -> None:
    clean = popup._get_clean_values()
    popup.set_current_values(clean)
    update_internal_vars(popup, clean)
    persist_active_profile(popup)


def update_rotation_row_state(popup: Any) -> None:
    if popup._rotation_row_state_cb is not None:
        popup._rotation_row_state_cb()


def handle_close(popup: Any) -> None:
    persist_active_profile(popup)
    popup.on_close(popup.profiles, popup.active_profile)
    popup.top.destroy()

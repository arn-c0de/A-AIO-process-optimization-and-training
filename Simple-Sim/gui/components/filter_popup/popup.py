"""Shared Image Filter Popup Component.

This popup is used by:
- Main GUI pipeline tab (gui/tabs/pipeline/tab.py)
- Debug preview tool (scripts/render_debug_previews.py)

Both tools share the same filter settings via outputs/gui/settings.json.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Set

from simple_sim.generators.filter_settings import (
    clean_filter_values_for_popup,
    default_filter_values_for_popup,
)

from .controller import (
    delete_profile,
    handle_close,
    load_profile_from_selection,
    new_profile,
    persist_active_profile,
    refresh_profile_list,
    rename_profile,
    reset_defaults,
    reset_to_one,
    save_profile,
    selected_profile_name,
    set_clean_filters,
    sync_to_external_vars,
    update_internal_vars,
    update_rotation_row_state,
)
from .preview_panel import (
    build_preview_panel,
    do_render_2d_preview,
    do_render_3d_preview,
    get_preview_filter_values,
    on_preview_profile_selected,
    populate_preview_profiles,
    render_preview,
    show_preview_image,
)
from .ui_custom import add_filter_sections, build_filter_controls
from .ui_realism import build_realism_controls, set_realism_defaults


class FilterPopup:
    """Image filter configuration popup with profile management."""

    def __init__(
        self,
        parent: tk.Widget,
        profiles: Dict[str, Dict[str, Any]],
        active_profile: str,
        on_close: Callable[[Dict[str, Dict[str, Any]], str], None],
        get_current_values: Callable[[], Dict[str, Any]],
        set_current_values: Callable[[Dict[str, Any]], None],
        float_or_default: Callable[[str, float], float],
        show_messagebox: Callable[[str, str, str], None],
        ask_string: Callable[[str, str, str], Optional[str]],
        ask_yes_no: Callable[[str, str], bool],
        sim_root: Optional[str] = None,
    ):
        self.profiles = profiles
        self.active_profile = active_profile
        self.on_close = on_close
        self.get_current_values = get_current_values
        self.set_current_values = set_current_values
        self.float_or_default = float_or_default
        self.show_messagebox = show_messagebox
        self.ask_string = ask_string
        self.ask_yes_no = ask_yes_no
        self._sim_root: Optional[str] = str(sim_root) if sim_root is not None else None

        self._preview_profiles_2d: Set[str] = set()
        self._preview_profiles_3d: Set[str] = set()
        self._preview_is_rendering: bool = False
        self._preview_photo: Any = None

        self.top = tk.Toplevel(parent)
        self.top.title("Image Filters")
        self.top.transient(parent.winfo_toplevel())
        self.top.resizable(True, True)
        min_w = 1310 if self._sim_root else 980
        self.top.minsize(min_w, 640)

        self._build_ui()
        self._bind_mousewheel()
        self.top.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _build_ui(self) -> None:
        root = ttk.Frame(self.top, padding=12)
        root.pack(fill="both", expand=True)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        self._build_profile_list(left)

        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_filter_controls(right)

        if self._sim_root:
            preview_col = ttk.Frame(root)
            preview_col.grid(row=0, column=2, sticky="nsew", padx=(12, 0))
            root.columnconfigure(2, minsize=340)
            self._build_preview_panel(preview_col)

    def _build_profile_list(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Filter Profiles", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.lb_profiles = tk.Listbox(parent, height=10, width=22, exportselection=False)
        self.lb_profiles.pack(fill="y", pady=(6, 6))

        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x")

        ttk.Button(btn_frame, text="New", width=7, command=self._new_profile).pack(side="left")
        ttk.Button(btn_frame, text="Save", width=9, command=self._save_profile).pack(side="left", padx=(4, 0))
        ttk.Button(btn_frame, text="Rename", width=11, command=self._rename_profile).pack(side="left", padx=(4, 0))
        ttk.Button(btn_frame, text="Delete", width=8, command=self._delete_profile).pack(side="left", padx=(4, 0))

        self.lb_profiles.bind("<<ListboxSelect>>", self._load_profile_from_selection)
        self.lb_profiles.bind("<Double-Button-1>", self._load_profile_from_selection)

        self._refresh_profile_list(select_name=self.active_profile)
        self.set_current_values(self.profiles.get(self.active_profile, {}))

    # UI delegates
    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        build_filter_controls(self, parent)

    def _build_realism_controls(self, parent: ttk.Frame, current: Dict[str, Any]) -> None:
        build_realism_controls(self, parent, current)

    def _set_realism_defaults(self) -> None:
        set_realism_defaults(self)

    # Preview delegates
    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        build_preview_panel(self, parent)

    def _populate_preview_profiles(self) -> None:
        populate_preview_profiles(self)

    def _on_preview_profile_selected(self, _evt: Optional[tk.Event] = None) -> None:
        on_preview_profile_selected(self, _evt)

    def _get_preview_filter_values(self) -> Dict[str, Any]:
        return get_preview_filter_values(self)

    def _render_preview(self) -> None:
        render_preview(self)

    def _do_render_2d_preview(self, profile_id: str, filter_vals: Dict[str, Any]) -> None:
        do_render_2d_preview(self, profile_id, filter_vals)

    def _do_render_3d_preview(self, profile_id: str, filter_vals: Dict[str, Any]) -> None:
        do_render_3d_preview(self, profile_id, filter_vals)

    def _show_preview_image(self, img_bgr: Any, profile_id: str = "", backend: str = "2D") -> None:
        show_preview_image(self, img_bgr, profile_id, backend)

    # Scrolling
    def _bind_mousewheel(self) -> None:
        self.top.bind("<MouseWheel>", self._on_mousewheel)
        self.top.bind("<Button-4>", self._on_mousewheel)
        self.top.bind("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event: tk.Event) -> str:
        canvas = getattr(self, "_scroll_canvas", None)
        if canvas is None:
            return "break"
        if getattr(event, "num", None) == 4:
            canvas.yview_scroll(-1, "units")
            return "break"
        if getattr(event, "num", None) == 5:
            canvas.yview_scroll(1, "units")
            return "break"
        delta = int(getattr(event, "delta", 0))
        if delta != 0:
            canvas.yview_scroll(-1 if delta > 0 else 1, "units")
        return "break"

    # Custom/Realism tab helpers
    def _add_filter_sections(self, parent: ttk.Frame, current: Dict[str, Any]) -> None:
        add_filter_sections(self, parent, current)

    # Controller delegates
    def _refresh_profile_list(self, select_name: Optional[str] = None) -> None:
        refresh_profile_list(self, select_name)

    def _selected_profile_name(self) -> Optional[str]:
        return selected_profile_name(self)

    def _persist_active_profile(self) -> None:
        persist_active_profile(self)

    def _sync_to_external_vars(self) -> None:
        sync_to_external_vars(self)

    def _update_internal_vars(self, data: Dict[str, Any]) -> None:
        update_internal_vars(self, data)

    def _load_profile_from_selection(self, _evt: Optional[tk.Event] = None) -> None:
        load_profile_from_selection(self, _evt)

    def _new_profile(self) -> None:
        new_profile(self)

    def _save_profile(self) -> None:
        save_profile(self)

    def _rename_profile(self) -> None:
        rename_profile(self)

    def _delete_profile(self) -> None:
        delete_profile(self)

    def _reset_defaults(self) -> None:
        reset_defaults(self)

    def _reset_to_one(self) -> None:
        reset_to_one(self)

    def _set_clean_filters(self) -> None:
        set_clean_filters(self)

    def _get_default_values(self) -> Dict[str, Any]:
        return default_filter_values_for_popup()

    def _get_clean_values(self) -> Dict[str, Any]:
        return clean_filter_values_for_popup()

    def _update_rotation_row_state(self) -> None:
        update_rotation_row_state(self)

    def _handle_close(self) -> None:
        handle_close(self)


def open_filter_popup(
    parent: tk.Widget,
    profiles: Dict[str, Dict[str, Any]],
    active_profile: str,
    on_close: Callable[[Dict[str, Dict[str, Any]], str], None],
    get_current_values: Callable[[], Dict[str, Any]],
    set_current_values: Callable[[Dict[str, Any]], None],
    float_or_default: Callable[[str, float], float],
    show_messagebox: Callable[[str, str, str], None],
    ask_string: Callable[[str, str, str], Optional[str]],
    ask_yes_no: Callable[[str, str], bool],
    sim_root: Optional[str] = None,
) -> None:
    FilterPopup(
        parent=parent,
        profiles=profiles,
        active_profile=active_profile,
        on_close=on_close,
        get_current_values=get_current_values,
        set_current_values=set_current_values,
        float_or_default=float_or_default,
        show_messagebox=show_messagebox,
        ask_string=ask_string,
        ask_yes_no=ask_yes_no,
        sim_root=sim_root,
    )

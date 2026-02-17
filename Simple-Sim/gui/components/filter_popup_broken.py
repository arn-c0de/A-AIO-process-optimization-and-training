"""Shared Image Filter Popup Component - V2 with proper variable sync.

This popup is used by:
- Main GUI pipeline tab (gui/tabs/pipeline/tab.py)
- Debug preview tool (scripts/render_debug_previews.py)

Both tools share the same filter settings via outputs/gui/settings.json.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, Callable


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
) -> None:
    """Open filter popup window.

    Args:
        parent: Parent widget
        profiles: Filter profiles dict (name -> settings)
        active_profile: Currently active profile name
        on_close: Callback(profiles, active_profile) when popup closes
        get_current_values: Function to get current filter values as dict
        set_current_values: Function to apply filter values from dict
        float_or_default: Function to parse float with default
        show_messagebox: Function to show message box (type, title, message)
        ask_string: Function to ask for string input (title, prompt, initial)
        ask_yes_no: Function to ask yes/no question (title, question)
    """
    # Track active profile
    current_active = [active_profile]  # Use list to allow modification in nested functions

    # Create popup window
    top = tk.Toplevel(parent)
    top.title("Image Filters")
    top.transient(parent.winfo_toplevel())
    top.grab_set()
    top.resizable(False, False)

    root = ttk.Frame(top, padding=12)
    root.pack(fill="both", expand=True)
    root.columnconfigure(1, weight=1)

    # Left: Profile list
    left = ttk.Frame(root)
    left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))

    # Right: Filter controls
    right = ttk.Frame(root)
    right.grid(row=0, column=1, sticky="nsew")

    # Build profile list
    ttk.Label(left, text="Filter Profiles", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    lb_profiles = tk.Listbox(left, height=10, width=22, exportselection=False)
    lb_profiles.pack(fill="y", pady=(6, 6))

    btn_frame = ttk.Frame(left)
    btn_frame.pack(fill="x")

    def _refresh_profile_list(select_name: Optional[str] = None) -> None:
        """Refresh profile list."""
        names = sorted(profiles.keys(), key=lambda s: s.lower())
        lb_profiles.delete(0, "end")
        for n in names:
            lb_profiles.insert("end", n)
        target = select_name or current_active[0]
        if target in names:
            idx = names.index(target)
            lb_profiles.selection_clear(0, "end")
            lb_profiles.selection_set(idx)
            lb_profiles.activate(idx)

    def _selected_profile_name() -> Optional[str]:
        """Get selected profile name."""
        sel = lb_profiles.curselection()
        if not sel:
            return None
        try:
            return str(lb_profiles.get(sel[0]))
        except Exception:
            return None

    def _persist_active_profile() -> None:
        """Save current filter values to active profile."""
        if current_active[0] in profiles:
            profiles[current_active[0]] = get_current_values()

    def _load_profile_from_selection(_evt: Optional[tk.Event] = None) -> None:
        """Load selected profile."""
        name = _selected_profile_name()
        if not name:
            return
        _persist_active_profile()
        data = profiles.get(name)
        if isinstance(data, dict):
            set_current_values(data)
            current_active[0] = name

    def _new_profile() -> None:
        """Create new profile."""
        name = ask_string("New Filter Profile", "Profile name:", "")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in profiles:
            show_messagebox("warning", "Exists", f"Profile '{name}' already exists.")
            return
        profiles[name] = get_current_values()
        current_active[0] = name
        _refresh_profile_list(select_name=name)

    def _save_profile() -> None:
        """Save current values to selected profile."""
        name = _selected_profile_name()
        if not name:
            show_messagebox("warning", "No selection", "Select a profile first.")
            return
        profiles[name] = get_current_values()
        current_active[0] = name

    def _rename_profile() -> None:
        """Rename selected profile."""
        old = _selected_profile_name()
        if not old:
            show_messagebox("warning", "No selection", "Select a profile first.")
            return
        new = ask_string("Rename Filter Profile", "New name:", old)
        if not new:
            return
        new = new.strip()
        if not new or new == old:
            return
        if new in profiles:
            show_messagebox("warning", "Exists", f"Profile '{new}' already exists.")
            return
        profiles[new] = profiles.pop(old)
        if current_active[0] == old:
            current_active[0] = new
        _refresh_profile_list(select_name=new)

    def _delete_profile() -> None:
        """Delete selected profile."""
        name = _selected_profile_name()
        if not name:
            show_messagebox("warning", "No selection", "Select a profile first.")
            return
        if len(profiles) <= 1:
            show_messagebox("warning", "Blocked", "At least one profile must remain.")
            return
        if not ask_yes_no("Delete Filter Profile", f"Delete profile '{name}'?"):
            return
        profiles.pop(name, None)
        if current_active[0] == name:
            current_active[0] = sorted(profiles.keys(), key=lambda s: s.lower())[0]
            set_current_values(profiles.get(current_active[0], {}))
        _refresh_profile_list(select_name=current_active[0])

    def _reset_defaults() -> None:
        """Reset all filters to default values."""
        defaults = {
            # Existing filters
            "cardinal_rotation_90": True,
            "enable_rotation": True,
            "enable_blur": True,
            "enable_grain": True,
            "enable_brightness": True,
            "enable_contrast": True,
            "rotation_strength": "1.00",
            "blur_strength": "1.00",
            "grain_strength": "1.00",
            "brightness_strength": "1.00",
            "contrast_strength": "1.00",
            # High priority (enabled)
            "enable_perspective": True,
            "perspective_strength": "1.00",
            "enable_motion_blur": True,
            "motion_blur_strength": "1.00",
            "enable_saturation": True,
            "saturation_factor": "1.00",
            "enable_hue_shift": True,
            "hue_shift_deg": "0.00",
            "enable_shadow": True,
            "shadow_strength": "0.30",
            "enable_reflection": True,
            "reflection_strength": "0.50",
            # Medium/Low priority (disabled)
            "enable_vignetting": False,
            "vignetting_strength": "1.00",
            "enable_chromatic_aberration": False,
            "chromatic_strength": "1.00",
            "enable_jpeg_compression": False,
            "jpeg_quality": "85",
            "enable_color_temperature": False,
            "color_temperature_kelvin": "5500",
            "enable_lens_distortion": False,
            "distortion_k1": "0.00",
            "enable_dust": False,
            "dust_density": "0.30",
            "enable_sharpen": False,
            "sharpen_strength": "1.00",
        }
        set_current_values(defaults)

    def _reset_to_one() -> None:
        """Reset selected profile to all 1.0."""
        name = _selected_profile_name()
        if not name:
            show_messagebox("warning", "No selection", "Select a profile first.")
            return
        reset_vals = {
            # Existing filters
            "cardinal_rotation_90": True,
            "enable_rotation": True,
            "enable_blur": True,
            "enable_grain": True,
            "enable_brightness": True,
            "enable_contrast": True,
            "rotation_strength": "1.00",
            "blur_strength": "1.00",
            "grain_strength": "1.00",
            "brightness_strength": "1.00",
            "contrast_strength": "1.00",
            # High priority (enabled)
            "enable_perspective": True,
            "perspective_strength": "1.00",
            "enable_motion_blur": True,
            "motion_blur_strength": "1.00",
            "enable_saturation": True,
            "saturation_factor": "1.00",
            "enable_hue_shift": True,
            "hue_shift_deg": "0.00",
            "enable_shadow": True,
            "shadow_strength": "0.30",
            "enable_reflection": True,
            "reflection_strength": "0.50",
            # Medium/Low priority (disabled)
            "enable_vignetting": False,
            "vignetting_strength": "1.00",
            "enable_chromatic_aberration": False,
            "chromatic_strength": "1.00",
            "enable_jpeg_compression": False,
            "jpeg_quality": "85",
            "enable_color_temperature": False,
            "color_temperature_kelvin": "5500",
            "enable_lens_distortion": False,
            "distortion_k1": "0.00",
            "enable_dust": False,
            "dust_density": "0.30",
            "enable_sharpen": False,
            "sharpen_strength": "1.00",
        }
        profiles[name] = reset_vals
        set_current_values(reset_vals)
        _refresh_profile_list(select_name=name)

    ttk.Button(btn_frame, text="New", width=7, command=_new_profile).pack(side="left")
    ttk.Button(btn_frame, text="Save", width=9, command=_save_profile).pack(side="left", padx=(4, 0))
    ttk.Button(btn_frame, text="Rename", width=11, command=_rename_profile).pack(side="left", padx=(4, 0))
    ttk.Button(btn_frame, text="Delete", width=8, command=_delete_profile).pack(side="left", padx=(4, 0))

    lb_profiles.bind("<<ListboxSelect>>", _load_profile_from_selection)
    lb_profiles.bind("<Double-Button-1>", _load_profile_from_selection)

    _refresh_profile_list(select_name=current_active[0])

    # Build filter controls (simplified - just show message)
    ttk.Label(right, text="Filter controls are managed by parent",
             font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    ttk.Label(right, text="Profile changes apply to the parent window controls.").pack(anchor="w", pady=(2, 8))

    # Bottom buttons
    btns = ttk.Frame(right)
    btns.pack(fill="x", pady=(10, 0))

    ttk.Button(btns, text="Reset Defaults", command=_reset_defaults).pack(side="left")
    ttk.Button(btns, text="Reset to 1.0", command=_reset_to_one).pack(side="left", padx=(6, 0))
    
    def _handle_close() -> None:
        """Handle popup close."""
        _persist_active_profile()
        on_close(profiles, current_active[0])
        top.destroy()
    
    ttk.Button(btns, text="Close", command=_handle_close).pack(side="right")
    top.protocol("WM_DELETE_WINDOW", _handle_close)

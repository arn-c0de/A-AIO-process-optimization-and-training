"""Shared Image Filter Popup Component.

This popup is used by:
- Main GUI pipeline tab (gui/tabs/pipeline/tab.py)
- Debug preview tool (scripts/render_debug_previews.py)

Both tools share the same filter settings via outputs/gui/settings.json.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, Optional, Callable


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
    ):
        """Initialize filter popup.

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
        self.profiles = profiles
        self.active_profile = active_profile
        self.on_close = on_close
        self.get_current_values = get_current_values
        self.set_current_values = set_current_values
        self.float_or_default = float_or_default
        self.show_messagebox = show_messagebox
        self.ask_string = ask_string
        self.ask_yes_no = ask_yes_no

        # Create popup window
        self.top = tk.Toplevel(parent)
        self.top.title("Image Filters")
        self.top.transient(parent.winfo_toplevel())
        # Don't grab_set() so user can interact with main window
        self.top.resizable(False, False)

        # Build UI
        self._build_ui()

        # Set up close handler
        self.top.protocol("WM_DELETE_WINDOW", self._handle_close)

    def _build_ui(self) -> None:
        """Build the popup UI."""
        root = ttk.Frame(self.top, padding=12)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)

        # Left: Profile list
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        self._build_profile_list(left)

        # Right: Filter controls
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_filter_controls(right)

    def _build_profile_list(self, parent: ttk.Frame) -> None:
        """Build profile list UI."""
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

    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        """Build filter controls UI."""
        ttk.Label(parent, text="Apply filters during image generation",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Configure strength and enable/disable filters below.").pack(anchor="w", pady=(2, 8))

        # Get current filter values
        current = self.get_current_values()

        # Cardinal rotation toggles
        row0 = ttk.Frame(parent)
        row0.pack(fill="x", pady=(0, 8))

        self.var_cardinal = tk.BooleanVar(value=current.get("cardinal_rotation_90", True))
        self.var_rotation = tk.BooleanVar(value=current.get("enable_rotation", True))

        ttk.Checkbutton(row0, text="Enable 90° base rotation", variable=self.var_cardinal).pack(side="left")
        ttk.Checkbutton(row0, text="Enable rotation jitter", variable=self.var_rotation).pack(side="left", padx=(12, 0))

        # Scrollable filter controls
        canvas = tk.Canvas(parent, height=400)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        controls_frame = ttk.Frame(canvas)

        controls_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=controls_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Store filter variables
        self.filter_vars: Dict[str, Any] = {
            "cardinal_rotation_90": self.var_cardinal,
            "enable_rotation": self.var_rotation,
        }

        # Add filter rows
        self._add_filter_sections(controls_frame, current)

        # Bottom buttons
        btns = ttk.Frame(parent)
        btns.pack(fill="x", pady=(10, 0))

        ttk.Button(btns, text="Reset Defaults", command=self._reset_defaults).pack(side="left")
        ttk.Button(btns, text="Reset to 1.0", command=self._reset_to_one).pack(side="left", padx=(6, 0))
        ttk.Button(btns, text="Close", command=self._handle_close).pack(side="right")

    def _add_filter_sections(self, parent: ttk.Frame, current: Dict[str, Any]) -> None:
        """Add all filter sections with controls."""

        def add_section(title: str) -> None:
            ttk.Label(parent, text=title, font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(6, 3))

        def add_filter_row(
            label: str,
            key_enable: str,
            key_strength: str,
            default_enable: bool,
            default_strength: float,
            min_v: float = 0.0,
            max_v: float = 2.0,
        ) -> None:
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=(3, 0))

            var_enable = tk.BooleanVar(value=current.get(key_enable, default_enable))
            var_strength = tk.DoubleVar(value=self.float_or_default(str(current.get(key_strength, default_strength)), default_strength))

            ttk.Checkbutton(row, text=label, variable=var_enable).pack(side="left")

            scale = ttk.Scale(row, from_=min_v, to=max_v, orient="horizontal",
                            variable=var_strength, length=210)
            scale.pack(side="left", padx=(10, 6))

            lbl = ttk.Label(row, width=5, anchor="e")
            lbl.pack(side="left")

            def update_label(*args):
                v = max(min_v, min(max_v, var_strength.get()))
                lbl.configure(text=f"{v:.2f}")

            var_strength.trace_add("write", lambda *args: update_label())
            update_label()

            self.filter_vars[key_enable] = var_enable
            self.filter_vars[key_strength] = var_strength

        # Existing filters
        add_section("━━━ Existing Filters ━━━")
        add_filter_row("Blur", "enable_blur", "blur_strength", True, 1.0)
        add_filter_row("Grain", "enable_grain", "grain_strength", True, 1.0, max_v=3.0)
        add_filter_row("Brightness", "enable_brightness", "brightness_strength", True, 1.0, max_v=2.0)
        add_filter_row("Contrast", "enable_contrast", "contrast_strength", True, 1.0, max_v=2.0)
        add_filter_row("Rotation", "enable_rotation", "rotation_strength", True, 1.0, max_v=2.0)

        # High priority
        add_section("━━━ High Priority (enabled) ━━━")
        add_filter_row("Perspective Transform", "enable_perspective", "perspective_strength", True, 1.0, max_v=2.0)
        add_filter_row("Motion Blur", "enable_motion_blur", "motion_blur_strength", True, 1.0, max_v=3.0)
        add_filter_row("Saturation", "enable_saturation", "saturation_factor", True, 1.0, min_v=0.5, max_v=1.5)
        add_filter_row("Hue Shift", "enable_hue_shift", "hue_shift_deg", True, 0.0, min_v=-30.0, max_v=30.0)
        add_filter_row("Shadow", "enable_shadow", "shadow_strength", True, 0.3, min_v=0.0, max_v=0.8)
        add_filter_row("Reflection/Glare", "enable_reflection", "reflection_strength", True, 0.5, min_v=0.0, max_v=1.0)

        # Medium priority
        add_section("━━━ Medium Priority (disabled) ━━━")
        add_filter_row("Vignetting", "enable_vignetting", "vignetting_strength", False, 1.0, max_v=2.0)
        add_filter_row("Chromatic Aberration", "enable_chromatic_aberration", "chromatic_strength", False, 1.0, max_v=2.0)
        add_filter_row("JPEG Quality", "enable_jpeg_compression", "jpeg_quality", False, 85.0, min_v=50.0, max_v=95.0)
        add_filter_row("Color Temperature", "enable_color_temperature", "color_temperature_kelvin", False, 5500.0, min_v=2500.0, max_v=7500.0)

        # Low priority
        add_section("━━━ Low Priority (disabled) ━━━")
        add_filter_row("Lens Distortion K1", "enable_lens_distortion", "distortion_k1", False, 0.0, min_v=-0.3, max_v=0.3)
        add_filter_row("Dust Particles", "enable_dust", "dust_density", False, 0.3, min_v=0.0, max_v=1.0)
        add_filter_row("Sharpen", "enable_sharpen", "sharpen_strength", False, 1.0, max_v=2.0)

    def _refresh_profile_list(self, select_name: Optional[str] = None) -> None:
        """Refresh profile list."""
        names = sorted(self.profiles.keys(), key=lambda s: s.lower())
        self.lb_profiles.delete(0, "end")
        for n in names:
            self.lb_profiles.insert("end", n)
        target = select_name or self.active_profile
        if target in names:
            idx = names.index(target)
            self.lb_profiles.selection_clear(0, "end")
            self.lb_profiles.selection_set(idx)
            self.lb_profiles.activate(idx)

    def _selected_profile_name(self) -> Optional[str]:
        """Get selected profile name."""
        sel = self.lb_profiles.curselection()
        if not sel:
            return None
        try:
            return str(self.lb_profiles.get(sel[0]))
        except Exception:
            return None

    def _persist_active_profile(self) -> None:
        """Save current filter values to active profile."""
        if self.active_profile in self.profiles:
            # First sync internal popup vars to external vars
            self._sync_to_external_vars()
            # Then get the values and save to profile
            self.profiles[self.active_profile] = self.get_current_values()

    def _sync_to_external_vars(self) -> None:
        """Sync internal popup variables to external variables via set_current_values."""
        current_popup_values = {}
        for key, var in self.filter_vars.items():
            if isinstance(var, tk.BooleanVar):
                current_popup_values[key] = var.get()
            elif isinstance(var, tk.DoubleVar):
                current_popup_values[key] = f"{var.get():.2f}"
            elif isinstance(var, tk.StringVar):
                current_popup_values[key] = var.get()
        # Push to external variables
        self.set_current_values(current_popup_values)

    def _update_internal_vars(self, data: Dict[str, Any]) -> None:
        """Update internal popup variables from profile data."""
        for key, var in self.filter_vars.items():
            if key in data:
                value = data[key]
                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                elif isinstance(var, (tk.DoubleVar, tk.StringVar)):
                    try:
                        if isinstance(var, tk.DoubleVar):
                            var.set(float(value))
                        else:
                            var.set(str(value))
                    except (ValueError, TypeError):
                        pass

    def _load_profile_from_selection(self, _evt: Optional[tk.Event] = None) -> None:
        """Load selected profile."""
        name = self._selected_profile_name()
        if not name:
            return
        self._persist_active_profile()
        data = self.profiles.get(name)
        if isinstance(data, dict):
            self.set_current_values(data)
            self.active_profile = name
            # Also update internal popup variables
            self._update_internal_vars(data)
            self._persist_active_profile()

    def _new_profile(self) -> None:
        """Create new profile."""
        name = self.ask_string("New Filter Profile", "Profile name:", "")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        if name in self.profiles:
            self.show_messagebox("warning", "Exists", f"Profile '{name}' already exists.")
            return
        # Sync popup values to external vars first
        self._sync_to_external_vars()
        self.profiles[name] = self.get_current_values()
        self.active_profile = name
        self._refresh_profile_list(select_name=name)

    def _save_profile(self) -> None:
        """Save current values to selected profile."""
        name = self._selected_profile_name()
        if not name:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        # Sync popup values to external vars first
        self._sync_to_external_vars()
        self.profiles[name] = self.get_current_values()
        self.active_profile = name

    def _rename_profile(self) -> None:
        """Rename selected profile."""
        old = self._selected_profile_name()
        if not old:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        new = self.ask_string("Rename Filter Profile", "New name:", old)
        if not new:
            return
        new = new.strip()
        if not new or new == old:
            return
        if new in self.profiles:
            self.show_messagebox("warning", "Exists", f"Profile '{new}' already exists.")
            return
        self.profiles[new] = self.profiles.pop(old)
        if self.active_profile == old:
            self.active_profile = new
        self._persist_active_profile()
        self._refresh_profile_list(select_name=new)

    def _delete_profile(self) -> None:
        """Delete selected profile."""
        name = self._selected_profile_name()
        if not name:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        if len(self.profiles) <= 1:
            self.show_messagebox("warning", "Blocked", "At least one profile must remain.")
            return
        if not self.ask_yes_no("Delete Filter Profile", f"Delete profile '{name}'?"):
            return
        self.profiles.pop(name, None)
        if self.active_profile == name:
            self.active_profile = sorted(self.profiles.keys(), key=lambda s: s.lower())[0]
            data = self.profiles.get(self.active_profile, {})
            self.set_current_values(data)
            self._update_internal_vars(data)
        self._persist_active_profile()
        self._refresh_profile_list(select_name=self.active_profile)

    def _reset_defaults(self) -> None:
        """Reset all filters to default values."""
        defaults = self._get_default_values()
        self.set_current_values(defaults)
        self._update_internal_vars(defaults)
        self._persist_active_profile()

    def _reset_to_one(self) -> None:
        """Reset selected profile to all 1.0."""
        name = self._selected_profile_name()
        if not name:
            self.show_messagebox("warning", "No selection", "Select a profile first.")
            return
        reset_vals = self._get_default_values()
        self.profiles[name] = reset_vals
        self.set_current_values(reset_vals)
        self._update_internal_vars(reset_vals)
        self._persist_active_profile()
        self._refresh_profile_list(select_name=name)

    def _get_default_values(self) -> Dict[str, Any]:
        """Get default filter values."""
        return {
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

    def _handle_close(self) -> None:
        """Handle popup close."""
        self._persist_active_profile()
        self.on_close(self.profiles, self.active_profile)
        self.top.destroy()


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
    """Open filter popup (convenience function).

    Args:
        parent: Parent widget
        profiles: Filter profiles dict
        active_profile: Active profile name
        on_close: Callback when closed
        get_current_values: Get current filter values
        set_current_values: Set filter values
        float_or_default: Parse float with default
        show_messagebox: Show message box
        ask_string: Ask for string input
        ask_yes_no: Ask yes/no question
    """
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
    )

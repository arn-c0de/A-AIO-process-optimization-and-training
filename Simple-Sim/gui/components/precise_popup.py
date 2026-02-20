"""Precise Mode Configuration Popup."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional


def open_precise_popup(
    parent: tk.Widget,
    settings: Dict[str, Any],
    on_apply: Callable[[Dict[str, Any]], None],
    get_class_names_func: Callable[[], List[str]],
) -> "PrecisePopup":
    """Open the precise mode settings popup and return the instance.

    The returned instance can be used to push live updates into the popup
    while it is open (e.g. via :meth:`PrecisePopup.refresh`).
    """
    return PrecisePopup(parent, settings, on_apply, get_class_names_func)


class PrecisePopup:
    """Toplevel popup for configuring per-class / per-profile sample counts."""

    # Expected keys in the *settings* dict with their defaults:
    _DEFAULTS: Dict[str, Any] = {
        "total": 100,
        "per_class": False,
        "class_names": ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"],
        "classes": {},           # class_name -> int
        "multi_profiles": [],    # list[str]; non-empty → multi-profile mode
        "multi_totals": {},      # profile_id -> int
        "multi_classes": {},     # profile_id -> {class_name -> int}
    }

    def __init__(
        self,
        parent: tk.Widget,
        settings: Dict[str, Any],
        on_apply: Callable[[Dict[str, Any]], None],
        get_class_names_func: Callable[[], List[str]],
    ) -> None:
        self._on_apply = on_apply
        self._get_class_names = get_class_names_func

        # Deep-copy the incoming settings so we never mutate the caller's dict
        self._s: Dict[str, Any] = {}
        for k, default in self._DEFAULTS.items():
            val = settings.get(k, default)
            if isinstance(default, dict):
                self._s[k] = dict(val) if isinstance(val, dict) else {}
            elif isinstance(default, list):
                self._s[k] = list(val) if isinstance(val, list) else list(default)
            else:
                self._s[k] = val
        # Nested dict-of-dicts needs one more level of copy
        self._s["multi_classes"] = {
            pid: dict(v)
            for pid, v in (settings.get("multi_classes") or {}).items()
        }

        # Tkinter control variables
        self._var_total = tk.StringVar(value=str(self._s["total"]))
        self._var_per_class = tk.BooleanVar(value=bool(self._s["per_class"]))

        # Per-content widget var caches (rebuilt on each _rebuild_content call)
        self._class_vars: Dict[str, tk.StringVar] = {}
        self._multi_total_vars: Dict[str, tk.StringVar] = {}
        self._multi_class_vars: Dict[str, Dict[str, tk.StringVar]] = {}

        self._content_frame: Optional[ttk.Frame] = None
        self._win = tk.Toplevel(parent)
        self._win.title("Precise Mode Settings")
        self._win.resizable(True, True)
        self._win.protocol("WM_DELETE_WINDOW", self._win.destroy)
        self._build_window()

    # ------------------------------------------------------------------
    # Public live-update API
    # ------------------------------------------------------------------

    def is_open(self) -> bool:
        """Return True if the popup window still exists."""
        try:
            return bool(self._win.winfo_exists())
        except Exception:
            return False

    def refresh(self, settings: Dict[str, Any]) -> None:
        """Push updated settings into the open popup and rebuild its content.

        Called externally when e.g. multi-profile is toggled while the popup
        is already visible.
        """
        if not self.is_open():
            return
        self._flush_widget_values()
        # Merge in the new top-level keys that change the layout
        for k in ("multi_profiles", "class_names", "multi_totals", "multi_classes"):
            if k in settings:
                self._s[k] = settings[k]
        # Ensure every new profile has at least empty entries
        default = self._s.get("total", 100)
        for pid in self._s.get("multi_profiles", []):
            self._s["multi_totals"].setdefault(pid, default)
            self._s["multi_classes"].setdefault(pid, {})
        self._rebuild_window()

    def _rebuild_window(self) -> None:
        """Destroy and rebuild all window widgets in-place."""
        for w in self._win.winfo_children():
            w.destroy()
        self._content_frame = None
        # Re-create control vars bound to the (still existing) window
        self._var_total = tk.StringVar(value=str(self._s["total"]))
        self._var_per_class = tk.BooleanVar(value=bool(self._s["per_class"]))
        self._build_window()

    # ------------------------------------------------------------------
    # Window construction
    # ------------------------------------------------------------------

    def _build_window(self) -> None:
        outer = ttk.Frame(self._win, padding=10)
        outer.pack(fill="both", expand=True)

        multi = bool(self._s["multi_profiles"])

        # Controls row ---------------------------------------------------
        ctrl = ttk.Frame(outer)
        ctrl.pack(fill="x", pady=(0, 8))

        if not multi:
            ttk.Label(ctrl, text="Samples per class:").pack(side="left")
            ttk.Entry(ctrl, textvariable=self._var_total, width=7).pack(side="left", padx=(4, 0))

        ttk.Checkbutton(
            ctrl, text="Per class",
            variable=self._var_per_class,
            command=self._on_per_class_toggle,
        ).pack(side="left", padx=(12 if not multi else 0, 0))

        ttk.Button(ctrl, text="↻", width=3, command=self._on_reload).pack(side="left", padx=(8, 0))

        # Dynamic content area -------------------------------------------
        self._content_frame = ttk.Frame(outer)
        self._content_frame.pack(fill="both", expand=True)
        self._rebuild_content()

        # OK / Cancel ----------------------------------------------------
        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=(10, 6))
        btns = ttk.Frame(outer)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", width=10, command=self._win.destroy).pack(side="right")
        ttk.Button(btns, text="OK", width=10, command=self._on_ok).pack(side="right", padx=(0, 6))

        self._win.minsize(380, 160)
        self._win.update_idletasks()

    # ------------------------------------------------------------------
    # Dynamic content
    # ------------------------------------------------------------------

    def _rebuild_content(self) -> None:
        if self._content_frame is None:
            return
        for w in self._content_frame.winfo_children():
            w.destroy()

        self._class_vars = {}
        self._multi_total_vars = {}
        self._multi_class_vars = {}

        multi = bool(self._s["multi_profiles"])
        per_class = self._var_per_class.get()
        class_names: List[str] = self._s["class_names"] or ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]
        default = str(self._s["total"])

        if multi:
            self._build_multi_table(class_names, per_class, default)
        elif per_class:
            self._build_single_per_class(class_names, default)

    def _build_single_per_class(self, class_names: List[str], default: str) -> None:
        assert self._content_frame is not None
        classes = self._s["classes"]
        for cls in class_names:
            val = str(classes.get(cls, default))
            var = tk.StringVar(value=val)
            self._class_vars[cls] = var
            row = ttk.Frame(self._content_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f"{cls}:", width=16).pack(side="left")
            ttk.Entry(row, textvariable=var, width=8).pack(side="left")

    def _build_multi_table(
        self, class_names: List[str], per_class: bool, default: str
    ) -> None:
        assert self._content_frame is not None
        profile_ids: List[str] = self._s["multi_profiles"]
        multi_totals: Dict[str, Any] = self._s["multi_totals"]
        multi_classes: Dict[str, Any] = self._s["multi_classes"]

        # Header row
        hdr = ttk.Frame(self._content_frame)
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Profile", width=22, font=("TkDefaultFont", 9, "bold")).pack(side="left")
        if per_class:
            for cls in class_names:
                ttk.Label(hdr, text=cls, width=8, font=("TkDefaultFont", 9, "bold")).pack(side="left")
        else:
            ttk.Label(hdr, text="Count", width=8, font=("TkDefaultFont", 9, "bold")).pack(side="left")

        ttk.Separator(self._content_frame, orient="horizontal").pack(fill="x", pady=(2, 4))

        for pid in profile_ids:
            label = pid if len(pid) <= 22 else pid[:20] + "…"
            row = ttk.Frame(self._content_frame)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=22).pack(side="left")

            if per_class:
                cls_dict: Dict[str, tk.StringVar] = {}
                for cls in class_names:
                    val = str((multi_classes.get(pid) or {}).get(cls, default))
                    var = tk.StringVar(value=val)
                    cls_dict[cls] = var
                    ttk.Entry(row, textvariable=var, width=8).pack(side="left")
                self._multi_class_vars[pid] = cls_dict
            else:
                val = str(multi_totals.get(pid, default))
                var = tk.StringVar(value=val)
                self._multi_total_vars[pid] = var
                ttk.Entry(row, textvariable=var, width=8).pack(side="left")

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_per_class_toggle(self) -> None:
        self._flush_widget_values()
        self._s["per_class"] = self._var_per_class.get()
        self._rebuild_content()

    def _on_reload(self) -> None:
        self._flush_widget_values()
        names = self._get_class_names()
        if names:
            self._s["class_names"] = names
        self._rebuild_content()

    def _on_ok(self) -> None:
        self._flush_widget_values()
        self._on_apply(dict(self._s))
        self._win.destroy()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush_widget_values(self) -> None:
        """Write current Entry/Var values back into _s before any rebuild."""
        try:
            self._s["total"] = int(self._var_total.get())
        except Exception:
            pass
        self._s["per_class"] = self._var_per_class.get()

        # Single per-class
        for cls, var in self._class_vars.items():
            try:
                self._s["classes"][cls] = int(var.get())
            except Exception:
                pass

        # Multi totals
        for pid, var in self._multi_total_vars.items():
            try:
                self._s["multi_totals"][pid] = int(var.get())
            except Exception:
                pass

        # Multi per-class
        for pid, cls_dict in self._multi_class_vars.items():
            entry = self._s["multi_classes"].setdefault(pid, {})
            for cls, var in cls_dict.items():
                try:
                    entry[cls] = int(var.get())
                except Exception:
                    pass

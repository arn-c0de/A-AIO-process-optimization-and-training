"""Realism tab UI helpers for the filter popup."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict

from .constants import DEFAULT_REALISM_PROFILE_ID, REALISM_GROUP_DEFAULTS, REALISM_K_DEFAULTS


def build_realism_controls(popup: Any, parent: ttk.Frame, current: Dict[str, Any]) -> None:
    """Build realism-mode controls."""
    top = ttk.Frame(parent)
    top.pack(fill="x", pady=(2, 6))
    ttk.Label(top, text="Grouped realism sampling (G0 always mild, sparse group mixing)."
              ).pack(anchor="w")

    grid = ttk.Frame(parent)
    grid.pack(fill="x", pady=(0, 8))

    def _f(key: str, default: float) -> tk.DoubleVar:
        return tk.DoubleVar(value=popup.float_or_default(str(current.get(key, default)), default))

    def _add_row(row: int, label: str, key: str, var: tk.DoubleVar) -> None:
        ttk.Label(grid, text=label, width=24).grid(row=row, column=0, sticky="w", pady=2)
        ent = ttk.Entry(grid, textvariable=var, width=10)
        ent.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        popup.filter_vars[key] = var

    popup.var_realism_constraints_enabled = tk.BooleanVar(
        value=bool(current.get("realism_constraints_enabled", True))
    )
    popup.filter_vars["realism_constraints_enabled"] = popup.var_realism_constraints_enabled
    ttk.Checkbutton(grid, text="Enable compatibility constraints", variable=popup.var_realism_constraints_enabled).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
    )

    popup.var_realism_profile_id = tk.StringVar(value=str(current.get("realism_profile_id", DEFAULT_REALISM_PROFILE_ID)))
    popup.filter_vars["realism_profile_id"] = popup.var_realism_profile_id
    ttk.Label(grid, text="Realism profile id", width=24).grid(row=1, column=0, sticky="w", pady=2)
    ttk.Entry(grid, textvariable=popup.var_realism_profile_id, width=22).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=2)

    popup.var_realism_k0 = _f("realism_k_prob_0", REALISM_K_DEFAULTS["realism_k_prob_0"])
    popup.var_realism_k1 = _f("realism_k_prob_1", REALISM_K_DEFAULTS["realism_k_prob_1"])
    popup.var_realism_k2 = _f("realism_k_prob_2", REALISM_K_DEFAULTS["realism_k_prob_2"])
    _add_row(2, "K=0 probability", "realism_k_prob_0", popup.var_realism_k0)
    _add_row(3, "K=1 probability", "realism_k_prob_1", popup.var_realism_k1)
    _add_row(4, "K=2 probability", "realism_k_prob_2", popup.var_realism_k2)

    row = 5
    for key, default in REALISM_GROUP_DEFAULTS.items():
        gname = key.replace("realism_group_", "").replace("_prob", "")
        var = _f(key, default)
        _add_row(row, f"{gname} probability", key, var)
        row += 1

    ttk.Button(parent, text="Load Realism Defaults", command=popup._set_realism_defaults).pack(anchor="w", pady=(6, 0))


def set_realism_defaults(popup: Any) -> None:
    """Apply the default realism group probabilities."""
    defaults: Dict[str, Any] = {
        "filter_mode": "realism",
        "realism_enabled": True,
        "realism_constraints_enabled": True,
        "realism_profile_id": DEFAULT_REALISM_PROFILE_ID,
    }
    defaults.update(REALISM_K_DEFAULTS)
    defaults.update(REALISM_GROUP_DEFAULTS)
    popup._update_internal_vars(defaults)
    popup._persist_active_profile()

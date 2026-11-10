"""UI for the Merge Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any, TYPE_CHECKING
from collections.abc import Callable
from datetime import datetime

from gui.theme.style import font
from gui.theme.tokens import SPACE
from gui.utils.formatting import fmt_bytes
from gui.utils.tooltip import ToolTip
from gui.widgets import FormGrid, LogView, LogViewHost, ScrolledTree, Section, Toolbar

if TYPE_CHECKING:
    from .tab import MergeTab


class MergeUI(LogViewHost):
    def __init__(self, tab: MergeTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=0)

        self.tree: ttk.Treeview
        self.selection_tree: ttk.Treeview
        self.bundle_tree: ttk.Treeview
        self.weights_view: ScrolledTree
        self.selection_view: ScrolledTree
        self.bundles_view: ScrolledTree
        self.lbl_selection_empty: ttk.Label
        self.lbl_bundles_empty: ttk.Label
        self.log_view: LogView
        self.txt: tk.Text

        self.var_bundle_name = tk.StringVar(value="merged_v1")
        self.var_add_timestamp = tk.BooleanVar(value=True)
        self.btn_merge: ttk.Button
        self.btn_merge_ensemble: ttk.Button
        self.var_status = tk.StringVar(value="Status: idle")

    def build_ui(self) -> None:
        top = Toolbar(self.frame)
        top.pack(fill="x")
        top.add(ttk.Label(top, text="Merge Weights", style="Title.TLabel"))
        top.icon("refresh", self.tab._scan_models, help_key="merge.refresh", right=True)

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(SPACE.M, 0))

        left = ttk.Frame(main)
        main.add(left, weight=3)
        self._build_weights_section(left)

        right = ttk.Frame(main)
        main.add(right, weight=2)
        self._build_selection_section(right)
        self._build_merge_section(right)
        self._build_bundles_section(right)
        self._build_log_section(right)

    # ------------------------------------------------------------------ sections
    def _build_weights_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Available Weights by Profile", help_key="merge.weights")
        sec.pack(fill="both", expand=True)

        self.weights_view = ScrolledTree(
            sec,
            columns=[
                ("Acc", "Accuracy", 75, "e"),
                ("F1", "F1", 70, "e"),
                ("Size", "Size", 75, "e"),
                ("Modified", "Modified", 130, "w"),
                ("Hash", "Profile Hash", 110, "w"),
            ],
            show_tree=True,
            tree_heading="Profile / Model",
            tree_width=260,
            selectmode="browse",
            sortable=False,  # hierarchical: profile -> group -> model
            striped=False,
        )
        self.weights_view.pack(fill="both", expand=True)
        self.tree = self.weights_view.tree
        self.tree.tag_configure("profile_header", font=font("BodyBold"))
        self.tree.tag_configure("group_header", font=font("SmallItalic"))
        # "selected_weight" rows also carry the themed "success" tag (see rebuild_tree).
        self.tree.bind("<Double-1>", self.tab._on_tree_double_click)
        ToolTip(self.tree, help_key="merge.tree")

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(SPACE.M, 0))
        bar.button("Select Weight for Profile", self.tab._select_weight, help_key="merge.select")
        bar.button("Deselect", self.tab._deselect_weight, help_key="merge.deselect")

    def _build_selection_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Selected Weights for Merge", help_key="merge.selection")
        sec.pack(fill="x", pady=(0, SPACE.M))
        self.lbl_selection_empty = ttk.Label(
            sec, text="No weights selected yet. Double-click a model on the left (one per profile).",
            style="CardMuted.TLabel", wraplength=360, justify="left")
        self.selection_view = ScrolledTree(sec, columns=[("Model", "Profile -> Model", 350, "w")], height=8)
        self.selection_view.pack(fill="x")
        self.selection_tree = self.selection_view.tree
        self._toggle_empty(self.lbl_selection_empty, self.selection_view, True)

    def _build_merge_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Merge", help_key="merge.merge")
        sec.pack(fill="x", pady=(0, SPACE.M))
        form = FormGrid(sec, style="Card.TFrame")
        form.pack(fill="x")
        form.row("Bundle Name", ttk.Entry(form, textvariable=self.var_bundle_name, width=30), help_key="merge.name")
        chk = ttk.Checkbutton(form, text="Append Timestamp", variable=self.var_add_timestamp, style="Card.TCheckbutton")
        form.row("", chk, help_key="merge.timestamp", sticky="w")

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(SPACE.M, 0))
        self.btn_merge = bar.button("Merge into Bundle", self.tab._do_merge, style="Accent.TButton",
                                    help_key="merge.bundle")
        self.btn_merge_ensemble = bar.button("Merge into Single .pt", self.tab._do_merge_ensemble,
                                             help_key="merge.ensemble")
        ttk.Label(sec, textvariable=self.var_status, style="CardMuted.TLabel").pack(anchor="w", pady=(SPACE.S, 0))

    def _build_bundles_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Existing Bundles", help_key="merge.bundles")
        sec.pack(fill="x", pady=(0, SPACE.M))
        self.lbl_bundles_empty = ttk.Label(sec, text="No bundles yet. Select weights above and press Merge into Bundle.",
                                           style="CardMuted.TLabel", wraplength=360, justify="left")
        self.bundles_view = ScrolledTree(
            sec,
            columns=[("Profiles", "Included Profiles", 250, "w"), ("Created", "Created", 140, "w")],
            height=5,
        )
        self.bundles_view.pack(fill="x")
        self.bundle_tree = self.bundles_view.tree
        self.bundle_tree.bind("<Double-1>", lambda _e: self.tab._show_bundle_info())
        self._toggle_empty(self.lbl_bundles_empty, self.bundles_view, True)

        bar = Toolbar(sec, style="Card.TFrame")
        bar.pack(fill="x", pady=(SPACE.M, 0))
        bar.button("Bundle Info…", self.tab._show_bundle_info, help_key="merge.bundle_info")
        bar.button("Load Into Selection", self.tab._load_bundle_into_selection, help_key="merge.load_bundle")
        bar.button("Delete Bundle", self.tab._delete_bundle, style="Danger.TButton", help_key="merge.delete_bundle",
                   right=True)

    def _build_log_section(self, parent: tk.Misc) -> None:
        sec = Section(parent, "Log")
        sec.pack(fill="both", expand=True)
        self.log_view = LogView(sec, height=8)
        self.log_view.pack(fill="both", expand=True)
        self.txt = self.log_view.text
        b_clear = ttk.Button(sec.actions, text="Clear", style="Icon.TButton", command=self.log_view.clear)
        b_clear.pack(side="right")
        ToolTip(b_clear, help_key="common.clear_log")
        b_copy = ttk.Button(sec.actions, text="Copy", style="Icon.TButton", command=self.log_view.copy_all)
        b_copy.pack(side="right", padx=(0, SPACE.S))
        ToolTip(b_copy, help_key="common.copy_log")

    @staticmethod
    def _toggle_empty(label: ttk.Label, view: tk.Widget, empty: bool) -> None:
        if not empty:
            label.pack_forget()
        elif not label.winfo_manager():
            label.pack(anchor="w", pady=(0, SPACE.S), before=view)

    def rebuild_tree(self, profile_models: dict[str, list[tuple[str, Path]]], iid_to_model: dict[str, tuple[str, Path]],
                     model_meta_cache: dict, selected_weights: dict[str, Path], favorites: dict[str, Any],
                     group_for_path: Callable[[Path], str], available_groups: list[str]) -> None:
        
        self.tree.delete(*self.tree.get_children())
        iid_to_model.clear()

        counter = 0
        for profile_id in sorted(profile_models.keys()):
            models = profile_models[profile_id]
            selected = selected_weights.get(profile_id)

            count = len(models)
            sel_marker = " [SELECTED]" if selected else ""
            profile_iid = f"profile::{profile_id}"
            self.tree.insert(
                "", "end", iid=profile_iid, text=f"{profile_id} ({count} weights){sel_marker}",
                values=("", "", "", "", ""), tags=("profile_header",), open=True,
            )

            grouped: dict[str, list[tuple[str, Path]]] = {g: [] for g in available_groups}
            for disp, p in models:
                g = group_for_path(p)
                if g not in grouped: grouped[g] = []
                grouped[g].append((disp, p))

            for group in available_groups:
                members = grouped.get(group, [])
                if not members: continue

                group_iid = f"group::{profile_id}::{group}"
                self.tree.insert(
                    profile_iid, "end", iid=group_iid, text=f"{group} ({len(members)})",
                    values=("", "", "", "", ""), tags=("group_header",), open=True,
                )

                for disp, p in members:
                    try:
                        st = p.stat()
                        size_s = self.fmt_bytes(int(st.st_size))
                        mt = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                    except Exception:
                        size_s = "?"
                        mt = "?"

                    _, phash, val_acc, val_f1 = self.tab.logic.load_model_meta(p, model_meta_cache)
                    acc_s = f"{val_acc:.4f}" if val_acc is not None else "-"
                    f1_s = f"{val_f1:.4f}" if val_f1 is not None else "-"
                    hash_short = phash.split(":")[1][:12] if phash and ":" in phash else (phash[:12] if phash else "")

                    iid_model = f"model::{counter}"
                    counter += 1

                    is_selected = selected is not None and selected.resolve() == p.resolve()
                    tags = ("selected_weight", "success") if is_selected else ()
                    marker = " *" if is_selected else ""

                    fav_s = "★" if self.tab.logic._rel(p) in favorites else ""
                    display_name = f"{fav_s} {disp}{marker}".strip()

                    self.tree.insert(
                        group_iid, "end", iid=iid_model, text=display_name,
                        values=(acc_s, f1_s, size_s, mt, hash_short), tags=tags,
                    )
                    iid_to_model[iid_model] = (profile_id, p)
    
    def refresh_selection_summary(self, selected_weights: dict[str, Path]) -> None:
        self.selection_view.clear()
        for profile_id in sorted(selected_weights.keys()):
            p = selected_weights[profile_id]
            self.selection_view.insert((f"{profile_id}  ->  {p.name}",))
        self._toggle_empty(self.lbl_selection_empty, self.selection_view, not selected_weights)

    def refresh_bundles(self, bundle_dirs: list[Path], read_bundle_meta_func: Callable[[Path], Any], fmt_bytes_func: Callable[[int], str]) -> None:
        self.bundles_view.clear()

        for bd in bundle_dirs:
            meta = read_bundle_meta_func(bd)
            if meta:
                profiles = ", ".join(sorted(meta.checkpoints.keys()))
                created = meta.created_at[:19] if meta.created_at else "?"
            else:
                pts = list(bd.glob("*.pt"))
                profiles = ", ".join(p.stem for p in pts) if pts else "(empty)"
                try: created = datetime.fromtimestamp(bd.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except Exception: created = "?"

            self.bundles_view.insert((profiles, created), iid=str(bd))
        self._toggle_empty(self.lbl_bundles_empty, self.bundles_view, not bundle_dirs)

    fmt_bytes = staticmethod(fmt_bytes)

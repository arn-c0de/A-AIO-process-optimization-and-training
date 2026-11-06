"""UI for the Datasets Tab."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip
from gui.widgets import ScrolledTree, Section, Toolbar, menu_button

if TYPE_CHECKING:
    from .tab import DatasetsTab


class DatasetsUI:
    def __init__(self, tab: DatasetsTab, parent: ttk.Frame) -> None:
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=0)
        self.var_count = tk.StringVar(value="")
        self.tree: ttk.Treeview
        self.list_view: ScrolledTree
        self.combo_category_filter: ttk.Combobox
        self.lbl_empty: ttk.Label
        self.menu: tk.Menu

    def build_ui(self) -> None:
        tab = self.tab

        # --- Toolbar: actions on the left, filters on the right.
        bar = Toolbar(self.frame)
        bar.pack(fill="x", pady=(0, SPACE.M))
        bar.icon("refresh", tab.refresh, help_key="datasets.refresh")
        bar.button("Show Details…", tab._show_details_selected, help_key="datasets.details")
        bar.separator()
        bar.button("Set Category…", tab._set_category_for_selected, help_key="datasets.set_category",
                   style="Accent.TButton")
        bar.add(menu_button(bar, "Categories", [
            ("New Category…", tab._new_category),
            ("Rename Category…", tab._rename_category),
            None,
            ("Delete Category…", tab._delete_category),
        ], help_key="datasets.categories_menu"))
        bar.separator()
        bar.button("Archive", lambda: tab._set_archive_for_selected(True), help_key="datasets.archive")
        bar.button("Unarchive", lambda: tab._set_archive_for_selected(False), help_key="datasets.unarchive")

        chk = ttk.Checkbutton(bar, text="Show Archived", variable=tab.var_show_archived, command=tab.refresh)
        ToolTip(chk, help_key="datasets.show_archived")
        bar.add_right(chk)
        self.combo_category_filter = ttk.Combobox(bar, textvariable=tab.var_category_filter, state="readonly", width=20)
        self.combo_category_filter.bind("<<ComboboxSelected>>", lambda _e: tab.refresh())
        ToolTip(self.combo_category_filter, help_key="datasets.category")
        bar.add_right(self.combo_category_filter)
        bar.add_right(ttk.Label(bar, text="Category", style="Field.TLabel"))

        # --- Dataset list.
        sec = Section(self.frame, "Datasets", help_key="datasets.list")
        sec.pack(fill="both", expand=True)
        ttk.Label(sec.actions, textvariable=self.var_count, style="CardMuted.TLabel").pack(side="right")
        self.lbl_empty = ttk.Label(sec, text="", style="CardMuted.TLabel", wraplength=640, justify="left")
        cols = [
            ("dataset", "Dataset", 280, "w"),
            ("category", "Category", 170, "w"),
            ("created", "Created", 170, "w"),
            ("archived", "Archived", 90, "center"),
            ("location", "Location", 360, "w"),
        ]
        self.list_view = ScrolledTree(sec, cols, selectmode="extended", height=16, hscroll=True)
        self.list_view.pack(fill="both", expand=True)
        self.tree = self.list_view.tree
        self.tree.bind("<Button-3>", tab._on_right_click)
        self.tree.bind("<Button-2>", tab._on_right_click)
        self.tree.bind("<Double-Button-1>", lambda _e: tab._show_details_selected())

        self.menu = tk.Menu(self.frame, tearoff=0)
        self.menu.add_command(label="Show Details…", command=tab._show_details_selected)
        self.menu.add_separator()
        self.menu.add_command(label="Set Category…", command=tab._set_category_for_selected)
        self.menu.add_separator()
        self.menu.add_command(label="Archive Selected", command=lambda: tab._set_archive_for_selected(True))
        self.menu.add_command(label="Unarchive Selected", command=lambda: tab._set_archive_for_selected(False))
        self.menu.add_separator()
        self.menu.add_command(label="Refresh", command=tab.refresh)

    def set_empty_hint(self, text: str) -> None:
        """Show a muted hint above the (empty) list, or hide it with ``""``."""
        if text:
            self.lbl_empty.configure(text=text)
            if not self.lbl_empty.winfo_manager():
                self.lbl_empty.pack(fill="x", pady=(0, SPACE.S), before=self.list_view)
        else:
            self.lbl_empty.pack_forget()

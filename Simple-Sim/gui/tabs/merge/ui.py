"""UI for the Merge Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .tab import MergeTab


class MergeUI:
    def __init__(self, tab: MergeTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)

        self.tree: ttk.Treeview
        self.selection_tree: ttk.Treeview
        self.bundle_tree: ttk.Treeview
        self.txt: tk.Text

        self.var_bundle_name = tk.StringVar(value="merged_v1")
        self.var_add_timestamp = tk.BooleanVar(value=True)
        self.btn_merge: ttk.Button
        self.btn_merge_ensemble: ttk.Button
        self.var_status = tk.StringVar(value="Status: idle")
        
    def build_ui(self) -> None:
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 8))

        ttk.Label(top, text="Merge", font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Refresh", command=self.tab._scan_models).pack(side="right")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(0, 8))

        left = ttk.Frame(main, padding=5)
        main.add(left, weight=3)

        ttk.Label(left, text="Available weights by profile type").pack(anchor="w")

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True, pady=(6, 0))

        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Acc", "F1", "Size", "Modified", "Hash"),
            show="tree headings",
            selectmode="browse",
        )
        self.tree.heading("#0", text="Profile / Model")
        self.tree.column("#0", width=260, anchor="w")
        self.tree.heading("Acc", text="Accuracy")
        self.tree.column("Acc", width=75, anchor="e")
        self.tree.heading("F1", text="F1")
        self.tree.column("F1", width=70, anchor="e")
        self.tree.heading("Size", text="Size")
        self.tree.column("Size", width=75, anchor="e")
        self.tree.heading("Modified", text="Modified")
        self.tree.column("Modified", width=130, anchor="w")
        self.tree.heading("Hash", text="Profile Hash")
        self.tree.column("Hash", width=110, anchor="w")
        self.tree.tag_configure("profile_header", font=("TkDefaultFont", 10, "bold"))
        self.tree.tag_configure("group_header", font=("TkDefaultFont", 9, "italic"))
        self.tree.tag_configure("selected_weight", foreground="green")

        scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.tree.bind("<Double-1>", self.tab._on_tree_double_click)

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_frame, text="Select weight for profile", command=self.tab._select_weight).pack(side="left")
        ttk.Button(btn_frame, text="Deselect", command=self.tab._deselect_weight).pack(side="left", padx=(8, 0))

        right = ttk.Frame(main, padding=5)
        main.add(right, weight=2)

        ttk.Label(right, text="Selected weights for merge").pack(anchor="w")

        self.selection_tree = ttk.Treeview(
            right,
            columns=("Model",),
            show="headings",
            selectmode="browse",
            height=8,
        )
        self.selection_tree.heading("Model", text="Profile -> Model")
        self.selection_tree.column("Model", width=350, anchor="w")
        self.selection_tree.pack(fill="x", pady=(6, 0))

        merge_frame = ttk.LabelFrame(right, text="Merge", padding=8)
        merge_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(merge_frame, text="Bundle Name:").grid(row=0, column=0, sticky="w")
        ttk.Entry(merge_frame, textvariable=self.var_bundle_name, width=30).grid(
            row=0, column=1, sticky="ew", padx=(6, 0)
        )
        merge_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(merge_frame, text="Append timestamp", variable=self.var_add_timestamp).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        self.btn_merge = ttk.Button(merge_frame, text="Merge -> Bundle", command=self.tab._do_merge)
        self.btn_merge.grid(row=2, column=0, sticky="w", pady=(8, 0))

        self.btn_merge_ensemble = ttk.Button(
            merge_frame, text="Merge -> Single .pt (Ensemble)", command=self.tab._do_merge_ensemble
        )
        self.btn_merge_ensemble.grid(row=2, column=1, sticky="e", pady=(8, 0))

        ttk.Label(merge_frame, textvariable=self.var_status).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        bundles_frame = ttk.LabelFrame(right, text="Existing Bundles", padding=8)
        bundles_frame.pack(fill="x", pady=(10, 0))

        self.bundle_tree = ttk.Treeview(
            bundles_frame,
            columns=("Profiles", "Created"),
            show="headings",
            selectmode="browse",
            height=5,
        )
        self.bundle_tree.heading("Profiles", text="Included Profiles")
        self.bundle_tree.column("Profiles", width=250, anchor="w")
        self.bundle_tree.heading("Created", text="Created")
        self.bundle_tree.column("Created", width=140, anchor="w")
        self.bundle_tree.pack(fill="x")

        bundle_btns = ttk.Frame(bundles_frame)
        bundle_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(bundle_btns, text="Bundle Info", command=self.tab._show_bundle_info).pack(side="left")
        ttk.Button(bundle_btns, text="Load Into Selection", command=self.tab._load_bundle_into_selection).pack(side="left", padx=(8, 0))
        ttk.Button(bundle_btns, text="Delete Bundle", command=self.tab._delete_bundle).pack(side="left", padx=(8, 0))

        ttk.Label(right, text="Log").pack(anchor="w", pady=(10, 0))
        self.txt = tk.Text(right, height=8, wrap="none")
        self.txt.pack(fill="both", expand=True, pady=(4, 0))
        self.txt.configure(state="disabled")

    def append_log(self, s: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", s)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def rebuild_tree(self, profile_models: Dict[str, List[Tuple[str, Path]]], iid_to_model: Dict[str, Tuple[str, Path]],
                     model_meta_cache: Dict, selected_weights: Dict[str, Path], favorites: Dict[str, Any],
                     group_for_path: Callable[[Path], str], available_groups: List[str]) -> None:
        
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

            grouped: Dict[str, List[Tuple[str, Path]]] = {g: [] for g in available_groups}
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
                    tags = ("selected_weight",) if is_selected else ()
                    marker = " *" if is_selected else ""

                    fav_s = "★" if self.tab.logic._rel(p) in favorites else ""
                    display_name = f"{fav_s} {disp}{marker}".strip()

                    self.tree.insert(
                        group_iid, "end", iid=iid_model, text=display_name,
                        values=(acc_s, f1_s, size_s, mt, hash_short), tags=tags,
                    )
                    iid_to_model[iid_model] = (profile_id, p)
    
    def refresh_selection_summary(self, selected_weights: Dict[str, Path]) -> None:
        self.selection_tree.delete(*self.selection_tree.get_children())
        for profile_id in sorted(selected_weights.keys()):
            p = selected_weights[profile_id]
            self.selection_tree.insert("", "end", values=(f"{profile_id}  ->  {p.name}",))

    def refresh_bundles(self, bundle_dirs: List[Path], read_bundle_meta_func: Callable[[Path], Any], fmt_bytes_func: Callable[[int], str]) -> None:
        self.bundle_tree.delete(*self.bundle_tree.get_children())

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

            self.bundle_tree.insert("", "end", iid=str(bd), values=(profiles, created))

    def fmt_bytes(self, n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        v = float(max(0, n))
        u = 0
        while v >= 1024.0 and u < len(units) - 1:
            v /= 1024.0
            u += 1
        if u == 0:
            return f"{int(v)} {units[u]}"
        return f"{v:.1f} {units[u]}"

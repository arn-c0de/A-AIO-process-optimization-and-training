"""UI for the Analysis Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING
from PIL import Image, ImageTk
import cv2

from gui import dialogs
from gui.components.overlay_renderer import draw_defect_overlay, draw_prediction_overlay
from gui.theme.style import on_theme_changed, palette
from gui.theme.tokens import SPACE
from gui.utils.tooltip import ToolTip
from gui.widgets import LogView, ScrolledTree, Section, Toolbar
from simple_sim.schema import MetaRow

if TYPE_CHECKING:
    from .tab import AnalysisTab


class AnalysisUI:
    def __init__(self, tab: AnalysisTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=SPACE.L)
        self.var_dataset: tk.StringVar
        self.var_dataset_stats: tk.StringVar
        self.dataset_combo: ttk.Combobox
        self.var_filter: tk.StringVar
        self.var_search: tk.StringVar
        self.var_run: tk.StringVar
        self.var_domain: tk.StringVar
        self.var_split: tk.StringVar
        self.var_profile: tk.StringVar
        self.var_profile_multi: tk.StringVar
        self.var_sort: tk.StringVar
        self.tree: ttk.Treeview
        self.canvas: tk.Canvas
        self.info_text: tk.Text
        self.combo_run: ttk.Combobox
        self.combo_domain: ttk.Combobox
        self.combo_class: ttk.Combobox
        self.combo_profile: ttk.Combobox
        self.combo_sort: ttk.Combobox
        self.photo: ImageTk.PhotoImage | None = None
        self._preview_max_px: int = 420
        self.sample_menu: tk.Menu
        self.var_feedback_status: tk.StringVar
        self._feedback_class_values: list[str]
        self.feedback_history_tree: ttk.Treeview
        self.feedback_history_view: ScrolledTree
        self.info_view: LogView
        self.lbl_tree_empty: ttk.Label
        self.lbl_history_empty: ttk.Label

    def build_ui(self):
        # ------------------------------------------------------------ action bar
        top = Toolbar(self.frame)
        top.pack(fill="x")
        top.add(ttk.Label(top, text="Dataset", style="Field.TLabel"))
        self.var_dataset = tk.StringVar(value="")
        self.dataset_combo = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=32)
        top.add(self.dataset_combo)
        self.dataset_combo.bind("<<ComboboxSelected>>", self.tab._on_dataset_selected)
        ToolTip(self.dataset_combo, help_key="common.dataset", text_func=self.var_dataset.get)
        top.icon("refresh", self.tab._refresh_datasets, help_key="common.refresh")
        top.separator()
        top.icon("prev", self.tab._prev_image, help_key="common.prev")
        top.icon("next", self.tab._next_image, help_key="common.next")
        top.separator()
        top.button("Delete Image…", self.tab._delete_current_image, style="Danger.TButton",
                   help_key="analysis.delete")
        top.button("Move To…", self.tab._move_current_image, help_key="analysis.move")
        top.button("Recompute With Feedback", self.tab._recompute_with_feedback, help_key="analysis.recompute",
                   right=True)
        top.button("Analyze Dataset", self.tab._analyze_dataset, style="Accent.TButton", help_key="analysis.batch",
                   right=True)

        stats_row = ttk.Frame(self.frame)
        stats_row.pack(fill="x", pady=(SPACE.M, 0))
        self.var_dataset_stats = tk.StringVar(value="Dataset Stats: -")
        ttk.Label(stats_row, textvariable=self.var_dataset_stats, style="Muted.TLabel").pack(side="left")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(SPACE.M, 0))

        # ------------------------------------------------------------ image list
        left = Section(main, "Images", help_key="analysis.images")
        main.add(left, weight=1)

        filters = ttk.Frame(left, style="Card.TFrame")
        filters.pack(fill="x")
        filters.columnconfigure(1, weight=1)
        filters.columnconfigure(3, weight=1)

        def field_label(text: str, row: int, col: int, help_key: str) -> None:
            lbl = ttk.Label(filters, text=text, style="CardMuted.TLabel")
            lbl.grid(row=row, column=col, sticky="e", padx=(SPACE.M if col else 0, SPACE.S), pady=SPACE.XS)
            ToolTip(lbl, help_key=help_key)

        def filter_combo(var: tk.StringVar, values: list[str], row: int, col: int, help_key: str,
                         width: int = 12) -> ttk.Combobox:
            cb = ttk.Combobox(filters, textvariable=var, values=values, state="readonly", width=width)
            cb.grid(row=row, column=col, sticky="ew", pady=SPACE.XS)
            ToolTip(cb, help_key=help_key)
            return cb

        self.var_search = tk.StringVar(value="")
        field_label("Search", 0, 0, "analysis.search")
        search_row = ttk.Frame(filters, style="Card.TFrame")
        search_row.grid(row=0, column=1, columnspan=3, sticky="ew", pady=SPACE.XS)
        ent_search = ttk.Entry(search_row, textvariable=self.var_search)
        ent_search.pack(side="left", fill="x", expand=True)
        ToolTip(ent_search, help_key="analysis.search")
        b_clear = ttk.Button(search_row, text="Clear", command=lambda: self.tab._set_search(""))
        b_clear.pack(side="left", padx=(SPACE.S, 0))
        ToolTip(b_clear, help_key="analysis.search_clear")
        ent_search.bind("<KeyRelease>", lambda e: self.tab._schedule_filter_images())

        def refilter(_e: object = None) -> None:
            self.tab._filter_images(display_first=True)

        self.var_run = tk.StringVar(value="All")
        field_label("Run", 1, 0, "analysis.run_filter")
        self.combo_run = filter_combo(self.var_run, ["All"], 1, 1, "analysis.run_filter")
        self.combo_run.bind("<<ComboboxSelected>>", refilter)

        self.var_domain = tk.StringVar(value="All")
        field_label("Domain", 1, 2, "analysis.domain_filter")
        self.combo_domain = filter_combo(self.var_domain, ["All"], 1, 3, "analysis.domain_filter")
        self.combo_domain.bind("<<ComboboxSelected>>", refilter)

        self.var_split = tk.StringVar(value="All")
        field_label("Split", 2, 0, "analysis.split_filter")
        combo_split = filter_combo(self.var_split, ["All", "train", "val", "test"], 2, 1, "analysis.split_filter")
        combo_split.bind("<<ComboboxSelected>>", refilter)

        self.var_filter = tk.StringVar(value="All")
        field_label("Class", 2, 2, "analysis.class_filter")
        self.combo_class = filter_combo(self.var_filter, ["All"], 2, 3, "analysis.class_filter")
        self.combo_class.bind("<<ComboboxSelected>>", refilter)

        self.var_sort = tk.StringVar(value="Default")
        field_label("Sort", 3, 0, "analysis.sort")
        self.combo_sort = filter_combo(
            self.var_sort, ["Default", "Newest first", "Oldest first", "ID A-Z", "ID Z-A"], 3, 1, "analysis.sort"
        )
        self.combo_sort.bind("<<ComboboxSelected>>", refilter)

        self.var_profile = tk.StringVar(value="All")
        field_label("Profile", 3, 2, "analysis.profile_filter")
        self.combo_profile = filter_combo(self.var_profile, ["All"], 3, 3, "analysis.profile_filter", width=16)
        self.combo_profile.bind("<<ComboboxSelected>>", self.tab._on_profile_filter_selected)

        multi_row = ttk.Frame(filters, style="Card.TFrame")
        multi_row.grid(row=4, column=3, sticky="ew", pady=(0, SPACE.XS))
        b_multi = ttk.Button(multi_row, text="Multi…", command=self.tab._open_profile_multi_select)
        b_multi.pack(side="left")
        ToolTip(b_multi, help_key="analysis.profile_multi")
        self.var_profile_multi = tk.StringVar(value="multi: off")
        ttk.Label(multi_row, textvariable=self.var_profile_multi, style="CardMuted.TLabel").pack(
            side="left", padx=(SPACE.S, 0))

        st = ScrolledTree(
            left,
            [("Class", "Class", 100, "w"), ("Date", "Added", 145, "w")],
            show_tree=True,
            tree_heading="ID",
            tree_width=200,
            height=20,
            selectmode="extended",
            sortable=False,
            striped=False,
        )
        st.pack(fill="both", expand=True, pady=(SPACE.M, 0))
        self.tree = st.tree
        self.lbl_tree_empty = ttk.Label(st, text="No samples match the current filters.", style="CardMuted.TLabel")
        self.tree.bind("<<TreeviewSelect>>", self.tab._on_tree_select)
        self.tree.bind("<Button-3>", self._on_tree_right_click)
        self.tree.bind("<Delete>", lambda _e: self.tab._delete_current_image())
        self.tree.bind("<Left>", lambda _e: self._on_tree_arrow(-1))
        self.tree.bind("<Right>", lambda _e: self._on_tree_arrow(+1))

        self.sample_menu = tk.Menu(self.tree, tearoff=0)
        self.sample_menu.add_command(label="Delete Image…", command=self.tab._delete_current_image)
        self.sample_menu.add_command(label="Move To…", command=self.tab._move_current_image)

        # ------------------------------------------------------------ viewer column
        right = ttk.Frame(main)
        main.add(right, weight=3)

        viewer = Section(right, "Image Viewer")
        viewer.pack(fill="x")
        self.canvas = tk.Canvas(viewer, background=palette(self.frame).canvas_bg, highlightthickness=0,
                                width=self._preview_max_px, height=self._preview_max_px, takefocus=1)
        self.canvas._simsim_role = "image"  # type: ignore[attr-defined]
        self.canvas.pack(fill="x", expand=False)
        on_theme_changed(self.canvas, lambda pal: self.canvas.configure(background=pal.canvas_bg))
        self.canvas.bind("<Button-1>", lambda _e: self.canvas.focus_set())
        self.canvas.bind("<Left>", lambda _e: (self.tab._prev_image(), "break")[1])
        self.canvas.bind("<Right>", lambda _e: (self.tab._next_image(), "break")[1])

        meta = Section(right, "Metadata")
        meta.pack(fill="both", expand=True, pady=(SPACE.M, 0))
        self.info_view = LogView(meta, height=12, wrap="word")
        self.info_view.pack(fill="both", expand=True)
        self.info_text = self.info_view.text

        feedback_frame = Section(right, "User Feedback", help_key="analysis.feedback")
        feedback_frame.pack(fill="x", pady=(SPACE.M, 0))
        self.var_feedback_status = tk.StringVar(value="No manual feedback yet.")
        ttk.Label(feedback_frame, textvariable=self.var_feedback_status, style="CardMuted.TLabel").pack(anchor="w")

        action_row = Toolbar(feedback_frame, style="Card.TFrame")
        action_row.pack(fill="x", pady=(SPACE.M, SPACE.S))
        action_row.button("Correct", self.tab._feedback_thumbs_up, help_key="analysis.feedback_up")
        action_row.button("Wrong…", self.tab._feedback_thumbs_down, help_key="analysis.feedback_down")
        action_row.button("History…", self.tab._open_feedback_history, help_key="analysis.feedback_history")
        self._feedback_class_values = []

        ttk.Label(feedback_frame, text="Recent Feedback", style="CardMuted.TLabel").pack(
            anchor="w", pady=(SPACE.S, SPACE.XS))
        hist = ScrolledTree(
            feedback_frame,
            [("Time", "Time", 130, "w"), ("Sample", "Sample", 240, "w"),
             ("Action", "Action", 90, "center"), ("To", "To", 110, "center")],
            height=6,
        )
        hist.pack(fill="x")
        self.feedback_history_view = hist
        self.feedback_history_tree = hist.tree
        ToolTip(self.feedback_history_tree, help_key="analysis.feedback_recent")
        self.lbl_history_empty = ttk.Label(hist, text="No feedback yet.", style="CardMuted.TLabel")
        self.lbl_history_empty.place(relx=0.5, rely=0.55, anchor="center")
        self.feedback_history_tree.bind("<<TreeviewSelect>>", self.tab._on_feedback_history_select)

    def _on_tree_arrow(self, step: int) -> str | None:
        """Left/Right on a sample row = previous/next image; on group rows keep collapse/expand."""
        sel = self.tree.selection()
        if not sel or "sample" not in self.tree.item(sel[0], "tags"):
            return None
        if step < 0:
            self.tab._prev_image()
        else:
            self.tab._next_image()
        return "break"

    def display_image(
        self,
        image_path: str,
        meta_row: MetaRow,
        label: str,
        prediction: tuple[str, float, dict] | None,
        original_label: str | None = None,
        feedback: dict | None = None,
    ):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Failed to read image")
            
            img_overlay = draw_defect_overlay(img, meta_row.defect, meta_row.nominal)
            if prediction:
                img_overlay = draw_prediction_overlay(img_overlay, label, prediction[0], prediction[1])
            
            img_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((self._preview_max_px, self._preview_max_px))
            self.photo = ImageTk.PhotoImage(img_pil)
            
            self.canvas.delete("all")
            cx = max(1, int(self.canvas.winfo_width() / 2))
            cy = max(1, int(self.canvas.winfo_height() / 2))
            self.canvas.create_image(cx, cy, image=self.photo, anchor="center")
            
            self.update_metadata_panel(meta_row, label, prediction, original_label=original_label, feedback=feedback)
        except Exception as e:
            raise RuntimeError(f"Failed to display image: {e}") from e

    def update_metadata_panel(
        self,
        meta_row: MetaRow,
        ground_truth: str,
        prediction: tuple[str, float, dict] | None,
        original_label: str | None = None,
        feedback: dict | None = None,
    ):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        
        info = f"Sample ID: {meta_row.id}\nGround Truth (effective): {ground_truth}\n"
        if original_label and original_label != ground_truth:
            info += f"Ground Truth (original): {original_label}\n"
        if prediction:
            predicted, confidence, _ = prediction
            status = "✓ CORRECT" if predicted == ground_truth else "✗ WRONG"
            info += f"Predicted: {predicted} ({confidence:.1%}) {status}\n"
        else:
            info += "Predicted: (no model loaded)\n"
        if feedback:
            verdict = str(feedback.get("verdict") or "")
            corrected = str(feedback.get("corrected_class") or "").strip()
            note = str(feedback.get("note") or "").strip()
            ts = str(feedback.get("timestamp") or "")
            info += f"Feedback: {verdict}"
            if corrected:
                info += f" -> {corrected}"
            if ts:
                info += f" @ {ts}"
            info += "\n"
            if note:
                info += f"Feedback note: {note}\n"
        
        info += "\nDefect Parameters:\n"
        for key, value in meta_row.defect.items():
            if isinstance(value, float):
                info += f"  {key}: {value:.2f}\n"
            else:
                info += f"  {key}: {value}\n"
        
        self.info_text.insert("1.0", info)
        self.info_text.configure(state="disabled")

    def populate_tree(
        self,
        hierarchy: dict,
        label_dict: dict,
        visible_sample_ids: list[str],
        visible_sample_item_by_id: dict[str, str],
        query: str,
        sample_date_dict: dict[str, str] | None = None,
    ):
        self.tree.delete(*self.tree.get_children())
        open_nodes = bool(query)
        
        for run_id in sorted(hierarchy.keys()):
            run_node = self.tree.insert("", "end", text=f"📁 {run_id}", values=("", ""), open=open_nodes,
                                        tags=("group",))
            for domain in sorted(hierarchy[run_id].keys()):
                domain_node = self.tree.insert(run_node, "end", text=f"🌐 {domain}", values=("", ""), open=open_nodes)
                for split in sorted(hierarchy[run_id][domain].keys(), key=lambda x: {"train": 0, "val": 1, "test": 2}.get(x, 3)):
                    samples = hierarchy[run_id][domain][split]
                    class_counts: dict[str, int] = {}
                    for sid in samples:
                        cls = label_dict.get(sid, "?")
                        if not cls:
                            continue
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    count_str = ", ".join(f"{cls}:{cnt}" for cls, cnt in sorted(class_counts.items()))
                    split_icon = {"train": "🔧", "val": "✓", "test": "🧪"}.get(split, "📂")
                    split_node = self.tree.insert(
                        domain_node, "end", text=f"{split_icon} {split} ({len(samples)})", values=(count_str, ""), open=open_nodes
                    )
                    
                    for sample_id in samples:
                        sample_idx = sample_id.split('/')[-1]
                        item_iid = sample_id
                        sample_date = (sample_date_dict or {}).get(sample_id, "-")
                        self.tree.insert(
                            split_node,
                            "end",
                            iid=item_iid,
                            text=f"  {sample_idx}",
                            values=(label_dict.get(sample_id, "?"), sample_date),
                            tags=("sample",),
                        )
                        visible_sample_ids.append(sample_id)
                        visible_sample_item_by_id[sample_id] = item_iid

        if visible_sample_ids:
            self.lbl_tree_empty.place_forget()
        else:
            self.lbl_tree_empty.place(relx=0.5, rely=0.4, anchor="center")

    def select_sample_in_tree(self, sample_id: str):
        self.tree.selection_set(sample_id)
        self.tree.focus(sample_id)
        self.tree.see(sample_id)

    def _on_tree_right_click(self, event) -> None:
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if "sample" not in self.tree.item(item, "tags"):
            return
        self.tree.selection_set(item)
        self.sample_menu.tk_popup(event.x_root, event.y_root)

    def set_feedback_class_values(self, values: list[str]) -> None:
        self._feedback_class_values = list(values)

    def set_feedback_status(self, text: str) -> None:
        self.var_feedback_status.set(text)

    def refresh_feedback_history(self, entries: list[dict]) -> None:
        self.feedback_history_tree.delete(*self.feedback_history_tree.get_children())
        for idx, entry in enumerate(entries):
            ts = str(entry.get("timestamp") or "")
            ts_short = ts.replace("T", " ")[:19] if ts else "-"
            sample_id = str(entry.get("id") or "")
            verdict = str(entry.get("verdict") or "")
            action = "Correct" if verdict == "thumbs_up" else "Wrong"
            corrected = str(entry.get("corrected_class") or "").strip() or "-"
            iid = f"fb::{idx}"
            self.feedback_history_view.insert((ts_short, sample_id, action, corrected), iid=iid)
        if entries:
            self.lbl_history_empty.place_forget()
        else:
            self.lbl_history_empty.place(relx=0.5, rely=0.55, anchor="center")

    def selected_feedback_sample_id(self) -> str | None:
        sel = self.feedback_history_tree.selection()
        if not sel:
            return None
        vals = self.feedback_history_tree.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return None
        sid = str(vals[1]).strip()
        return sid or None

    def prompt_thumbs_down_feedback(self) -> tuple[str, str] | None:
        """Ask user for corrected class and optional note on thumbs-down."""
        if not self._feedback_class_values:
            return None
        dlg = dialogs.Dialog(self.frame, "Mark As Wrong")
        ttk.Label(dlg.body, text="Correct class (required)", style="Field.TLabel").pack(anchor="w")
        var_class = tk.StringVar(value=self._feedback_class_values[0])
        combo = ttk.Combobox(dlg.body, textvariable=var_class, values=self._feedback_class_values,
                             state="readonly", width=18)
        combo.pack(anchor="w", pady=(SPACE.S, SPACE.L))
        ToolTip(combo, help_key="analysis.feedback_class")

        ttk.Label(dlg.body, text="Note (optional)", style="Field.TLabel").pack(anchor="w")
        var_note = tk.StringVar(value="")
        entry = ttk.Entry(dlg.body, textvariable=var_note, width=42)
        entry.pack(fill="x", pady=(SPACE.S, 0))
        ToolTip(entry, help_key="analysis.feedback_note")

        def _save() -> None:
            cls = var_class.get().strip()
            if not cls:
                return
            dlg.ok((cls, var_note.get().strip()))

        dlg.buttons([("Save", _save, "Accent.TButton"), ("Cancel", dlg.cancel, None)])
        combo.after_idle(combo.focus_set)
        result = dlg.show()
        if not result:
            return None
        return result[0], result[1]

"""UI for the Analysis Tab."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from PIL import Image, ImageTk
import cv2

from gui.components.overlay_renderer import draw_defect_overlay, draw_prediction_overlay
from gui.utils.tooltip import ToolTip
from simple_sim.schema import MetaRow

if TYPE_CHECKING:
    from .tab import AnalysisTab


class AnalysisUI:
    def __init__(self, tab: AnalysisTab, parent: ttk.Frame):
        self.tab = tab
        self.frame = ttk.Frame(parent, padding=10)
        self.var_dataset: tk.StringVar
        self.var_dataset_stats: tk.StringVar
        self.dataset_combo: ttk.Combobox
        self.var_filter: tk.StringVar
        self.var_search: tk.StringVar
        self.var_run: tk.StringVar
        self.var_domain: tk.StringVar
        self.var_split: tk.StringVar
        self.tree: ttk.Treeview
        self.canvas: tk.Canvas
        self.info_text: tk.Text
        self.combo_run: ttk.Combobox
        self.combo_domain: ttk.Combobox
        self.combo_class: ttk.Combobox
        self.photo: Optional[ImageTk.PhotoImage] = None
        self._preview_max_px: int = 420
        self.sample_menu: tk.Menu
        self.var_feedback_status: tk.StringVar
        self._feedback_class_values: List[str]
        self.feedback_history_tree: ttk.Treeview

    def build_ui(self):
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar(value="")
        self.dataset_combo = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=32)
        self.dataset_combo.pack(side="left", padx=(5, 6))
        self.dataset_combo.bind("<<ComboboxSelected>>", self.tab._on_dataset_selected)
        ToolTip(self.dataset_combo, text_func=lambda: self.var_dataset.get())
        ttk.Button(top, text="↻", width=3, command=self.tab._refresh_datasets).pack(side="left", padx=(0, 15))

        ttk.Button(top, text="←", command=self.tab._prev_image, width=3).pack(side="left", padx=2)
        ttk.Button(top, text="→", command=self.tab._next_image, width=3).pack(side="left", padx=2)
        ttk.Button(top, text="Delete Image", command=self.tab._delete_current_image).pack(side="left", padx=(10, 0))
        ttk.Button(top, text="Move To...", command=self.tab._move_current_image).pack(side="left", padx=(6, 0))
        ttk.Button(top, text="Analyze Dataset", command=self.tab._analyze_dataset).pack(side="left", padx=(15, 0))
        ttk.Button(top, text="Recompute with Feedback", command=self.tab._recompute_with_feedback).pack(side="left", padx=(6, 0))
        
        stats_row = ttk.Frame(self.frame)
        stats_row.pack(fill="x", pady=(0, 8))
        self.var_dataset_stats = tk.StringVar(value="Dataset Stats: -")
        ttk.Label(stats_row, textvariable=self.var_dataset_stats).pack(side="left")

        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, padding=5)
        main.add(left, weight=1)
        ttk.Label(left, text="Images").pack(anchor="w")

        filters = ttk.Frame(left)
        filters.pack(fill="x", pady=(5, 0))
        filters.columnconfigure(1, weight=1)

        self.var_search = tk.StringVar(value="")
        ttk.Label(filters, text="Search:").grid(row=0, column=0, sticky="w")
        ent_search = ttk.Entry(filters, textvariable=self.var_search)
        ent_search.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        ttk.Button(filters, text="Clear", command=lambda: self.tab._set_search("")).grid(row=0, column=2, sticky="e")
        ent_search.bind("<KeyRelease>", lambda e: self.tab._schedule_filter_images())

        row2 = ttk.Frame(filters)
        row2.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(3, 0))

        self.var_run = tk.StringVar(value="All")
        ttk.Label(row2, text="Run").pack(side="left")
        self.combo_run = ttk.Combobox(row2, textvariable=self.var_run, values=["All"], state="readonly", width=12)
        self.combo_run.pack(side="left", padx=(4, 10))
        self.combo_run.bind("<<ComboboxSelected>>", lambda e: self.tab._filter_images(display_first=True))

        self.var_domain = tk.StringVar(value="All")
        ttk.Label(row2, text="Domain").pack(side="left")
        self.combo_domain = ttk.Combobox(row2, textvariable=self.var_domain, values=["All"], state="readonly", width=12)
        self.combo_domain.pack(side="left", padx=(4, 10))
        self.combo_domain.bind("<<ComboboxSelected>>", lambda e: self.tab._filter_images(display_first=True))

        self.var_split = tk.StringVar(value="All")
        ttk.Label(row2, text="Split").pack(side="left")
        combo_split = ttk.Combobox(row2, textvariable=self.var_split, values=["All", "train", "val", "test"], state="readonly", width=8)
        combo_split.pack(side="left", padx=(4, 10))
        combo_split.bind("<<ComboboxSelected>>", lambda e: self.tab._filter_images(display_first=True))

        self.var_filter = tk.StringVar(value="All")
        ttk.Label(row2, text="Class").pack(side="left")
        self.combo_class = ttk.Combobox(row2, textvariable=self.var_filter, values=["All"], state="readonly", width=12)
        self.combo_class.pack(side="left", padx=(4, 0))
        self.combo_class.bind("<<ComboboxSelected>>", lambda e: self.tab._filter_images(display_first=True))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.tree = ttk.Treeview(tree_frame, columns=("Class",), show="tree headings", height=20, selectmode="extended")
        self.tree.heading("#0", text="ID")
        self.tree.column("#0", width=200)
        self.tree.heading("Class", text="Class")
        self.tree.column("Class", width=100)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.tab._on_tree_select)
        self.tree.bind("<Button-3>", self._on_tree_right_click)
        self.tree.bind("<Delete>", lambda _e: self.tab._delete_current_image())

        self.sample_menu = tk.Menu(self.tree, tearoff=0)
        self.sample_menu.add_command(label="Delete Image", command=self.tab._delete_current_image)
        self.sample_menu.add_command(label="Move To...", command=self.tab._move_current_image)

        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)
        ttk.Label(right, text="Image Viewer").pack(anchor="w")

        canvas_frame = ttk.Frame(right, relief="sunken", borderwidth=2)
        canvas_frame.pack(fill="x", expand=False, pady=(5, 10))
        self.canvas = tk.Canvas(canvas_frame, bg="gray20", width=self._preview_max_px, height=self._preview_max_px)
        self.canvas.pack(fill="x", expand=False)
        
        ttk.Label(right, text="Metadata:").pack(anchor="w")
        info_frame = ttk.Frame(right, relief="sunken", borderwidth=1)
        info_frame.pack(fill="both", expand=True, pady=(5, 0))
        self.info_text = tk.Text(info_frame, height=12, wrap="word", state="disabled")
        self.info_text.pack(fill="both", expand=True)

        feedback_frame = ttk.LabelFrame(right, text="User Feedback", padding=10)
        feedback_frame.pack(fill="x", pady=(10, 0))
        self.var_feedback_status = tk.StringVar(value="No manual feedback yet.")
        ttk.Label(feedback_frame, textvariable=self.var_feedback_status).pack(anchor="w")

        action_row = tk.Frame(feedback_frame, bg="#d4d7db")
        action_row.pack(fill="x", pady=(8, 4))
        tk.Button(
            action_row,
            text="👍 Correct",
            command=self.tab._feedback_thumbs_up,
            bd=0,
            relief="flat",
            bg="#4b5563",
            fg="white",
            activebackground="#374151",
            activeforeground="white",
            padx=16,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=(8, 6), pady=8)
        tk.Button(
            action_row,
            text="👎 Wrong",
            command=self.tab._feedback_thumbs_down,
            bd=0,
            relief="flat",
            bg="#6b7280",
            fg="white",
            activebackground="#4b5563",
            activeforeground="white",
            padx=16,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=6, pady=8)
        tk.Button(
            action_row,
            text="History",
            command=self.tab._open_feedback_history,
            bd=0,
            relief="flat",
            bg="#9ca3af",
            fg="white",
            activebackground="#6b7280",
            activeforeground="white",
            padx=14,
            pady=8,
            cursor="hand2",
        ).pack(side="left", padx=(6, 8), pady=8)
        self._feedback_class_values = []

        ttk.Label(feedback_frame, text="Recent feedback").pack(anchor="w", pady=(6, 2))
        hist_frame = ttk.Frame(feedback_frame)
        hist_frame.pack(fill="x")
        self.feedback_history_tree = ttk.Treeview(
            hist_frame,
            columns=("Time", "Sample", "Action", "To"),
            show="headings",
            height=6,
        )
        self.feedback_history_tree.heading("Time", text="Time")
        self.feedback_history_tree.heading("Sample", text="Sample")
        self.feedback_history_tree.heading("Action", text="Action")
        self.feedback_history_tree.heading("To", text="To")
        self.feedback_history_tree.column("Time", width=130, anchor="w")
        self.feedback_history_tree.column("Sample", width=240, anchor="w")
        self.feedback_history_tree.column("Action", width=90, anchor="center")
        self.feedback_history_tree.column("To", width=110, anchor="center")
        hist_scroll = ttk.Scrollbar(hist_frame, orient="vertical", command=self.feedback_history_tree.yview)
        self.feedback_history_tree.configure(yscrollcommand=hist_scroll.set)
        self.feedback_history_tree.pack(side="left", fill="x", expand=True)
        hist_scroll.pack(side="right", fill="y")
        self.feedback_history_tree.bind("<<TreeviewSelect>>", self.tab._on_feedback_history_select)

    def display_image(
        self,
        image_path: str,
        meta_row: MetaRow,
        label: str,
        prediction: Optional[Tuple[str, float, dict]],
        original_label: Optional[str] = None,
        feedback: Optional[Dict] = None,
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
            raise RuntimeError(f"Failed to display image: {e}")

    def update_metadata_panel(
        self,
        meta_row: MetaRow,
        ground_truth: str,
        prediction: Optional[Tuple[str, float, dict]],
        original_label: Optional[str] = None,
        feedback: Optional[Dict] = None,
    ):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        
        info = f"Sample ID: {meta_row.id}\\nGround Truth (effective): {ground_truth}\\n"
        if original_label and original_label != ground_truth:
            info += f"Ground Truth (original): {original_label}\\n"
        if prediction:
            predicted, confidence, _ = prediction
            status = "✓ CORRECT" if predicted == ground_truth else "✗ WRONG"
            info += f"Predicted: {predicted} ({confidence:.1%}) {status}\\n"
        else:
            info += "Predicted: (no model loaded)\\n"
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
            info += "\\n"
            if note:
                info += f"Feedback note: {note}\\n"
        
        info += f"\\nDefect Parameters:\\n"
        for key, value in meta_row.defect.items():
            if isinstance(value, float):
                info += f"  {key}: {value:.2f}\n"
            else:
                info += f"  {key}: {value}\n"
        
        self.info_text.insert("1.0", info)
        self.info_text.configure(state="disabled")

    def populate_tree(self, hierarchy: Dict, label_dict: Dict, visible_sample_ids: List[str], visible_sample_item_by_id: Dict[str, str], query: str):
        self.tree.delete(*self.tree.get_children())
        open_nodes = bool(query)
        
        for run_id in sorted(hierarchy.keys()):
            run_node = self.tree.insert("", "end", text=f"📁 {run_id}", values=("",), open=open_nodes)
            for domain in sorted(hierarchy[run_id].keys()):
                domain_node = self.tree.insert(run_node, "end", text=f"🌐 {domain}", values=("",), open=open_nodes)
                for split in sorted(hierarchy[run_id][domain].keys(), key=lambda x: {"train": 0, "val": 1, "test": 2}.get(x, 3)):
                    samples = hierarchy[run_id][domain][split]
                    class_counts: Dict[str, int] = {}
                    for sid in samples:
                        cls = label_dict.get(sid, "?")
                        if not cls:
                            continue
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                    count_str = ", ".join(f"{cls}:{cnt}" for cls, cnt in sorted(class_counts.items()))
                    split_icon = {"train": "🔧", "val": "✓", "test": "🧪"}.get(split, "📂")
                    split_node = self.tree.insert(domain_node, "end", text=f"{split_icon} {split} ({len(samples)})", values=(count_str,), open=open_nodes)
                    
                    for sample_id in sorted(samples):
                        sample_idx = sample_id.split('/')[-1]
                        item_iid = sample_id
                        self.tree.insert(split_node, "end", iid=item_iid, text=f"  {sample_idx}", values=(label_dict.get(sample_id, "?"),), tags=("sample",))
                        visible_sample_ids.append(sample_id)
                        visible_sample_item_by_id[sample_id] = item_iid
                        
        self.tree.tag_configure("sample", foreground="black")

    def select_sample_in_tree(self, sample_id: str):
        self.tree.selection_set(sample_id)
        self.tree.see(sample_id)

    def _on_tree_right_click(self, event) -> None:
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if "sample" not in self.tree.item(item, "tags"):
            return
        self.tree.selection_set(item)
        self.sample_menu.tk_popup(event.x_root, event.y_root)

    def set_feedback_class_values(self, values: List[str]) -> None:
        self._feedback_class_values = list(values)

    def set_feedback_status(self, text: str) -> None:
        self.var_feedback_status.set(text)

    def refresh_feedback_history(self, entries: List[Dict]) -> None:
        self.feedback_history_tree.delete(*self.feedback_history_tree.get_children())
        for idx, entry in enumerate(entries):
            ts = str(entry.get("timestamp") or "")
            ts_short = ts.replace("T", " ")[:19] if ts else "-"
            sample_id = str(entry.get("id") or "")
            verdict = str(entry.get("verdict") or "")
            action = "👍" if verdict == "thumbs_up" else "👎"
            corrected = str(entry.get("corrected_class") or "").strip() or "-"
            iid = f"fb::{idx}"
            self.feedback_history_tree.insert("", "end", iid=iid, values=(ts_short, sample_id, action, corrected))

    def selected_feedback_sample_id(self) -> Optional[str]:
        sel = self.feedback_history_tree.selection()
        if not sel:
            return None
        vals = self.feedback_history_tree.item(sel[0], "values")
        if not vals or len(vals) < 2:
            return None
        sid = str(vals[1]).strip()
        return sid or None

    def prompt_thumbs_down_feedback(self) -> Optional[Tuple[str, str]]:
        """Ask user for corrected class and optional note on thumbs-down."""
        if not self._feedback_class_values:
            return None
        dlg = tk.Toplevel(self.frame.winfo_toplevel())
        dlg.title("Mark As Wrong")
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()
        dlg.configure(bg="#d4d7db")

        wrap = ttk.Frame(dlg, padding=12)
        wrap.pack(fill="both", expand=True)
        ttk.Label(wrap, text="Select correct class (required):").pack(anchor="w")

        var_class = tk.StringVar(value=self._feedback_class_values[0])
        combo = ttk.Combobox(wrap, textvariable=var_class, values=self._feedback_class_values, state="readonly", width=18)
        combo.pack(anchor="w", pady=(4, 10))
        combo.focus_set()

        ttk.Label(wrap, text="Note (optional):").pack(anchor="w")
        var_note = tk.StringVar(value="")
        entry = ttk.Entry(wrap, textvariable=var_note, width=42)
        entry.pack(fill="x", pady=(4, 12))

        result: dict[str, str] = {}

        def _save() -> None:
            cls = var_class.get().strip()
            if not cls:
                return
            result["class"] = cls
            result["note"] = var_note.get().strip()
            dlg.destroy()

        btn_row = ttk.Frame(wrap)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side="right")
        ttk.Button(btn_row, text="Save", command=_save).pack(side="right", padx=(0, 8))

        dlg.wait_window()
        if "class" not in result:
            return None
        return result["class"], result.get("note", "")

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
        
    def display_image(self, image_path: str, meta_row: MetaRow, label: str, prediction: Optional[Tuple[str, float, dict]]):
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
            
            self.update_metadata_panel(meta_row, label, prediction)
        except Exception as e:
            raise RuntimeError(f"Failed to display image: {e}")

    def update_metadata_panel(self, meta_row: MetaRow, ground_truth: str, prediction: Optional[Tuple[str, float, dict]]):
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        
        info = f"Sample ID: {meta_row.id}\\nGround Truth: {ground_truth}\\n"
        if prediction:
            predicted, confidence, _ = prediction
            status = "✓ CORRECT" if predicted == ground_truth else "✗ WRONG"
            info += f"Predicted: {predicted} ({confidence:.1%}) {status}\\n"
        else:
            info += "Predicted: (no model loaded)\\n"
        
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

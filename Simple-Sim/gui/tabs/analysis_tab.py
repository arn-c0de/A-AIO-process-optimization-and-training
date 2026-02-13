"""Analysis Tab - Image browser with defect overlays and predictions."""

from __future__ import annotations
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import threading
import json

import cv2
import numpy as np
from PIL import Image, ImageTk

from .base_tab import BaseTab
from gui.state import UiState
from gui.components.image_cache import ImageCache
from gui.components.overlay_renderer import draw_defect_overlay, draw_prediction_overlay
from gui.utils.model_inference import load_model, ModelWrapper

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_sim.schema import read_jsonl, MetaRow, LabelRow


class AnalysisTab(BaseTab):
    """Tab 2: Interactive image browser with analysis."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.image_cache = ImageCache()
        self.model: Optional[ModelWrapper] = None

        self.sample_ids: list[str] = []
        self.meta_dict: dict[str, MetaRow] = {}
        self.label_dict: dict[str, str] = {}
        self.current_sample_id: Optional[str] = None
        self.visible_sample_ids: list[str] = []
        self.visible_sample_item_by_id: dict[str, str] = {}
        self._filter_after_id: Optional[str] = None

        # UI components
        self.var_dataset: tk.StringVar
        self.var_filter: tk.StringVar  # class
        self.var_search: tk.StringVar
        self.var_run: tk.StringVar
        self.var_domain: tk.StringVar
        self.var_split: tk.StringVar
        self.tree: ttk.Treeview
        self.canvas: tk.Canvas
        self.canvas_image: Optional[int] = None
        self.info_text: tk.Text
        self.combo_run: ttk.Combobox
        self.combo_domain: ttk.Combobox
        self._preview_max_px: int = 420

    def build_ui(self) -> None:
        """Build the analysis UI."""
        # Note: Don't pack self.frame - it's managed by the notebook

        # Top controls
        top = ttk.Frame(self.frame)
        top.pack(fill="x", pady=(0, 10))

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.var_dataset = tk.StringVar()
        dataset_entry = ttk.Entry(top, textvariable=self.var_dataset, width=50, state="readonly")
        dataset_entry.pack(side="left", padx=(5, 15))

        ttk.Button(top, text="←", command=self._prev_image, width=3).pack(side="left", padx=2)
        ttk.Button(top, text="→", command=self._next_image, width=3).pack(side="left", padx=2)

        ttk.Button(top, text="Analyze Dataset", command=self._analyze_dataset).pack(side="left", padx=(15, 0))

        # Main split
        main = ttk.Panedwindow(self.frame, orient="horizontal")
        main.pack(fill="both", expand=True)

        # Left: Image list
        left = ttk.Frame(main, padding=5)
        main.add(left, weight=1)

        ttk.Label(left, text="Images").pack(anchor="w")

        # List-local search + filters (compact)
        filters = ttk.Frame(left)
        filters.pack(fill="x", pady=(5, 0))
        filters.columnconfigure(1, weight=1)

        self.var_search = tk.StringVar(value="")
        ttk.Label(filters, text="Search:").grid(row=0, column=0, sticky="w")
        ent_search = ttk.Entry(filters, textvariable=self.var_search)
        ent_search.grid(row=0, column=1, sticky="ew", padx=(5, 5))
        ttk.Button(filters, text="Clear", command=lambda: self._set_search("")).grid(row=0, column=2, sticky="e")
        ent_search.bind("<KeyRelease>", lambda e: self._schedule_filter_images())

        # Second row: filters in one line to reduce vertical height
        row2 = ttk.Frame(filters)
        row2.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(3, 0))

        self.var_run = tk.StringVar(value="All")
        ttk.Label(row2, text="Run").pack(side="left")
        self.combo_run = ttk.Combobox(row2, textvariable=self.var_run, values=["All"], state="readonly", width=12)
        self.combo_run.pack(side="left", padx=(4, 10))
        self.combo_run.bind("<<ComboboxSelected>>", lambda e: self._filter_images(display_first=True))

        self.var_domain = tk.StringVar(value="All")
        ttk.Label(row2, text="Domain").pack(side="left")
        self.combo_domain = ttk.Combobox(row2, textvariable=self.var_domain, values=["All"], state="readonly", width=12)
        self.combo_domain.pack(side="left", padx=(4, 10))
        self.combo_domain.bind("<<ComboboxSelected>>", lambda e: self._filter_images(display_first=True))

        self.var_split = tk.StringVar(value="All")
        ttk.Label(row2, text="Split").pack(side="left")
        combo_split = ttk.Combobox(row2, textvariable=self.var_split, values=["All", "train", "val", "test"], state="readonly", width=8)
        combo_split.pack(side="left", padx=(4, 10))
        combo_split.bind("<<ComboboxSelected>>", lambda e: self._filter_images(display_first=True))

        self.var_filter = tk.StringVar(value="All")
        ttk.Label(row2, text="Class").pack(side="left")
        combo_class = ttk.Combobox(
            row2,
            textvariable=self.var_filter,
            values=["All", "OK", "MISSING", "MISALIGNED", "TOMBSTONE"],
            state="readonly",
            width=12,
        )
        combo_class.pack(side="left", padx=(4, 0))
        combo_class.bind("<<ComboboxSelected>>", lambda e: self._filter_images(display_first=True))

        # Tree with scrollbar
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.tree = ttk.Treeview(tree_frame, columns=("Class",), show="tree headings", height=20)
        self.tree.heading("#0", text="ID")
        self.tree.heading("Class", text="Class")
        self.tree.column("#0", width=200)
        self.tree.column("Class", width=100)

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Right: Image viewer + metadata
        right = ttk.Frame(main, padding=5)
        main.add(right, weight=3)

        ttk.Label(right, text="Image Viewer").pack(anchor="w")

        # Canvas for image display
        canvas_frame = ttk.Frame(right, relief="sunken", borderwidth=2)
        # Keep the preview a bit smaller so the metadata/log area stays readable.
        canvas_frame.pack(fill="x", expand=False, pady=(5, 10))

        self.canvas = tk.Canvas(canvas_frame, bg="gray20", width=self._preview_max_px, height=self._preview_max_px)
        self.canvas.pack(fill="x", expand=False)

        # Metadata panel
        ttk.Label(right, text="Metadata:").pack(anchor="w")

        info_frame = ttk.Frame(right, relief="sunken", borderwidth=1)
        info_frame.pack(fill="both", expand=True, pady=(5, 0))

        self.info_text = tk.Text(info_frame, height=12, wrap="word")
        self.info_text.pack(fill="both", expand=True)
        self.info_text.configure(state="disabled")

        # Load initial dataset
        self.on_dataset_changed()

    def on_dataset_changed(self) -> None:
        """Handle dataset change from other tabs."""
        if self.state.dataset_dir:
            self.var_dataset.set(str(self.state.dataset_dir))
            self._load_dataset()
            self._try_load_model()

    def _load_dataset(self) -> None:
        """Load dataset metadata."""
        if not self.state.dataset_dir:
            return

        try:
            meta_rows = read_jsonl(self.state.dataset_dir / "meta.jsonl", MetaRow)
            label_rows = read_jsonl(self.state.dataset_dir / "labels.jsonl", LabelRow)

            self.meta_dict = {row.id: row for row in meta_rows}
            self.label_dict = {row.id: row.class_name for row in label_rows}
            self.sample_ids = [row.id for row in meta_rows]

            # Update filter dropdown values based on the dataset.
            run_ids: set[str] = set()
            domains: set[str] = set()
            for sid in self.sample_ids:
                parts = sid.split("/")
                if len(parts) >= 2:
                    run_ids.add(parts[0])
                    domains.add(parts[1])

            if hasattr(self, "combo_run"):
                run_values = ["All"] + sorted(run_ids)
                self.combo_run.configure(values=run_values)
                if self.var_run.get() not in run_values:
                    self.var_run.set("All")

            if hasattr(self, "combo_domain"):
                domain_values = ["All"] + sorted(domains)
                self.combo_domain.configure(values=domain_values)
                if self.var_domain.get() not in domain_values:
                    self.var_domain.set("All")

            self._populate_tree()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load dataset:\n{e}")

    def _populate_tree(self) -> None:
        """Populate tree with hierarchical structure: Dataset → Domain → Split → Samples."""
        self.tree.delete(*self.tree.get_children())

        self.visible_sample_ids = []
        self.visible_sample_item_by_id = {}

        filter_class = self.var_filter.get().strip()
        filter_run = self.var_run.get().strip()
        filter_domain = self.var_domain.get().strip()
        filter_split = self.var_split.get().strip()
        query = self.var_search.get().strip().lower()

        # Organize samples by hierarchy
        hierarchy = {}  # run_id -> domain -> split -> [sample_ids]

        for sample_id in self.sample_ids:
            class_name = self.label_dict.get(sample_id, "?")

            # Apply filter
            if filter_class != "All" and class_name != filter_class:
                continue

            # Parse sample_id: "run_0001/domain_A/train/000042"
            parts = sample_id.split('/')
            if len(parts) >= 3:
                run_id = parts[0]
                domain = parts[1]
                split = parts[2]

                if filter_run != "All" and run_id != filter_run:
                    continue
                if filter_domain != "All" and domain != filter_domain:
                    continue
                if filter_split != "All" and split != filter_split:
                    continue

                if query:
                    sample_idx = parts[-1]
                    haystack = f"{sample_id} {sample_idx} {class_name}".lower()
                    if query not in haystack:
                        continue

                if run_id not in hierarchy:
                    hierarchy[run_id] = {}
                if domain not in hierarchy[run_id]:
                    hierarchy[run_id][domain] = {}
                if split not in hierarchy[run_id][domain]:
                    hierarchy[run_id][domain][split] = []

                hierarchy[run_id][domain][split].append(sample_id)

        # Build tree hierarchically
        open_nodes = bool(query)
        for run_id in sorted(hierarchy.keys()):
            # Insert run_id node
            run_node = self.tree.insert("", "end", text=f"📁 {run_id}", values=("",), open=open_nodes)

            for domain in sorted(hierarchy[run_id].keys()):
                # Insert domain node
                domain_node = self.tree.insert(run_node, "end", text=f"🌐 {domain}", values=("",), open=open_nodes)

                for split in sorted(hierarchy[run_id][domain].keys(),
                                  key=lambda x: {"train": 0, "val": 1, "test": 2}.get(x, 3)):
                    samples = hierarchy[run_id][domain][split]

                    # Count classes in this split
                    class_counts = {}
                    for sid in samples:
                        cls = self.label_dict.get(sid, "?")
                        class_counts[cls] = class_counts.get(cls, 0) + 1

                    count_str = ", ".join(f"{cls}:{cnt}" for cls, cnt in sorted(class_counts.items()))

                    # Insert split node with count
                    split_icon = {"train": "🔧", "val": "✓", "test": "🧪"}.get(split, "📂")
                    split_node = self.tree.insert(
                        domain_node, "end",
                        text=f"{split_icon} {split} ({len(samples)})",
                        values=(count_str,),
                        open=open_nodes
                    )

                    # Insert individual samples
                    for sample_id in sorted(samples):
                        class_name = self.label_dict.get(sample_id, "?")
                        # Just show the sample index, not full path
                        sample_idx = sample_id.split('/')[-1] if '/' in sample_id else sample_id
                        item_iid = sample_id
                        self.tree.insert(
                            split_node, "end",
                            iid=item_iid,
                            text=f"  {sample_idx}",
                            values=(class_name,),
                            tags=("sample",)
                        )
                        self.visible_sample_ids.append(sample_id)
                        self.visible_sample_item_by_id[sample_id] = item_iid

        # Tag configuration for styling
        self.tree.tag_configure("sample", foreground="black")

    def _filter_images(self, *, display_first: bool = True) -> None:
        """Filter images by search/filters and refresh the list."""
        self._populate_tree()
        # Keep current selection if it still exists, otherwise select the first visible sample.
        if self.current_sample_id and self.current_sample_id in self.visible_sample_item_by_id:
            self._select_sample_id(self.current_sample_id, display=False)
        elif self.visible_sample_ids:
            self._select_sample_id(self.visible_sample_ids[0], display=display_first)

    def _schedule_filter_images(self) -> None:
        """Debounce filtering while typing."""
        if self._filter_after_id is not None:
            try:
                self.frame.after_cancel(self._filter_after_id)
            except Exception:
                pass
        self._filter_after_id = self.frame.after(150, lambda: self._filter_images(display_first=False))

    def _set_search(self, value: str) -> None:
        self.var_search.set(value)
        self._filter_images(display_first=True)

    def _on_tree_select(self, event) -> None:
        """Handle tree selection."""
        selection = self.tree.selection()
        if not selection:
            return

        item = selection[0]
        tags = self.tree.item(item, "tags")

        # Only display if it's a sample (leaf node)
        if "sample" not in tags:
            return

        # Sample leaf nodes use the full sample_id as iid.
        self._display_image(item)

    def _display_image(self, sample_id: str) -> None:
        """Display selected image with overlays."""
        if sample_id not in self.meta_dict:
            return

        self.current_sample_id = sample_id
        meta_row = self.meta_dict[sample_id]
        image_path = self.state.dataset_dir / meta_row.image_path

        if not image_path.exists():
            messagebox.showerror("Error", f"Image not found:\n{image_path}")
            return

        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError("Failed to read image")

            # Draw defect overlay
            img_overlay = draw_defect_overlay(img, meta_row.defect, meta_row.nominal)

            # Draw prediction overlay if model available
            if self.model:
                try:
                    predicted, confidence, _ = self.model.predict(image_path)
                    ground_truth = self.label_dict.get(sample_id, "?")
                    img_overlay = draw_prediction_overlay(img_overlay, ground_truth, predicted, confidence)
                except Exception as e:
                    print(f"Prediction failed: {e}")

            # Convert to PhotoImage and display
            img_rgb = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((self._preview_max_px, self._preview_max_px))

            self.photo = ImageTk.PhotoImage(img_pil)

            # Update canvas
            self.canvas.delete("all")
            cx = max(1, int(self.canvas.winfo_width() / 2))
            cy = max(1, int(self.canvas.winfo_height() / 2))
            self.canvas.create_image(cx, cy, image=self.photo, anchor="center")

            # Update metadata panel
            self._update_metadata_panel(meta_row, sample_id)

        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display image:\n{e}")

    def _select_sample_id(self, sample_id: str, *, display: bool = True) -> None:
        """Select a sample in the tree (and optionally display it)."""
        item = self.visible_sample_item_by_id.get(sample_id)
        if not item:
            return
        self.tree.selection_set(item)
        self.tree.see(item)
        if display:
            self._display_image(sample_id)

    def _update_metadata_panel(self, meta_row: MetaRow, sample_id: str) -> None:
        """Update metadata info panel."""
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")

        ground_truth = self.label_dict.get(sample_id, "?")

        info = f"Sample ID: {sample_id}\n"
        info += f"Ground Truth: {ground_truth}\n"

        # Prediction info
        if self.model:
            try:
                predicted, confidence, probs = self.model.predict(self.state.dataset_dir / meta_row.image_path)
                status = "✓ CORRECT" if predicted == ground_truth else "✗ WRONG"
                info += f"Predicted: {predicted} ({confidence:.1%}) {status}\n"
            except:
                info += "Predicted: (error)\n"
        else:
            info += "Predicted: (no model loaded)\n"

        info += f"\nDefect Parameters:\n"
        for key, value in meta_row.defect.items():
            if isinstance(value, float):
                info += f"  {key}: {value:.2f}\n"
            else:
                info += f"  {key}: {value}\n"

        self.info_text.insert("1.0", info)
        self.info_text.configure(state="disabled")

    def _prev_image(self) -> None:
        """Navigate to previous image."""
        if not self.visible_sample_ids:
            return

        if self.current_sample_id in self.visible_sample_ids:
            idx = self.visible_sample_ids.index(self.current_sample_id)  # O(n), fine for GUI scale
        else:
            idx = 0
        prev_id = self.visible_sample_ids[(idx - 1) % len(self.visible_sample_ids)]
        self._select_sample_id(prev_id)

    def _next_image(self) -> None:
        """Navigate to next image."""
        if not self.visible_sample_ids:
            return

        if self.current_sample_id in self.visible_sample_ids:
            idx = self.visible_sample_ids.index(self.current_sample_id)  # O(n), fine for GUI scale
        else:
            idx = 0
        next_id = self.visible_sample_ids[(idx + 1) % len(self.visible_sample_ids)]
        self._select_sample_id(next_id)

    def _try_load_model(self) -> None:
        """Try to load trained model for dataset."""
        if not self.state.dataset_dir:
            return

        model_path = self.sim_root / "outputs" / "models" / f"{self.state.dataset_dir.name}.pt"

        if model_path.exists():
            try:
                self.model = load_model(model_path)
                print(f"Loaded model: {model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.model = None
        else:
            self.model = None

    def _analyze_dataset(self) -> None:
        """Run batch analysis on entire dataset."""
        if not self.state.dataset_dir or not self.model:
            messagebox.showinfo("Info", "No model loaded. Train a model first.")
            return

        # Run in background thread
        def analyze():
            try:
                results = {
                    'dataset_path': str(self.state.dataset_dir),
                    'model_path': str(self.model.model_path),
                    'samples': [],
                    'summary': {}
                }

                correct_count = 0
                total_count = len(self.sample_ids)

                for sample_id in self.sample_ids:
                    meta_row = self.meta_dict[sample_id]
                    ground_truth = self.label_dict[sample_id]
                    image_path = self.state.dataset_dir / meta_row.image_path

                    try:
                        predicted, confidence, _ = self.model.predict(image_path)
                        is_correct = (predicted == ground_truth)

                        if is_correct:
                            correct_count += 1

                        results['samples'].append({
                            'id': sample_id,
                            'ground_truth': ground_truth,
                            'predicted': predicted,
                            'confidence': confidence,
                            'correct': is_correct
                        })
                    except Exception as e:
                        print(f"Failed to analyze {sample_id}: {e}")

                results['summary'] = {
                    'total': total_count,
                    'correct': correct_count,
                    'accuracy': correct_count / total_count if total_count > 0 else 0
                }

                # Save results
                output_path = self.state.dataset_dir / "analysis_results.json"
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)

                messagebox.showinfo("Success",
                                   f"Analysis complete!\n\n"
                                   f"Accuracy: {results['summary']['accuracy']:.1%}\n"
                                   f"Results saved to:\n{output_path}")

            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed:\n{e}")

        thread = threading.Thread(target=analyze, daemon=True)
        thread.start()

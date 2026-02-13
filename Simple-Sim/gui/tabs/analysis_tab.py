"""Analysis Tab - Image browser with defect overlays and predictions."""

from __future__ import annotations
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional
import threading
import json
import os

import torch

import cv2
import numpy as np
from PIL import Image, ImageTk

from .base_tab import BaseTab
from gui.state import UiState
from gui.components.image_cache import ImageCache
from gui.components.overlay_renderer import draw_defect_overlay, draw_prediction_overlay
from gui.utils.model_inference import load_model, ModelWrapper
from gui.utils.tooltip import ToolTip
from gui.utils.settings_store import SettingsStore

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from simple_sim.schema import read_jsonl, MetaRow, LabelRow
from simple_sim.manifest import hash_file


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
        self.dataset_combo: ttk.Combobox
        self._dataset_by_display: dict[str, Path] = {}
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
        self.var_dataset = tk.StringVar(value="")
        self.dataset_combo = ttk.Combobox(top, textvariable=self.var_dataset, state="readonly", width=32)
        self.dataset_combo.pack(side="left", padx=(5, 6))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._on_dataset_selected)
        ToolTip(self.dataset_combo, text_func=lambda: self.var_dataset.get())
        ttk.Button(top, text="↻", width=3, command=self._refresh_datasets).pack(side="left", padx=(0, 15))

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
        self._refresh_datasets()
        self.on_dataset_changed()
        self._load_persisted_settings()
        self._wire_settings_autosave()

    def on_dataset_changed(self) -> None:
        """Handle dataset change from other tabs."""
        if not self.state.dataset_dir:
            return

        # Show a human-friendly dataset name (not full path).
        display = self._display_for_dataset(self.state.dataset_dir)
        # Prefer keeping current selection if it matches mapping.
        if display in self._dataset_by_display:
            self.var_dataset.set(display)
        else:
            # Refresh list and try again.
            self._refresh_datasets()
            display = self._display_for_dataset(self.state.dataset_dir)
            self.var_dataset.set(display)

        self._load_dataset()
        self._try_load_model()

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None:
            return
        v = st.get("analysis.dataset_selection")
        if isinstance(v, str) and v:
            try:
                self.var_dataset.set(v)
                self._on_dataset_selected()
            except Exception:
                pass
        for key, var in [
            ("analysis.search", self.var_search),
            ("analysis.class", self.var_filter),
            ("analysis.run", self.var_run),
            ("analysis.domain", self.var_domain),
            ("analysis.split", self.var_split),
        ]:
            vv = st.get(key)
            if vv is None:
                continue
            try:
                var.set(vv)
            except Exception:
                pass
        try:
            self._filter_images(display_first=False)
        except Exception:
            pass

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None:
            return

        def bind(var, key: str) -> None:
            def cb(*_a) -> None:
                try:
                    st.set(key, var.get())
                    st.schedule_save(self.frame)
                except Exception:
                    pass

            try:
                var.trace_add("write", cb)
            except Exception:
                try:
                    var.trace("w", cb)
                except Exception:
                    pass

        bind(self.var_dataset, "analysis.dataset_selection")
        bind(self.var_search, "analysis.search")
        bind(self.var_filter, "analysis.class")
        bind(self.var_run, "analysis.run")
        bind(self.var_domain, "analysis.domain")
        bind(self.var_split, "analysis.split")

    def _sim_data_roots(self) -> tuple[Path, Path]:
        sim_data = self.sim_root / "outputs" / "sim_data"
        return sim_data / "runs", sim_data / "versions"

    def _display_for_dataset(self, p: Path) -> str:
        """Compact label for datasets in the dropdown."""
        try:
            runs, versions = self._sim_data_roots()
            rp = p.resolve()
            if str(rp).startswith(str(runs.resolve()) + os.sep):
                return rp.name
            if str(rp).startswith(str(versions.resolve()) + os.sep):
                # versions/<group>/<snapshot_dir>
                return f"{rp.parent.name}:{rp.name}"
        except Exception:
            pass
        return p.name

    def _refresh_datasets(self) -> None:
        """Populate dataset dropdown from runs/ and versions/ snapshots."""
        runs, versions = self._sim_data_roots()
        runs.mkdir(parents=True, exist_ok=True)
        versions.mkdir(parents=True, exist_ok=True)

        cand: list[Path] = []
        cand.extend([p for p in runs.iterdir() if p.is_dir()])
        for p in versions.glob("*/*"):
            if p.is_dir():
                cand.append(p)

        # Sort runs by name, snapshots by mtime desc
        def sort_key(p: Path) -> tuple:
            try:
                rp = p.resolve()
                is_ver = str(rp).startswith(str(versions.resolve()) + os.sep)
                mt = p.stat().st_mtime
            except Exception:
                is_ver = False
                mt = 0.0
            if is_ver:
                return (1, -mt, p.name)
            return (0, p.name, -mt)

        cand.sort(key=sort_key)

        self._dataset_by_display = {}
        displays: list[str] = []
        seen: dict[str, int] = {}
        for p in cand:
            base = self._display_for_dataset(p)
            n = seen.get(base, 0) + 1
            seen[base] = n
            disp = base if n == 1 else f"{base} ({n})"
            self._dataset_by_display[disp] = p
            displays.append(disp)

        try:
            self.dataset_combo["values"] = displays
        except Exception:
            return

        # Keep selection if possible; otherwise choose state.dataset_dir or latest run.
        cur = self.var_dataset.get().strip()
        if cur and cur in self._dataset_by_display:
            return
        if self.state.dataset_dir:
            want = self._display_for_dataset(self.state.dataset_dir)
            # Might have been disambiguated.
            for d, pp in self._dataset_by_display.items():
                if pp == self.state.dataset_dir:
                    self.var_dataset.set(d)
                    return
                if d == want:
                    self.var_dataset.set(d)
                    return
        if displays:
            self.var_dataset.set(displays[0])

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        disp = self.var_dataset.get().strip()
        if not disp:
            return
        ds = self._dataset_by_display.get(disp)
        if not ds:
            return
        self.state.dataset_dir = ds
        # Notify other tabs / keep global state coherent.
        try:
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception:
            pass
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
        self.model = None
        ds = self.state.dataset_dir
        if not ds:
            self.state.current_model_path = None
            return

        manifest_path = ds / "dataset_manifest.json"
        manifest_hash = None
        if manifest_path.exists():
            try:
                manifest_hash = hash_file(manifest_path)
            except Exception as e:
                print(f"Failed to hash dataset manifest: {e}")

        explicit = self.state.current_model_path
        if explicit and explicit.exists():
            if manifest_hash is None or self._checkpoint_matches_manifest(explicit, manifest_hash):
                if self._load_model_from_path(explicit):
                    return

        dataset_model = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        if dataset_model.exists() and self._load_model_from_path(dataset_model):
            return

        if manifest_hash:
            match = self._find_model_by_manifest_hash(manifest_hash)
            if match and self._load_model_from_path(match):
                return

        self.state.current_model_path = None

    def _load_model_from_path(self, path: Path) -> bool:
        """Load the model from disk, handling errors."""
        try:
            self.model = load_model(path)
            self.state.current_model_path = path
            print(f"Loaded model: {path}")
            return True
        except Exception as e:
            print(f"Failed to load model {path}: {e}")
            self.model = None
            return False

    def _find_model_by_manifest_hash(self, manifest_hash: str) -> Optional[Path]:
        """Search for a checkpoint whose manifest hash matches the dataset."""
        models_root = self.sim_root / "outputs" / "models"
        candidates = []
        for glob_pattern in [
            models_root.glob("*.pt"),
            (models_root / "imports").glob("*.pt"),
            (models_root / "versions").glob("**/*.pt"),
        ]:
            for p in glob_pattern:
                if p.is_file():
                    candidates.append(p)

        seen: dict[str, Path] = {}
        for p in candidates:
            try:
                rp = str(p.resolve())
            except Exception:
                rp = str(p)
            if rp not in seen:
                seen[rp] = p

        ordered = sorted(
            seen.values(),
            key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
            reverse=True,
        )

        for path in ordered:
            ckpt_hash = self._checkpoint_manifest_hash(path)
            if ckpt_hash == manifest_hash:
                return path

        return None

    def _checkpoint_matches_manifest(self, path: Path, manifest_hash: str) -> bool:
        """Return True if checkpoint manifest hash equals target."""
        ckpt_hash = self._checkpoint_manifest_hash(path)
        return ckpt_hash == manifest_hash if ckpt_hash else False

    def _checkpoint_manifest_hash(self, path: Path) -> Optional[str]:
        try:
            checkpoint = torch.load(path, map_location="cpu")
        except Exception as exc:
            print(f"Failed to read checkpoint metadata {path}: {exc}")
            return None
        return checkpoint.get("dataset_manifest_hash")
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

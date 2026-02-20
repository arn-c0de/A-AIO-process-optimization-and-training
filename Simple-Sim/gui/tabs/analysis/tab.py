"""Analysis Tab"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Dict, List
import threading

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore
from gui.utils.dataset_ops import delete_samples, move_samples
from .ui import AnalysisUI
from .logic import AnalysisLogic
from simple_sim.schema import MetaRow


class AnalysisTab(BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.ui = AnalysisUI(self, self.frame)
        self.logic = AnalysisLogic(sim_root)
        self.sample_ids: list[str] = []
        self.meta_dict: dict[str, MetaRow] = {}
        self.label_dict: dict[str, str] = {}
        self.current_sample_id: Optional[str] = None
        self.visible_sample_ids: list[str] = []
        self.visible_sample_item_by_id: dict[str, str] = {}
        self._filter_after_id: Optional[str] = None
        self._dataset_by_display: dict[str, Path] = {}

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self._refresh_datasets()
        self.on_dataset_changed()
        self._load_persisted_settings()
        self._wire_settings_autosave()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir: return
        display = self._display_for_dataset(self.state.dataset_dir)
        if display in self._dataset_by_display:
            self.ui.var_dataset.set(display)
        else:
            self._refresh_datasets()
            self.ui.var_dataset.set(self._display_for_dataset(self.state.dataset_dir))
        self._load_dataset()
        self._try_load_model()

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None: return
        if v := st.get("analysis.dataset_selection"):
            try:
                self.ui.var_dataset.set(v)
                self._on_dataset_selected()
            except Exception: pass
        for key, var in [("analysis.search", self.ui.var_search), ("analysis.class", self.ui.var_filter), 
                         ("analysis.run", self.ui.var_run), ("analysis.domain", self.ui.var_domain), 
                         ("analysis.split", self.ui.var_split)]:
            if vv := st.get(key):
                try: var.set(vv)
                except Exception: pass
        self._filter_images(display_first=False)

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None: return
        for var, key in [(self.ui.var_dataset, "analysis.dataset_selection"), (self.ui.var_search, "analysis.search"),
                         (self.ui.var_filter, "analysis.class"), (self.ui.var_run, "analysis.run"),
                         (self.ui.var_domain, "analysis.domain"), (self.ui.var_split, "analysis.split")]:
            var.trace_add("write", lambda *a, v=var, k=key: (st.set(k, v.get()), st.schedule_save(self.frame)))

    def _display_for_dataset(self, p: Path) -> str:
        try:
            runs, versions = self.logic.sim_data_roots()
            rp = p.resolve()
            if str(rp).startswith(str(runs.resolve()) + "/"): return rp.name
            if str(rp).startswith(str(versions.resolve()) + "/"): return f"{rp.parent.name}:{rp.name}"
        except Exception: pass
        return p.name

    def _refresh_datasets(self) -> None:
        cand = self.logic.get_datasets()
        cand.sort(key=lambda p: (1, -p.stat().st_mtime, p.name) if "versions" in str(p) else (0, p.name, -p.stat().st_mtime))
        
        self._dataset_by_display.clear()
        displays: list[str] = []
        seen: dict[str, int] = {}
        for p in cand:
            base = self._display_for_dataset(p)
            disp = f"{base} ({seen.get(base, 0)})" if (seen.update({base: seen.get(base, 0) + 1})) else base
            self._dataset_by_display[disp] = p
            displays.append(disp)
        
        self.ui.dataset_combo["values"] = displays
        if (cur := self.ui.var_dataset.get().strip()) and cur in self._dataset_by_display: return
        
        if self.state.dataset_dir:
            want = self._display_for_dataset(self.state.dataset_dir)
            for d, pp in self._dataset_by_display.items():
                if pp == self.state.dataset_dir or d == want:
                    self.ui.var_dataset.set(d)
                    return
        if displays: self.ui.var_dataset.set(displays[0])

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        if not (ds := self._dataset_by_display.get(self.ui.var_dataset.get().strip())): return
        self.state.dataset_dir = ds
        try: self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception: pass
        self._load_dataset()
        self._try_load_model()

    def _load_dataset(self) -> None:
        if not self.state.dataset_dir: return
        try:
            self.meta_dict, self.label_dict, self.sample_ids = self.logic.load_dataset(self.state.dataset_dir)
            
            run_ids = {"All"} | {sid.split('/')[0] for sid in self.sample_ids if '/' in sid}
            domains = {"All"} | {sid.split('/')[1] for sid in self.sample_ids if len(sid.split('/')) > 1}
            classes = {"All"} | {c for c in self.label_dict.values() if c and c != "?"}
            
            self.ui.combo_run.configure(values=sorted(list(run_ids)))
            self.ui.combo_domain.configure(values=sorted(list(domains)))
            self.ui.combo_class.configure(values=["All"] + self._sorted_class_values(classes - {"All"}))

            self._filter_images()
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load dataset:\\n{e}")

    def _sorted_class_values(self, classes: set[str]) -> list[str]:
        priority = ["OK", "MISSING", "MISALIGNED", "TOMBSTONE"]
        return [p for p in priority if p in classes] + sorted([c for c in classes if c not in priority])

    def _filter_images(self, display_first: bool = True):
        hierarchy, self.visible_sample_ids, self.visible_sample_item_by_id = {}, [], {}
        query = self.ui.var_search.get().strip().lower()
        filters = {
            'class': self.ui.var_filter.get().strip(),
            'run': self.ui.var_run.get().strip(),
            'domain': self.ui.var_domain.get().strip(),
            'split': self.ui.var_split.get().strip()
        }

        for sid in self.sample_ids:
            if filters['class'] != "All" and self.label_dict.get(sid, "?") != filters['class']: continue
            parts = sid.split('/')
            if len(parts) >= 3:
                run, domain, split = parts[0], parts[1], parts[2]
                if filters['run'] != "All" and run != filters['run']: continue
                if filters['domain'] != "All" and domain != filters['domain']: continue
                if filters['split'] != "All" and split != filters['split']: continue
                if query and query not in f"{sid} {sid.split('/')[-1]} {self.label_dict.get(sid, '?')}".lower(): continue

                hierarchy.setdefault(run, {}).setdefault(domain, {}).setdefault(split, []).append(sid)
        
        self.ui.populate_tree(hierarchy, self.label_dict, self.visible_sample_ids, self.visible_sample_item_by_id, query)
        
        if self.current_sample_id not in self.visible_sample_item_by_id:
            if self.visible_sample_ids: self._select_sample_id(self.visible_sample_ids[0], display=display_first)
        else:
            self._select_sample_id(self.current_sample_id, display=False)

    def _schedule_filter_images(self) -> None:
        if self._filter_after_id: self.frame.after_cancel(self._filter_after_id)
        self._filter_after_id = self.frame.after(150, lambda: self._filter_images(display_first=False))

    def _set_search(self, value: str) -> None:
        self.ui.var_search.set(value)
        self._filter_images(display_first=True)

    def _on_tree_select(self, event) -> None:
        if not (sel := self.ui.tree.selection()): return
        if "sample" in self.ui.tree.item(sel[0], "tags"): self._display_image(sel[0])

    def _display_image(self, sample_id: str):
        if sample_id not in self.meta_dict or not self.state.dataset_dir: return
        self.current_sample_id = sample_id
        
        prediction = None
        if self.logic.model:
            try: prediction = self.logic.model.predict(self.state.dataset_dir / self.meta_dict[sample_id].image_path)
            except Exception as e: print(f"Prediction failed: {e}")
        
        try:
            self.ui.display_image(str(self.state.dataset_dir / self.meta_dict[sample_id].image_path),
                                  self.meta_dict[sample_id], self.label_dict.get(sample_id, "?"), prediction)
        except Exception as e:
            messagebox.showerror("Display Error", str(e))

    def _select_sample_id(self, sample_id: str, display: bool = True):
        if not (item := self.visible_sample_item_by_id.get(sample_id)): return
        self.ui.select_sample_in_tree(item)
        if display: self._display_image(sample_id)

    def _prev_image(self) -> None:
        if not self.visible_sample_ids: return
        idx = self.visible_sample_ids.index(self.current_sample_id) if self.current_sample_id in self.visible_sample_ids else 0
        self._select_sample_id(self.visible_sample_ids[(idx - 1) % len(self.visible_sample_ids)])

    def _next_image(self) -> None:
        if not self.visible_sample_ids: return
        idx = self.visible_sample_ids.index(self.current_sample_id) if self.current_sample_id in self.visible_sample_ids else 0
        self._select_sample_id(self.visible_sample_ids[(idx + 1) % len(self.visible_sample_ids)])

    def _try_load_model(self) -> None:
        if not self.state.dataset_dir: return
        self.logic.model = self.logic.try_load_model(self.state.dataset_dir, self.state.current_model_path)
        self.state.current_model_path = self.logic.model.model_path if self.logic.model else None

    def _analyze_dataset(self) -> None:
        if not self.state.dataset_dir or not self.logic.model:
            messagebox.showinfo("Info", "No model loaded. Train a model first.")
            return

        def analyze():
            try:
                results = self.logic.analyze_dataset(self.state.dataset_dir, self.sample_ids, self.meta_dict, self.label_dict)
                messagebox.showinfo("Success", f"Analysis complete!\\n\\nAccuracy: {results['summary']['accuracy']:.1%}\\nResults saved to:\\n{self.state.dataset_dir / 'analysis_results.json'}")
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed:\\n{e}")

        threading.Thread(target=analyze, daemon=True).start()

    def _selected_sample_id(self) -> Optional[str]:
        if self.current_sample_id in self.meta_dict:
            return self.current_sample_id
        if self.visible_sample_ids:
            return self.visible_sample_ids[0]
        return None

    def _delete_current_image(self) -> None:
        if not self.state.dataset_dir:
            return
        sample_id = self._selected_sample_id()
        if not sample_id:
            messagebox.showinfo("Info", "No sample selected.")
            return
        if not messagebox.askyesno("Delete image", f"Delete sample '{sample_id}' from dataset '{self.state.dataset_dir.name}'?"):
            return

        try:
            delete_samples(self.state.dataset_dir, [sample_id])
            self._refresh_datasets()
            self._load_dataset()
            self._try_load_model()
            try:
                self.parent.event_generate("<<DatasetChanged>>", when="tail")
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc))

    def _move_current_image(self) -> None:
        if not self.state.dataset_dir:
            return
        sample_id = self._selected_sample_id()
        if not sample_id:
            messagebox.showinfo("Info", "No sample selected.")
            return

        source_ds = self.state.dataset_dir
        targets = [(display, path) for display, path in self._dataset_by_display.items() if path.resolve() != source_ds.resolve()]
        if not targets:
            messagebox.showinfo("Move image", "No target dataset available.")
            return

        dialog = tk.Toplevel(self.frame.winfo_toplevel())
        dialog.title("Move Image")
        dialog.transient(self.frame.winfo_toplevel())
        dialog.grab_set()
        frm = ttk.Frame(dialog, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text=f"Select target dataset for '{sample_id}':").pack(anchor="w")
        lb = tk.Listbox(frm, height=min(12, max(5, len(targets))), exportselection=False)
        lb.pack(fill="both", expand=True, pady=(6, 8))
        for label, _ in targets:
            lb.insert("end", label)
        lb.selection_set(0)

        def _confirm_move() -> None:
            idxs = lb.curselection()
            if not idxs:
                return
            target_display, target_dir = targets[int(idxs[0])]
            if not messagebox.askyesno(
                "Move image",
                f"Move sample '{sample_id}'\nfrom '{source_ds.name}'\nto '{target_display}'?",
            ):
                return
            try:
                move_samples(source_ds, target_dir, [sample_id], enforce_profile_match=True)
                dialog.destroy()
                self._refresh_datasets()
                self._load_dataset()
                self._try_load_model()
                try:
                    self.parent.event_generate("<<DatasetChanged>>", when="tail")
                except Exception:
                    pass
            except Exception as exc:
                messagebox.showerror("Move failed", str(exc))

        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=dialog.destroy).pack(side="right")
        ttk.Button(btns, text="Move", command=_confirm_move).pack(side="right", padx=(0, 8))

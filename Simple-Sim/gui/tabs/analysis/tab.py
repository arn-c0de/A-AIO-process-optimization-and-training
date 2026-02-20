"""Analysis Tab"""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Dict, List
import threading
from collections import Counter

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore
from gui.utils.dataset_ops import delete_samples, move_samples
from gui.utils.dataset_catalog import DEFAULT_CATEGORY, DatasetCatalog, dataset_display_name
from gui.utils.feedback_manager import (
    FeedbackManager,
    VERDICT_THUMBS_DOWN,
    VERDICT_THUMBS_UP,
)
from .ui import AnalysisUI
from .logic import AnalysisLogic
from simple_sim.schema import MetaRow
from simple_sim.config import VALID_CLASSES


class AnalysisTab(BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.ui = AnalysisUI(self, self.frame)
        self.logic = AnalysisLogic(sim_root)
        self.sample_ids: list[str] = []
        self.meta_dict: dict[str, MetaRow] = {}
        self.label_dict: dict[str, str] = {}
        self.effective_label_dict: dict[str, str] = {}
        self.profile_dict: dict[str, str] = {}
        self.feedback_manager: Optional[FeedbackManager] = None
        self.feedback_latest_by_sample: dict[str, dict] = {}
        self.current_sample_id: Optional[str] = None
        self.visible_sample_ids: list[str] = []
        self.visible_sample_item_by_id: dict[str, str] = {}
        self._filter_after_id: Optional[str] = None
        self._dataset_by_display: dict[str, Path] = {}
        self._dataset_catalog = DatasetCatalog(sim_root, state.settings_store)
        self._selected_profiles: set[str] = set()
        self._profile_options: list[str] = []

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
        cat = self._dataset_catalog.category_for(self.state.dataset_dir)
        if cat and cat != DEFAULT_CATEGORY:
            display = f"[{cat}] {display}"
        if display in self._dataset_by_display:
            self.ui.var_dataset.set(display)
        else:
            self._refresh_datasets()
            if display in self._dataset_by_display:
                self.ui.var_dataset.set(display)
            elif values := list(self.ui.dataset_combo["values"]):
                self.ui.var_dataset.set(values[0])
                self._on_dataset_selected()
                return
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
        for key, var in [
            ("analysis.search", self.ui.var_search),
            ("analysis.class", self.ui.var_filter),
            ("analysis.run", self.ui.var_run),
            ("analysis.domain", self.ui.var_domain),
            ("analysis.split", self.ui.var_split),
            ("analysis.profile", self.ui.var_profile),
            ("analysis.sort", self.ui.var_sort),
        ]:
            if vv := st.get(key):
                try: var.set(vv)
                except Exception: pass
        if vv := st.get("analysis.profile_multi"):
            try:
                self._selected_profiles = {p for p in str(vv).split(",") if p.strip()}
            except Exception:
                self._selected_profiles = set()
        self._update_profile_multi_label()
        self._filter_images(display_first=False)

    def _wire_settings_autosave(self) -> None:
        st = self._store()
        if st is None: return
        for var, key in [
            (self.ui.var_dataset, "analysis.dataset_selection"),
            (self.ui.var_search, "analysis.search"),
            (self.ui.var_filter, "analysis.class"),
            (self.ui.var_run, "analysis.run"),
            (self.ui.var_domain, "analysis.domain"),
            (self.ui.var_split, "analysis.split"),
            (self.ui.var_profile, "analysis.profile"),
            (self.ui.var_sort, "analysis.sort"),
        ]:
            var.trace_add("write", lambda *a, v=var, k=key: (st.set(k, v.get()), st.schedule_save(self.frame)))

    def _display_for_dataset(self, p: Path) -> str:
        runs, versions = self.logic.sim_data_roots()
        return dataset_display_name(p, runs_root=runs, versions_root=versions)

    def _refresh_datasets(self) -> None:
        self._dataset_catalog.reload()
        cand = self.logic.get_datasets()
        cand.sort(key=lambda p: (1, -p.stat().st_mtime, p.name) if "versions" in str(p) else (0, p.name, -p.stat().st_mtime))
        self._dataset_catalog.prune_unknown(cand)
        self._dataset_catalog.save(self.frame)
        cand = [p for p in cand if not self._dataset_catalog.is_archived(p)]
        
        self._dataset_by_display.clear()
        displays: list[str] = []
        seen: dict[str, int] = {}
        for p in cand:
            base = self._display_for_dataset(p)
            category = self._dataset_catalog.category_for(p)
            if category and category != DEFAULT_CATEGORY:
                base = f"[{category}] {base}"
            n = seen.get(base, 0) + 1
            seen[base] = n
            disp = base if n == 1 else f"{base} ({n})"
            self._dataset_by_display[disp] = p
            displays.append(disp)
        
        self.ui.dataset_combo["values"] = displays
        if (cur := self.ui.var_dataset.get().strip()) and cur in self._dataset_by_display: return
        
        if self.state.dataset_dir:
            want = self._display_for_dataset(self.state.dataset_dir)
            cat = self._dataset_catalog.category_for(self.state.dataset_dir)
            if cat and cat != DEFAULT_CATEGORY:
                want = f"[{cat}] {want}"
            for d, pp in self._dataset_by_display.items():
                if pp == self.state.dataset_dir or d == want:
                    self.ui.var_dataset.set(d)
                    return
        if displays: self.ui.var_dataset.set(displays[0])

    def refresh(self) -> None:
        if not self.initialized:
            return
        self._refresh_datasets()

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
            self.meta_dict, self.label_dict, self.profile_dict, self.sample_ids = self.logic.load_dataset(self.state.dataset_dir)
            self._reload_feedback()
            
            run_ids = {"All"} | {sid.split('/')[0] for sid in self.sample_ids if '/' in sid}
            domains = {"All"} | {sid.split('/')[1] for sid in self.sample_ids if len(sid.split('/')) > 1}
            classes = {"All"} | {c for c in self.effective_label_dict.values() if c and c != "?"}
            
            self.ui.combo_run.configure(values=sorted(list(run_ids)))
            self.ui.combo_domain.configure(values=sorted(list(domains)))
            self.ui.combo_class.configure(values=["All"] + self._sorted_class_values(classes - {"All"}))
            self.ui.set_feedback_class_values(self._sorted_class_values(set(VALID_CLASSES) | (classes - {"All"})))
            profiles = sorted({str(v).strip() for v in self.profile_dict.values() if str(v).strip()})
            self._profile_options = profiles
            self.ui.combo_profile.configure(values=["All"] + profiles)
            if self.ui.var_profile.get().strip() not in ({"All"} | set(profiles)):
                self.ui.var_profile.set("All")
            self.ui.combo_sort.configure(values=["Default", "Newest first", "Oldest first", "ID A-Z", "ID Z-A"])
            if self.ui.var_sort.get().strip() not in {"Default", "Newest first", "Oldest first", "ID A-Z", "ID Z-A"}:
                self.ui.var_sort.set("Default")
            self._selected_profiles = {p for p in self._selected_profiles if p in set(profiles)}
            self._update_profile_multi_label()

            self._filter_images()
            self._update_dataset_stats_label()
            self._update_feedback_panel()
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
            'split': self.ui.var_split.get().strip(),
            'profile': self.ui.var_profile.get().strip(),
        }

        for sid in self.sample_ids:
            if filters['class'] != "All" and self.effective_label_dict.get(sid, "?") != filters['class']: continue
            sid_profile = str(self.profile_dict.get(sid, "")).strip() or "unknown_or_legacy"
            if self._selected_profiles:
                if sid_profile not in self._selected_profiles:
                    continue
            elif filters['profile'] != "All" and sid_profile != filters['profile']:
                continue
            parts = sid.split('/')
            if len(parts) >= 3:
                run, domain, split = parts[0], parts[1], parts[2]
                if filters['run'] != "All" and run != filters['run']: continue
                if filters['domain'] != "All" and domain != filters['domain']: continue
                if filters['split'] != "All" and split != filters['split']: continue
                if query and query not in f"{sid} {sid.split('/')[-1]} {self.effective_label_dict.get(sid, '?')}".lower(): continue

                hierarchy.setdefault(run, {}).setdefault(domain, {}).setdefault(split, []).append(sid)

        sort_mode = self.ui.var_sort.get().strip() or "Default"
        for run in hierarchy.values():
            for domain in run.values():
                for split, sample_list in domain.items():
                    domain[split] = self._sorted_sample_ids(sample_list, sort_mode)
        
        self.ui.populate_tree(hierarchy, self.effective_label_dict, self.visible_sample_ids, self.visible_sample_item_by_id, query)
        self._update_dataset_stats_label()
        
        if self.current_sample_id not in self.visible_sample_item_by_id:
            if self.visible_sample_ids: self._select_sample_id(self.visible_sample_ids[0], display=display_first)
        else:
            self._select_sample_id(self.current_sample_id, display=False)

    def _update_dataset_stats_label(self) -> None:
        if not self.sample_ids:
            self.ui.var_dataset_stats.set("Dataset Stats: -")
            return

        total = len(self.sample_ids)
        split_counts = Counter()
        for sid in self.sample_ids:
            parts = sid.split("/")
            split = parts[2] if len(parts) > 2 else "unknown"
            split_counts[split] += 1

        profile_counts = Counter()
        for sid in self.sample_ids:
            pid = str(self.profile_dict.get(sid, "")).strip() or "unknown_or_legacy"
            profile_counts[pid] += 1

        split_text = (
            f"train={split_counts.get('train', 0)}, "
            f"val={split_counts.get('val', 0)}, "
            f"test={split_counts.get('test', 0)}"
        )
        profiles_text = ", ".join(
            f"{pid}:{cnt}" for pid, cnt in sorted(profile_counts.items(), key=lambda x: (-x[1], x[0]))
        )
        self.ui.var_dataset_stats.set(f"Dataset Stats: total={total} | {split_text} | profiles: {profiles_text}")

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
            original_label = self.label_dict.get(sample_id, "?")
            effective_label = self.effective_label_dict.get(sample_id, original_label)
            feedback = self.feedback_latest_by_sample.get(sample_id)
            self.ui.display_image(str(self.state.dataset_dir / self.meta_dict[sample_id].image_path),
                                  self.meta_dict[sample_id], effective_label, prediction,
                                  original_label=original_label, feedback=feedback)
            self._update_feedback_panel()
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
        self._run_analysis(use_feedback=False)

    def _recompute_with_feedback(self) -> None:
        self._run_analysis(use_feedback=True)

    def _run_analysis(self, use_feedback: bool) -> None:
        if not self.state.dataset_dir or not self.logic.model:
            messagebox.showinfo("Info", "No model loaded. Train a model first.")
            return

        def analyze():
            try:
                effective = self.effective_label_dict if use_feedback else None
                feedback_summary = self.feedback_manager.summary() if (use_feedback and self.feedback_manager) else None
                results = self.logic.analyze_dataset(
                    self.state.dataset_dir,
                    self.sample_ids,
                    self.meta_dict,
                    self.label_dict,
                    effective_label_dict=effective,
                    feedback_summary=feedback_summary,
                )
                mode = "with feedback" if use_feedback else "baseline"
                messagebox.showinfo(
                    "Success",
                    f"Analysis complete ({mode})!\\n\\nAccuracy: {results['summary']['accuracy']:.1%}"
                    f"\\nResults saved to:\\n{self.state.dataset_dir / 'analysis_results.json'}",
                )
            except Exception as e:
                messagebox.showerror("Error", f"Analysis failed:\\n{e}")

        threading.Thread(target=analyze, daemon=True).start()

    def _reload_feedback(self) -> None:
        if not self.state.dataset_dir:
            self.feedback_manager = None
            self.feedback_latest_by_sample = {}
            self.effective_label_dict = dict(self.label_dict)
            self.ui.refresh_feedback_history([])
            return
        self.feedback_manager = FeedbackManager(self.state.dataset_dir)
        self.feedback_latest_by_sample = self.feedback_manager.latest_feedback_by_sample()
        self.effective_label_dict = {
            sid: self.feedback_manager.effective_label(sid, self.label_dict.get(sid, "?"))
            for sid in self.sample_ids
        }
        self.ui.refresh_feedback_history(self.feedback_manager.history(limit=200))

    def _feedback_thumbs_up(self) -> None:
        self._save_feedback(verdict=VERDICT_THUMBS_UP)

    def _feedback_thumbs_down(self) -> None:
        picked = self.ui.prompt_thumbs_down_feedback()
        if not picked:
            return
        corrected_class, note = picked
        self._save_feedback(verdict=VERDICT_THUMBS_DOWN, corrected_class=corrected_class, note=note)

    def _save_feedback(self, verdict: str, corrected_class: str = "", note: str = "") -> None:
        if not self.current_sample_id:
            messagebox.showinfo("Feedback", "No sample selected.")
            return
        if not self.feedback_manager:
            messagebox.showerror("Feedback", "Feedback manager not available.")
            return
        if verdict == VERDICT_THUMBS_DOWN and not corrected_class:
            messagebox.showinfo("Feedback", "Select the correct class for thumbs down.")
            return

        try:
            self.feedback_manager.append_feedback(
                sample_id=self.current_sample_id,
                verdict=verdict,
                corrected_class=corrected_class if verdict == VERDICT_THUMBS_DOWN else None,
                note=note,
            )
            self.feedback_latest_by_sample = self.feedback_manager.latest_feedback_by_sample()
            self.effective_label_dict[self.current_sample_id] = self.feedback_manager.effective_label(
                self.current_sample_id,
                self.label_dict.get(self.current_sample_id, "?"),
            )
            self.ui.refresh_feedback_history(self.feedback_manager.history(limit=200))
            self._filter_images(display_first=False)
            self._display_image(self.current_sample_id)
        except Exception as exc:
            messagebox.showerror("Feedback", f"Failed to save feedback:\\n{exc}")

    def _update_feedback_panel(self) -> None:
        if not self.current_sample_id:
            self.ui.set_feedback_status("Feedback: no sample selected")
            return
        sid = self.current_sample_id
        original = self.label_dict.get(sid, "?")
        effective = self.effective_label_dict.get(sid, original)
        latest = self.feedback_latest_by_sample.get(sid)
        if not latest:
            self.ui.set_feedback_status(f"Feedback: none | label={effective}")
            return
        verdict = str(latest.get("verdict") or "")
        corrected = str(latest.get("corrected_class") or "").strip()
        ts = str(latest.get("timestamp") or "")
        status = f"Feedback: {verdict}"
        if corrected:
            status += f" -> {corrected}"
        if original != effective:
            status += f" (original: {original})"
        if ts:
            status += f" @ {ts}"
        self.ui.set_feedback_status(status)

    def _on_profile_filter_selected(self, _evt: Optional[object] = None) -> None:
        self._selected_profiles = set()
        self._update_profile_multi_label()
        st = self._store()
        if st is not None:
            st.set("analysis.profile_multi", "")
            st.schedule_save(self.frame)
        self._filter_images(display_first=True)

    def _open_profile_multi_select(self) -> None:
        if not self._profile_options:
            messagebox.showinfo("Profile Filter", "No profiles available in current dataset.")
            return
        dlg = tk.Toplevel(self.frame.winfo_toplevel())
        dlg.title("Select Multiple Profiles")
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()
        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select one or more profiles:").pack(anchor="w")
        lb = tk.Listbox(frm, selectmode="multiple", exportselection=False, height=min(14, max(5, len(self._profile_options))))
        lb.pack(fill="both", expand=True, pady=(6, 8))
        for p in self._profile_options:
            lb.insert("end", p)
        for i, p in enumerate(self._profile_options):
            if p in self._selected_profiles:
                lb.selection_set(i)

        def _apply() -> None:
            picked = {self._profile_options[int(i)] for i in lb.curselection()}
            self._selected_profiles = picked
            self.ui.var_profile.set("All")
            self._update_profile_multi_label()
            st = self._store()
            if st is not None:
                st.set("analysis.profile_multi", ",".join(sorted(self._selected_profiles)))
                st.schedule_save(self.frame)
            dlg.destroy()
            self._filter_images(display_first=True)

        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="Clear", command=lambda: [lb.selection_clear(0, "end")]).pack(side="left")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right")
        ttk.Button(btns, text="Apply", command=_apply).pack(side="right", padx=(0, 8))

    def _update_profile_multi_label(self) -> None:
        if not self._selected_profiles:
            self.ui.var_profile_multi.set("multi: off")
            return
        self.ui.var_profile_multi.set(f"multi: {len(self._selected_profiles)} selected")

    def _sorted_sample_ids(self, sample_ids: list[str], sort_mode: str) -> list[str]:
        if sort_mode == "ID A-Z":
            return sorted(sample_ids)
        if sort_mode == "ID Z-A":
            return sorted(sample_ids, reverse=True)
        if sort_mode in {"Newest first", "Oldest first"}:
            reverse = sort_mode == "Newest first"
            return sorted(sample_ids, key=self._sample_mtime_key, reverse=reverse)
        return list(sample_ids)

    def _sample_mtime_key(self, sample_id: str) -> float:
        if not self.state.dataset_dir:
            return 0.0
        try:
            meta = self.meta_dict.get(sample_id)
            if not meta:
                return 0.0
            path = self.state.dataset_dir / meta.image_path
            return path.stat().st_mtime if path.exists() else 0.0
        except Exception:
            return 0.0

    def _on_feedback_history_select(self, _event: Optional[object] = None) -> None:
        sid = self.ui.selected_feedback_sample_id()
        if not sid:
            return
        if sid not in self.meta_dict:
            messagebox.showinfo("Feedback History", f"Sample not found in current dataset:\n{sid}")
            return
        self._set_search("")
        self._filter_images(display_first=False)
        self._select_sample_id(sid, display=True)

    def _open_feedback_history(self) -> None:
        if not self.feedback_manager:
            messagebox.showinfo("Feedback History", "No dataset selected.")
            return
        entries = self.feedback_manager.history(limit=500)
        if not entries:
            messagebox.showinfo("Feedback History", "No feedback entries yet.")
            return

        dlg = tk.Toplevel(self.frame.winfo_toplevel())
        dlg.title("Feedback History")
        dlg.transient(self.frame.winfo_toplevel())
        dlg.geometry("860x360")
        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)

        tree = ttk.Treeview(frm, columns=("Time", "Sample", "Action", "To", "Note"), show="headings")
        tree.heading("Time", text="Time")
        tree.heading("Sample", text="Sample")
        tree.heading("Action", text="Action")
        tree.heading("To", text="To")
        tree.heading("Note", text="Note")
        tree.column("Time", width=140, anchor="w")
        tree.column("Sample", width=300, anchor="w")
        tree.column("Action", width=80, anchor="center")
        tree.column("To", width=120, anchor="center")
        tree.column("Note", width=180, anchor="w")
        scroll = ttk.Scrollbar(frm, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        for idx, entry in enumerate(entries):
            ts = str(entry.get("timestamp") or "")
            ts_short = ts.replace("T", " ")[:19] if ts else "-"
            sid = str(entry.get("id") or "")
            verdict = str(entry.get("verdict") or "")
            action = "👍" if verdict == VERDICT_THUMBS_UP else "👎"
            corrected = str(entry.get("corrected_class") or "").strip() or "-"
            note = str(entry.get("note") or "").strip()
            tree.insert("", "end", iid=f"h::{idx}", values=(ts_short, sid, action, corrected, note))

        def _open_selected(_event: Optional[object] = None) -> None:
            sel = tree.selection()
            if not sel:
                return
            vals = tree.item(sel[0], "values")
            if not vals or len(vals) < 2:
                return
            sid = str(vals[1]).strip()
            if not sid:
                return
            if sid not in self.meta_dict:
                messagebox.showinfo("Feedback History", f"Sample not found in current dataset:\n{sid}")
                return
            self._set_search("")
            self._filter_images(display_first=False)
            self._select_sample_id(sid, display=True)
            dlg.destroy()

        tree.bind("<Double-1>", _open_selected)
        btns = ttk.Frame(dlg, padding=(10, 0, 10, 10))
        btns.pack(fill="x")
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="right")
        ttk.Button(btns, text="Open Selected", command=_open_selected).pack(side="right", padx=(0, 8))

    def _selected_sample_ids(self) -> List[str]:
        selected: List[str] = []
        try:
            for item in self.ui.tree.selection():
                if item in self.meta_dict:
                    selected.append(item)
        except Exception:
            selected = []

        if selected:
            seen = set()
            ordered: List[str] = []
            for sample_id in selected:
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                ordered.append(sample_id)
            return ordered

        if self.current_sample_id in self.meta_dict:
            return [self.current_sample_id]
        if self.visible_sample_ids:
            return [self.visible_sample_ids[0]]
        return []

    def _delete_current_image(self) -> None:
        if not self.state.dataset_dir:
            return
        sample_ids = self._selected_sample_ids()
        if not sample_ids:
            messagebox.showinfo("Info", "No sample selected.")
            return
        if not messagebox.askyesno(
            "Delete image",
            f"Delete {len(sample_ids)} sample(s) from dataset '{self.state.dataset_dir.name}'?",
        ):
            return

        try:
            delete_samples(self.state.dataset_dir, sample_ids)
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
        sample_ids = self._selected_sample_ids()
        if not sample_ids:
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
        ttk.Label(frm, text=f"Select target dataset for {len(sample_ids)} selected sample(s):").pack(anchor="w")
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
                f"Move {len(sample_ids)} sample(s)\nfrom '{source_ds.name}'\nto '{target_display}'?",
            ):
                return
            try:
                move_samples(source_ds, target_dir, sample_ids, enforce_profile_match=True)
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

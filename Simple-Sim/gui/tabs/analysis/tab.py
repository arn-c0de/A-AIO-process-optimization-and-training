"""Analysis Tab"""

from __future__ import annotations
from tkinter import ttk
from pathlib import Path
from collections import Counter
from datetime import datetime

from gui import dialogs
from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore, restore_persisted_vars, wire_vars_autosave
from gui.utils.dataset_ops import delete_samples, move_samples
from gui.utils.dataset_catalog import DatasetCatalog, catalog_labels
from gui.widgets import ScrolledTree
from gui.core.tasks import runner_for
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
        self.sample_date_dict: dict[str, str] = {}
        self.feedback_manager: FeedbackManager | None = None
        self.feedback_latest_by_sample: dict[str, dict] = {}
        self.current_sample_id: str | None = None
        self.visible_sample_ids: list[str] = []
        self.visible_sample_item_by_id: dict[str, str] = {}
        self._filter_after_id: str | None = None
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

    def _store(self) -> SettingsStore | None:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None: return
        if self.state.dataset_dir is None and (v := st.get("analysis.dataset_selection")):
            try:
                self.ui.var_dataset.set(v)
                self._on_dataset_selected()
            except Exception: pass
        # Empty strings keep the widget defaults (filters persisted as "" mean "unset").
        restore_persisted_vars(st, self._persisted_var_bindings()[1:], skip_empty_strings=True)
        if vv := st.get("analysis.profile_multi"):
            try:
                self._selected_profiles = {p for p in str(vv).split(",") if p.strip()}
            except Exception:
                self._selected_profiles = set()
        self._update_profile_multi_label()
        self._filter_images(display_first=False)

    def _persisted_var_bindings(self) -> list:
        # First entry (dataset selection) is restored separately because it triggers a dataset load.
        return [
            ("analysis.dataset_selection", self.ui.var_dataset),
            ("analysis.search", self.ui.var_search),
            ("analysis.class", self.ui.var_filter),
            ("analysis.run", self.ui.var_run),
            ("analysis.domain", self.ui.var_domain),
            ("analysis.split", self.ui.var_split),
            ("analysis.profile", self.ui.var_profile),
            ("analysis.sort", self.ui.var_sort),
        ]

    def _wire_settings_autosave(self) -> None:
        wire_vars_autosave(self._store(), self._persisted_var_bindings(), self.frame)

    def _display_for_dataset(self, p: Path) -> str:
        """Category-prefixed display label (before duplicate suffixing)."""
        runs, versions = self.logic.sim_data_roots()
        return self._dataset_catalog.display_label(p, runs_root=runs, versions_root=versions)

    def _refresh_datasets(self) -> None:
        self._dataset_by_display = catalog_labels(self.sim_root, self._dataset_catalog, save_widget=self.frame)
        displays = list(self._dataset_by_display)
        self.ui.dataset_combo["values"] = displays
        if (cur := self.ui.var_dataset.get().strip()) and cur in self._dataset_by_display: return

        if self.state.dataset_dir:
            want = self._display_for_dataset(self.state.dataset_dir)
            for d, pp in self._dataset_by_display.items():
                if pp == self.state.dataset_dir or d == want:
                    self.ui.var_dataset.set(d)
                    return
        if displays: self.ui.var_dataset.set(displays[0])

    def refresh(self) -> None:
        if not self.initialized:
            return
        self._refresh_datasets()

    def _on_dataset_selected(self, _evt: object | None = None) -> None:
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
            self.sample_date_dict = self._build_sample_date_dict()
            
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
            dialogs.show_error(self.frame, "Load Error", e, hint=f"Failed to load dataset {self.state.dataset_dir}.")

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
        
        self.ui.populate_tree(
            hierarchy,
            self.effective_label_dict,
            self.visible_sample_ids,
            self.visible_sample_item_by_id,
            query,
            sample_date_dict=self.sample_date_dict,
        )
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
            dialogs.show_error(self.frame, "Display Error", e)

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
        if not dialogs.require(self.state.dataset_dir and self.logic.model, self.frame,
                               "No model loaded. Train a model first.", title="No Model"):
            return

        # Snapshot state on the main thread; dialogs are shown back on the main thread via after().
        dataset_dir = self.state.dataset_dir
        sample_ids = list(self.sample_ids)
        meta_dict = dict(self.meta_dict)
        label_dict = dict(self.label_dict)
        effective = dict(self.effective_label_dict) if use_feedback else None
        feedback_summary = self.feedback_manager.summary() if (use_feedback and self.feedback_manager) else None
        mode = "with feedback" if use_feedback else "baseline"

        def analyze(handle) -> dict:
            return self.logic.analyze_dataset(
                dataset_dir,
                sample_ids,
                meta_dict,
                label_dict,
                effective_label_dict=effective,
                feedback_summary=feedback_summary,
                progress=handle.progress,
                cancelled=handle.cancelled,
            )

        def on_done(results: dict) -> None:
            acc = results["summary"]["accuracy"]
            out = dataset_dir / "analysis_results.json"
            ctx = getattr(self.state, "ctx", None)
            if ctx is not None:
                ctx.status(f"Analysis complete ({mode}): accuracy {acc:.1%}, saved to {out}", "success")
            else:
                dialogs.show_info(self.frame, "Analysis Complete",
                                  f"Analysis complete ({mode})!\n\nAccuracy: {acc:.1%}\nResults saved to:\n{out}")

        def on_error(exc: BaseException) -> None:
            dialogs.show_error(self.frame, "Analysis Failed", exc)

        runner_for(self.state, self.frame).run(
            analyze, name=f"Analyze dataset ({mode})", on_done=on_done, on_error=on_error)

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
        if not dialogs.require(self.current_sample_id, self.frame, "Select a sample first."):
            return
        if not self.feedback_manager:
            dialogs.show_error(self.frame, "Feedback", "Feedback manager not available.",
                               hint="Select a dataset first.")
            return
        if not dialogs.require(verdict != VERDICT_THUMBS_DOWN or corrected_class, self.frame,
                               "Select the correct class for thumbs down."):
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
            dialogs.show_error(self.frame, "Feedback", exc, hint="Failed to save feedback.")

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

    def _on_profile_filter_selected(self, _evt: object | None = None) -> None:
        self._selected_profiles = set()
        self._update_profile_multi_label()
        st = self._store()
        if st is not None:
            st.set("analysis.profile_multi", "")
            st.schedule_save(self.frame)
        self._filter_images(display_first=True)

    def _open_profile_multi_select(self) -> None:
        if not dialogs.require(self._profile_options, self.frame, "No profiles available in current dataset.",
                               title="Profile Filter"):
            return
        picked = dialogs.pick_from_list(
            self.frame,
            "Select Multiple Profiles",
            self._profile_options,
            multi=True,
            selected=sorted(self._selected_profiles),
            prompt="Show only samples of the selected profiles. Select none to turn the multi filter off.",
        )
        if picked is None:
            return
        self._selected_profiles = set(picked)
        self.ui.var_profile.set("All")
        self._update_profile_multi_label()
        st = self._store()
        if st is not None:
            st.set("analysis.profile_multi", ",".join(sorted(self._selected_profiles)))
            st.schedule_save(self.frame)
        self._filter_images(display_first=True)

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

    def _build_sample_date_dict(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if not self.state.dataset_dir:
            return out
        for sid in self.sample_ids:
            ts = self._sample_mtime_key(sid)
            out[sid] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts > 0 else "-"
        return out

    def _on_feedback_history_select(self, _event: object | None = None) -> None:
        sid = self.ui.selected_feedback_sample_id()
        if not sid:
            return
        if sid not in self.meta_dict:
            dialogs.show_info(self.frame, "Feedback History", f"Sample not found in current dataset:\n{sid}")
            return
        self._set_search("")
        self._filter_images(display_first=False)
        self._select_sample_id(sid, display=True)

    def _open_feedback_history(self) -> None:
        if not dialogs.require(self.feedback_manager, self.frame, "Select a dataset first.",
                               title="Feedback History"):
            return
        entries = self.feedback_manager.history(limit=500)
        if not entries:
            dialogs.show_info(self.frame, "Feedback History", "No feedback entries yet.")
            return

        # Non-modal: the user can keep browsing while the history is open.
        dlg = dialogs.Dialog(self.frame, "Feedback History")
        view = ScrolledTree(
            dlg.body,
            [("Time", "Time", 140, "w"), ("Sample", "Sample", 300, "w"), ("Action", "Action", 80, "center"),
             ("To", "To", 120, "center"), ("Note", "Note", 180, "w")],
        )
        tree = view.tree

        for idx, entry in enumerate(entries):
            ts = str(entry.get("timestamp") or "")
            ts_short = ts.replace("T", " ")[:19] if ts else "-"
            sid = str(entry.get("id") or "")
            verdict = str(entry.get("verdict") or "")
            action = "Correct" if verdict == VERDICT_THUMBS_UP else "Wrong"
            corrected = str(entry.get("corrected_class") or "").strip() or "-"
            note = str(entry.get("note") or "").strip()
            view.insert((ts_short, sid, action, corrected, note), iid=f"h::{idx}")

        def _open_selected(_event: object | None = None) -> None:
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
                dialogs.show_info(dlg, "Feedback History", f"Sample not found in current dataset:\n{sid}")
                return
            self._set_search("")
            self._filter_images(display_first=False)
            self._select_sample_id(sid, display=True)
            dlg.destroy()

        tree.bind("<Double-1>", _open_selected)
        bar = dlg.buttons([("Open Selected", _open_selected, "Accent.TButton"), ("Close", dlg.cancel, None)])
        # buttons() packs at the bottom; the table fills the rest.
        view.pack(fill="both", expand=True, before=bar)
        dlg.geometry("860x360")

    def _selected_sample_ids(self) -> list[str]:
        selected: list[str] = []
        try:
            for item in self.ui.tree.selection():
                if item in self.meta_dict:
                    selected.append(item)
        except Exception:
            selected = []

        if selected:
            seen = set()
            ordered: list[str] = []
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
        if not dialogs.require(sample_ids, self.frame, "Select a sample first."):
            return
        if not dialogs.confirm(
            self.frame,
            "Delete Images",
            f"Delete {len(sample_ids)} sample(s) (image files and labels) from dataset "
            f"'{self.state.dataset_dir.name}'? This cannot be undone.",
            danger=True,
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
            dialogs.show_error(self.frame, "Delete Failed", exc)

    def _move_current_image(self) -> None:
        if not self.state.dataset_dir:
            return
        sample_ids = self._selected_sample_ids()
        if not dialogs.require(sample_ids, self.frame, "Select a sample first."):
            return

        source_ds = self.state.dataset_dir
        targets = [(display, path) for display, path in self._dataset_by_display.items() if path.resolve() != source_ds.resolve()]
        if not targets:
            dialogs.show_info(self.frame, "Move Images", "No target dataset available.")
            return

        labels = [label for label, _ in targets]
        picked = dialogs.pick_from_list(
            self.frame,
            "Move Images",
            labels,
            selected=labels[:1],
            prompt=f"Select the target dataset for {len(sample_ids)} selected sample(s):",
        )
        if not picked:
            return
        target_display = picked[0]
        target_dir = dict(targets)[target_display]
        if not dialogs.confirm(
            self.frame,
            "Move Images",
            f"Move {len(sample_ids)} sample(s)\nfrom '{source_ds.name}'\nto '{target_display}'?",
            ok_text="Move",
        ):
            return
        try:
            move_samples(source_ds, target_dir, sample_ids, enforce_profile_match=True)
            self._refresh_datasets()
            self._load_dataset()
            self._try_load_model()
            try:
                self.parent.event_generate("<<DatasetChanged>>", when="tail")
            except Exception:
                pass
        except Exception as exc:
            dialogs.show_error(self.frame, "Move Failed", exc)

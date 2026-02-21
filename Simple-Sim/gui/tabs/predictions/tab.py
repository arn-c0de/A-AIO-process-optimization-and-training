"""Predictions Tab"""

from __future__ import annotations

import json
import queue
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

from gui.tabs.core.base import BaseTab
from gui.state import UiState
from gui.utils.settings_store import SettingsStore

from .ui import PredictionsUI
from .logic import PredictionsLogic


class PredictionsTab(BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.log_q: queue.Queue[str] = queue.Queue()
        self.ui = PredictionsUI(self, self.frame)
        self.logic = PredictionsLogic(sim_root, self.log_q)

        self._pred_rows: List[Dict[str, Any]] = []
        self._pred_row_by_id: Dict[str, Dict[str, Any]] = {}
        self._active_dataset_dir: Optional[Path] = None

        self._multi_dataset_paths: List[str] = []
        self._multi_results: List[Dict[str, Any]] = []
        self._multi_result_by_key: Dict[str, Dict[str, Any]] = {}
        
        self._dataset_dirs: List[Path] = []
        self._dataset_by_label: Dict[str, Path] = {}

        self._ui_tick_id: Optional[str] = None

    def build_ui(self) -> None:
        self.ui.build_ui()
        self.ui.frame.pack(fill="both", expand=True)
        self.on_dataset_changed()
        self._refresh_datasets()
        self._refresh_models()
        self._refresh_profile_models()
        self._load_persisted_settings()
        try:
            self._on_dataset_selected()
        except Exception:
            pass
        self._wire_settings_autosave()
        self._update_multi_sel_label()
        self._apply_multi_mode_ui()
        self._tick_ui()

    def on_dataset_changed(self) -> None:
        if not self.state.dataset_dir:
            return
        try:
            if self._dataset_by_label:
                for label, p in self._dataset_by_label.items():
                    if p == self.state.dataset_dir:
                        self.ui.var_dataset.set(label)
                        break
                else:
                    self.ui.var_dataset.set(self.state.dataset_dir.name)
            else:
                self.ui.var_dataset.set(self.state.dataset_dir.name)
        except Exception:
            self.ui.var_dataset.set(self.state.dataset_dir.name)
        try:
            self._refresh_datasets()
            self._refresh_models()
        except Exception:
            pass

    def _store(self) -> Optional[SettingsStore]:
        return self.state.settings_store

    def _load_persisted_settings(self) -> None:
        st = self._store()
        if st is None:
            return
        for key, var in [
            ("pred.dataset_selection", self.ui.var_dataset),
            ("pred.model_selection", self.ui.var_model),
            ("pred.split", self.ui.var_split),
            ("pred.device", self.ui.var_device),
            ("pred.max_samples", self.ui.var_max_samples),
            ("pred.profile_model", self.ui.var_profile_model),
        ]:
            v = st.get(key)
            if v is None:
                continue
            try:
                var.set(v)
            except Exception:
                pass
        try:
            self.ui.chk_save_preds.set(bool(st.get("pred.save_preds", True)))
            self.ui.chk_auto_train_bundle.set(bool(st.get("pred.auto_train_bundle", False)))
            self.ui.var_multi_datasets.set(bool(st.get("pred.multi_datasets", False)))
            raw = st.get("pred.multi_dataset_paths_json")
            if isinstance(raw, str) and raw.strip():
                obj = json.loads(raw)
                if isinstance(obj, list):
                    self._multi_dataset_paths = [str(x) for x in obj if str(x).strip()]
        except Exception:
            self._multi_dataset_paths = []

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
            var.trace_add("write", cb)

        bind(self.ui.var_dataset, "pred.dataset_selection")
        bind(self.ui.var_model, "pred.model_selection")
        bind(self.ui.var_split, "pred.split")
        bind(self.ui.var_device, "pred.device")
        bind(self.ui.var_max_samples, "pred.max_samples")
        bind(self.ui.var_profile_model, "pred.profile_model")
        bind(self.ui.var_multi_datasets, "pred.multi_datasets")
        bind(self.ui.chk_save_preds, "pred.save_preds")
        bind(self.ui.chk_auto_train_bundle, "pred.auto_train_bundle")

    def _persist_multi_dataset_paths(self) -> None:
        st = self._store()
        if st is None:
            return
        try:
            st.set("pred.multi_dataset_paths_json", json.dumps(self._multi_dataset_paths, ensure_ascii=True))
            st.schedule_save(self.frame)
        except Exception:
            pass

    def _update_multi_sel_label(self) -> None:
        n = len([p for p in self._multi_dataset_paths if str(p).strip()])
        try:
            self.ui.var_multi_sel.set(f"selected: {n}")
        except Exception:
            pass

    def _on_multi_toggle(self) -> None:
        self._apply_multi_mode_ui()
        self._update_multi_sel_label()

    def _apply_multi_mode_ui(self) -> None:
        try:
            if bool(self.ui.var_multi_datasets.get()):
                self.ui.combo_dataset.configure(state="disabled")
                if self.ui.ds_summary_frame.winfo_ismapped() == 0:
                    self.ui.ds_summary_frame.pack(fill="x", pady=(0, 8))
            else:
                self.ui.combo_dataset.configure(state="readonly")
                if self.ui.ds_summary_frame.winfo_ismapped() != 0:
                    self.ui.ds_summary_frame.pack_forget()
                self.ui.tree_ds.delete(*self.ui.tree_ds.get_children())
        except Exception:
            pass

    def _open_multi_dataset_dialog(self) -> None:
        self._refresh_datasets()
        dlg = tk.Toplevel(self.frame)
        dlg.title("Select Datasets")
        dlg.transient(self.frame.winfo_toplevel())
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, text="Select one or more datasets (Ctrl/Shift):").pack(anchor="w")

        list_frame = ttk.Frame(frm)
        list_frame.pack(fill="both", expand=True, pady=(8, 8))
        lb = tk.Listbox(list_frame, selectmode="extended", height=16)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        labels = list(self._dataset_by_label.keys())
        for x in labels:
            lb.insert("end", x)

        want = {str(p).strip() for p in self._multi_dataset_paths if str(p).strip()}
        for idx, lab in enumerate(labels):
            p = self._dataset_by_label.get(lab)
            if not p:
                continue
            try:
                rel = str(p.resolve().relative_to(self.sim_root.resolve()))
            except Exception:
                rel = str(p.resolve())
            if rel in want:
                lb.selection_set(idx)

        def on_ok() -> None:
            sel = list(lb.curselection())
            picked: List[str] = []
            for i in sel:
                try:
                    lab = labels[int(i)]
                    p = self._dataset_by_label.get(lab)
                    if not p: continue
                    picked.append(str(p.resolve().relative_to(self.sim_root.resolve())))
                except Exception:
                    continue
            self._multi_dataset_paths = picked
            self._persist_multi_dataset_paths()
            self._update_multi_sel_label()
            dlg.destroy()

        btns = ttk.Frame(frm)
        btns.pack(fill="x")
        ttk.Button(btns, text="OK", command=on_ok).pack(side="right")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right", padx=(0, 8))
        dlg.geometry("560x420")

    def _active_data_dir(self) -> Optional[Path]:
        return self._active_dataset_dir or self.state.dataset_dir

    def _tick_ui(self) -> None:
        while True:
            try:
                line = self.log_q.get_nowait()
                self.ui.append_log(line)
                if "Report saved to:" in line:
                    p = line.split("Report saved to:", 1)[-1].strip()
                    self.ui.var_report_path.set(f"report: {p}")
                if "Predictions saved to:" in line:
                    p = line.split("Predictions saved to:", 1)[-1].strip()
                    self.ui.var_preds_path.set(f"preds: {p}")
            except queue.Empty:
                break
        self._ui_tick_id = self.frame.after(150, self._tick_ui)

    def _stop_predictions(self) -> None:
        self.logic.stop_predictions()
        self.ui.var_status.set("status: stopping...")

    def _selected_dataset_paths(self) -> List[Path]:
        if hasattr(self.ui, "var_multi_datasets") and bool(self.ui.var_multi_datasets.get()):
            out: List[Path] = [self.logic.resolve_dataset_path(s) for s in self._multi_dataset_paths if s.strip()]
            seen = set()
            return [p for p in out if not (str(p) in seen or seen.add(str(p)))]
        ds = self.state.dataset_dir
        return [ds] if ds else []

    def _run_predictions(self) -> None:
        if self.logic.proc and self.logic.proc.poll() is None:
            messagebox.showinfo("Info", "Predictions already running.")
            return

        model_s = self.ui.var_model.get().strip()
        model_path = self.logic.resolve_model_path(model_s)
        if not model_path.exists():
            messagebox.showerror("Error", f"Model not found:\\n{model_path}")
            return

        dataset_dirs = self._selected_dataset_paths()
        if not dataset_dirs:
            messagebox.showinfo("Info", "No dataset selected.")
            return
        
        profile_model_s = self.ui.var_profile_model.get().strip()
        pm_path: Optional[Path] = None
        if profile_model_s:
            pm_path = self.logic.resolve_model_path(profile_model_s)
            if not pm_path.exists():
                messagebox.showerror("Error", f"Profile model not found:\\n{pm_path}")
                return
        
        self.ui.var_status.set("status: running...")
        self.ui.var_report_path.set("report: -")
        self.ui.var_preds_path.set("preds: -")
        self.ui.set_metrics_text("Running batch prediction...\\n")
        self.ui.clear_confusion_matrix()
        self._pred_rows = []
        self._pred_row_by_id = {}
        self._refresh_tree()
        self._multi_results = []
        self._multi_result_by_key = {}
        self.ui.tree_ds.delete(*self.ui.tree_ds.get_children())

        self.ui.btn_run.configure(state="disabled")
        self.ui.btn_stop.configure(state="normal")

        self.logic.run_predictions(
            model_path,
            dataset_dirs,
            self.ui.var_split.get().strip() or "test",
            self.ui.var_device.get().strip() or "auto",
            self.ui.var_max_samples.get().strip(),
            bool(self.ui.chk_save_preds.get()),
            bool(self.ui.chk_auto_train_bundle.get()),
            pm_path,
            self._on_prediction_complete,
        )

    def _on_prediction_complete(self, last_report_path, last_preds_path, last_dataset_dir, multi_results, multi_result_by_key, was_stopped):
        self.ui.btn_run.configure(state="normal")
        self.ui.btn_stop.configure(state="disabled")
        self.ui.var_status.set("status: stopped" if was_stopped else "status: done")

        if bool(self.ui.var_multi_datasets.get()):
            if last_dataset_dir:
                self._active_dataset_dir = last_dataset_dir
            self._multi_results = multi_results
            self._multi_result_by_key = multi_result_by_key
            self._render_multi_summary()
        else:
            if last_dataset_dir:
                self._active_dataset_dir = last_dataset_dir
            if last_report_path and last_report_path.exists():
                self._load_report(last_report_path)
            if last_preds_path and last_preds_path.exists():
                self._load_preds(last_preds_path)
    
    def _refresh_datasets(self) -> None:
        def same_path(a: Path, b: Path) -> bool:
            try:
                return a.resolve() == b.resolve()
            except Exception:
                return str(a) == str(b)

        prev_selected: Optional[Path] = self._selected_dataset_dir()
        if prev_selected is None and self.state.dataset_dir is not None:
            prev_selected = self.state.dataset_dir

        cand = self.logic.get_datasets()

        def is_version(p: Path) -> bool:
            try:
                return str(p.resolve()).startswith(str(self.sim_root / "outputs" / "sim_data" / "versions").resolve() + os.sep)
            except Exception: return False

        def sort_key(p: Path) -> tuple:
            return (1, -p.stat().st_mtime, p.name) if is_version(p) else (0, p.name)

        cand.sort(key=sort_key)
        self._dataset_dirs = cand

        def display(p: Path) -> str:
            try:
                rp = p.resolve()
                runs_base = (self.sim_root / "outputs" / "sim_data" / "runs").resolve()
                versions_base = (self.sim_root / "outputs" / "sim_data" / "versions").resolve()
                if str(rp).startswith(str(runs_base) + os.sep): return rp.name
                if str(rp).startswith(str(versions_base) + os.sep): return f"{rp.parent.name}:{rp.name}"
            except Exception: pass
            return p.name

        self._dataset_by_label = {}
        labels: List[str] = []
        seen: Dict[str, int] = {}
        for p in cand:
            base = display(p)
            n = seen.get(base, 0) + 1
            seen[base] = n
            label = base if n == 1 else f"{base} ({n})"
            labels.append(label)
            self._dataset_by_label[label] = p

        self.ui.combo_dataset["values"] = labels
        cur = self.ui.var_dataset.get().strip()
        if cur and cur in self._dataset_by_label:
            self.state.dataset_dir = self._dataset_by_label[cur]
            return
        if prev_selected is not None:
            for label, p in self._dataset_by_label.items():
                if same_path(p, prev_selected):
                    self.ui.var_dataset.set(label)
                    self.state.dataset_dir = p
                    return
        if self.state.dataset_dir:
            for label, p in self._dataset_by_label.items():
                if same_path(p, self.state.dataset_dir):
                    self.ui.var_dataset.set(label)
                    return
        if labels:
            self.ui.var_dataset.set(labels[0])
            self._on_dataset_selected()

    def _selected_dataset_dir(self) -> Optional[Path]:
        disp = self.ui.var_dataset.get().strip()
        if not disp: return None
        return self._dataset_by_label.get(disp)

    def _on_dataset_selected(self, _evt: Optional[object] = None) -> None:
        ds = self._selected_dataset_dir()
        if not ds: return
        self.state.dataset_dir = ds
        self._active_dataset_dir = ds
        try:
            self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception:
            pass
        self._refresh_models()

    def _refresh_models(self) -> None:
        paths = self.logic.get_models()
        values = [str(p.resolve().relative_to(self.sim_root.resolve())) if p.is_relative_to(self.sim_root) else str(p) for p in paths]
        self.ui.combo_model["values"] = values

        default_model = self.sim_root / "outputs" / "models" / f"{self.state.dataset_dir.name}.pt" if self.state.dataset_dir else None
        cur = self.ui.var_model.get().strip()
        if cur and cur in values: return

        if default_model and default_model.exists():
            rel = str(default_model.resolve().relative_to(self.sim_root.resolve()))
            if rel in values:
                self.ui.var_model.set(rel)
                return
        if values:
            self.ui.var_model.set(values[0])

    def _refresh_profile_models(self) -> None:
        paths = self.logic.get_profile_models()
        values = [""] + [str(p.resolve().relative_to(self.sim_root.resolve())) if p.is_relative_to(self.sim_root) else str(p) for p in paths]
        self.ui.combo_profile_model["values"] = values
        
        cur = self.ui.var_profile_model.get().strip()
        if not (cur and cur in values):
            self.ui.var_profile_model.set("")

    def _load_report(self, report_path: Path) -> None:
        obj, err = self.logic.load_report(report_path)
        if err:
            self.ui.set_metrics_text(err)
            return
        if not obj: return

        metrics = obj.get("metrics", {})
        cm = metrics.get("confusion_matrix")
        class_names = list(metrics.get("per_class", {}).keys())

        lines = [f"Report: {report_path.name}", f"Model:  {Path(obj.get('model_path', '-')).name}", f"Split:  {obj.get('split', '-')}", f"Seen:   {obj.get('seen_samples', '-')}"]
        if "img_per_s" in obj: lines.append(f"Speed:  {obj.get('img_per_s', 0.0):.1f} img/s")
        if "accuracy" in metrics: lines.append(f"\\nAccuracy: {metrics.get('accuracy', 0.0):.4f}")
        if "macro_f1" in metrics: lines.append(f"Macro F1:  {metrics.get('macro_f1', 0.0):.4f}")
        if cfn := metrics.get("critical_fn_rates"):
            lines.append("\\nCritical FN rates:")
            lines.extend([f"  {k}: {v:.4f}" for k, v in sorted(cfn.items())])
        
        self.ui.set_metrics_text("\\n".join(lines) + "\\n")
        self.ui.render_confusion_matrix(cm, class_names) if cm and class_names else self.ui.clear_confusion_matrix()

    def _load_preds(self, preds_path: Path) -> None:
        rows, err = self.logic.load_preds(preds_path)
        if err:
            messagebox.showerror("Error", err)
            return
        if not rows: return
        
        try:
            self.logic.enrich_rows_with_gt_profile(rows, self._active_data_dir())
        except Exception as e:
            self.ui.append_log(f"[warn] failed to enrich rows with GT profile: {e}\\n")
        
        self._pred_rows = rows
        self._pred_row_by_id = {str(r.get("id")): r for r in rows if r.get("id")}
        self._refresh_tree()

    def _on_tree_select(self, _evt: Optional[object] = None) -> None:
        sel = self.ui.tree.selection()
        if not sel: return
        sid = sel[0]
        row = self._pred_row_by_id.get(sid)
        data_dir = self._active_data_dir()
        if not row or not data_dir: return

        image_rel = row.get("image_path")
        if not image_rel: return
        image_path = data_dir / str(image_rel)
        if not image_path.exists():
            messagebox.showerror("Error", f"Image not found:\\n{image_path}")
            return
        
        try:
            self.ui.display_image(str(image_path), row)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image:\\n{e}")

    def _on_tree_ds_select(self, _evt: Optional[object] = None) -> None:
        sel = self.ui.tree_ds.selection()
        if not sel: return
        key = sel[0]
        res = self._multi_result_by_key.get(key)
        if not res: return
        
        if isinstance(data_dir := res.get("dataset_dir"), Path):
            self._active_dataset_dir = data_dir
        
        if isinstance(report_path := res.get("report_path"), Path) and report_path.exists():
            self._load_report(report_path)
        if isinstance(preds_path := res.get("preds_path"), Path) and preds_path.exists():
            self._load_preds(preds_path)
        else:
            self._pred_rows, self._pred_row_by_id = [], {}
            self._refresh_tree()

    def _refresh_tree(self) -> None:
        self.ui.refresh_tree(self._pred_rows, self.ui.var_filter.get())
        
    def _render_multi_summary(self) -> None:
        """Populate the dataset-results table from collected report JSONs."""
        self._apply_multi_mode_ui()
        self.ui.tree_ds.delete(*self.ui.tree_ds.get_children())
        
        rows: List[Tuple[str, str, str, str, str, str]] = []
        for res in self._multi_results:
            key, label, report_path = str(res.get("key") or ""), str(res.get("label") or ""), res.get("report_path")
            acc_s, f1_s, seen_s, prof_acc_s = "", "", "", ""
            if isinstance(report_path, Path) and report_path.exists():
                try:
                    obj = json.loads(report_path.read_text(encoding="utf-8"))
                    metrics, seen, prof_acc = obj.get("metrics", {}), obj.get("seen_samples"), obj.get("profile_accuracy")
                    acc, f1 = metrics.get("accuracy"), metrics.get("macro_f1")
                    if acc is not None: acc_s = f"{float(acc):.4f}"
                    if f1 is not None: f1_s = f"{float(f1):.4f}"
                    if seen is not None: seen_s = str(int(seen))
                    if prof_acc is not None: prof_acc_s = f"{float(prof_acc):.4f}"
                except Exception: pass
            
            if not key:
                key = f"{label}:{len(rows)}"
                res["key"] = key
                self._multi_result_by_key[key] = res
            rows.append((key, label, acc_s, f1_s, seen_s, prof_acc_s))

        for key, label, acc_s, f1_s, seen_s, prof_acc_s in rows:
            self.ui.tree_ds.insert("", "end", iid=key, values=(label, acc_s, f1_s, seen_s, prof_acc_s))
            
        if kids := list(self.ui.tree_ds.get_children()):
            self.ui.tree_ds.selection_set(kids[0])
            self.ui.tree_ds.see(kids[0])
            self._on_tree_ds_select()

"""Dataset actions (delete, move, create, snapshot, rename) and train / eval of the selection.

Mixin of :class:`gui.tabs.pipeline.tab.PipelineControlTab`; it relies on the attributes that class sets up
(``ui``, ``logic``, ``state``, ``sim_root`` ...).
"""

from __future__ import annotations
import logging
import os
import threading
from pathlib import Path
import shutil


from gui import dialogs
from gui.utils.dataset_ops import delete_samples, list_dataset_samples, move_samples, DatasetSampleInfo
from gui.utils.pipeline_helpers import (
    chain_commands,
    continue_train_argv,
    eval_argv,
    multi_train_argvs,
    train_argv,
    validate_argv,
)

from simple_sim.io_utils import now_tag
from simple_sim.model_bundle import bundle_checkpoint_path


_LOG = logging.getLogger(__name__)


class PipelineDatasetActionsMixin:
    def _delete_dataset(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before deleting datasets."); return
        if not dialogs.confirm(self.frame, "Delete Dataset", f"Permanently delete the dataset folder and all its images?\n\n{ds}", danger=True): return
        try: shutil.rmtree(ds); self._append_log(f"\n[deleted dataset {ds}]\n")
        except Exception as e: self.ui.show_messagebox("error", "Delete failed", str(e))
        self._refresh_datasets()

    def _delete_images_from_dataset(self) -> None:
        ds = self._selected_dataset_dir()
        if not ds:
            self.ui.show_messagebox("info", "Delete images", "No dataset selected.")
            return
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "Stop the running process before editing datasets.")
            return

        sample_ids = self._open_sample_selector(ds, title=f"Delete images from {ds.name}", action_text="Delete")
        if not sample_ids:
            return

        if not self.ui.ask_yes_no(
            "Delete images",
            f"Delete {len(sample_ids)} sample(s) from dataset '{ds.name}'?",
        ):
            return

        try:
            summary = delete_samples(ds, sample_ids)
            self._append_log(f"[dataset edit] deleted {summary.affected} sample(s) from {ds}\n")
            self._refresh_datasets()
            if label := self._dataset_label_for_path(ds):
                self.ui.var_dataset.set(label)
            self._on_dataset_selected()
            try:
                self.parent.event_generate("<<DatasetChanged>>", when="tail")
            except Exception:
                pass
        except Exception as exc:
            self.ui.show_messagebox("error", "Delete images failed", str(exc))

    def _move_images_between_datasets(self) -> None:
        src_ds = self._selected_dataset_dir()
        if not src_ds:
            self.ui.show_messagebox("info", "Move images", "No source dataset selected.")
            return
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "Stop the running process before editing datasets.")
            return

        target = self._open_target_dataset_selector(src_ds)
        if target is None:
            return

        sample_ids = self._open_sample_selector(src_ds, title=f"Move images from {src_ds.name}", action_text="Move")
        if not sample_ids:
            return

        if not self.ui.ask_yes_no(
            "Move images",
            f"Move {len(sample_ids)} sample(s)\nfrom '{src_ds.name}'\nto '{target.name}'?",
        ):
            return

        try:
            summary = move_samples(src_ds, target, sample_ids, enforce_profile_match=True)
            self._append_log(
                f"[dataset edit] moved {summary.affected} sample(s): {src_ds} -> {target}\n"
            )
            self._refresh_datasets()
            if label := self._dataset_label_for_path(src_ds):
                self.ui.var_dataset.set(label)
            self._on_dataset_selected()
            try:
                self.parent.event_generate("<<DatasetChanged>>", when="tail")
            except Exception:
                pass
        except Exception as exc:
            self.ui.show_messagebox("error", "Move images failed", str(exc))

    def _open_sample_selector(self, dataset_dir: Path, *, title: str, action_text: str) -> list[str]:
        try:
            samples = list_dataset_samples(dataset_dir)
        except Exception as exc:
            self.ui.show_messagebox("error", "Dataset read failed", str(exc))
            return []

        if not samples:
            self.ui.show_messagebox("info", "Dataset is empty", f"No samples found in '{dataset_dir.name}'.")
            return []

        samples_sorted = sorted(samples, key=lambda s: s.sample_id)
        ids = [row.sample_id for row in samples_sorted]
        picked = dialogs.pick_from_list(
            self.frame, title, ids, multi=True, labels=[self._sample_list_label(r) for r in samples_sorted],
            prompt=f"Dataset: {dataset_dir.name}. Select one or more samples.", size=(860, 520), ok_text=action_text,
        )
        return list(picked or [])

    def _open_target_dataset_selector(self, source_dataset: Path) -> Path | None:
        targets = [p for p in self._dataset_dirs if p.resolve() != source_dataset.resolve()]
        if not targets:
            self.ui.show_messagebox("info", "Move Images", "No target dataset available.")
            return None
        targets_sorted = sorted(targets, key=lambda p: self._dataset_label_for_path(p) or p.name)
        keys = [str(p) for p in targets_sorted]
        picked = dialogs.pick_from_list(
            self.frame, "Select Target Dataset", keys,
            labels=[self._dataset_label_for_path(p) or p.name for p in targets_sorted],
            prompt=f"Source dataset: {source_dataset.name}. Choose the dataset the images move to.",
            size=(620, 420), ok_text="Use Target",
        )
        if not picked:
            return None
        return targets_sorted[keys.index(picked[0])]

    def _sample_list_label(self, sample: DatasetSampleInfo) -> str:
        return f"{sample.sample_id} | class={sample.class_name} | split={sample.split} | image={Path(sample.image_path).name}"

    def _create_new_dataset(self) -> None:
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before creating datasets."); return

        initial = self.logic.slugify_name(self.ui.var_name.get())
        if not (new_name := self.ui.ask_string("Create dataset", "New dataset folder name (will be created under outputs/sim_data/runs):", initialvalue=initial)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name: self.ui.show_messagebox("error", "Error", "Name must not contain path separators."); return

        out_dir = (self.sim_root / "outputs" / "sim_data" / "runs" / new_name).resolve()
        if out_dir.exists():
            # Never fall through: creating the skeleton would empty an existing dataset's meta / labels.
            if self.ui.ask_yes_no("Dataset exists", f"Dataset already exists:\n\n{out_dir}\n\nSelect it?"): self._select_dataset(out_dir)
            return

        if not (config_candidate := Path(self.ui.var_config.get())).exists(): self.ui.show_messagebox("error", "Missing config", f"Config not found:\n\n{config_candidate}\n\nPick a valid Config first, then retry."); return
        if not (profile_id := self.ui.var_profile.get().strip()): self.ui.show_messagebox("error", "Missing profile", "No profile selected."); return

        try:
            if self.logic.ensure_dataset_skeleton(profile_id, config_candidate, out_dir, self._append_log):
                self.ui.var_dataset_mode.set("new"); self.ui.var_out.set(str(out_dir)); self.ui.var_name.set(new_name)
                model_path = (self.sim_root / "outputs" / "models" / f"{new_name}.pt").resolve()
                self.ui.var_model.set(str(model_path)); self._refresh_models()
                self._refresh_datasets(); self._select_dataset(out_dir)
                self._append_log(f"[create dataset] {out_dir}\n")
        except Exception as e: self.ui.show_messagebox("error", "Create failed", str(e))

    def _snapshot_dataset_selected(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before snapshotting datasets."); return

        versions_root = self.sim_root / "outputs" / "sim_data" / "versions"
        versions_root.mkdir(parents=True, exist_ok=True)
        group, tag = ds.name, now_tag(micro=False)
        dst = versions_root / group / f"{group}_{tag}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): self.ui.show_messagebox("error", "Error", f"Snapshot destination already exists:\n{dst}"); return
        if not self.ui.ask_yes_no("Snapshot dataset", f"Create snapshot?\n\nFrom:\n{ds}\n\nTo:\n{dst}"): return
        self._append_log(f"\n[snapshot dataset] {ds} -> {dst}\n")

        def link_or_copy_file(src: Path, dstp: Path) -> None:
            dstp.parent.mkdir(parents=True, exist_ok=True)
            try: os.link(src, dstp)
            except Exception: shutil.copy2(src, dstp)

        def worker() -> None:
            try:
                for dirpath, _dirnames, filenames in os.walk(ds):
                    rel = Path(dirpath).relative_to(ds)
                    for fn in filenames: link_or_copy_file(Path(dirpath) / fn, dst / rel / fn)
                self.log_q.put(f"[snapshot dataset] done: {dst}\n")
            except Exception as e: self.log_q.put(f"[snapshot dataset] failed: {e}\n"); shutil.rmtree(dst, ignore_errors=True)
            finally: self.post(self._refresh_datasets)
        threading.Thread(target=worker, daemon=True).start()

    def _rename_dataset_selected(self) -> None:
        if not (ds := self._selected_dataset_dir()): return
        if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "Stop the running process before renaming datasets."); return

        sim_data = (self.sim_root / "outputs" / "sim_data").resolve()
        runs = (sim_data / "runs").resolve()
        versions = (sim_data / "versions").resolve()
        rp = ds.resolve()
        kind, old_name, parent = "", rp.name, rp.parent
        if str(rp).startswith(str(runs) + os.sep): kind = "run"
        elif str(rp).startswith(str(versions) + os.sep): kind = "snapshot"
        else: self.ui.show_messagebox("error", "Error", f"Can only rename datasets under:\n{runs}\n{versions}\n\nSelected:\n{ds}"); return

        if not (new_name := self.ui.ask_string("Rename dataset", f"New folder name for this {kind} dataset:", initialvalue=old_name)): return
        if not (new_name := new_name.strip()) or "/" in new_name or "\\" in new_name: self.ui.show_messagebox("error", "Error", "Name must not contain path separators."); return
        
        dst = parent / new_name
        if dst.exists(): self.ui.show_messagebox("error", "Error", f"Target already exists:\n{dst}"); return
        if not self.ui.ask_yes_no("Rename dataset", f"Rename?\n\nFrom:\n{rp}\n\nTo:\n{dst}"): return

        # The catalog is keyed by relative path: carry category / archive flag over to the new name.
        old_category, was_archived = self._dataset_catalog.category_for(rp), self._dataset_catalog.is_archived(rp)
        try: rp.rename(dst)
        except Exception as e: self.ui.show_messagebox("error", "Error", f"Rename failed:\n{e}"); return
        try:
            self._dataset_catalog.set_category([dst], old_category)
            if was_archived: self._dataset_catalog.set_archived([dst], True)
            self._dataset_catalog.save(self.frame)
        except Exception: pass

        self.state.dataset_dir = dst; self._refresh_datasets()
        try:
            label = self._display_for_dataset(dst, runs=runs, versions=versions)
            for k, v in self._dataset_by_label.items():
                if v == dst or k == label: self.ui.var_dataset.set(k); break
        except Exception: pass
        self._on_dataset_selected()

        cur_model_str = self.ui.var_model.get().strip(); cur_resolved_model_path = str(self.logic._resolve_model_path(cur_model_str))
        old_default_model_path = str((self.sim_root / "outputs" / "models" / f"{old_name}.pt").resolve())
        if cur_resolved_model_path == old_default_model_path: self.ui.var_model.set(str(self.sim_root / "outputs" / "models" / f"{new_name}.pt"))
        
        cur_out_str = self.ui.var_out.get().strip()
        try:
            cur_resolved_out_path = str(self.logic._resolve_out_dir(cur_out_str))
            old_out_path = str((self.sim_root / "outputs" / "sim_data" / "runs" / old_name).resolve())
            if cur_resolved_out_path == old_out_path: self.ui.var_out.set(str(self.sim_root / "outputs" / "sim_data" / "runs" / new_name))
        except Exception: pass

        self._rename_models_for_dataset(old_name, new_name)
        self._append_log(f"[rename dataset] {rp} -> {dst}\n")
        try: self.parent.event_generate("<<DatasetChanged>>", when="tail")
        except Exception: pass

    def _rename_models_for_dataset(self, old_name: str, new_name: str) -> None:
        models_root = self.sim_root / "outputs" / "models"
        for suffix in ["", "_last"]:
            old, new = models_root / f"{old_name}{suffix}.pt", models_root / f"{new_name}{suffix}.pt"
            if old.exists():
                try: old.rename(new)
                except Exception as e: _LOG.warning("Failed to rename model %s -> %s: %s", old, new, e)
            old_meta, new_meta = models_root / f"{old_name}{suffix}.pt.meta.json", models_root / f"{new_name}{suffix}.pt.meta.json"
            if old_meta.exists():
                try: old_meta.rename(new_meta)
                except Exception as e: _LOG.warning("Failed to rename meta %s -> %s: %s", old_meta, new_meta, e)

        versions_root = models_root / "versions"
        old_dir, new_dir = versions_root / old_name, versions_root / new_name
        if old_dir.exists():
            try: old_dir.rename(new_dir)
            except Exception as e: _LOG.warning("Failed to rename versioned models %s -> %s: %s", old_dir, new_dir, e)

    def _validate_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        if len(dss) == 1: self._run_simple_cmd(validate_argv(dss[0])); return

        self._append_log(f"\n=== Validate {len(dss)} datasets ===\n")
        self._run_simple_cmd([chain_commands([validate_argv(ds) for ds in dss])])

    def _train_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        if len(dss) > 1:
            if not (strat := self._ask_multi_dataset_train_strategy(dss)): return
            if strat in ("merge_keep", "merge_temp"):
                if self.logic.is_busy(): self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first."); return
                
                ts = now_tag(micro=False)
                name = ""
                if strat == "merge_keep":
                    default = f"{dss[0].name}_merged_{len(dss)}"
                    if not (name := self.ui.ask_string("Merge datasets", "Output dataset folder name (will be created under outputs/sim_data/runs):", initialvalue=default)): return
                    if not (name := name.strip()) or "/" in name or "\\" in name: self.ui.show_messagebox("error", "Error", "Invalid dataset name."); return
                else: name = f"__tmp_merge_{ts}"
                
                out_ds = (self.sim_root / "outputs" / "sim_data" / "runs" / name).resolve()
                if out_ds.exists(): self.ui.show_messagebox("error", "Error", f"Merge output already exists:\n{out_ds}"); return

                model_s = self.ui.var_model.get().strip()
                model_out = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{name}.pt").resolve()
                delete_after = (strat == "merge_temp")

                self._append_log(f"\n=== Merge {len(dss)} datasets -> {out_ds.name} ===\n")
                for ds in dss: self._append_log(f"  - {ds}\n")
                self._append_log("\n")

                self.logic.loop_running = True; self.logic.stop_evt.clear()  # keep Busy guards closed until training starts
                def worker() -> None:
                    try:
                        self.log_q.put(f"[merge] building merged dataset: {out_ds}\n")
                        self.logic.merge_datasets_for_training(dss, out_ds, self.log_q.put, self.log_q.put)
                        self.log_q.put(f"[merge] done: {out_ds}\n")

                        def kick_off_train() -> None:
                            self.logic.loop_running = False
                            if self.logic.stop_evt.is_set():
                                self._append_log("[merge] stopped by user; training not started\n")
                                if delete_after: shutil.rmtree(out_ds, ignore_errors=True)
                                self._refresh_datasets(); return
                            try:
                                if strat == "merge_keep": self._select_dataset(out_ds)
                                else: self._refresh_datasets()
                            except Exception: pass

                            cmd = chain_commands([train_argv(out_ds, model_out)])
                            self._run_simple_cmd([cmd], cleanup_dir=out_ds if delete_after else None)

                        self.post(kick_off_train)
                    except Exception as e:
                        self.log_q.put(f"[merge] failed: {e}\n")
                        if delete_after: shutil.rmtree(out_ds, ignore_errors=True)
                        def _merge_failed_ui() -> None: self.logic.loop_running = False
                        self.post(_merge_failed_ui)
                self.ui.set_run_buttons_state(True)
                try: threading.Thread(target=worker, daemon=True).start()
                except Exception: self.logic.loop_running = False; raise
                return

        model_s = self.ui.var_model.get().strip()
        model_out = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()

        if len(dss) == 1: self._run_simple_cmd(train_argv(dss[0], model_out)); return

        allow_mixed_profiles = False
        if self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_out):
            if not self.ui.ask_yes_no("Allow mixed-profile training into a single model?", "You selected datasets with different component profiles.\n\nRecommended: train into a multi-model bundle (.bundle) so each profile gets its own checkpoint.\n\nIf you continue anyway, the training will resume across different profiles by passing --allow-profile-mismatch (warm-start). This is not guaranteed to work and can degrade accuracy.\n\nContinue?"): return
            allow_mixed_profiles = True

        extra_s = self.ui.var_continue_epochs.get().strip() or "10"
        try: extra = int(extra_s)
        except Exception: self.ui.show_messagebox("error", "Error", "+epochs must be an integer."); return
        if extra < 0: self.ui.show_messagebox("error", "Error", "+epochs must be >= 0."); return

        out_mode = self.ui.var_continue_out_mode.get().strip() or "best"
        if out_mode not in ("last", "best"): out_mode = "best"

        argvs = multi_train_argvs(
            dss, model_out, [self.logic.get_dataset_profile_id(ds) for ds in dss],
            is_bundle=self.logic._is_model_bundle_target(model_out), extra_epochs=extra, out_mode=out_mode,
            allow_profile_mismatch=allow_mixed_profiles,
        )

        self._append_log(f"\n=== Multi-dataset train: {len(dss)} datasets ===\n")
        for ds in dss: self._append_log(f"  - {ds}\n")
        self._append_log(f"out: {model_out}\n+epochs per subsequent dataset: {extra} (out-mode={out_mode})\n\n")
        self._run_simple_cmd([chain_commands(argvs)])

    def _eval_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return
        
        model_s = self.ui.var_model.get().strip()
        model_in = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_in.exists(): self.ui.show_messagebox("error", "Missing model", f"Model not found:\n{model_in}\n\nRun Train first."); return
        if len(dss) == 1: self._run_simple_cmd(eval_argv(dss[0], model_in)); return

        allow_mixed_profiles = False
        if self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_in):
            if not self.ui.ask_yes_no("Allow mixed-profile eval on a single model?", "You selected datasets with different component profiles.\n\nRecommended: evaluate a multi-model bundle (.bundle) so each dataset uses its matching checkpoint.\n\nIf you continue anyway, eval will pass --allow-profile-mismatch. Results may be meaningless.\n\nContinue?"): return
            allow_mixed_profiles = True

        self._append_log(f"\n=== Eval on {len(dss)} datasets ===\n")
        self._run_simple_cmd([chain_commands([eval_argv(ds, model_in, allow_profile_mismatch=allow_mixed_profiles) for ds in dss])])

    def _continue_train_selected(self) -> None:
        dss = self._selected_dataset_dirs();
        if not dss: return

        model_s = self.ui.var_model.get().strip()
        model_path = self.logic._resolve_model_path(model_s) if model_s else (self.sim_root / "outputs" / "models" / f"{dss[0].name}.pt").resolve()
        if not model_path.exists(): self.ui.show_messagebox("error", "Missing model", f"Model not found to resume from:\n{model_path}\n\nRun Train first."); return

        extra_s = self.ui.var_continue_epochs.get().strip() or "10"
        try: extra = int(extra_s)
        except Exception: self.ui.show_messagebox("error", "Error", "Extra epochs must be an integer."); return
        if extra < 1: self.ui.show_messagebox("error", "Error", "Extra epochs must be >= 1."); return

        out_mode = self.ui.var_continue_out_mode.get().strip() or "last"
        if out_mode not in ("last", "best"): out_mode = "last"

        allow_mixed_profiles = False
        if len(dss) > 1 and self.logic._datasets_require_bundle(dss, self.logic.get_dataset_profile_id) and not self.logic._is_model_bundle_target(model_path):
            if not self.ui.ask_yes_no("Allow mixed-profile resume into a single model?", "You selected datasets with different component profiles.\n\nRecommended: resume into a multi-model bundle (.bundle).\n\nIf you continue anyway, the resume will pass --allow-profile-mismatch. This is not guaranteed to work.\n\nContinue?"): return
            allow_mixed_profiles = True

        if len(dss) > 1 and self.logic._is_model_bundle_target(model_path):
            missing: list[str] = []
            for ds in dss:
                if not (pid := self.logic.get_dataset_profile_id(ds)): missing.append(f"{ds} (missing profile metadata)"); continue
                try:
                    if not bundle_checkpoint_path(model_path, pid, kind="best").exists(): missing.append(f"{pid} (no checkpoint in bundle)")
                except Exception: missing.append(f"{pid} (bundle lookup failed)")
            if missing: self.ui.show_messagebox("error", "Missing bundle checkpoints", "The selected model bundle does not have checkpoints for all selected datasets.\n\n" + "\n".join(missing)); return

        if len(dss) == 1:
            ds = dss[0]
            self._append_log(f"\n=== Continue training: +{extra} epochs (out={out_mode}) ===\n" f"data:  {ds}\n" f"model: {model_path}\n\n")
            self._run_simple_cmd(continue_train_argv(ds, model_path, extra_epochs=extra, out_mode=out_mode))
            return

        argvs = [
            continue_train_argv(ds, model_path, extra_epochs=extra, out_mode=out_mode, allow_profile_mismatch=allow_mixed_profiles)
            for ds in dss
        ]

        self._append_log(f"\n=== Continue training on {len(dss)} datasets: +{extra} epochs each (out={out_mode}) ===\n" f"model: {model_path}\n")
        for ds in dss: self._append_log(f"data:  {ds}\n")
        self._append_log("\n")

        self._run_simple_cmd([chain_commands(argvs)])

"""Starting / stopping pipeline runs: run specs, config patching and the command runner.

Mixin of :class:`gui.tabs.pipeline.tab.PipelineControlTab`; it relies on the attributes that class sets up
(``ui``, ``logic``, ``state``, ``sim_root`` ...).
"""

from __future__ import annotations
import os
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Any
import shutil
import shlex


from gui.utils.pipeline_helpers import (
    apply_render_overrides,
    mixed_profile_config,
    parse_run_count,
    precise_counts_for_profile,
    profile_run_targets,
    render_overrides,
    render_overrides_text,
    resolve_under,
    unique_timestamped_dir,
)

from simple_sim.io_utils import load_yaml, now_tag, rel_to

from .start_summary import start_summary
import contextlib


class PipelineRunMixin:
    def _confirm_pipeline_start(
        self,
        *,
        task: str,
        out_dir: str,
        run_mode: str,
        run_count: int,
        dataset_mode: str,
        profile_ids: list[str],
        multi_enabled: bool,
        multi_mode: str,
        image_filters: dict[str, Any],
    ) -> bool:
        model_path = self.ui.var_model.get().strip()
        header, filter_details, precise_details = start_summary(
            task=task, out_dir=out_dir, run_mode=run_mode, run_count=run_count, dataset_mode=dataset_mode,
            profile_ids=profile_ids, multi_enabled=multi_enabled, multi_mode=multi_mode, image_filters=image_filters,
            precise_settings=self._precise_settings, model_path=model_path)

        top = tk.Toplevel(self.frame.winfo_toplevel())
        top.title("Start pipeline")
        top.transient(self.frame.winfo_toplevel())
        top.resizable(True, True)
        top.minsize(740, 460)
        top.geometry("860x620")

        result = {"ok": False}

        outer = ttk.Frame(top, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text=header, justify="left", anchor="w").pack(fill="x", pady=(0, 10))

        sep = ttk.Separator(outer, orient="horizontal")
        sep.pack(fill="x", pady=(0, 8))

        details_wrap = ttk.Frame(outer)
        details_wrap.pack(fill="both", expand=True)

        def _add_collapsible(parent: ttk.Frame, title: str, text: str) -> None:
            block = ttk.Frame(parent)
            block.pack(fill="x", pady=(0, 6))
            state = {"open": False}
            body = ttk.Frame(block)
            btn = ttk.Button(block, text=f"\u25b8 {title}")
            btn.pack(anchor="w")

            height = min(14, max(3, len(text.splitlines())))
            txt = tk.Text(body, height=height, wrap="word")
            txt.insert("1.0", text)
            txt.configure(state="disabled")
            txt.pack(fill="x", expand=True)

            def _toggle() -> None:
                if state["open"]:
                    body.pack_forget()
                    btn.configure(text=f"\u25b8 {title}")
                    state["open"] = False
                else:
                    body.pack(fill="x", pady=(4, 0))
                    btn.configure(text=f"▼ {title}")
                    state["open"] = True

            btn.configure(command=_toggle)

        _add_collapsible(details_wrap, "Filter Details", filter_details)
        _add_collapsible(details_wrap, "Precise Details", precise_details)
        if task != "generate_only":
            _add_collapsible(details_wrap, "Model Details", f"model_path={model_path}")

        btns = ttk.Frame(outer)
        btns.pack(fill="x", pady=(8, 0))

        def _on_cancel() -> None:
            result["ok"] = False
            top.destroy()

        def _on_start() -> None:
            result["ok"] = True
            top.destroy()

        ttk.Button(btns, text="Cancel", width=12, command=_on_cancel).pack(side="right")
        ttk.Button(btns, text="Start", width=12, command=_on_start).pack(side="right", padx=(0, 8))

        top.bind("<Escape>", lambda _e: _on_cancel())
        top.protocol("WM_DELETE_WINDOW", _on_cancel)
        top.grab_set()
        top.wait_window()
        return bool(result["ok"])

    def _confirm_overwrite_existing_new_dataset(self, out_dir: str) -> bool:
        """Double confirmation before overwriting an existing dataset in Create New mode."""
        out_path = self.logic._resolve_out_dir(out_dir)
        if not out_path.exists():
            return True
        if not self.ui.ask_yes_no(
            "Output already exists",
            "Dataset Mode is 'Create New', but the output folder already exists.\n\n"
            f"{out_path}\n\n"
            "Continuing can overwrite existing dataset contents.\n\nContinue?",
        ):
            return False
        return self.ui.ask_yes_no(
            "Confirm overwrite",
            "Final confirmation:\n\n"
            f"Overwrite existing dataset at:\n{out_path}\n\n"
            "Proceed?",
        )

    def _ask_multi_dataset_train_strategy(self, dss: list[Path]) -> str | None:
        msg = (
            f"{len(dss)} datasets selected.\n\n"
            "Yes: Merge and keep merged dataset\n"
            "No: Merge into temporary dataset and auto-cleanup\n"
            "Cancel: Abort"
        )
        choice = messagebox.askyesnocancel("Multi-dataset training", msg)
        if choice is None:
            return None
        return "merge_keep" if choice else "merge_temp"

    def start_pipeline(self) -> None: self._start_pipeline(task="full")
    def start_pipeline_generate_only(self) -> None: self._start_pipeline(task="generate_only")

    def _rel_to_sim_root(self, p: Path) -> str:
        return rel_to(self.sim_root, p)

    def _build_run_specs(
        self,
        *,
        task: str,
        run_count: int,
        dataset_mode: str,
        multi_enabled: bool,
        multi_mode: str,
        profile_ids: list[str],
        render_backend: str,
        out_dir: str,
    ) -> tuple[list[dict[str, str]], str, Any, str] | None:
        run_specs: list[dict[str, str]] = []
        effective_task = task
        run_complete_callback = None

        if multi_enabled and multi_mode == "mixed":
            if task != "generate_only":
                self.ui.show_messagebox("error", "Error", "Multi-profile mode 'mixed' is supported only for Generate Only.")
                return None
            if len(profile_ids) < 2:
                self.ui.show_messagebox("error", "Error", "Mixed dataset mode requires at least 2 selected profiles.")
                return None
            if run_count != 1:
                self.ui.show_messagebox("error", "Error", "Mixed dataset generation currently supports only a single run.")
                return None

            resolved_configs: list[tuple[str, Path, str, str]] = []
            backends_used: set[str] = set()
            for pid in profile_ids:
                cfg_info = self.logic.config_for_profile(pid, want_backend=render_backend or None)
                if not cfg_info:
                    cfg_info = self.logic.config_for_profile(pid, want_backend=None)
                if not cfg_info:
                    self.ui.show_messagebox("error", "Error", f"No matching config found for profile:\n{pid}\n\nPick or create a config with run.component_profile={pid}.")
                    return None
                cfg_path = Path(cfg_info["path"])
                run_id = str(cfg_info.get("run_id") or pid)
                try:
                    cfg_data = load_yaml(cfg_path) or {}
                except Exception:
                    cfg_data = {}
                backend = str((cfg_data.get("render") or {}).get("backend") or "opencv_2d").strip() or "opencv_2d"
                resolved_configs.append((pid, cfg_path, run_id, backend))
                backends_used.add(backend)

            if backends_used == {"opencv_2d"}:
                cfg_str = self.ui.var_config.get().strip()
                cfg_path = resolve_under(self.sim_root, cfg_str) if cfg_str else None
                if not cfg_path or not cfg_path.exists():
                    self.ui.show_messagebox("error", "Error", f"Config file not found:\n{cfg_str}")
                    return None
                try:
                    base_cfg = load_yaml(cfg_path) or {}
                except Exception as e:
                    self.ui.show_messagebox("error", "Error", f"Failed to read config:\n{cfg_path}\n\n{e}")
                    return None

                try:
                    tmp_cfg_path = self._write_temp_cfg(mixed_profile_config(base_cfg, profile_ids), prefix="run_profile_cls_selected_")
                except Exception as e:
                    self.ui.show_messagebox("error", "Error", f"Failed to write temporary config under outputs/live:\n\n{e}")
                    return None

                if dataset_mode == "extend":
                    # generate-profile-dataset has no extend mode: it would replace the selected dataset.
                    self.ui.show_messagebox("error", "Error", "Extend Existing is not supported for a mixed multi-profile dataset (2D only).\n\nUse 'Create New', or extend with a single profile.")
                    return None
                run_specs.append({
                    "profile_id": "multi", "config": self._rel_to_sim_root(tmp_cfg_path), "out_dir": out_dir,
                    "model_path": self.ui.var_model.get().strip(),
                })
                effective_task = "generate_mixed"
            else:
                final_out = self.logic._resolve_out_dir(out_dir)
                replace_existing_final_out = False
                if final_out.exists() and dataset_mode == "extend":
                    replace_existing_final_out = True
                    self._append_log(f"[mixed] reusing selected output path: {final_out}\n")
                elif final_out.exists():
                    final_out = unique_timestamped_dir(final_out, now_tag(micro=False))
                    out_dir = str(final_out)
                    self.ui.var_out.set(rel_to(self.sim_root, final_out))
                    self._append_log(f"[mixed] output exists, using new dataset path: {final_out}\n")

                live_dir = (self.sim_root / "outputs" / "live")
                live_dir.mkdir(parents=True, exist_ok=True)
                tmp_root = live_dir / f"mixed_profile_gen_{int(time.time())}_{os.getpid()}"
                try:
                    tmp_root.mkdir(parents=True, exist_ok=False)
                except Exception as e:
                    self.ui.show_messagebox("error", "Error", f"Failed to create temporary mixed-generation directory:\n{tmp_root}\n\n{e}")
                    return None

                self._append_log(f"[mixed] Cross-backend profiles detected ({', '.join(sorted(backends_used))}). Generating per profile and auto-merging.\n")
                for idx, (pid, cfg_path, _run_id, backend) in enumerate(resolved_configs):
                    tmp_out = tmp_root / f"{idx:02d}_{self.logic.slugify_name(pid)}"
                    run_specs.append({
                        "profile_id": pid,
                        "config": self._rel_to_sim_root(cfg_path),
                        "out_dir": str(tmp_out),
                        "model_path": self.ui.var_model.get().strip(),
                    })
                    self._append_log(f"[mixed] {pid} -> backend={backend} tmp={tmp_out}\n")
                effective_task = "generate_only"

                def _merge_after_run(_run_i: int, specs: list[dict[str, str]], log_callback) -> bool:
                    ok = True
                    src_dirs: list[Path] = []
                    try:
                        src_dirs = [Path(spec["out_dir"]).resolve() for spec in specs]
                        merge_sources = list(src_dirs)
                        merge_target = final_out
                        tmp_target: Path | None = None

                        if replace_existing_final_out and final_out.exists():
                            if dataset_mode == "extend":
                                merge_sources = [final_out] + merge_sources
                                tmp_target = final_out.parent / f"{final_out.name}.__merge_tmp_{int(time.time())}_{os.getpid()}"
                                if tmp_target.exists():
                                    shutil.rmtree(tmp_target, ignore_errors=True)
                                merge_target = tmp_target
                                log_callback(f"[mixed] extend mode: including existing dataset as merge source: {final_out}\n")
                            else:
                                shutil.rmtree(final_out, ignore_errors=True)
                                log_callback(f"[mixed] removed previous dataset at target: {final_out}\n")

                        log_callback(f"[mixed] merging {len(merge_sources)} source datasets into: {merge_target}\n")
                        self.logic.merge_datasets_for_training(merge_sources, merge_target, log_callback, log_callback)
                        if tmp_target is not None and merge_target == tmp_target:
                            if final_out.exists():
                                shutil.rmtree(final_out, ignore_errors=True)
                            tmp_target.rename(final_out)
                            merge_target = final_out
                            log_callback(f"[mixed] extend mode: replaced target dataset with merged result: {final_out}\n")
                        log_callback(f"[mixed] merged dataset ready: {final_out}\n")
                    except Exception as e:
                        log_callback(f"[mixed] merge failed: {e}\n")
                        ok = False
                    if not ok:
                        # Keep freshly generated samples so a failed merge can be retried manually.
                        log_callback(f"[mixed] temp generation data kept after failed merge: {tmp_root}\n")
                    else:
                        try:
                            for p in src_dirs:
                                if p.exists():
                                    shutil.rmtree(p, ignore_errors=True)
                            if tmp_root.exists():
                                shutil.rmtree(tmp_root, ignore_errors=True)
                            log_callback(f"[mixed] cleaned temp generation data: {tmp_root}\n")
                        except Exception as e:
                            log_callback(f"[mixed] temp cleanup warning: {e}\n")
                    return ok

                run_complete_callback = _merge_after_run
        else:
            base_out = Path(out_dir)
            base_model = Path(self.ui.var_model.get().strip())
            for pid in profile_ids:
                cfg_info = self.logic.config_for_profile(pid, want_backend=render_backend or None)
                if not cfg_info:
                    self.ui.show_messagebox("error", "Error", f"No matching config found for profile:\n{pid}\n\nPick or create a config with run.component_profile={pid}.")
                    return None
                cfg_path = Path(cfg_info["path"])
                cfg_rel = self._rel_to_sim_root(cfg_path)
                run_id = str(cfg_info.get("run_id") or pid)
                out_i, model_i = profile_run_targets(
                    base_out, base_model, run_id, multi=len(profile_ids) > 1, models_root=self.sim_root / "outputs" / "models"
                )

                run_specs.append({
                    "profile_id": pid, "config": cfg_rel, "out_dir": str(out_i), "model_path": str(model_i),
                })

        return run_specs, effective_task, run_complete_callback, out_dir

    def _apply_precise_counts_to_run_specs(self, run_specs: list[dict[str, str]]) -> list[dict[str, str]] | None:
        counts = self._get_precise_counts()
        if counts is None:
            return None
        patched_specs: list[dict[str, str]] = []
        for spec in run_specs:
            cfg_path = resolve_under(self.sim_root, spec["config"])
            spec_counts = precise_counts_for_profile(counts, spec.get("profile_id", ""))
            try:
                patched_path = self._write_precise_config(cfg_path, spec_counts)
                patched_specs.append({**spec, "config": self._rel_to_sim_root(patched_path)})
            except Exception as e:
                self.ui.show_messagebox("error", "Error", f"Failed to write precise config:\n\n{e}")
                return None
        return patched_specs
    
    def _render_overrides(self) -> dict[str, Any]:
        """``render.look`` / ``render.board`` chosen in the Source section (empty: the configs' own values)."""
        return render_overrides(look=self.ui.var_look.get(), routing=self.ui.var_board_routing.get(),
                                drc=self.ui.var_board_drc.get(), design_rules=self.ui.var_board_rules.get())

    def _apply_render_overrides_to_run_specs(self, run_specs: list[dict[str, str]],
                                             overrides: dict[str, Any]) -> list[dict[str, str]] | None:
        """Each spec's config rewritten as a temp config under outputs/live with *overrides* applied."""
        patched_specs: list[dict[str, str]] = []
        for spec in run_specs:
            cfg_path = resolve_under(self.sim_root, spec["config"])
            try:
                data = load_yaml(cfg_path) or {}
                data["render"] = apply_render_overrides(dict(data.get("render") or {}), overrides)
                patched_path = self._write_temp_cfg(data, prefix="render_")
                patched_specs.append({**spec, "config": self._rel_to_sim_root(patched_path)})
            except Exception as e:
                self.ui.show_messagebox("error", "Error", f"Failed to write render config:\n\n{e}")
                return None
        return patched_specs

    def _start_pipeline(self, *, task: str) -> None:
        if self.logic.is_busy(): return
        
        run_mode, dataset_mode = self.ui.var_run_mode.get(), self.ui.var_dataset_mode.get()
        out_dir = ""
        if dataset_mode == "extend":
            if not (selected_ds := self._selected_dataset_dir()):
                self.ui.show_messagebox("error", "Error", "No dataset selected to extend.\n\nSelect a dataset or use 'Create New' mode.")
                return
            try:
                if str(selected_ds.resolve()).startswith(str((self.sim_root / "outputs" / "sim_data" / "versions").resolve()) + os.sep):
                    self.ui.show_messagebox("error", "Error", "Cannot extend/overwrite a dataset snapshot under outputs/sim_data/versions.\n\nSelect a dataset under outputs/sim_data/runs instead.")
                    return
            except Exception: pass
            if not self.ui.ask_yes_no("Extend dataset?", f"Extend Existing will APPEND new samples into the selected dataset folder.\n\nDataset:\n{selected_ds}\n\nContinue?"): return
            out_dir = str(selected_ds)
            self._append_log(f"\n=== Extending selected dataset: {out_dir} ===\n")
        else:
            out_dir = self.ui.var_out.get().strip()
            self._append_log(f"\n=== Creating new dataset: {out_dir} ===\n")

        try: run_count = parse_run_count(run_mode, self.ui.var_run_count.get())
        except ValueError:
            self.ui.show_messagebox("error", "Error", "Invalid run count. Please enter a positive integer.")
            return

        self._append_log(f"Run mode: {run_mode}" + (f" ({run_count}x)" if run_count > 0 else " (continuous)") + "\n\n")

        render_backend = self.ui.var_render_backend.get().strip() or ""
        
        multi_enabled = bool(self.ui.var_profiles_multi.get())
        multi_mode = self.ui.var_profiles_multi_mode.get().strip() or "separate"
        
        profile_ids = []
        if multi_enabled:
            profile_ids = self._profiles_multi_list(available=self._profiles_multi_available_values())
            if not profile_ids:
                self.ui.show_messagebox("error", "Error", "Multi-profile mode is enabled but no profiles are selected.\n\nClick Pick and select at least 1 profile.")
                return
        else:
            if p := self.ui.var_profile.get().strip(): profile_ids = [p]

        try:
            if profile_ids: self._autoselect_config_for_profile(profile_ids[0], prefer_quiet=False)
        except Exception: pass

        built_specs = self._build_run_specs(
            task=task,
            run_count=run_count,
            dataset_mode=dataset_mode,
            multi_enabled=multi_enabled,
            multi_mode=multi_mode,
            profile_ids=profile_ids,
            render_backend=render_backend,
            out_dir=out_dir,
        )
        if built_specs is None:
            return
        run_specs, effective_task, run_complete_callback, out_dir = built_specs

        if dataset_mode == "new":
            # Several profiles in separate mode write runs/<run_id> each, not the base folder.
            targets = [out_dir] if run_complete_callback is not None else list(dict.fromkeys(s["out_dir"] for s in run_specs))
            if not all(self._confirm_overwrite_existing_new_dataset(t) for t in targets):
                return
        
        # Precise mode: patch each spec's config with user-specified sample counts
        if run_mode == "precise":
            patched_specs = self._apply_precise_counts_to_run_specs(run_specs)
            if patched_specs is None:
                return
            run_specs = patched_specs

        # Look / board selectors: patch each spec's config like precise mode does (generate --extend then
        # compares the look options with the dataset's config.yaml as for any config edit).
        try:
            overrides = self._render_overrides()
        except ValueError as e:
            self.ui.show_messagebox("error", "Error", str(e))
            return
        if overrides:
            self._append_log(f"Render overrides: {render_overrides_text(overrides)}\n")
            patched_specs = self._apply_render_overrides_to_run_specs(run_specs, overrides)
            if patched_specs is None:
                return
            run_specs = patched_specs

        image_filters = self._image_filters_payload()

        if not self._confirm_pipeline_start(
            task=effective_task, out_dir=out_dir, run_mode=run_mode, run_count=run_count,
            dataset_mode=dataset_mode, profile_ids=profile_ids, multi_enabled=multi_enabled, multi_mode=multi_mode,
            image_filters=image_filters,
        ): return

        # Mixed cross-backend generation runs through temporary per-profile output dirs.
        # Those dirs are always newly created and must not use --extend.
        runtime_dataset_mode = dataset_mode
        if (
            dataset_mode == "extend"
            and multi_enabled
            and multi_mode == "mixed"
            and effective_task == "generate_only"
        ):
            runtime_dataset_mode = "new"
            self._append_log("[mixed] using internal dataset mode 'new' for temporary generation dirs; extend is applied during final merge.\n")

        # The confirm dialogs above run a nested event loop (F5 / buttons stay live): check again.
        if self.logic.is_busy(): return
        self.ui.set_run_buttons_state(True)
        self.logic.stop_evt.clear()

        status_vars = {
            "phase": self.ui.var_phase, "run_progress": self.ui.var_run_progress,
            "epoch": self.ui.var_epoch, "ips": self.ui.var_ips, "last": self.ui.var_last,
            "dataset_dir": self.state.dataset_dir,
            "snap_every": self._snap_every_n(), "snap_keep": self._snap_keep_n(), "autosnap": self.ui.var_autosnap.get()
        }
        self.logic.start_pipeline(
            run_specs=run_specs, run_count=run_count, dataset_mode=runtime_dataset_mode, task=effective_task,
            image_filters=image_filters,
            event_callback=self._handle_event, log_callback=self._append_log,
            ui_update_callback=lambda: self.post(self._after_pipeline_run_ui_update),
            ui_reset_callback=lambda: self.post(self._reset_ui_on_pipeline_end),
            status_vars=status_vars,
            run_complete_callback=run_complete_callback,
        )

    def _after_pipeline_run_ui_update(self) -> None:
        self._refresh_datasets()
        if self.ui.var_dataset_mode.get() != "extend":
            out_path = self.logic._resolve_out_dir(self.ui.var_out.get())
            runs_base = (self.sim_root / "outputs" / "sim_data" / "runs").resolve()
            try:
                out_path.resolve().relative_to(runs_base)
                self.ui.var_dataset.set(self._dataset_label_for_path(out_path) or out_path.name)
                self._on_dataset_selected()
            except Exception: pass
        else:
            if self.state.dataset_dir: self._start_dataset_size_calc(self.state.dataset_dir)

    def _reset_ui_on_pipeline_end(self) -> None:
        for p in self._temp_cfg_paths:
            with contextlib.suppress(Exception): p.unlink(missing_ok=True)
        self._temp_cfg_paths.clear()
        self.ui.set_run_buttons_state(False)
        self.ui.var_phase.set("phase: idle")
        self.ui.var_run_progress.set("run: -")
        self.logic.current_run_iteration = 0
        self.logic.total_run_count = 0

    def stop_pipeline(self) -> None:
        self.logic.stop_pipeline()

    def _run_simple_cmd(self, args: list[str], extra_env: dict[str, str] | None = None, *, cleanup_dir: Path | None = None) -> bool:
        # cleanup_dir (or a pending logic._temp_merge_cleanup_dir) is removed when this process exits with rc=0.
        cleanup_dir = cleanup_dir or self.logic._temp_merge_cleanup_dir
        self.logic._temp_merge_cleanup_dir = None
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first.")
            if cleanup_dir is not None: self.logic.cleanup_temp_merge_dir(cleanup_dir, 0, self._append_log)
            return False

        self.ui.set_run_buttons_state(True)
        self.logic.stop_evt.clear()
        
        env = os.environ.copy()
        env["EVENT_LOG"] = str(self.event_log_path)
        env["SIMPLE_SIM_EVENT_LOG"] = str(self.event_log_path)
        env["PYTHONUNBUFFERED"] = "1"
        if extra_env: env.update(extra_env)

        # A single element is a pre-built shell command line; multiple elements are argv and get quoted
        # so dataset/model paths containing spaces survive bash -lc.
        shell_cmd = args[0] if len(args) == 1 else shlex.join(args)
        cmd = ["bash", "-lc", shell_cmd]

        def on_exit(rc: int | None) -> None:  # reader thread; logging goes through log_q
            if cleanup_dir is not None: self.logic.cleanup_temp_merge_dir(cleanup_dir, rc, self._append_log)

        try: self.logic.launch_process(cmd, env, self._append_log, self._handle_event, on_exit=on_exit)
        except Exception as e:
            self._append_log(f"\nError starting command: {e}\n"); self.ui.set_run_buttons_state(False)
            if cleanup_dir is not None: self.logic.cleanup_temp_merge_dir(cleanup_dir, 0, self._append_log)
            return False
        self.ui.var_phase.set("running cmd")
        return True

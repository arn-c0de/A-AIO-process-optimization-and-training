"""Dataset / model / profile selection widgets of the pipeline tab.

Mixin of :class:`gui.tabs.pipeline.tab.PipelineControlTab`; it relies on the attributes that class sets up
(``ui``, ``logic``, ``state``, ``sim_root`` ...).
"""

from __future__ import annotations
import json
import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from gui import dialogs
from gui.utils.checkpoint_meta import short_profile_hash
from gui.utils.dataset_catalog import catalog_labels, dataset_display_name, sim_data_roots
from gui.utils.pipeline_helpers import (
    decode_dataset_paths_json,
    encode_dataset_paths_json,
    parse_profile_ids,
    profile_compat_status,
    profile_model_paths,
    resolve_under,
    summarize_labels,
    summarize_profile_ids,
)

from simple_sim.config import DEFAULT_COMPONENT_PROFILE
from simple_sim.dataset_files import CONFIG_YAML, DATASET_MANIFEST, LABELS_JSONL, META_JSONL
from simple_sim.io_utils import load_yaml, now_tag, rel_to
from simple_sim.dataset_index import DatasetIndex
from simple_sim.model_bundle import bundle_checkpoint_path
import contextlib


_LOG = logging.getLogger(__name__)


def load_dataset_view(ds: Path, n_recent: int = 12) -> dict[str, Any]:
    """Tk-free (worker thread): lookups for thumbnails/overlays + the newest images of ``ds``.

    meta/labels come from :class:`DatasetIndex` (parsed once per file version, shared with the
    library). Same content as the former inline ``read_jsonl`` calls; a read error leaves the maps
    empty and is reported in ``"error"`` (logged by the caller, as before).
    """
    out: dict[str, Any] = {"image_to_sample": {}, "meta_by_sample": {}, "meta_by_image_rel": {},
                           "label_dict": {}, "recent_imgs": [], "has_images": False, "error": None}
    try:
        idx = DatasetIndex(ds)
        if (ds / META_JSONL).exists():
            meta_rows = idx.meta_rows()
            out["image_to_sample"] = {row.image_path: row.id for row in meta_rows}
            out["meta_by_sample"] = {row.id: row for row in meta_rows}
            out["meta_by_image_rel"] = {row.image_path: row for row in meta_rows}
        if (ds / LABELS_JSONL).exists():
            out["label_dict"] = {row.id: row.class_name for row in idx.label_rows()}
    except Exception as e:
        out["error"] = e
    if (img_dir := ds / "images").exists():
        out["has_images"] = True
        out["recent_imgs"] = [str(p) for p in sorted(img_dir.glob("*.png"))[-n_recent:]]
    return out


class PipelineSelectionMixin:
    def _refresh_profile_models(self) -> None:
        paths = self.logic.get_profile_classifier_models()
        values = [""] + [str(p.resolve().relative_to(self.sim_root.resolve())) if p.is_relative_to(self.sim_root) else str(p) for p in paths]
        self.ui.set_profile_model_combo_values(values)
        if (cur := self.ui.var_profile_model.get().strip()) and cur in values: return
        if not cur: self.ui.var_profile_model.set("")

    def _apply_profile_model_lock(self) -> None:
        locked = bool(self.ui.var_profile_model_lock.get())
        if locked and not self.ui.var_profile_model.get().strip():
            self.ui.var_profile_model_lock.set(False)
            return
        self.ui.set_profile_model_combo_state("disabled" if locked else "readonly")

    def _build_profile_model(self) -> None:
        if self.logic.is_busy():
            self.ui.show_messagebox("warning", "Busy", "A process is already running. Stop it first.")
            return

        preset = (self.ui.var_profile_build_preset.get().strip() or "quick").lower()
        if preset not in ("quick", "full"): preset = "quick"

        cfg, out_ds, out_model = profile_model_paths(self.sim_root, preset, now_tag(micro=False))
        if not cfg.exists():
            self.ui.show_messagebox("error", "Missing config", f"Config not found:\n{cfg}")
            return

        rel_model = rel_to(self.sim_root, out_model)
        self.ui.var_profile_model.set(rel_model)
        self.ui.var_profile_model_lock.set(True)
        self._apply_profile_model_lock()
        
        def run_simple_cmd_callback(cmd_args: list[str], extra_env: dict[str, str] | None = None) -> None:
            self._run_simple_cmd(cmd_args, extra_env)

        self.logic.build_profile_model(preset, str(cfg), str(out_ds), str(out_model), self._append_log, run_simple_cmd_callback)
        self.ui.var_profile_model.set(rel_model)

    def _apply_dataset_settings(self, ds: Path) -> None:
        if not ds: return
        if (config_path := ds / CONFIG_YAML).exists(): self.ui.var_config.set(str(config_path))
        if profile_id := self.logic.get_dataset_profile_id(ds): self._set_profile_value(profile_id)
        
        self.ui.var_name.set(ds.name)
        self.ui.var_out.set(str(ds.resolve()))
        
        model_path = self.sim_root / "outputs" / "models" / f"{ds.name}.pt"
        try: self.ui.var_model.set(str(model_path.resolve()))
        except Exception: self.ui.var_model.set(str(model_path))
        if not bool(self.ui.var_profiles_multi.get()):
            self.ui.var_dataset_mode.set("extend")
        self._update_model_dropdown_for_dataset()

    def _set_profile_value(self, profile_id: str) -> None:
        self.ui.var_profile.set(profile_id)
        self._last_profile_id = profile_id

    def _refresh_datasets(self) -> None:
        self._dataset_by_label = catalog_labels(self.sim_root, self._dataset_catalog, save_widget=self.frame)
        self._dataset_labels = list(self._dataset_by_label)
        self._dataset_dirs = list(self._dataset_by_label.values())

        self.ui.set_dataset_combo_values(self._dataset_labels)
        self._sync_multi_selection_after_dataset_refresh()

        if (cur := self.ui.var_dataset.get().strip()) and cur in self._dataset_by_label: return
        if self.state.dataset_dir:
            runs, versions = sim_data_roots(self.sim_root, create=False)
            want = self._dataset_catalog.display_label(self.state.dataset_dir, runs_root=runs, versions_root=versions)
            for label, p in self._dataset_by_label.items():
                if p == self.state.dataset_dir or label == want: self.ui.var_dataset.set(label); return
        if self._dataset_labels: self.ui.var_dataset.set(self._dataset_labels[0]); self._on_dataset_selected()

    def _dataset_multi_tooltip_text(self) -> str:
        labels = self._selected_dataset_labels()
        if not labels: return "(no datasets selected)"
        if len(labels) <= 12: return "\n".join(labels)
        return "\n".join(labels[:12]) + f"\n... (+{len(labels) - 12} more)"

    def _on_dataset_multi_toggle(self) -> None:
        enabled = bool(self.ui.var_dataset_multi.get())

        if enabled:
            paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
            if not paths and (ds := self._selected_dataset_dir_single()) is not None:
                self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json([ds]))

        self.ui.toggle_dataset_multi_widgets(enabled)
        self._sync_multi_summary()
        
        if enabled:
            if ds0 := self._selected_dataset_dir():
                if label0 := self._dataset_label_for_path(ds0): self.ui.var_dataset.set(label0)
        self._on_dataset_selected()

    def _open_dataset_multi_picker(self) -> None:
        if not self._dataset_labels:
            self.ui.show_messagebox("info", "Datasets", "No datasets found. Click Refresh after generating datasets.")
            return

        selected_paths = {str(p.resolve()) for p in self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())}
        if not selected_paths and (ds := self._selected_dataset_dir_single()) is not None:
            selected_paths.add(str(ds.resolve()))
        preselected = [
            label for label in self._dataset_labels
            if (p := self._dataset_by_label.get(label)) is not None and str(p.resolve()) in selected_paths
        ]
        picked = dialogs.pick_from_list(
            self.frame, "Select Datasets", self._dataset_labels, multi=True, selected=preselected,
            prompt="Select one or more datasets. One model is trained on all of them.", size=(640, 480),
        )
        if picked is None:
            return
        paths: list[Path] = [self._dataset_by_label[label] for label in picked if label in self._dataset_by_label]
        if not paths and (ds := self._selected_dataset_dir_single()) is not None:
            paths = [ds]

        self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json(paths))
        self._sync_multi_summary()
        if paths:
            if label0 := self._dataset_label_for_path(paths[0]):
                self.ui.var_dataset.set(label0)
        self._on_dataset_selected()

    def _decode_dataset_paths_json(self, s: str) -> list[Path]:
        return decode_dataset_paths_json(s, self.sim_root)

    def _encode_dataset_paths_json(self, paths: list[Path]) -> str:
        return encode_dataset_paths_json(paths, self.sim_root)

    def _dataset_label_for_path(self, p: Path) -> str | None:
        rp = p.resolve()
        for label, pp in self._dataset_by_label.items():
            if pp.resolve() == rp: return label
        for label, pp in self._dataset_by_label.items():
            if pp.name == p.name: return label
        return None

    def _selected_dataset_labels(self) -> list[str]:
        return [label for ds in self._selected_dataset_dirs() if (label := self._dataset_label_for_path(ds))]

    def _sync_multi_summary(self) -> None:
        self.ui.var_dataset_multi_summary.set(summarize_labels(self._selected_dataset_labels()))

    def _sync_multi_selection_after_dataset_refresh(self) -> None:
        if not hasattr(self.ui, "var_dataset_multi_paths"): return
        paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
        if not paths: self._sync_multi_summary(); return

        valid = {str(p.resolve()) for p in self._dataset_dirs}
        filtered = [p for p in paths if str(p.resolve()) in valid]

        if filtered != paths: self.ui.var_dataset_multi_paths.set(self._encode_dataset_paths_json(filtered))
        if bool(self.ui.var_dataset_multi.get()) and filtered:
            if label0 := self._dataset_label_for_path(filtered[0]): self.ui.var_dataset.set(label0)
        self._sync_multi_summary()

    def _display_for_dataset(self, p: Path, *, runs: Path, versions: Path) -> str:
        return dataset_display_name(p, runs_root=runs, versions_root=versions)

    @staticmethod
    def _profile_from_meta(meta: dict[str, Any] | None) -> tuple[str | None, str | None]:
        comp = (meta or {}).get("component_profile")
        return (comp.get("profile_id"), comp.get("profile_hash")) if isinstance(comp, dict) else (None, None)

    def _refresh_models(self) -> None:
        paths = self.logic.get_all_model_paths()
        self._model_paths = paths
        self._model_profile_cache.clear()
        self._model_meta_gen += 1
        gen, uncached = self._model_meta_gen, []
        # One resolve per path; cached_checkpoint_meta is stat-first (path, mtime_ns, size) and never
        # opens a checkpoint it has already read, so this loop is one stat per file.
        self._model_resolved = resolved = {p: p.resolve() for p in paths}
        for p in self._model_paths:
            key = str(resolved[p])
            if p.is_dir(): self._model_profile_cache[key] = ("multi", None); continue
            if (meta := self.logic.cached_checkpoint_meta(p)) is None: uncached.append(p); continue
            self._model_profile_cache[key] = self._profile_from_meta(meta)

        if uncached:
            # torch.load of every checkpoint is slow: do it off the Tk thread and re-filter when done.
            def worker() -> None:
                res = {str(p.resolve()): self._profile_from_meta(self.logic.read_checkpoint_meta(p)) for p in uncached}
                def apply() -> None:
                    if gen != self._model_meta_gen: return
                    self._model_profile_cache.update(res)
                    self._update_model_dropdown_for_dataset(); self._schedule_profile_compat_check()
                with contextlib.suppress(Exception): self.post(apply)
            threading.Thread(target=worker, daemon=True).start()

        root_resolved = self.sim_root.resolve()
        values = [str(resolved[p].relative_to(root_resolved)) if p.is_relative_to(self.sim_root) else str(p) for p in paths]
        self._all_model_combo_values = values
        self._update_model_dropdown_for_dataset()

    def _update_model_dropdown_for_dataset(self) -> None:
        ds = self.state.dataset_dir
        values = list(self._all_model_combo_values)
        if ds:
            profile_id, profile_hash = self._dataset_profile(ds)
            if profile_id:
                exact, id_only, multi, legacy = [], [], [], []
                resolved = self._model_resolved
                root_resolved = self.sim_root.resolve()
                for p in self._model_paths:
                    rp = resolved.get(p) or p.resolve()
                    rel_path = str(rp)
                    pid, phash = self._model_profile_cache.get(rel_path, (None, None))
                    if pid == "multi" or (isinstance(pid, str) and pid.lower().startswith("multi")):
                        try: multi.append(str(rp.relative_to(root_resolved)))
                        except Exception: multi.append(str(p))
                        continue
                    if pid is None:
                        try: legacy.append(str(rp.relative_to(root_resolved)))
                        except Exception: legacy.append(str(p))
                        continue
                    if pid != profile_id: continue
                    if profile_hash and phash and phash == profile_hash: exact.append(str(p.relative_to(self.sim_root)))
                    else: id_only.append(str(p.relative_to(self.sim_root)))
                values = exact + id_only + multi + legacy
        
        self.ui.set_model_combo_values(values)
        current = self.ui.var_model.get().strip()
        if current:
            # The field may hold an absolute path of a listed model, or the name of a model that is
            # about to be trained (Apply Name / Create dataset): neither must be replaced.
            cur_path = resolve_under(self.sim_root, current)
            try: cur_rel = str(cur_path.resolve().relative_to(self.sim_root.resolve()))
            except Exception: cur_rel = current
            if current in values or cur_rel in values or not cur_path.exists(): return
        self.ui.var_model.set(values[0] if values else "")

    def _dataset_profile(self, ds: Path) -> tuple[str | None, str | None]:
        """(profile_id, profile_hash) of ``ds``'s manifest; parsed again only when the manifest changes."""
        return self._manifest_cache.cached(("profile", str(ds)), (ds / DATASET_MANIFEST,),
                                           lambda: (self.logic.get_dataset_profile_id(ds), self.logic.get_dataset_profile_hash(ds)))

    def _add_new_model(self) -> None:
        dss = self._selected_dataset_dirs()
        default_name = dss[0].name if dss else ""
        is_multi = bool(self.ui.var_dataset_multi.get()) and len(dss) > 1
        if is_multi and default_name and "multitrained" not in default_name.lower(): default_name = f"{default_name}_MultiTrained"
        
        name = self.ui.ask_string("New Model", "Model name (without .pt):", initialvalue=default_name)
        if not (name := (name or "").strip()): return
        if is_multi and "multitrained" not in name.lower(): name = f"{name}_MultiTrained"
        if "/" in name or "\\" in name:
            self.ui.show_messagebox("error", "Error", "Name must not contain path separators.")
            return

        suffix = ".bundle" if self.ui.var_model_bundle.get() else ".pt"
        model_path = self.sim_root / "outputs" / "models" / f"{name}{suffix}"
        rel = str(model_path.resolve().relative_to(self.sim_root.resolve()))

        if rel not in self._all_model_combo_values: self._all_model_combo_values.insert(0, rel)
        if rel not in (current_values := list(self.ui.model_combo["values"])):
            current_values.insert(0, rel)
            self.ui.set_model_combo_values(current_values)
        self.ui.var_model.set(rel)
        self._append_log(f"[model] New model target: {rel}\n")

    def _refresh_profiles(self) -> None:
        self.ui.set_profile_combo_values(self.logic.get_profile_paths(self.ui.var_render_backend.get().strip() or "opencv_2d"))

        cur = self.ui.var_profile.get().strip()
        if not cur or cur not in self.ui.profile_combo["values"]:
            if DEFAULT_COMPONENT_PROFILE in self.ui.profile_combo["values"]: self.ui.var_profile.set(DEFAULT_COMPONENT_PROFILE)
            elif self.ui.profile_combo["values"]: self.ui.var_profile.set(self.ui.profile_combo["values"][0])

        if bool(self.ui.var_profiles_multi.get()): self._set_profiles_multi(self._profiles_multi_list(available=self._profiles_multi_available_values()))
        self._on_profile_selected()

    def _profiles_multi_available_values(self) -> list[str]:
        mode = self.ui.var_profiles_multi_mode.get().strip() or "separate"
        if mode == "mixed":
            all_profiles = set(self.logic.get_profile_paths("opencv_2d")) | set(self.logic.get_profile_paths("blender_3d"))
            return sorted(all_profiles)
        return list(self.ui.profile_combo["values"] or [])

    def _profiles_multi_list(self, *, available: list[str] | None = None) -> list[str]:
        available_set = set(available or list(self.ui.profile_combo["values"] or []))
        return parse_profile_ids(self.ui.var_profiles_multi_json.get(), available_set)

    def _set_profiles_multi(self, pids: list[str]) -> None:
        pids = [str(x).strip() for x in (pids or []) if str(x).strip()]
        with contextlib.suppress(Exception): self.ui.var_profiles_multi_json.set(json.dumps(pids, ensure_ascii=True))
        self.ui.var_profiles_multi_summary.set(summarize_profile_ids(pids))

    def _on_profiles_multi_toggle(self) -> None:
        enabled = bool(self.ui.var_profiles_multi.get())
        self.ui.set_profiles_multi_state(enabled)
        if enabled:
            self.ui.var_dataset_mode.set("new")
            cur = self._profiles_multi_list(available=self._profiles_multi_available_values())
            if not cur and (pid := self.ui.var_profile.get().strip()): cur = [pid]
            self._set_profiles_multi(cur)
        with contextlib.suppress(Exception): self._refresh_precise_classes()

    def _pick_profiles_multi(self) -> None:
        values = self._profiles_multi_available_values()
        if not values:
            self.ui.show_messagebox("info", "Profiles", "No profiles available.")
            return
        cur = set(self._profiles_multi_list(available=values))
        sel = dialogs.pick_from_list(
            self.frame, "Select Profiles", values, multi=True, selected=[v for v in values if v in cur],
            prompt="Select one or more component profiles to generate.",
        )
        if sel is None:
            return
        self._set_profiles_multi(sel)
        if sel:
            self.ui.var_profile.set(sel[0])
        with contextlib.suppress(Exception):
            self._refresh_precise_classes()

    def _on_render_backend_changed(self, _event: object | None = None) -> None:
        try:
            self._refresh_profiles()
            if pid := self.ui.var_profile.get().strip(): self._autoselect_config_for_profile(pid, prefer_quiet=True)
        except Exception: pass

    def _on_profile_selected(self, _event: object | None = None) -> None:
        if not (profile_id := self.ui.var_profile.get().strip()): return

        with contextlib.suppress(Exception): self._autoselect_config_for_profile(profile_id, prefer_quiet=True)

        # Do not open profile-switch/create popups automatically.
        # Keep profile selection passive and only update state.
        self._last_profile_id = profile_id

    def _select_dataset(self, ds: Path) -> None:
        self._refresh_datasets()
        runs, versions = self.sim_root / "outputs" / "sim_data" / "runs", self.sim_root / "outputs" / "sim_data" / "versions"
        label = self._display_for_dataset(ds, runs=runs, versions=versions)
        for key, value in self._dataset_by_label.items():
            if value == ds or key == label: self.ui.var_dataset.set(key); self._on_dataset_selected(); return
        self.state.dataset_dir = ds
        self.ui.var_dataset.set(label)
        self._on_dataset_selected()

    def _autoselect_config_for_profile(self, profile_id: str, *, prefer_quiet: bool) -> None:
        want_backend = (self.ui.var_render_backend.get().strip() or None)
        cfg_info = self.logic.config_for_profile(profile_id, want_backend=want_backend)
        if not cfg_info and want_backend: cfg_info = self.logic.config_for_profile(profile_id, want_backend=None)
        if not cfg_info: return

        cfg_path: Path = cfg_info["path"]
        try: rel = str(cfg_path.resolve().relative_to(self.sim_root.resolve()))
        except Exception: rel = str(cfg_path)

        if (cur := self.ui.var_config.get().strip()) == rel: return

        if prefer_quiet and cur and want_backend:
            try:
                cur_path = (self.sim_root / cur) if not Path(cur).is_absolute() else Path(cur)
                if cur_path.exists():
                    data = load_yaml(cur_path)
                    cur_backend = ((data.get("render") or {}).get("backend") or "").strip()
                    cur_profile = ((data.get("run") or {}).get("component_profile") or "")
                    if cur_backend == want_backend and str(cur_profile).strip() == str(profile_id).strip(): return
            except Exception: return

        self.ui.var_config.set(rel)
        with contextlib.suppress(Exception):
            self._refresh_precise_classes()

    def _show_profile_info(self) -> None:
        profile_id = self.ui.var_profile.get()
        if not profile_id: self.ui.show_messagebox("info", "Profile Info", "No profile selected"); return

        profiles_dir = self.sim_root / "configs" / "profiles"
        profile_path = profiles_dir / f"{profile_id}.yaml"
        if not profile_path.exists(): self.ui.show_messagebox("error", "Error", f"Profile file not found:\n{profile_path}"); return

        try:
            profile_data = load_yaml(profile_path)
            info_text = f"Profile ID: {profile_id}\nPath: {profile_path}\n\n" + "="*60 + "\n\n" + yaml.dump(profile_data, default_flow_style=False, sort_keys=False)
            dialogs.show_text(self.frame, f"Profile Info: {profile_id}", info_text, size=(640, 520))

        except Exception as e: self.ui.show_messagebox("error", "Error", f"Failed to load profile:\n{str(e)}")

    def _schedule_profile_compat_check(self, delay_ms: int = 300) -> None:
        # Debounce: var_model is written repeatedly while typing/refreshing; check only after edits settle.
        if self._compat_after_id is not None:
            with contextlib.suppress(Exception): self.frame.after_cancel(self._compat_after_id)
        self._compat_after_id = self.frame.after(delay_ms, self._run_scheduled_profile_compat_check)

    def _run_scheduled_profile_compat_check(self) -> None:
        self._compat_after_id = None
        self._check_profile_compatibility()

    def _check_profile_compatibility(self) -> None:
        try:
            if not (ds := self._selected_dataset_dir()): self.ui.var_profile_compat.set(""); return
            manifest_path = ds / DATASET_MANIFEST
            if not manifest_path.exists(): self.ui.var_profile_compat.set(""); return

            with open(manifest_path) as f: manifest = json.load(f)
            mver = int(manifest.get('manifest_version', 1) or 1)
            ds_profile_id = (manifest.get('component_profile', {}).get('profile_id') if mver == 1 else "multi")
            ds_profile_hash = (manifest.get('component_profile', {}).get('profile_hash') if mver == 1 else "multi")

            if not ds_profile_id: self.ui.var_profile_compat.set(""); return

            if not (model_path_str := self.ui.var_model.get().strip()): self.ui.var_profile_compat.set(""); return
            model_path = self.logic._resolve_model_path(model_path_str)
            if not model_path.exists(): self.ui.var_profile_compat.set(""); return
            
            if model_path.is_dir():
                if not (cand := bundle_checkpoint_path(model_path, ds_profile_id, kind="best")).exists():
                    self.ui.var_profile_compat.set(f"⚠ Multi-model: no weights for {ds_profile_id}"); self.ui.set_compat_color("orange"); return
                model_path = cand

            if (meta := self.logic.cached_checkpoint_meta(model_path)) is None:
                # Uncached: load metadata in the background, then re-run this check on the Tk thread.
                self.ui.var_profile_compat.set("… checking model profile"); self.ui.set_compat_color("gray")
                if (key := str(model_path.resolve())) not in self._compat_loading:
                    self._compat_loading.add(key)
                    def worker() -> None:
                        try: self.logic.read_checkpoint_meta(model_path)
                        finally:
                            def done() -> None: self._compat_loading.discard(key); self._schedule_profile_compat_check(0)
                            with contextlib.suppress(Exception): self.post(done)
                    threading.Thread(target=worker, daemon=True).start()
                return
            if meta.get("error"): self.ui.var_profile_compat.set(""); return
            text, color = profile_compat_status(ds_profile_id, ds_profile_hash, meta.get("component_profile"))
            self.ui.var_profile_compat.set(text); self.ui.set_compat_color(color)

        except Exception as e: _LOG.warning("Profile compatibility check failed: %s", e); self.ui.var_profile_compat.set("")

    def _show_dataset_profile_info(self) -> None:
        if not (ds := self._selected_dataset_dir()): self.ui.show_messagebox("info", "Dataset Profile", "No dataset selected"); return
        manifest_path = ds / DATASET_MANIFEST
        if not manifest_path.exists(): self.ui.show_messagebox("error", "No Manifest", f"Dataset has no manifest (legacy dataset).\n\nUse the backfill tool to add a manifest:\n  .venv/bin/python tools/backfill_manifest.py --data {ds}"); return

        try:
            with open(manifest_path) as f: manifest = json.load(f)
            profile_id, profile_hash, profile_path_str = manifest.get('component_profile', {}).get('profile_id', 'unknown'), manifest.get('component_profile', {}).get('profile_hash', ''), manifest.get('component_profile', {}).get('profile_path', '')
            profiles_dir = self.sim_root / "configs" / "profiles"
            profile_path = profiles_dir / f"{profile_id}.yaml"

            info_text = f"Dataset: {ds.name}\nProfile ID: {profile_id}\nProfile Hash: {profile_hash}\nProfile Path: {profile_path_str}\n\n" + "="*60 + "\nMANIFEST METADATA\n" + "="*60 + "\n\n" + json.dumps(manifest, indent=2)
            if profile_path.exists(): info_text += "\n\n" + "="*60 + "\nPROFILE DETAILS\n" + "="*60 + "\n\n" + yaml.dump(load_yaml(profile_path), default_flow_style=False, sort_keys=False)
            else: info_text += f"\n\n⚠ Profile file not found at: {profile_path}"
            dialogs.show_text(self.frame, f"Dataset Profile: {ds.name}", info_text, size=(680, 560))

        except Exception as e: self.ui.show_messagebox("error", "Error", f"Failed to load dataset profile info:\n{str(e)}")

    def _apply_name_to_out_and_model(self) -> None:
        base = self.logic.slugify_name(self.ui.var_name.get())
        name = f"{base}_{now_tag(micro=False)}" if self.ui.var_name_ts.get() else base
        out_rel, model_rel = f"outputs/sim_data/runs/{name}", f"outputs/models/{name}.{'bundle' if self.ui.var_model_bundle.get() else 'pt'}"
        self.ui.var_out.set(out_rel); self.ui.var_model.set(model_rel)
        self._refresh_models()
        self._append_log(f"[naming] out={out_rel} model={model_rel}\n")

    def _selected_dataset_dir(self) -> Path | None:
        dss = self._selected_dataset_dirs(); return dss[0] if dss else None

    def _selected_dataset_dir_single(self) -> Path | None:
        label = self.ui.var_dataset.get().strip(); return self._dataset_by_label.get(label)

    def _selected_dataset_dirs(self) -> list[Path]:
        if self.ui.var_dataset_multi.get():
            paths = self._decode_dataset_paths_json(self.ui.var_dataset_multi_paths.get())
            return [p for p in paths if p.exists() and p.is_dir() and not self._dataset_catalog.is_archived(p)]
        return [ds] if (ds := self._selected_dataset_dir_single()) is not None else []

    def refresh(self) -> None:
        if not self.initialized:
            return
        self._refresh_datasets()

    def _on_dataset_selected(self, _evt: object | None = None) -> None:
        ds = self._selected_dataset_dir()
        if not ds: self.ui.var_dataset_profile.set("Profile: -"); self.ui.var_ds_samples.set("Samples: -"); return
        
        self.state.dataset_dir, self._recent_imgs = ds, []
        self._label_dict, self._image_to_sample, self._meta_by_sample, self._meta_by_image_rel = {}, {}, {}, {}
        self._start_dataset_size_calc(ds)
        # meta/labels (DatasetIndex, stat-cached) and the image listing are read in a worker; the
        # thumbnails follow when they are in (S6). The rest below only reads the small manifest.
        self._start_dataset_index_load(ds)

        try:
            if (manifest_path := ds / DATASET_MANIFEST).exists():
                with open(manifest_path) as f: manifest = json.load(f)
                profile_id = manifest.get('component_profile', {}).get('profile_id', 'unknown')
                profile_hash = manifest.get('component_profile', {}).get('profile_hash', '')
                stats = manifest.get("dataset_stats", {}); total_samples = stats.get("total_samples")
                splits = stats.get("splits", {}); train, val, test = splits.get("train"), splits.get("val"), splits.get("test")
                self.ui.var_dataset_profile.set(f"Profile: {profile_id} ({short_profile_hash(profile_hash)}...)")
                self._set_dataset_samples_from_manifest(ds, total_samples, train, val, test)
            else:
                self.ui.var_dataset_profile.set("Profile: ⚠ No manifest (legacy dataset)")
                self._set_dataset_samples_info("Samples: legacy dataset", "black")
        except Exception as e:
            self.ui.var_dataset_profile.set("Profile: ⚠ Error loading manifest"); self._set_dataset_samples_info("Samples: error", "")
            _LOG.warning("Failed to load dataset manifest: %s", e)

        self._schedule_profile_compat_check(); self._apply_dataset_settings(ds)

        with contextlib.suppress(Exception): self.parent.event_generate("<<DatasetChanged>>", when="tail")
        
        with contextlib.suppress(Exception): self._refresh_models()

    def _start_dataset_index_load(self, ds: Path) -> None:
        self._ds_load_gen += 1
        gen = self._ds_load_gen

        def on_done(res: dict[str, Any]) -> None:
            if gen != self._ds_load_gen or self.state.dataset_dir != ds: return  # another dataset was picked
            if res.get("error"): _LOG.warning("Failed to load metadata/labels: %s", res["error"])
            self._image_to_sample, self._meta_by_sample = res["image_to_sample"], res["meta_by_sample"]
            self._meta_by_image_rel, self._label_dict = res["meta_by_image_rel"], res["label_dict"]
            if res["has_images"]:
                # Images pushed by a run in the meantime stay newest.
                self._recent_imgs = (res["recent_imgs"] + [p for p in self._recent_imgs if p not in res["recent_imgs"]])[-12:]
                self._refresh_thumbnails()

        self.quiet_tasks.run(lambda _t: load_dataset_view(ds), name="Dataset index", on_done=on_done)

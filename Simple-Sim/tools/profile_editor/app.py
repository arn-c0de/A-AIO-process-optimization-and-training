from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.profile_editor.blender_live_preview import BlenderLivePreviewWorker, RenderRequest
    from tools.profile_editor.form_renderer import FormRenderer
    from tools.profile_editor.hq_preview_panel import HQPreviewPanel
    from tools.profile_editor.preview_3d import Preview3D
    from tools.profile_editor.state_store import EditorStateStore
    from tools.profile_editor.system_monitor import format_system_stats, read_system_stats
    from tools.profile_editor.sync_controller import SyncController
    from tools.profile_editor.yaml_editor import YamlEditorPanel
    from tools.profile_editor.yaml_io import save_yaml
else:
    from .blender_live_preview import BlenderLivePreviewWorker, RenderRequest
    from .form_renderer import FormRenderer
    from .hq_preview_panel import HQPreviewPanel
    from .preview_3d import Preview3D
    from .state_store import EditorStateStore
    from .system_monitor import format_system_stats, read_system_stats
    from .sync_controller import SyncController
    from .yaml_editor import YamlEditorPanel
    from .yaml_io import save_yaml


class ProfileEditorApp:
    def __init__(self, root: tk.Tk, sim_root: Path):
        self.root = root
        self.sim_root = Path(sim_root)

        self.store = EditorStateStore()
        self.controller = SyncController(self.sim_root, self.store)
        self.store.add_listener(self._on_state_event)

        self._hq_after_id: str | None = None
        self._worker = BlenderLivePreviewWorker(self.sim_root, self._on_hq_status_from_worker)
        self._closed = False
        self._settings_after_id: str | None = None
        self._session_settings = self._load_session_settings()
        self.dark_mode_var = tk.BooleanVar(value=bool(self._session_settings.get("dark_mode", False)))

        self._build_ui()
        self._apply_session_settings_to_ui()
        self._load_initial_choices()
        self._schedule_registry_poll()
        self._schedule_system_stats()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.title("Simple-Sim Profile Editor")
        self.root.geometry("1750x980")

        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="Profile:").pack(side="left")
        self.profile_var = tk.StringVar(value="")
        self.profile_combo = ttk.Combobox(top, textvariable=self.profile_var, width=42, state="readonly")
        self.profile_combo.pack(side="left", padx=(6, 12))
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_profile_selected())

        ttk.Label(top, text="Run:").pack(side="left")
        self.run_var = tk.StringVar(value="")
        self.run_combo = ttk.Combobox(top, textvariable=self.run_var, width=34, state="readonly")
        self.run_combo.pack(side="left", padx=(6, 12))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_run_selected())

        ttk.Button(top, text="Save Profile", command=self._save_profile).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Copy Profile", command=self._copy_profile).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Save Run", command=self._save_run).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Render HQ Now", command=self._trigger_hq_render_now).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Refresh Files", command=self._refresh_choices).pack(side="left")
        ttk.Checkbutton(top, text="Night Mode", variable=self.dark_mode_var, command=self._on_theme_toggled).pack(
            side="left", padx=(8, 0)
        )
        ttk.Label(top, text="").pack(side="left", expand=True)
        self.system_stats_var = tk.StringVar(value="CPU n/a | GPU n/a | RAM n/a")
        self.system_stats_label = ttk.Label(top, textvariable=self.system_stats_var)
        self.system_stats_label.pack(side="right")

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x", padx=10)

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(paned)
        middle = ttk.Frame(paned)
        right = ttk.Frame(paned)

        paned.add(left, weight=3)
        paned.add(middle, weight=3)
        paned.add(right, weight=4)

        left_tabs = ttk.Notebook(left)
        left_tabs.pack(fill="both", expand=True)

        self.preview = Preview3D(
            left_tabs,
            on_settings_changed=self._schedule_settings_save,
            on_geometry_changed=self._on_preview_geometry_changed,
        )
        self.hq_preview = HQPreviewPanel(left_tabs)

        left_tabs.add(self.preview, text="Fast 3D")
        left_tabs.add(self.hq_preview, text="Blender HQ")

        middle_tabs = ttk.Notebook(middle)
        middle_tabs.pack(fill="both", expand=True)

        self.profile_form = FormRenderer(middle_tabs, "profile", self._on_form_commit)
        self.run_form = FormRenderer(middle_tabs, "run", self._on_form_commit)

        middle_tabs.add(self.profile_form, text="Profile Fields")
        middle_tabs.add(self.run_form, text="Run Fields")

        self.yaml_editor = YamlEditorPanel(right, self._on_yaml_apply)
        self.yaml_editor.pack(fill="both", expand=True)
        self._apply_theme(bool(self.dark_mode_var.get()))

    def _load_initial_choices(self) -> None:
        self._refresh_choices()
        profiles = list(self.profile_combo.cget("values"))
        if not profiles:
            return

        preferred_profile = str(self._session_settings.get("selected_profile_id") or "").strip()
        if preferred_profile and preferred_profile in profiles:
            self.profile_var.set(preferred_profile)
        else:
            self.profile_var.set(profiles[0])
        self._on_profile_selected()

        run_values = list(self.run_combo.cget("values"))
        preferred_run = str(self._session_settings.get("selected_run_id") or "").strip()
        if preferred_run and preferred_run in run_values:
            self.run_var.set(preferred_run)
            self._on_run_selected()

    def _refresh_choices(self) -> None:
        choices = self.controller.refresh_registry()
        self.profile_combo["values"] = choices.profile_ids
        self.run_combo["values"] = choices.run_ids

        cur_profile = self.profile_var.get()
        if cur_profile not in choices.profile_ids:
            self.profile_var.set(choices.profile_ids[0] if choices.profile_ids else "")

        cur_run = self.run_var.get()
        if cur_run not in choices.run_ids:
            self.run_var.set(choices.run_ids[0] if choices.run_ids else "")

        self._update_run_choices_for_profile()
        self.status_var.set(
            f"Discovered {len(choices.profile_ids)} profiles and {len(choices.run_ids)} run configs"
        )
        self._schedule_settings_save()

    def _update_run_choices_for_profile(self) -> None:
        matching = self.controller.matching_runs_for_current_profile()
        if matching:
            self.run_combo["values"] = matching
            if self.run_var.get() not in matching:
                self.run_var.set(matching[0])
        else:
            snapshot = self.controller.snapshot
            all_runs = sorted(snapshot.run_files.keys()) if snapshot else []
            self.run_combo["values"] = all_runs
            if self.run_var.get() not in all_runs:
                self.run_var.set(all_runs[0] if all_runs else "")

    def _on_profile_selected(self) -> None:
        profile_id = self.profile_var.get().strip()
        if not profile_id:
            return
        try:
            self.controller.load_profile(profile_id)
        except Exception as exc:
            self.status_var.set(f"Failed to load profile '{profile_id}': {exc}")
            return

        self._update_run_choices_for_profile()
        current_run = self.run_var.get().strip()
        if current_run:
            self._on_run_selected()
        self._schedule_settings_save()

    def _on_run_selected(self) -> None:
        run_id = self.run_var.get().strip()
        if not run_id:
            return
        try:
            self.controller.load_run(run_id)
        except Exception as exc:
            self.status_var.set(f"Failed to load run '{run_id}': {exc}")
        self._schedule_settings_save()

    def _on_form_commit(self, target: str, path: tuple[str, ...], raw: str) -> None:
        try:
            value = self.store.parse_form_input(raw)
            self.store.update_leaf(target, path, value, source="form")
            self.status_var.set(f"Updated {target}.{'.'.join(path)}")
        except Exception as exc:
            self.status_var.set(f"Invalid input for {target}.{'.'.join(path)}: {exc}")

    def _on_yaml_apply(self, target: str, text: str) -> None:
        self.store.set_text(target, text, source="yaml")
        doc = self.store.profile if target == "profile" else self.store.run
        if doc.text_error:
            self.yaml_editor.set_error(target, doc.text_error)
            self.status_var.set(f"YAML error in {target}: {doc.text_error}")
        else:
            self.yaml_editor.clear_error(target)
            self.status_var.set(f"Applied {target} YAML")

    def _save_profile(self) -> None:
        try:
            self.controller.save_profile()
            p = self.store.profile.path
            self.status_var.set(f"Saved profile: {p}")
        except Exception as exc:
            self.status_var.set(f"Save profile failed: {exc}")

    def _save_run(self) -> None:
        try:
            self.controller.save_run()
            p = self.store.run.path
            self.status_var.set(f"Saved run: {p}")
        except Exception as exc:
            self.status_var.set(f"Save run failed: {exc}")

    def _copy_profile(self) -> None:
        src = self.store.profile.data or {}
        src_pid = self.store.current_profile_id()
        if not src or not src_pid:
            self.status_var.set("Copy profile failed: no active profile loaded")
            return

        suffix = simpledialog.askstring(
            "Copy Profile",
            "Suffix/Name für neues Profil (wird an profile_id angehängt):",
            initialvalue="_custom",
            parent=self.root,
        )
        if suffix is None:
            return
        suffix = suffix.strip()
        if not suffix:
            self.status_var.set("Copy profile aborted: empty suffix")
            return

        safe_suffix = self._sanitize_profile_token(suffix)
        if not safe_suffix:
            self.status_var.set("Copy profile failed: invalid suffix")
            return

        new_profile_id = f"{src_pid}{safe_suffix}"
        dst_path = self.sim_root / "configs" / "profiles" / f"{new_profile_id}.yaml"
        if dst_path.exists():
            ok = messagebox.askyesno(
                "Overwrite existing profile?",
                f"Profile exists already:\n{dst_path}\n\nOverwrite?",
                parent=self.root,
            )
            if not ok:
                self.status_var.set("Copy profile aborted")
                return

        data = copy.deepcopy(src)
        profile_obj = data.setdefault("profile", {})
        if isinstance(profile_obj, dict):
            profile_obj["profile_id"] = new_profile_id
        else:
            data["profile"] = {"profile_id": new_profile_id}

        try:
            save_yaml(dst_path, data)
            self._refresh_choices()
            self.profile_var.set(new_profile_id)
            self._on_profile_selected()
            self.status_var.set(f"Copied profile to {dst_path}")
            self._schedule_settings_save()
        except Exception as exc:
            self.status_var.set(f"Copy profile failed: {exc}")

    @staticmethod
    def _sanitize_profile_token(raw: str) -> str:
        token = raw.strip().replace(" ", "_")
        token = re.sub(r"[^a-zA-Z0-9_@.+-]+", "_", token)
        token = re.sub(r"_+", "_", token)
        if token and not token.startswith(("_", "-", "+")):
            token = "_" + token
        return token[:80]

    def _on_state_event(self, event: str, source: str) -> None:
        profile_data = self.store.profile.data or {}
        run_data = self.store.run.data or {}

        if source != "yaml":
            self.yaml_editor.set_text("profile", self.store.get_text("profile"))
            self.yaml_editor.set_text("run", self.store.get_text("run"))

        if source in {"yaml", "system"}:
            if event.startswith("profile"):
                self.profile_form.set_data(profile_data)
            if event.startswith("run") or event.startswith("profile"):
                self.run_form.set_data(run_data)

        if self.store.profile.text_error:
            self.yaml_editor.set_error("profile", self.store.profile.text_error)
        else:
            self.yaml_editor.clear_error("profile")

        if self.store.run.text_error:
            self.yaml_editor.set_error("run", self.store.run.text_error)
        else:
            self.yaml_editor.clear_error("run")

        self.preview.set_profile_data(profile_data)
        self._update_title()
        self._schedule_hq_render()

    def _on_preview_geometry_changed(self, updates: dict[tuple[str, ...], object]) -> None:
        for path, value in updates.items():
            self.store.update_leaf("profile", path, value, source="preview3d")

    def _on_theme_toggled(self) -> None:
        self._apply_theme(bool(self.dark_mode_var.get()))
        self._schedule_settings_save()

    def _apply_theme(self, dark: bool) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        if dark:
            bg = "#171b20"
            panel = "#1f2329"
            fg = "#f5f7fa"
            field_bg = "#11161c"
            field_fg = "#f5f7fa"
            tab_active = "#2a3138"
            tab_selected = "#303841"
            self.root.configure(bg=bg)
            self.system_stats_label.configure(foreground="#d9e1ea")
            self.status_label.configure(foreground="#d9e1ea")
        else:
            bg = "#f0f0f0"
            panel = "#ffffff"
            fg = "#000000"
            field_bg = "#ffffff"
            field_fg = "#000000"
            tab_active = "#e7e7e7"
            tab_selected = "#ffffff"
            self.root.configure(bg=bg)
            self.system_stats_label.configure(foreground="#4d5656")
            self.status_label.configure(foreground="#34495e")

        style.configure(".", background=bg, foreground=fg)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=panel, foreground=fg)
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TMenubutton", background=panel, foreground=fg)
        style.configure("TNotebook", background=bg)
        style.configure("TNotebook.Tab", background=panel, foreground=fg)
        style.map(
            "TNotebook.Tab",
            background=[("selected", tab_selected), ("active", tab_active), ("!selected", panel)],
            foreground=[("selected", fg), ("active", fg), ("!selected", fg)],
        )
        style.configure("TPanedwindow", background=bg)
        style.configure("TScale", background=bg)
        style.configure("TEntry", fieldbackground=field_bg, foreground=field_fg)
        style.configure("Dark.TEntry", fieldbackground="#11161c", foreground="#f5f7fa")
        style.configure("Light.TEntry", fieldbackground="#ffffff", foreground="#000000")
        style.configure(
            "TCombobox",
            fieldbackground=field_bg,
            background=panel,
            foreground=field_fg,
            arrowcolor=field_fg,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", field_bg)],
            foreground=[("readonly", field_fg)],
            selectbackground=[("readonly", field_bg)],
            selectforeground=[("readonly", field_fg)],
        )

        # ttk.Combobox popup list uses a Tk Listbox; theme it explicitly.
        self.root.option_add("*TCombobox*Listbox.background", field_bg)
        self.root.option_add("*TCombobox*Listbox.foreground", field_fg)
        self.root.option_add("*TCombobox*Listbox.selectBackground", tab_active if dark else "#d9e8ff")
        self.root.option_add("*TCombobox*Listbox.selectForeground", field_fg)

        self.preview.apply_theme(dark=dark)
        self.hq_preview.apply_theme(dark=dark)
        self.profile_form.apply_theme(dark=dark)
        self.run_form.apply_theme(dark=dark)
        self.yaml_editor.apply_theme(dark=dark)

    def _schedule_hq_render(self) -> None:
        if self._hq_after_id:
            self.root.after_cancel(self._hq_after_id)
        self.hq_preview.set_status("HQ preview queued...")
        self._hq_after_id = self.root.after(900, self._trigger_hq_render_now)

    def _trigger_hq_render_now(self) -> None:
        self._hq_after_id = None
        if self.store.profile.text_error or self.store.run.text_error:
            self.hq_preview.set_status("HQ preview skipped: YAML errors")
            return
        profile_data = self.store.profile.data or {}
        run_data = self.store.run.data or {}
        if not profile_data or not run_data:
            self.hq_preview.set_status("HQ preview skipped: missing profile/run")
            return
        self.hq_preview.set_status("HQ preview rendering...")
        self._worker.submit(RenderRequest(profile_data=profile_data, run_data=run_data))

    def _on_hq_status_from_worker(self, status: str, image_path: Path | None) -> None:
        self.root.after(0, lambda: self._apply_hq_status(status, image_path))

    def _apply_hq_status(self, status: str, image_path: Path | None) -> None:
        if status == "rendering":
            self.hq_preview.set_status("HQ preview rendering...")
            return
        if status == "ready":
            self.hq_preview.set_status("HQ preview ready")
            self.hq_preview.set_image(image_path)
            return
        self.hq_preview.set_status(status)

    def _update_title(self) -> None:
        profile_dirty = "*" if self.store.profile.dirty else ""
        run_dirty = "*" if self.store.run.dirty else ""
        pid = self.store.current_profile_id() or "(no profile)"
        run_id = str(((self.store.run.data or {}).get("run") or {}).get("run_id") or "(no run)")
        self.root.title(f"Simple-Sim Profile Editor | {pid}{profile_dirty} | {run_id}{run_dirty}")

    def _schedule_registry_poll(self) -> None:
        if self._closed:
            return
        updated = self.controller.maybe_refresh_registry()
        if updated is not None:
            current_profile = self.profile_var.get().strip()
            current_run = self.run_var.get().strip()

            self.profile_combo["values"] = updated.profile_ids
            if current_profile not in updated.profile_ids and updated.profile_ids:
                self.profile_var.set(updated.profile_ids[0])
                self._on_profile_selected()
            elif not current_profile and updated.profile_ids:
                self.profile_var.set(updated.profile_ids[0])
                self._on_profile_selected()

            self._update_run_choices_for_profile()
            if current_run and current_run not in self.run_combo.cget("values"):
                values = list(self.run_combo.cget("values"))
                if values:
                    self.run_var.set(values[0])
                    self._on_run_selected()
            self.status_var.set("Detected config file changes and refreshed profile/run lists")

        self.root.after(1400, self._schedule_registry_poll)

    def _schedule_system_stats(self) -> None:
        if self._closed:
            return
        try:
            stats = read_system_stats()
            self.system_stats_var.set(format_system_stats(stats))
        except Exception:
            self.system_stats_var.set("CPU n/a | GPU n/a | RAM n/a")
        self.root.after(1000, self._schedule_system_stats)

    def _session_settings_path(self) -> Path:
        return self.sim_root / "outputs" / "profile_editor" / "settings.json"

    def _load_session_settings(self) -> dict:
        path = self._session_settings_path()
        try:
            if not path.exists():
                return {}
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def _apply_session_settings_to_ui(self) -> None:
        geom = str(self._session_settings.get("window_geometry") or "").strip()
        if geom:
            try:
                self.root.geometry(geom)
            except Exception:
                pass
        preview_settings = self._session_settings.get("preview3d")
        if isinstance(preview_settings, dict):
            self.preview.apply_user_settings(preview_settings)
        self.dark_mode_var.set(bool(self._session_settings.get("dark_mode", False)))
        self._apply_theme(bool(self.dark_mode_var.get()))

    def _collect_session_settings(self) -> dict:
        return {
            "window_geometry": str(self.root.geometry()),
            "selected_profile_id": str(self.profile_var.get() or ""),
            "selected_run_id": str(self.run_var.get() or ""),
            "dark_mode": bool(self.dark_mode_var.get()),
            "preview3d": self.preview.get_user_settings(),
        }

    def _schedule_settings_save(self) -> None:
        if self._closed:
            return
        if self._settings_after_id:
            self.root.after_cancel(self._settings_after_id)
        self._settings_after_id = self.root.after(350, self._flush_session_settings)

    def _flush_session_settings(self) -> None:
        self._settings_after_id = None
        data = self._collect_session_settings()
        path = self._session_settings_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            tmp.replace(path)
            self._session_settings = data
        except Exception:
            pass

    def _on_close(self) -> None:
        self._closed = True
        self._flush_session_settings()
        try:
            self._worker.stop()
        finally:
            self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple-Sim profile editor")
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to Simple-Sim root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    app = ProfileEditorApp(root, args.sim_root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()

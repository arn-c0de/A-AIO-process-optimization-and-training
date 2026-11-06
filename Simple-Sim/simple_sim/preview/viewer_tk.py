"""Interactive Tkinter viewer for debug previews (re-render, filter popup, zoom).

Kept separate from :mod:`simple_sim.preview.render` so the rendering logic can be
imported without tkinter. Tk and Pillow are imported lazily when the viewer opens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from simple_sim.coerce import as_bool
from simple_sim.filter_profiles import (
    load_shared_filter_profiles,
    normalized_with_extras,
    save_shared_filter_profiles,
    shared_settings_path,
    to_profile_store,
)
from simple_sim.generation.common import project_root
from simple_sim.preview.render import (
    BACKEND_2D,
    BACKEND_3D,
    PreviewOptions,
    PreviewRec,
    ProfileEntry,
    backends_for_mode,
    new_run_id,
    profile_backends,
    render_profile_backend,
    resolve_defect_types,
    sanitize_dir_name,
    write_index_html,
)
from simple_sim.profile_hash import load_profile


log = logging.getLogger(__name__)


def open_tk_viewer(
    out_root: Path,
    previews: list[PreviewRec],
    *,
    sim_root: Path | None = None,
    profiles_dir: Path | None = None,
    configs_dir: Path | None = None,
    all_profiles: list[ProfileEntry] | None = None,
    options: PreviewOptions | None = None,
    settings_file: Path | None = None,
    selected_profile_ids: list[str] | None = None,
) -> None:
    """Open an interactive Tkinter viewer for rendered previews with rerun capabilities."""
    # Import lazily so the script can still run on systems without Tk installed.
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox
    except ImportError as exc:
        raise RuntimeError(f"Tkinter not available: {exc}") from exc

    try:
        from PIL import Image, ImageTk
    except ImportError as exc:
        raise RuntimeError(f"Pillow (PIL) not available: {exc}") from exc

    import queue
    import threading

    out_root = Path(out_root)
    settings_file = Path(settings_file) if settings_file else shared_settings_path(Path(sim_root) if sim_root is not None else project_root())

    # State for interactive mode
    class ViewerState:
        def __init__(self):
            self.previews = previews
            self.is_rendering = False
            self.backend_filter = "both"

    state = ViewerState()

    def _load_items():
        """Load image items from current preview list."""
        items: list[tuple[str, str, str, Path]] = []
        for pid, backend, defect, rel in state.previews:
            if state.backend_filter != "both" and backend != state.backend_filter:
                continue
            p = (out_root / rel).resolve()
            if p.exists():
                items.append((pid, backend, defect, p))
        return items

    root = tk.Tk()
    root.title("Simple-Sim Debug Previews")
    root.geometry("1200x900")

    top = ttk.Frame(root, padding=10)
    top.pack(fill="both", expand=True)

    # Header with controls
    header = ttk.Frame(top)
    header.pack(fill="x", pady=(0, 10))

    ttk.Label(header, text="Simple-Sim Debug Previews", font=("TkDefaultFont", 14, "bold")).pack(side="left")
    ttk.Label(
        header,
        text=str(out_root),
        foreground="#444",
    ).pack(side="left", padx=12)

    # Interactive controls (only if we have the necessary context)
    if all_profiles and sim_root and profiles_dir and configs_dir and options:
        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(0, 10))

        def _profiles_for_backend(mode: str) -> list[str]:
            mode = str(mode or "both")
            if mode == "both":
                return [pid for pid, _prof, _b in all_profiles]
            return [pid for pid, _prof, b in all_profiles if mode in b]

        # Backend selection / filter
        ttk.Label(controls, text="Backend:").pack(side="left", padx=(0, 5))
        backend_var = tk.StringVar(value=str(options.backend_mode or "both"))
        backend_combo = ttk.Combobox(
            controls,
            textvariable=backend_var,
            values=["both", BACKEND_2D, BACKEND_3D],
            width=10,
            state="readonly",
        )
        backend_combo.pack(side="left", padx=(0, 15))
        state.backend_filter = backend_var.get()

        # Profile selection
        ttk.Label(controls, text="Profile:").pack(side="left", padx=(0, 5))

        profile_var = tk.StringVar()
        profile_ids = _profiles_for_backend(state.backend_filter)
        profile_combo = ttk.Combobox(controls, textvariable=profile_var, values=profile_ids, width=35, state="readonly")
        if profile_ids:
            # Pre-select the first selected profile if it was explicitly chosen
            initial_profile = None
            if selected_profile_ids and len(selected_profile_ids) > 0:
                for pid in selected_profile_ids:
                    if pid in profile_ids:
                        initial_profile = pid
                        break
            if initial_profile:
                profile_var.set(initial_profile)
            else:
                profile_combo.current(0)
        profile_combo.pack(side="left", padx=(0, 15))

        def _on_backend_change(_evt=None):
            state.backend_filter = backend_var.get()
            ids = _profiles_for_backend(state.backend_filter)
            profile_combo["values"] = ids
            if ids:
                profile_combo.current(0)
            _refresh_images()

        backend_combo.bind("<<ComboboxSelected>>", _on_backend_change)

        # Seed controls
        ttk.Label(controls, text="Seed:").pack(side="left", padx=(0, 5))
        seed_var = tk.StringVar(value=str(options.seed_base))
        seed_entry = ttk.Entry(controls, textvariable=seed_var, width=8)
        seed_entry.pack(side="left", padx=(0, 10))

        variable_seeds_var = tk.BooleanVar(value=True)  # Default to variable for reruns
        variable_seeds_check = ttk.Checkbutton(controls, text="Variable", variable=variable_seeds_var)
        variable_seeds_check.pack(side="left", padx=(0, 15))

        shared_profiles, shared_active_profile, shared_current_filter = load_shared_filter_profiles(settings_file)
        # Current filter values in profile-store (popup) format. Unknown keys are kept
        # so a save never strips filters this viewer does not know about.
        filter_values: dict[str, Any] = to_profile_store(
            shared_current_filter,
            shared_profiles.get(shared_active_profile),
        )
        cardinal_rotation_90_var = tk.BooleanVar(
            value=as_bool(filter_values.get("cardinal_rotation_90", True), default=True)
        )

        def _current_filter_store() -> dict[str, Any]:
            """Current filter values in profile-store format (main thread only)."""
            d = dict(filter_values)
            d["cardinal_rotation_90"] = bool(cardinal_rotation_90_var.get())
            return d

        def _current_filters_from_vars() -> dict[str, Any]:
            """Normalized current filter values (main thread only)."""
            return normalized_with_extras(_current_filter_store())

        def _apply_filters_to_vars(d: dict[str, Any]) -> None:
            filter_values.clear()
            filter_values.update(to_profile_store(d))
            cardinal_rotation_90_var.set(as_bool(filter_values.get("cardinal_rotation_90", True), default=True))

        def _persist_filters(active_name: str | None = None) -> None:
            nonlocal shared_active_profile
            cur = _current_filter_store()
            if active_name:
                shared_active_profile = active_name
            shared_profiles[shared_active_profile] = to_profile_store(
                cur,
                shared_profiles.get(shared_active_profile),
            )
            save_shared_filter_profiles(
                settings_file,
                shared_profiles,
                shared_active_profile,
                cur,
            )

        def _open_filter_popup() -> None:
            """Open shared image filter popup."""
            from gui.components.filter_popup import open_filter_popup

            def on_close(updated_profiles: dict[str, dict[str, Any]], updated_active: str) -> None:
                nonlocal shared_active_profile, shared_profiles
                shared_profiles = updated_profiles
                shared_active_profile = updated_active
                _apply_filters_to_vars(shared_profiles.get(shared_active_profile, {}))
                save_shared_filter_profiles(
                    settings_file,
                    shared_profiles,
                    shared_active_profile,
                    _current_filter_store(),
                )

            def show_messagebox(msg_type: str, title: str, message: str) -> None:
                if msg_type == "warning":
                    messagebox.showwarning(title, message, parent=root)
                elif msg_type == "error":
                    messagebox.showerror(title, message, parent=root)
                else:
                    messagebox.showinfo(title, message, parent=root)

            def ask_string(title: str, prompt: str, initial: str = "") -> str | None:
                from tkinter import simpledialog
                return simpledialog.askstring(title, prompt, initialvalue=initial, parent=root)

            def ask_yes_no(title: str, question: str) -> bool:
                return messagebox.askyesno(title, question, parent=root)

            def float_or_default(val: str, default: float) -> float:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            open_filter_popup(
                parent=root,
                profiles=shared_profiles,
                active_profile=shared_active_profile,
                on_close=on_close,
                get_current_values=lambda: to_profile_store(_current_filter_store()),
                set_current_values=_apply_filters_to_vars,
                float_or_default=float_or_default,
                show_messagebox=show_messagebox,
                ask_string=ask_string,
                ask_yes_no=ask_yes_no,
                sim_root=str(sim_root),
            )

        cardinal_rotation_90_check = ttk.Checkbutton(controls, text="90° Rotation", variable=cardinal_rotation_90_var)
        cardinal_rotation_90_check.pack(side="left", padx=(0, 15))
        ttk.Button(controls, text="Image Filters", command=_open_filter_popup).pack(side="left", padx=(0, 15))

        status_label = ttk.Label(controls, text="Ready", foreground="#060")
        status_label.pack(side="left", padx=(10, 10))

        # Worker threads never touch Tk: they post callables here, and the main
        # thread runs them from a root.after poll loop.
        ui_queue: queue.Queue = queue.Queue()

        def _post_ui(fn) -> None:
            ui_queue.put(fn)

        def _poll_ui_queue() -> None:
            while True:
                try:
                    fn = ui_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    fn()
                except Exception as exc:
                    log.info(f"[viewer] UI update failed: {exc}")
            root.after(100, _poll_ui_queue)

        root.after(100, _poll_ui_queue)

        def _snapshot_render_inputs() -> dict[str, Any]:
            """Read every Tk-backed render input on the main thread."""
            try:
                seed_base = int(seed_var.get())
            except ValueError:
                seed_base = int(options.seed_base)
            image_filters = _current_filters_from_vars()
            return {
                "seed_base": seed_base,
                "use_variable_seeds": bool(variable_seeds_var.get()),
                "mode": str(backend_var.get()),
                "image_filters": image_filters,
                "enable_cardinal_rotation_90": bool(image_filters.get("cardinal_rotation_90", True)),
            }

        def _render_profile_backend(
            pid: str,
            prof: dict[str, Any],
            backend: str,
            defect_types: list[str],
            run_id: int | None,
            snap: dict[str, Any],
            jobs_path: Path,
        ) -> list[PreviewRec]:
            """Render one profile/backend in a worker thread (no Tk access)."""
            return render_profile_backend(
                pid,
                prof,
                backend,
                defect_types,
                opts=options,
                sim_root=sim_root,
                configs_dir=configs_dir,
                out_root=out_root,
                seed_base=int(snap["seed_base"]),
                run_id=run_id,
                image_filters=snap["image_filters"],
                enable_cardinal_rotation_90=snap["enable_cardinal_rotation_90"],
                jobs_path=jobs_path,
            )

        def _finish_render(
            new_previews: list[PreviewRec],
            rendered_pairs: set[tuple[str, str]],
            message: str,
            error: str | None = None,
        ) -> None:
            """Main thread: merge results, rewrite index.html, refresh the grid."""
            if rendered_pairs:
                # Replace only (profile, backend) pairs that were actually rendered.
                state.previews = [rec for rec in state.previews if (rec[0], rec[1]) not in rendered_pairs]
                state.previews.extend(new_previews)
                write_index_html(out_root, state.previews)
                _refresh_images()
            if error is None:
                status_label.config(text=message, foreground="#060")
            else:
                status_label.config(text=error, foreground="#b00")
            state.is_rendering = False
            render_btn.config(state="normal")
            render_all_btn.config(state="normal")

        def _begin_render(status_text: str) -> dict[str, Any]:
            state.is_rendering = True
            render_btn.config(state="disabled")
            render_all_btn.config(state="disabled")
            status_label.config(text=status_text, foreground="#c60")
            _persist_filters(shared_active_profile)
            return _snapshot_render_inputs()

        def _start_render():
            if state.is_rendering:
                return

            selected_pid = profile_var.get()
            if not selected_pid:
                status_label.config(text="No profile selected", foreground="#b00")
                return

            snap = _begin_render("Rendering...")

            def _render_thread():
                new_previews: list[PreviewRec] = []
                rendered_pairs: set[tuple[str, str]] = set()
                error: str | None = None
                try:
                    # Always reload profile from disk so edits are picked up without restart.
                    selected_prof = load_profile(selected_pid, profiles_dir)
                    selected_backends = profile_backends(selected_prof)
                    if not selected_backends:
                        raise RuntimeError(f"Profile {selected_pid!r} has no supported backends")

                    defect_types = resolve_defect_types(selected_prof, options.states)

                    # Determine which backend(s) to render in this run
                    mode = snap["mode"]
                    backends_to_render = backends_for_mode(mode, selected_backends)
                    if not backends_to_render:
                        raise RuntimeError(f"Profile {selected_pid!r} does not support backend={mode!r}")

                    run_id = new_run_id(snap["use_variable_seeds"])
                    for backend in backends_to_render:
                        new_previews.extend(
                            _render_profile_backend(
                                selected_pid,
                                selected_prof,
                                backend,
                                defect_types,
                                run_id,
                                snap,
                                out_root / "blender_jobs_previews.jsonl",
                            )
                        )
                        rendered_pairs.add((selected_pid, backend))
                except Exception as exc:
                    error = f"Error: {exc}"
                _post_ui(lambda: _finish_render(new_previews, rendered_pairs, "Render complete", error))

            thread = threading.Thread(target=_render_thread, daemon=True)
            thread.start()

        def _start_render_all():
            """Render all available profiles."""
            if state.is_rendering:
                return

            if not all_profiles:
                status_label.config(text="No profiles available", foreground="#b00")
                return

            total = len(all_profiles)
            snap = _begin_render(f"Rendering all {total} profiles...")

            def _render_all_thread():
                new_previews: list[PreviewRec] = []
                rendered_pairs: set[tuple[str, str]] = set()
                error: str | None = None
                try:
                    mode = snap["mode"]
                    for idx, (pid, _prof_cached, _backends_cached) in enumerate(all_profiles, 1):
                        # Always reload profile from disk so edits are picked up without restart.
                        prof = load_profile(pid, profiles_dir)
                        backends = profile_backends(prof)

                        # Skip profiles not in the current backend filter.
                        if mode != "both" and mode not in backends:
                            continue

                        _post_ui(lambda i=idx, p=pid: status_label.config(
                            text=f"Rendering {i}/{total}: {p}", foreground="#c60"))

                        defect_types = resolve_defect_types(prof, options.states)
                        run_id = new_run_id(snap["use_variable_seeds"])
                        backends_to_render = [b for b in [BACKEND_2D, BACKEND_3D] if b in backends] if mode == "both" else [mode]

                        for backend in backends_to_render:
                            new_previews.extend(
                                _render_profile_backend(
                                    pid,
                                    prof,
                                    backend,
                                    defect_types,
                                    run_id,
                                    snap,
                                    out_root / f"blender_jobs_previews_{sanitize_dir_name(pid)}.jsonl",
                                )
                            )
                            rendered_pairs.add((pid, backend))
                except Exception as exc:
                    error = f"Error: {exc}"
                n_rendered = len({p for p, _b in rendered_pairs})
                _post_ui(lambda: _finish_render(new_previews, rendered_pairs, f"Rendered {n_rendered} profiles", error))

            thread = threading.Thread(target=_render_all_thread, daemon=True)
            thread.start()

        render_btn = ttk.Button(controls, text="Render", command=_start_render)
        render_btn.pack(side="left", padx=(0, 5))

        render_all_btn = ttk.Button(controls, text="Render ALL", command=_start_render_all)
        render_all_btn.pack(side="left", padx=(0, 10))

    # Container for images
    container = ttk.Frame(top)
    container.pack(fill="both", expand=True)

    canvas = tk.Canvas(container, highlightthickness=0)
    vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)

    vbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = ttk.Frame(canvas, padding=6)
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_inner_configure(_evt=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _on_canvas_configure(evt):
        canvas.itemconfigure(inner_id, width=evt.width)

    inner.bind("<Configure>", _on_inner_configure)
    canvas.bind("<Configure>", _on_canvas_configure)

    # Mouse wheel scrolling
    def _on_mousewheel(evt):
        if getattr(evt, "delta", 0):
            canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")

    def _on_button4(_evt):
        canvas.yview_scroll(-3, "units")

    def _on_button5(_evt):
        canvas.yview_scroll(3, "units")

    canvas.bind_all("<MouseWheel>", _on_mousewheel)
    canvas.bind_all("<Button-4>", _on_button4)
    canvas.bind_all("<Button-5>", _on_button5)

    # Build grid
    thumb_max = 240
    pad = 10
    cols = 4

    photo_refs: list[Any] = []
    zoom_photo_refs: list[Any] = []

    def _open_zoom_popup(path: Path, title: str) -> None:
        try:
            im = Image.open(path)
        except Exception:
            return

        win = tk.Toplevel(root)
        win.title(title)
        win.transient(root)

        sw = max(800, int(root.winfo_screenwidth() * 0.9))
        sh = max(600, int(root.winfo_screenheight() * 0.9))

        w, h = im.size
        scale = min(sw / max(1, w), sh / max(1, h), 1.0)
        vw = max(1, int(w * scale))
        vh = max(1, int(h * scale))
        if scale < 1.0:
            im = im.resize((vw, vh), Image.Resampling.LANCZOS)

        ph = ImageTk.PhotoImage(im)
        zoom_photo_refs.append(ph)

        frm = ttk.Frame(win, padding=8)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, image=ph).pack(fill="both", expand=True)

        # Keep popup size snug to image while respecting screen bounds.
        win.geometry(f"{min(vw + 24, sw)}x{min(vh + 24, sh)}")

    def _refresh_images():
        """Clear and rebuild the image grid."""
        # Clear existing widgets
        for widget in inner.winfo_children():
            widget.destroy()

        photo_refs.clear()

        items = _load_items()

        if not items:
            msg = "No images found yet. Click 'Render' to generate previews."
            ttk.Label(inner, text=msg, foreground="#b00").grid(row=0, column=0, pady=16)
            root.title("Simple-Sim Debug Previews (0 images)")
            return

        root.title(f"Simple-Sim Debug Previews ({len(items)} images)")

        items.sort(key=lambda t: (t[0], t[1], t[2]))
        for idx, (pid, backend, defect, img_path) in enumerate(items):
            r = idx // cols
            c = idx % cols

            card = ttk.Frame(inner, padding=pad)
            card.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)

            cap = f"{pid}\n{backend}\n{defect}"
            ttk.Label(card, text=cap, justify="left").pack(anchor="w")

            try:
                im = Image.open(img_path)
                im.thumbnail((thumb_max, thumb_max))
                ph = ImageTk.PhotoImage(im)
            except Exception as exc:
                ttk.Label(card, text=f"[failed to load]\n{img_path.name}\n{exc}", foreground="#b00").pack()
                continue

            photo_refs.append(ph)
            lbl = ttk.Label(card, image=ph)
            lbl.pack()

            lbl.bind(
                "<Button-1>",
                lambda _e, p=img_path, t=f"{pid} | {backend} | {defect}": _open_zoom_popup(p, t),
            )

        for c in range(cols):
            inner.grid_columnconfigure(c, weight=1)

    # Initial load
    _refresh_images()

    root.mainloop()

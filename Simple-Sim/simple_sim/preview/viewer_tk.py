"""Interactive Tkinter viewer for debug previews (re-render, filter popup, zoom).

Kept separate from :mod:`simple_sim.preview.render` so the rendering logic can be
imported without tkinter. Tk and Pillow are imported lazily when the viewer opens.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
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
    PreviewRequest,
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


class _Toolkit:
    """Lazily imported Tk / Pillow modules, so the module itself imports without them."""

    def __init__(self) -> None:
        try:
            import tkinter as tk
            from tkinter import messagebox, simpledialog, ttk
        except ImportError as exc:
            raise RuntimeError(f"Tkinter not available: {exc}") from exc
        try:
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(f"Pillow (PIL) not available: {exc}") from exc
        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.simpledialog = simpledialog
        self.Image = Image
        self.ImageTk = ImageTk


@dataclass(frozen=True)
class RenderSnapshot:
    """Every Tk-backed render input, read once on the main thread and handed to the worker."""

    seed_base: int
    use_variable_seeds: bool
    mode: str
    image_filters: dict[str, Any]
    enable_cardinal_rotation_90: bool


@dataclass
class RenderOutcome:
    """What a render worker produced (filled in the worker, consumed on the main thread)."""

    previews: list[PreviewRec] = field(default_factory=list)
    rendered_pairs: set[tuple[str, str]] = field(default_factory=set)
    error: str | None = None


class PreviewViewer:
    """Tk window showing the preview grid; with a render context it can re-render profiles."""

    THUMB_MAX = 240
    CARD_PAD = 10
    COLUMNS = 4
    UI_POLL_MS = 100

    def __init__(
        self,
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
        self.kit = _Toolkit()
        self.out_root = Path(out_root)
        self.previews = previews
        self.sim_root = sim_root
        self.profiles_dir = profiles_dir
        self.configs_dir = configs_dir
        self.all_profiles = all_profiles
        self.options = options
        self.settings_file = (
            Path(settings_file) if settings_file
            else shared_settings_path(Path(sim_root) if sim_root is not None else project_root())
        )
        self.selected_profile_ids = selected_profile_ids

        self.is_rendering = False
        self.backend_filter = "both"
        self._photo_refs: list[Any] = []
        # Worker threads never touch Tk: they post callables here, and the main
        # thread runs them from a root.after poll loop.
        self._ui_queue: queue.Queue = queue.Queue()

        tk, ttk = self.kit.tk, self.kit.ttk
        self.root = tk.Tk()
        self.root.title("Simple-Sim Debug Previews")
        self.root.geometry("1200x900")

        self.top = ttk.Frame(self.root, padding=10)
        self.top.pack(fill="both", expand=True)

        self._build_header()
        # Interactive controls (only if we have the necessary context)
        if all_profiles and sim_root and profiles_dir and configs_dir and options:
            self._build_controls()
        self._build_grid_container()

    def run(self) -> None:
        """Show the grid and block in the Tk main loop."""
        self._refresh_images()
        self.root.mainloop()

    # ------------------------------------------------------------------ layout
    def _build_header(self) -> None:
        ttk = self.kit.ttk
        header = ttk.Frame(self.top)
        header.pack(fill="x", pady=(0, 10))
        ttk.Label(header, text="Simple-Sim Debug Previews", font=("TkDefaultFont", 14, "bold")).pack(side="left")
        ttk.Label(header, text=str(self.out_root), foreground="#444").pack(side="left", padx=12)

    def _build_controls(self) -> None:
        tk, ttk = self.kit.tk, self.kit.ttk
        options = self.options
        controls = ttk.Frame(self.top)
        controls.pack(fill="x", pady=(0, 10))

        # Backend selection / filter
        ttk.Label(controls, text="Backend:").pack(side="left", padx=(0, 5))
        self.backend_var = tk.StringVar(value=str(options.backend_mode or "both"))
        backend_combo = ttk.Combobox(
            controls,
            textvariable=self.backend_var,
            values=["both", BACKEND_2D, BACKEND_3D],
            width=10,
            state="readonly",
        )
        backend_combo.pack(side="left", padx=(0, 15))
        self.backend_filter = self.backend_var.get()

        # Profile selection
        ttk.Label(controls, text="Profile:").pack(side="left", padx=(0, 5))
        self.profile_var = tk.StringVar()
        profile_ids = self._profiles_for_backend(self.backend_filter)
        self.profile_combo = ttk.Combobox(controls, textvariable=self.profile_var, values=profile_ids, width=35, state="readonly")
        if profile_ids:
            # Pre-select the first selected profile if it was explicitly chosen
            initial_profile = next((pid for pid in self.selected_profile_ids or [] if pid in profile_ids), None)
            if initial_profile:
                self.profile_var.set(initial_profile)
            else:
                self.profile_combo.current(0)
        self.profile_combo.pack(side="left", padx=(0, 15))
        backend_combo.bind("<<ComboboxSelected>>", self._on_backend_change)

        # Seed controls
        ttk.Label(controls, text="Seed:").pack(side="left", padx=(0, 5))
        self.seed_var = tk.StringVar(value=str(options.seed_base))
        ttk.Entry(controls, textvariable=self.seed_var, width=8).pack(side="left", padx=(0, 10))
        self.variable_seeds_var = tk.BooleanVar(value=True)  # Default to variable for reruns
        ttk.Checkbutton(controls, text="Variable", variable=self.variable_seeds_var).pack(side="left", padx=(0, 15))

        self.shared_profiles, self.shared_active_profile, shared_current_filter = load_shared_filter_profiles(self.settings_file)
        # Current filter values in profile-store (popup) format. Unknown keys are kept
        # so a save never strips filters this viewer does not know about.
        self.filter_values: dict[str, Any] = to_profile_store(
            shared_current_filter,
            self.shared_profiles.get(self.shared_active_profile),
        )
        self.cardinal_rotation_90_var = tk.BooleanVar(
            value=as_bool(self.filter_values.get("cardinal_rotation_90", True), default=True)
        )
        ttk.Checkbutton(controls, text="90° Rotation", variable=self.cardinal_rotation_90_var).pack(side="left", padx=(0, 15))
        ttk.Button(controls, text="Image Filters", command=self._open_filter_popup).pack(side="left", padx=(0, 15))

        self.status_label = ttk.Label(controls, text="Ready", foreground="#060")
        self.status_label.pack(side="left", padx=(10, 10))

        self.root.after(self.UI_POLL_MS, self._poll_ui_queue)

        self.render_btn = ttk.Button(controls, text="Render", command=self._start_render)
        self.render_btn.pack(side="left", padx=(0, 5))
        self.render_all_btn = ttk.Button(controls, text="Render ALL", command=self._start_render_all)
        self.render_all_btn.pack(side="left", padx=(0, 10))

    def _build_grid_container(self) -> None:
        tk, ttk = self.kit.tk, self.kit.ttk
        container = ttk.Frame(self.top)
        container.pack(fill="both", expand=True)

        canvas = self.canvas = tk.Canvas(container, highlightthickness=0)
        vbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.inner = ttk.Frame(canvas, padding=6)
        inner_id = canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", lambda _evt=None: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda evt: canvas.itemconfigure(inner_id, width=evt.width))

        # Mouse wheel scrolling
        canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        canvas.bind_all("<Button-4>", lambda _evt: canvas.yview_scroll(-3, "units"))
        canvas.bind_all("<Button-5>", lambda _evt: canvas.yview_scroll(3, "units"))

    def _on_mousewheel(self, evt: Any) -> None:
        if getattr(evt, "delta", 0):
            self.canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")

    # ------------------------------------------------------------------ profiles / filters
    def _profiles_for_backend(self, mode: str) -> list[str]:
        mode = str(mode or "both")
        if mode == "both":
            return [pid for pid, _prof, _b in self.all_profiles]
        return [pid for pid, _prof, b in self.all_profiles if mode in b]

    def _on_backend_change(self, _evt: Any = None) -> None:
        self.backend_filter = self.backend_var.get()
        ids = self._profiles_for_backend(self.backend_filter)
        self.profile_combo["values"] = ids
        if ids:
            self.profile_combo.current(0)
        self._refresh_images()

    def _current_filter_store(self) -> dict[str, Any]:
        """Current filter values in profile-store format (main thread only)."""
        d = dict(self.filter_values)
        d["cardinal_rotation_90"] = bool(self.cardinal_rotation_90_var.get())
        return d

    def _apply_filters_to_vars(self, d: dict[str, Any]) -> None:
        self.filter_values.clear()
        self.filter_values.update(to_profile_store(d))
        self.cardinal_rotation_90_var.set(as_bool(self.filter_values.get("cardinal_rotation_90", True), default=True))

    def _persist_filters(self) -> None:
        cur = self._current_filter_store()
        self.shared_profiles[self.shared_active_profile] = to_profile_store(
            cur,
            self.shared_profiles.get(self.shared_active_profile),
        )
        save_shared_filter_profiles(self.settings_file, self.shared_profiles, self.shared_active_profile, cur)

    def _on_filter_popup_close(self, updated_profiles: dict[str, dict[str, Any]], updated_active: str) -> None:
        self.shared_profiles = updated_profiles
        self.shared_active_profile = updated_active
        self._apply_filters_to_vars(self.shared_profiles.get(self.shared_active_profile, {}))
        save_shared_filter_profiles(
            self.settings_file,
            self.shared_profiles,
            self.shared_active_profile,
            self._current_filter_store(),
        )

    def _show_messagebox(self, msg_type: str, title: str, message: str) -> None:
        box = self.kit.messagebox
        show = {"warning": box.showwarning, "error": box.showerror}.get(msg_type, box.showinfo)
        show(title, message, parent=self.root)

    def _open_filter_popup(self) -> None:
        """Open shared image filter popup."""
        from gui.components.filter_popup import open_filter_popup

        def float_or_default(val: str, default: float) -> float:
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        open_filter_popup(
            parent=self.root,
            profiles=self.shared_profiles,
            active_profile=self.shared_active_profile,
            on_close=self._on_filter_popup_close,
            get_current_values=lambda: to_profile_store(self._current_filter_store()),
            set_current_values=self._apply_filters_to_vars,
            float_or_default=float_or_default,
            show_messagebox=self._show_messagebox,
            ask_string=lambda title, prompt, initial="": self.kit.simpledialog.askstring(
                title, prompt, initialvalue=initial, parent=self.root),
            ask_yes_no=lambda title, question: self.kit.messagebox.askyesno(title, question, parent=self.root),
            sim_root=str(self.sim_root),
        )

    # ------------------------------------------------------------------ rendering
    def _post_ui(self, fn: Callable[[], None]) -> None:
        self._ui_queue.put(fn)

    def _poll_ui_queue(self) -> None:
        while True:
            try:
                fn = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception as exc:
                log.info(f"[viewer] UI update failed: {exc}")
        self.root.after(self.UI_POLL_MS, self._poll_ui_queue)

    def _set_status(self, text: str, color: str) -> None:
        self.status_label.config(text=text, foreground=color)

    def _snapshot_render_inputs(self) -> RenderSnapshot:
        """Read every Tk-backed render input on the main thread."""
        try:
            seed_base = int(self.seed_var.get())
        except ValueError:
            seed_base = int(self.options.seed_base)
        image_filters = normalized_with_extras(self._current_filter_store())
        return RenderSnapshot(
            seed_base=seed_base,
            use_variable_seeds=bool(self.variable_seeds_var.get()),
            mode=str(self.backend_var.get()),
            image_filters=image_filters,
            enable_cardinal_rotation_90=bool(image_filters.get("cardinal_rotation_90", True)),
        )

    def _render_into(
        self,
        outcome: RenderOutcome,
        pid: str,
        prof: dict[str, Any],
        backends: list[str],
        snap: RenderSnapshot,
        jobs_path: Path,
    ) -> None:
        """Render one profile on the given backends in a worker thread (no Tk access)."""
        defect_types = resolve_defect_types(prof, self.options.states)
        request = PreviewRequest(
            pid=pid,
            prof=prof,
            defect_types=defect_types,
            opts=self.options,
            out_root=self.out_root,
            seed_base=snap.seed_base,
            run_id=new_run_id(snap.use_variable_seeds),
            image_filters=snap.image_filters,
            enable_cardinal_rotation_90=snap.enable_cardinal_rotation_90,
            jobs_path=jobs_path,
        )
        for backend in backends:
            outcome.previews.extend(
                render_profile_backend(request, backend, sim_root=self.sim_root, configs_dir=self.configs_dir)
            )
            outcome.rendered_pairs.add((pid, backend))

    def _finish_render(self, outcome: RenderOutcome, message: str) -> None:
        """Main thread: merge results, rewrite index.html, refresh the grid."""
        error = outcome.error
        try:
            if outcome.rendered_pairs:
                # Replace only (profile, backend) pairs that were actually rendered.
                self.previews = [rec for rec in self.previews if (rec[0], rec[1]) not in outcome.rendered_pairs]
                self.previews.extend(outcome.previews)
                write_index_html(self.out_root, self.previews)
                self._refresh_images()
        except Exception as exc:  # e.g. output dir removed, disk full
            error = f"Preview update failed: {exc}"
        finally:
            # Always leave the "rendering" state, or both render buttons stay disabled until restart.
            self.is_rendering = False
            self.render_btn.config(state="normal")
            self.render_all_btn.config(state="normal")
        if error is None:
            self._set_status(message, "#060")
        else:
            self._set_status(error, "#b00")

    def _begin_render(self, status_text: str) -> RenderSnapshot:
        self.is_rendering = True
        self.render_btn.config(state="disabled")
        self.render_all_btn.config(state="disabled")
        self._set_status(status_text, "#c60")
        self._persist_filters()
        return self._snapshot_render_inputs()

    def _run_render_worker(self, work: Callable[[RenderOutcome], None], message: Callable[[RenderOutcome], str]) -> None:
        """Run ``work`` in a daemon thread and hand its outcome back to the main thread."""

        def _thread() -> None:
            outcome = RenderOutcome()
            try:
                work(outcome)
            except Exception as exc:
                outcome.error = f"Error: {exc}"
            self._post_ui(lambda: self._finish_render(outcome, message(outcome)))

        threading.Thread(target=_thread, daemon=True).start()

    def _start_render(self) -> None:
        if self.is_rendering:
            return
        selected_pid = self.profile_var.get()
        if not selected_pid:
            self._set_status("No profile selected", "#b00")
            return
        snap = self._begin_render("Rendering...")

        def work(outcome: RenderOutcome) -> None:
            # Always reload profile from disk so edits are picked up without restart.
            prof = load_profile(selected_pid, self.profiles_dir)
            supported = profile_backends(prof)
            if not supported:
                raise RuntimeError(f"Profile {selected_pid!r} has no supported backends")
            backends = backends_for_mode(snap.mode, supported)
            if not backends:
                raise RuntimeError(f"Profile {selected_pid!r} does not support backend={snap.mode!r}")
            self._render_into(outcome, selected_pid, prof, backends, snap, self.out_root / "blender_jobs_previews.jsonl")

        self._run_render_worker(work, lambda _outcome: "Render complete")

    def _start_render_all(self) -> None:
        """Render all available profiles."""
        if self.is_rendering:
            return
        if not self.all_profiles:
            self._set_status("No profiles available", "#b00")
            return
        total = len(self.all_profiles)
        snap = self._begin_render(f"Rendering all {total} profiles...")

        def work(outcome: RenderOutcome) -> None:
            mode = snap.mode
            for idx, (pid, _prof_cached, _backends_cached) in enumerate(self.all_profiles, 1):
                # Always reload profile from disk so edits are picked up without restart.
                prof = load_profile(pid, self.profiles_dir)
                supported = profile_backends(prof)
                # Skip profiles not in the current backend filter.
                if mode != "both" and mode not in supported:
                    continue
                self._post_ui(lambda i=idx, p=pid: self._set_status(f"Rendering {i}/{total}: {p}", "#c60"))
                backends = [b for b in [BACKEND_2D, BACKEND_3D] if b in supported] if mode == "both" else [mode]
                self._render_into(
                    outcome, pid, prof, backends, snap,
                    self.out_root / f"blender_jobs_previews_{sanitize_dir_name(pid)}.jsonl",
                )

        self._run_render_worker(
            work, lambda outcome: f"Rendered {len({p for p, _b in outcome.rendered_pairs})} profiles")

    # ------------------------------------------------------------------ image grid
    def _load_items(self) -> list[tuple[str, str, str, Path]]:
        """Existing image files of the current preview list, limited to the backend filter."""
        items: list[tuple[str, str, str, Path]] = []
        for pid, backend, defect, rel in self.previews:
            if self.backend_filter != "both" and backend != self.backend_filter:
                continue
            p = (self.out_root / rel).resolve()
            if p.exists():
                items.append((pid, backend, defect, p))
        return items

    def _open_zoom_popup(self, path: Path, title: str) -> None:
        tk, ttk, Image = self.kit.tk, self.kit.ttk, self.kit.Image
        try:
            im = Image.open(path)
        except Exception:
            return

        root = self.root
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

        ph = self.kit.ImageTk.PhotoImage(im)
        win._photo = ph  # keeps the image alive for this window only (a shared list grew with every zoom)

        frm = ttk.Frame(win, padding=8)
        frm.pack(fill="both", expand=True)
        ttk.Label(frm, image=ph).pack(fill="both", expand=True)

        # Keep popup size snug to image while respecting screen bounds.
        win.geometry(f"{min(vw + 24, sw)}x{min(vh + 24, sh)}")

    def _refresh_images(self) -> None:
        """Clear and rebuild the image grid."""
        ttk, inner = self.kit.ttk, self.inner
        for widget in inner.winfo_children():
            widget.destroy()
        self._photo_refs.clear()

        items = self._load_items()
        if not items:
            msg = "No images found yet. Click 'Render' to generate previews."
            ttk.Label(inner, text=msg, foreground="#b00").grid(row=0, column=0, pady=16)
            self.root.title("Simple-Sim Debug Previews (0 images)")
            return

        self.root.title(f"Simple-Sim Debug Previews ({len(items)} images)")

        items.sort(key=lambda t: (t[0], t[1], t[2]))
        for idx, (pid, backend, defect, img_path) in enumerate(items):
            card = ttk.Frame(inner, padding=self.CARD_PAD)
            card.grid(row=idx // self.COLUMNS, column=idx % self.COLUMNS, sticky="nsew", padx=6, pady=6)
            ttk.Label(card, text=f"{pid}\n{backend}\n{defect}", justify="left").pack(anchor="w")

            try:
                im = self.kit.Image.open(img_path)
                im.thumbnail((self.THUMB_MAX, self.THUMB_MAX))
                ph = self.kit.ImageTk.PhotoImage(im)
            except Exception as exc:
                ttk.Label(card, text=f"[failed to load]\n{img_path.name}\n{exc}", foreground="#b00").pack()
                continue

            self._photo_refs.append(ph)
            lbl = ttk.Label(card, image=ph)
            lbl.pack()
            lbl.bind(
                "<Button-1>",
                lambda _e, p=img_path, t=f"{pid} | {backend} | {defect}": self._open_zoom_popup(p, t),
            )

        for c in range(self.COLUMNS):
            inner.grid_columnconfigure(c, weight=1)


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
    PreviewViewer(
        out_root,
        previews,
        sim_root=sim_root,
        profiles_dir=profiles_dir,
        configs_dir=configs_dir,
        all_profiles=all_profiles,
        options=options,
        settings_file=settings_file,
        selected_profile_ids=selected_profile_ids,
    ).run()

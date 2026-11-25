"""Component preview panels of the filter popup.

The popup shows two of them (inline next to the filter controls, and a compact one below the
filter-profile list). Both are :class:`ProfilePreviewPanel` instances that differ only in layout
and texts; each renders the selected component profile with the popup's *current* filter values.
"""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any

from simple_sim.preview.render import BACKEND_2D, BACKEND_3D, discover_profiles
from gui.utils.tk_safe import quiet

_HEADER_PREFIX = "──"
_HEADER_2D = "── 2D ──"
_HEADER_3D = "── 3D ──"
_NO_PIL = "Pillow not installed.\nRun: pip install Pillow"


def _open_zoom_image_popup(popup: Any, pil_img: Any, title: str, on_error: Callable[[str], None]) -> None:
    """Open a modal-ish window with a larger version of the selected preview image."""
    try:
        from PIL import Image, ImageTk  # type: ignore
    except ImportError:
        on_error(_NO_PIL)
        return

    if pil_img is None:
        return

    prev = getattr(popup, "_zoom_img_window", None)
    if prev is not None and getattr(prev, "winfo_exists", lambda: False)():
        prev.destroy()

    win = tk.Toplevel(popup.top)
    win.title(title)
    win.transient(popup.top)
    win.lift()
    win.focus_force()

    max_w = max(240, int(win.winfo_screenwidth() * 0.9))
    max_h = max(240, int(win.winfo_screenheight() * 0.85))
    img = pil_img.copy()
    w, h = img.size
    if w > max_w or h > max_h:
        scale = min(max_w / float(w), max_h / float(h))
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    photo = ImageTk.PhotoImage(img, master=win)
    lbl = tk.Label(win, image=photo, anchor="center", bd=0, highlightthickness=0)
    lbl.image = photo  # keep reference
    lbl.pack(fill="both", expand=True, padx=8, pady=8)
    lbl.bind("<Double-Button-1>", lambda _evt: win.destroy())
    win.bind("<Escape>", lambda _evt: win.destroy())

    popup._zoom_photo = photo
    popup._zoom_pil = img
    popup._zoom_img_window = win

    # Grab can fail if called before the window is viewable on some Tk backends.
    def _safe_grab() -> None:
        try:
            if win.winfo_exists():
                win.grab_set()
        except tk.TclError:
            pass

    win.after_idle(_safe_grab)


def get_preview_filter_values(popup: Any) -> dict[str, Any]:
    """Collect current filter values directly from the popup's internal vars."""
    result: dict[str, Any] = {}
    for key, var in popup.filter_vars.items():
        if isinstance(var, tk.BooleanVar):
            result[key] = bool(var.get())
        elif isinstance(var, tk.DoubleVar):
            with quiet():
                result[key] = float(var.get())
        elif isinstance(var, tk.StringVar):
            result[key] = var.get()
    return result


def _run_in_background(popup: Any, work: Callable[[], Any], on_ok: Callable[[Any], None],
                       on_error: Callable[[str], None], on_settled: Callable[[], None]) -> None:
    """Run *work* in a thread; deliver its result on the Tk thread (polled, no Tk calls from the worker).

    The popup is also opened from the standalone viewer, which has no app dispatcher, hence the poll.
    """
    results: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            results.put(("ok", work()))
        except Exception as exc:  # noqa: BLE001 - shown in the popup
            results.put(("error", f"{type(exc).__name__}: {exc}"))

    def poll() -> None:
        try:
            if not popup.top.winfo_exists():
                return
        except tk.TclError:
            return
        try:
            kind, value = results.get_nowait()
        except queue.Empty:
            popup.top.after(50, poll)
            return
        on_settled()
        (on_ok if kind == "ok" else on_error)(value)

    threading.Thread(target=target, daemon=True).start()
    popup.top.after(50, poll)


def do_render_2d_preview(popup: Any, profile_id: str, filter_vals: dict[str, Any]) -> tuple[Any, str]:
    """2D preview render (call from a worker thread). Returns ``(BGR image, config source)``."""
    from .preview_render import render_filter_preview_2d

    return render_filter_preview_2d(_sim_root(popup), profile_id, filter_vals)


def do_render_3d_preview(popup: Any, profile_id: str, filter_vals: dict[str, Any],
                         out_subdir: str = "filter_preview_3d") -> tuple[Any, str]:
    """3D (Blender) preview render (call from a worker thread). Returns ``(BGR image, config source)``."""
    from .preview_render import render_filter_preview_3d

    root = _sim_root(popup)
    return render_filter_preview_3d(root, profile_id, filter_vals, root / "outputs" / out_subdir)


def _sim_root(popup: Any) -> Path:
    sim_root = Path(popup._sim_root)
    if str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    return sim_root


def _to_photo(popup: Any, img_bgr: Any, max_px: int) -> tuple[Any, Any]:
    """(full-size PIL image, Tk photo scaled to fit *max_px*)."""
    from PIL import Image, ImageTk  # type: ignore
    import numpy as np  # type: ignore

    pil_img = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]).astype(np.uint8))
    thumb = pil_img
    w, h = pil_img.size
    if w > max_px or h > max_px:
        scale = max_px / max(w, h)
        thumb = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return pil_img, ImageTk.PhotoImage(thumb, master=popup.top)


def scan_preview_profiles(sim_root: Path) -> tuple[list[str], list[str]]:
    """``(2D profile ids, 3D profile ids)`` under ``configs/profiles`` (a profile may be in both)."""
    entries = discover_profiles(Path(sim_root) / "configs" / "profiles", backend_mode="both")
    return ([pid for pid, _prof, b in entries if BACKEND_2D in b],
            [pid for pid, _prof, b in entries if BACKEND_3D in b])


def combo_values(vals_2d: list[str], vals_3d: list[str]) -> list[str]:
    """Combobox entries: each backend group under a (non-selectable) header line."""
    out: list[str] = []
    if vals_2d:
        out += [_HEADER_2D, *vals_2d]
    if vals_3d:
        out += [_HEADER_3D, *vals_3d]
    return out


def skip_header(value: str, values: list[str]) -> str:
    """*value*, or the first real entry after it when it is a group header ("" if there is none)."""
    if not value.startswith(_HEADER_PREFIX):
        return value
    try:
        idx = values.index(value)
    except ValueError:
        return ""
    return next((v for v in values[idx + 1:] if not v.startswith(_HEADER_PREFIX)), "")


@dataclass(frozen=True)
class PreviewPanelSpec:
    """Layout and texts of one preview panel."""

    title: str
    combo_width: int
    wraplength: int
    max_px: int
    button_text: str
    zoom_title: str  # "{profile_id}"/"{backend}" placeholders
    ready_text: str
    invalid_text: str
    out_subdir_3d: str = "filter_preview_3d"
    show_details: bool = False  # status line with "defect: OK" and the config source
    fill_y: bool = True


INLINE_PANEL = PreviewPanelSpec(
    title="Filter Preview", combo_width=28, wraplength=330, max_px=300, button_text="Render Preview",
    zoom_title="Filter Preview - {profile_id} ({backend})",
    ready_text="Click 'Render Preview' to see the current filter effect.",
    invalid_text="Select a valid profile first.", show_details=True)

LEFT_PANEL = PreviewPanelSpec(
    title="Component Preview", combo_width=16, wraplength=170, max_px=180, button_text="Render",
    zoom_title="Component Preview - {profile_id} ({backend})",
    ready_text="Click 'Render' to preview with current filter settings.",
    invalid_text="Select a valid component profile first.", out_subdir_3d="filter_preview_left", fill_y=False)


class ProfilePreviewPanel:
    """Profile combobox + Render button + status line + image (click to zoom)."""

    def __init__(self, popup: Any, parent: ttk.Frame, spec: PreviewPanelSpec) -> None:
        self.popup = popup
        self.spec = spec
        self.profiles_2d: set[str] = set()
        self.profiles_3d: set[str] = set()
        self.is_rendering = False
        self.photo: Any = None
        self.full_pil: Any = None

        lf = ttk.LabelFrame(parent, text=spec.title, padding=8 if spec.fill_y else 6)
        if spec.fill_y:
            lf.pack(fill="both", expand=True)
        else:
            lf.pack(fill="x", pady=(8, 0))

        sel_row = ttk.Frame(lf)
        sel_row.pack(fill="x", pady=(0, 4))
        ttk.Label(sel_row, text="Profile:").pack(side="left")
        self.profile_var = tk.StringVar()
        self.combo = ttk.Combobox(sel_row, textvariable=self.profile_var, width=spec.combo_width, state="readonly")
        self.combo.pack(side="left", padx=(6 if spec.fill_y else 4, 0), fill="x", expand=True)
        self.combo.bind("<<ComboboxSelected>>", self._on_selected)

        ttk.Button(lf, text=spec.button_text, command=self.render).pack(anchor="w", pady=(4, 4))

        self.status_var = tk.StringVar(value="")
        ttk.Label(lf, textvariable=self.status_var, style="Muted.TLabel", wraplength=spec.wraplength,
                  justify="left").pack(anchor="w", pady=(0, 6 if spec.fill_y else 4))

        self.img_label = ttk.Label(lf, text="No preview yet", anchor="center")
        self.img_label.pack(fill="both", expand=True)

        try:
            self.populate()
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            self.status_var.set(f"Preview init error: {exc}")

    def populate(self) -> None:
        """Scan configs/profiles/ and fill the combobox with all component profiles (2D + 3D)."""
        if not getattr(self.popup, "_sim_root", None):
            self.status_var.set("Preview unavailable: sim_root not set.")
            self.combo["values"] = ()
            return
        profiles_dir = Path(self.popup._sim_root) / "configs" / "profiles"
        if not profiles_dir.exists():
            self.status_var.set(f"profiles dir not found:\n{profiles_dir}")
            return

        vals_2d, vals_3d = scan_preview_profiles(Path(self.popup._sim_root))
        self.profiles_2d, self.profiles_3d = set(vals_2d), set(vals_3d)
        values = combo_values(vals_2d, vals_3d)
        self.combo["values"] = values

        current = self.profile_var.get()
        if current not in values or current.startswith(_HEADER_PREFIX):
            if vals_2d:
                self.profile_var.set(vals_2d[0])
            elif vals_3d:
                self.profile_var.set(vals_3d[0])
        self.status_var.set(self.spec.ready_text if values else "No component profiles found.")

    def _on_selected(self, _evt: tk.Event | None = None) -> None:
        """Skip separator header items in the combobox."""
        self.profile_var.set(skip_header(self.profile_var.get(), list(self.combo["values"])))

    def render(self) -> None:
        """Start a background render with the current filter settings."""
        if self.is_rendering:
            return
        pid = self.profile_var.get().strip()
        if not pid or pid.startswith(_HEADER_PREFIX):
            self.status_var.set(self.spec.invalid_text)
            return

        filter_vals = get_preview_filter_values(self.popup)
        is_3d_only = pid in self.profiles_3d and pid not in self.profiles_2d
        backend = "3D" if is_3d_only else "2D"
        self.status_var.set(f"Rendering ({backend})…")
        popup, out_subdir = self.popup, self.spec.out_subdir_3d
        work = (lambda: do_render_3d_preview(popup, pid, filter_vals, out_subdir)) if is_3d_only else (
            lambda: do_render_2d_preview(popup, pid, filter_vals))

        def settled() -> None:
            self.is_rendering = False

        self.is_rendering = True
        _run_in_background(popup, work, lambda res: self.show_image(res[0], pid, backend, res[1]),
                           lambda err: self.status_var.set(f"Render error: {err}"), settled)

    def show_image(self, img_bgr: Any, profile_id: str, backend: str = "2D", source: str = "") -> None:
        """Display a rendered BGR numpy array in the panel."""
        try:
            self.full_pil, self.photo = _to_photo(self.popup, img_bgr, self.spec.max_px)
            self.img_label.configure(image=self.photo, text="", cursor="hand2")
            title = self.spec.zoom_title.format(profile_id=profile_id, backend=backend)
            self.img_label.bind("<Button-1>", lambda _evt: _open_zoom_image_popup(
                self.popup, self.full_pil, title, self.status_var.set))
            if self.spec.show_details:
                src = f"\n{source}" if source else ""
                self.status_var.set(f"Profile: {profile_id}  ·  defect: OK  ·  {backend}{src}")
            else:
                self.status_var.set(f"Profile: {profile_id}  ·  {backend}")
        except ImportError:
            self.status_var.set(_NO_PIL)
        except Exception as exc:  # noqa: BLE001 - shown in the panel
            self.status_var.set(f"Display error: {exc}")

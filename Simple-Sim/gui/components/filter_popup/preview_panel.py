"""Preview panel helpers for the filter popup."""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import Any


def _open_zoom_image_popup(popup: Any, pil_img: Any, title: str) -> None:
    """Open a modal-ish window with a larger version of the selected preview image."""
    try:
        from PIL import Image, ImageTk  # type: ignore
    except ImportError:
        popup._preview_status_var.set("Pillow not installed.\nRun: pip install Pillow")
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


def build_preview_panel(popup: Any, parent: ttk.Frame) -> None:
    """Build the live filter preview panel."""
    lf = ttk.LabelFrame(parent, text="Filter Preview", padding=8)
    lf.pack(fill="both", expand=True)

    sel_row = ttk.Frame(lf)
    sel_row.pack(fill="x", pady=(0, 4))

    ttk.Label(sel_row, text="Profile:").pack(side="left")
    popup._preview_profile_var = tk.StringVar()
    popup._preview_profile_combo = ttk.Combobox(
        sel_row,
        textvariable=popup._preview_profile_var,
        width=30,
        state="readonly",
    )
    popup._preview_profile_combo.pack(side="left", padx=(6, 0), fill="x", expand=True)
    popup._preview_profile_combo.bind("<<ComboboxSelected>>", popup._on_preview_profile_selected)

    btn_row = ttk.Frame(lf)
    btn_row.pack(fill="x", pady=(4, 4))
    ttk.Button(btn_row, text="Render Preview", command=popup._render_preview).pack(side="left")

    popup._preview_status_var = tk.StringVar(value="")
    ttk.Label(lf, textvariable=popup._preview_status_var, style="Muted.TLabel", wraplength=310, justify="left"
              ).pack(anchor="w", pady=(0, 6))

    popup._preview_img_label = ttk.Label(lf, text="No preview yet", anchor="center")
    popup._preview_img_label.pack(fill="both", expand=True)

    populate_preview_profiles(popup)


def populate_preview_profiles(popup: Any) -> None:
    """Scan configs/profiles/ and populate the profile combobox."""
    if not getattr(popup, "_sim_root", None):
        popup._preview_status_var.set("Preview unavailable: sim_root not provided.")
        popup._preview_profile_combo["values"] = ()
        return

    try:
        import yaml as _yaml  # type: ignore
    except ImportError:
        popup._preview_status_var.set("PyYAML not installed – cannot load profiles.")
        return

    profiles_dir = Path(popup._sim_root) / "configs" / "profiles"
    if not profiles_dir.exists():
        popup._preview_status_var.set("profiles dir not found.")
        return

    vals_2d: list = []
    vals_3d: list = []
    for p in sorted(profiles_dir.glob("*.yaml")):
        try:
            data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            backends = list((data.get("profile") or {}).get("supported_render_backends") or [])
            pid = p.stem
            if "opencv_2d" in backends:
                vals_2d.append(pid)
            if "blender_3d" in backends:
                vals_3d.append(pid)
        except Exception:
            continue

    popup._preview_profiles_2d = set(vals_2d)
    popup._preview_profiles_3d = set(vals_3d)

    combo_vals: list = []
    if vals_2d:
        combo_vals.append("── 2D ──")
        combo_vals.extend(vals_2d)
    if vals_3d:
        combo_vals.append("── 3D ──")
        combo_vals.extend(vals_3d)

    popup._preview_profile_combo["values"] = combo_vals

    if vals_2d:
        popup._preview_profile_var.set(vals_2d[0])
        popup._preview_status_var.set("Click 'Render Preview' to see the current filter effect.")
    elif combo_vals:
        popup._preview_profile_var.set(combo_vals[0])
    else:
        popup._preview_status_var.set("No component profiles found.")


def on_preview_profile_selected(popup: Any, _evt: tk.Event | None = None) -> None:
    """Skip separator header items in the combobox."""
    v = popup._preview_profile_var.get()
    if not v.startswith("──"):
        return
    vals = list(popup._preview_profile_combo["values"])
    try:
        idx = vals.index(v)
        for nxt in vals[idx + 1:]:
            if not nxt.startswith("──"):
                popup._preview_profile_var.set(nxt)
                return
    except ValueError:
        pass
    popup._preview_profile_var.set("")


def get_preview_filter_values(popup: Any) -> dict[str, Any]:
    """Collect current filter values directly from the popup's internal vars."""
    result: dict[str, Any] = {}
    for key, var in popup.filter_vars.items():
        if isinstance(var, tk.BooleanVar):
            result[key] = bool(var.get())
        elif isinstance(var, tk.DoubleVar):
            try:
                result[key] = float(var.get())
            except Exception:
                pass
        elif isinstance(var, tk.StringVar):
            result[key] = var.get()
    return result


def _run_in_background(popup: Any, work: Callable[[], Any], on_ok: Callable[[Any], None],
                       on_error: Callable[[str], None], busy_attr: str) -> None:
    """Run *work* in a thread; deliver its result on the Tk thread (polled, no Tk calls from the worker)."""
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
        setattr(popup, busy_attr, False)
        (on_ok if kind == "ok" else on_error)(value)

    setattr(popup, busy_attr, True)
    threading.Thread(target=target, daemon=True).start()
    popup.top.after(50, poll)


def render_preview(popup: Any) -> None:
    """Start a background render with the current filter settings."""
    if popup._preview_is_rendering:
        return

    pid = popup._preview_profile_var.get().strip()
    if not pid or pid.startswith("──"):
        popup._preview_status_var.set("Select a valid profile first.")
        return

    is_3d_only = pid in popup._preview_profiles_3d and pid not in popup._preview_profiles_2d
    backend = "3D" if is_3d_only else "2D"
    popup._preview_status_var.set(f"Rendering ({backend})…")
    filter_vals = get_preview_filter_values(popup)
    work = (lambda: do_render_3d_preview(popup, pid, filter_vals)) if is_3d_only else (
        lambda: do_render_2d_preview(popup, pid, filter_vals))
    _run_in_background(
        popup, work,
        lambda res: show_preview_image(popup, res[0], pid, backend=backend, source=res[1]),
        lambda err: popup._preview_status_var.set(f"Render error: {err}"),
        "_preview_is_rendering",
    )


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


def show_preview_image(popup: Any, img_bgr: Any, profile_id: str = "", backend: str = "2D", source: str = "") -> None:
    """Display a rendered BGR numpy array in the preview label."""
    try:
        popup._preview_full_pil, popup._preview_photo = _to_photo(popup, img_bgr, 300)
        popup._preview_img_label.configure(image=popup._preview_photo, text="", cursor="hand2")
        popup._preview_img_label.bind(
            "<Button-1>",
            lambda _evt: _open_zoom_image_popup(
                popup,
                getattr(popup, "_preview_full_pil", None),
                f"Filter Preview - {profile_id} ({backend})",
            ),
        )
        src = f"\n{source}" if source else ""
        popup._preview_status_var.set(f"Profile: {profile_id}  ·  defect: OK  ·  {backend}{src}")

    except ImportError:
        popup._preview_status_var.set("Pillow not installed.\nRun: pip install Pillow")
    except Exception as exc:
        popup._preview_status_var.set(f"Display error: {exc}")



# ---------------------------------------------------------------------------
# Left-column profile preview panel
# ---------------------------------------------------------------------------

def build_left_preview_panel(popup: Any, parent: ttk.Frame) -> None:
    """Build a compact preview panel below the filter-profiles listbox.

    The dropdown lists all real component profiles from configs/profiles/ (both
    2D and 3D).  Render always uses the currently active filter settings from
    the popup's filter_vars – not any stored profile dict.
    """
    lf = ttk.LabelFrame(parent, text="Component Preview", padding=6)
    lf.pack(fill="x", pady=(8, 0))

    sel_row = ttk.Frame(lf)
    sel_row.pack(fill="x", pady=(0, 4))
    ttk.Label(sel_row, text="Profile:").pack(side="left")
    popup._left_preview_profile_var = tk.StringVar()
    popup._left_preview_profile_combo = ttk.Combobox(
        sel_row,
        textvariable=popup._left_preview_profile_var,
        width=16,
        state="readonly",
    )
    popup._left_preview_profile_combo.pack(side="left", padx=(4, 0), fill="x", expand=True)
    popup._left_preview_profile_combo.bind(
        "<<ComboboxSelected>>", popup._on_left_preview_profile_selected
    )

    ttk.Button(lf, text="Render", command=popup._render_left_preview).pack(anchor="w", pady=(0, 4))

    popup._left_preview_status_var = tk.StringVar(value="")
    ttk.Label(
        lf,
        textvariable=popup._left_preview_status_var,
        style="Muted.TLabel",
        wraplength=170,
        justify="left",
    ).pack(anchor="w", pady=(0, 4))

    popup._left_preview_img_label = ttk.Label(lf, text="No preview yet", anchor="center")
    popup._left_preview_img_label.pack(fill="both", expand=True)

    popup._left_preview_is_rendering = False
    popup._left_preview_photo = None
    popup._left_preview_profiles_2d: set = set()
    popup._left_preview_profiles_3d: set = set()

    populate_left_filter_profiles(popup)


def populate_left_filter_profiles(popup: Any) -> None:
    """Scan configs/profiles/ and fill the left combo with all component profiles (2D + 3D)."""
    combo = getattr(popup, "_left_preview_profile_combo", None)
    if combo is None:
        return
    status = getattr(popup, "_left_preview_status_var", None)

    if not getattr(popup, "_sim_root", None):
        if status:
            status.set("Preview unavailable: sim_root not set.")
        combo["values"] = ()
        return

    try:
        import yaml as _yaml  # type: ignore
    except ImportError:
        if status:
            status.set("PyYAML not installed.")
        return

    profiles_dir = Path(popup._sim_root) / "configs" / "profiles"
    if not profiles_dir.exists():
        if status:
            status.set(f"profiles dir not found:\n{profiles_dir}")
        return

    vals_2d: list = []
    vals_3d: list = []
    for p in sorted(profiles_dir.glob("*.yaml")):
        try:
            data = _yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            backends = list((data.get("profile") or {}).get("supported_render_backends") or [])
            pid = p.stem
            if "opencv_2d" in backends:
                vals_2d.append(pid)
            if "blender_3d" in backends:
                vals_3d.append(pid)
        except Exception:
            continue

    popup._left_preview_profiles_2d = set(vals_2d)
    popup._left_preview_profiles_3d = set(vals_3d)

    combo_vals: list = []
    if vals_2d:
        combo_vals.append("── 2D ──")
        combo_vals.extend(vals_2d)
    if vals_3d:
        combo_vals.append("── 3D ──")
        combo_vals.extend(vals_3d)

    combo["values"] = combo_vals

    current = popup._left_preview_profile_var.get()
    if current not in combo_vals or current.startswith("──"):
        if vals_2d:
            popup._left_preview_profile_var.set(vals_2d[0])
        elif vals_3d:
            popup._left_preview_profile_var.set(vals_3d[0])

    if status:
        if combo_vals:
            status.set("Click 'Render' to preview with current filter settings.")
        else:
            status.set("No component profiles found.")


def on_left_preview_profile_selected(popup: Any, _evt: tk.Event | None = None) -> None:
    """Skip separator header items in the left combo."""
    v = popup._left_preview_profile_var.get()
    if not v.startswith("──"):
        return
    vals = list(popup._left_preview_profile_combo["values"])
    try:
        idx = vals.index(v)
        for nxt in vals[idx + 1:]:
            if not nxt.startswith("──"):
                popup._left_preview_profile_var.set(nxt)
                return
    except ValueError:
        pass
    popup._left_preview_profile_var.set("")




def render_left_preview(popup: Any) -> None:
    """Render the selected component profile with the CURRENT active filter settings."""
    if getattr(popup, "_left_preview_is_rendering", False):
        return

    pid = popup._left_preview_profile_var.get().strip()
    if not pid or pid.startswith("──"):
        popup._left_preview_status_var.set("Select a valid component profile first.")
        return

    # Always use the live filter settings currently shown in the UI controls.
    filter_vals = get_preview_filter_values(popup)

    profiles_2d = getattr(popup, "_left_preview_profiles_2d", set())
    profiles_3d = getattr(popup, "_left_preview_profiles_3d", set())
    is_3d_only = pid in profiles_3d and pid not in profiles_2d
    backend = "3D" if is_3d_only else "2D"
    popup._left_preview_status_var.set(f"Rendering ({backend})…")
    work = (lambda: do_render_3d_preview(popup, pid, filter_vals, "filter_preview_left")) if is_3d_only else (
        lambda: do_render_2d_preview(popup, pid, filter_vals))
    _run_in_background(
        popup, work,
        lambda res: _show_left_preview_image(popup, res[0], pid, backend),
        lambda err: popup._left_preview_status_var.set(f"Render error: {err}"),
        "_left_preview_is_rendering",
    )


def _show_left_preview_image(
    popup: Any, img_bgr: Any, profile_id: str, backend: str = "2D"
) -> None:
    """Display a rendered image in the left preview label."""
    try:
        popup._left_preview_full_pil, popup._left_preview_photo = _to_photo(popup, img_bgr, 180)
        popup._left_preview_img_label.configure(image=popup._left_preview_photo, text="", cursor="hand2")
        popup._left_preview_img_label.bind(
            "<Button-1>",
            lambda _evt: _open_zoom_image_popup(
                popup,
                getattr(popup, "_left_preview_full_pil", None),
                f"Component Preview - {profile_id} ({backend})",
            ),
        )
        popup._left_preview_status_var.set(f"Profile: {profile_id}  ·  {backend}")

    except ImportError:
        popup._left_preview_status_var.set("Pillow not installed.\nRun: pip install Pillow")
    except Exception as exc:
        popup._left_preview_status_var.set(f"Display error: {exc}")

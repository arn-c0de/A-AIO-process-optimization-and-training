"""Profile Preview tab: render a few images per defect state of one profile with chosen settings.

Lets you check how a profile looks (style, supersampling up to "Ultra", camera preset,
Blender quality) before generating a dataset. Rendering runs in a worker thread.
"""

from __future__ import annotations

import random
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from gui import dialogs
from gui.widgets.layout import keep_pane_visible
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from PIL import Image, ImageTk

from gui.components.overlay_renderer import draw_defect_overlay
from gui.core.tasks import DispatchQueue
from gui.tabs.core.base import BaseTab
from gui.utils.pipeline_helpers import BOARD_DRC_CHOICES, BOARD_ROUTING_CHOICES, BOARD_RULES_CHOICES, LOOK_CHOICES
from gui.utils.process import open_path
from gui.utils.settings_store import bind_persisted_vars
from simple_sim.preview.render import BACKEND_2D, BACKEND_3D, resolve_defect_types

from .logic import (
    CONFIG_AUTO,
    QUALITY_PRESETS,
    PreviewCancelled,
    PreviewImage,
    PreviewRequest,
    ProfilePreviewLogic,
    read_sidecar,
)
from gui.utils.tk_safe import quiet

from .ui import FROM_CONFIG, MAX_PER_STATE, QUALITY_CUSTOM, ProfilePreviewUIMixin

if TYPE_CHECKING:
    from gui.state import UiState



#: Refresh of the "... Ns elapsed" status while rendering (the results themselves are pushed).
_ELAPSED_TICK_MS = 500


class ProfilePreviewTab(ProfilePreviewUIMixin, BaseTab):
    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        super().__init__(parent, sim_root, state)
        self.logic = ProfilePreviewLogic(sim_root)
        self._profiles: dict[str, set[str]] = {}
        self._profile_data: dict[str, dict[str, Any]] = {}
        # Worker put()s (kind, payload); drained on the Tk thread via the dispatcher (no 100 ms poll).
        self._queue: DispatchQueue = DispatchQueue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._images: list[PreviewImage] = []
        self._thumb_refs: list[ImageTk.PhotoImage] = []
        self._cells: list[ttk.Frame] = []
        # Decoded preview images (path, mtime, overlay) -> PIL, so relayouts (resize, thumbnail size,
        # overlay toggle) do not re-read and re-decode every PNG.
        self._pil_cache: dict[tuple[str, int, bool], Image.Image] = {}
        self._poll_id: str | None = None
        self._applying_preset = False
        self._t_start = 0.0
        self._expected = 0
        self._layout_cols = 0
        self._last_image_s = 0.0

    # ------------------------------------------------------------------ UI

    def build_ui(self) -> None:
        self._create_vars()

        paned = ttk.PanedWindow(self.frame, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=(0, 0, 8, 0))
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        keep_pane_visible(paned, 0, left, min_width=300)  # settings column must never collapse

        self._build_settings(left)
        self._build_gallery(right)

        self._load_profiles()
        bind_persisted_vars(
            self.state.settings_store,
            [
                ("profile_preview.profile", self.var_profile),
                ("profile_preview.quality", self.var_quality),
                ("profile_preview.supersampling", self.var_ss),
                ("profile_preview.style", self.var_style),
                ("profile_preview.camera", self.var_camera),
                ("profile_preview.samples", self.var_samples),
                ("profile_preview.engine", self.var_engine),
                ("profile_preview.device", self.var_device),
                ("profile_preview.look", self.var_look),
                ("profile_preview.board_routing", self.var_board_routing),
                ("profile_preview.board_drc", self.var_board_drc),
                ("profile_preview.board_rules", self.var_board_rules),
                ("profile_preview.variants", self.var_variants),
                ("profile_preview.seed", self.var_seed),
                ("profile_preview.cardinal", self.var_cardinal),
                ("profile_preview.filters", self.var_filters),
                ("profile_preview.thumb", self.var_thumb),
                ("profile_preview.overlay", self.var_overlay),
            ],
            self.frame,
            skip_empty_strings=True,
        )
        if self.var_profile.get() not in self._profiles and self._profiles:
            self.var_profile.set(next(iter(self._profiles)))
        self._on_profile_changed()
        self.var_quality.trace_add("write", lambda *_: self._apply_quality())
        for var in (self.var_ss, self.var_samples):
            var.trace_add("write", lambda *_: self._mark_custom())
        self._show_existing()

    # ------------------------------------------------------------------ settings logic

    def _load_profiles(self) -> None:
        self._profiles.clear()
        self._profile_data.clear()
        for pid, prof, backends in self.logic.profiles():
            self._profiles[pid] = backends
            self._profile_data[pid] = prof
        self.cb_profile.configure(values=list(self._profiles))

    def _on_profile_changed(self) -> None:
        pid = self.var_profile.get()
        backends = [b for b in (BACKEND_2D, BACKEND_3D) if b in self._profiles.get(pid, set())]
        self.cb_backend.configure(values=backends)
        if self.var_backend.get() not in backends:
            self.var_backend.set(backends[0] if backends else "")
        self.lb_states.delete(0, "end")
        for s in resolve_defect_types(self._profile_data.get(pid, {}), ""):
            self.lb_states.insert("end", s)
        self.lb_states.selection_set(0, "end")
        self._update_total()
        self._on_backend_changed()

    def _on_backend_changed(self) -> None:
        names = self.logic.configs_for(self.var_profile.get(), self.var_backend.get()) if self.var_backend.get() else []
        self.cb_config.configure(values=[CONFIG_AUTO, *names])
        if self.var_config.get() not in (CONFIG_AUTO, *names):
            self.var_config.set(CONFIG_AUTO)
        is_3d = self.var_backend.get() == BACKEND_3D
        self.cb_style.configure(state="disabled" if is_3d else "readonly")
        for w in (self.cb_engine, self.cb_device):
            w.configure(state="readonly" if is_3d else "disabled")
        self.sp_samples.configure(state="normal" if is_3d else "disabled")
        self._show_existing()

    def _update_total(self) -> None:
        """'= N images total' next to the per-state count."""
        try:
            per_state = max(1, min(MAX_PER_STATE, int(self.var_variants.get())))
        except (tk.TclError, ValueError):
            self.var_total.set("")
            return
        n_states = len(self.lb_states.curselection())
        self.var_total.set(f"x {n_states} states = {per_state * n_states} images")

    def _apply_quality(self) -> None:
        preset = QUALITY_PRESETS.get(self.var_quality.get())
        if preset is None:
            return
        self._applying_preset = True
        try:
            self.var_ss.set(preset["supersampling"])
            self.var_samples.set(preset["samples"])
        finally:
            self._applying_preset = False

    def _mark_custom(self) -> None:
        if self._applying_preset:
            return
        preset = QUALITY_PRESETS.get(self.var_quality.get())
        try:
            current = {"supersampling": int(self.var_ss.get()), "samples": int(self.var_samples.get())}
        except (tk.TclError, ValueError):
            return
        if preset is not None and preset != current:
            match = next((k for k, v in QUALITY_PRESETS.items() if v == current), QUALITY_CUSTOM)
            self.var_quality.set(match)

    def _new_seed(self) -> None:
        self.var_seed.set(str(random.randint(0, 999_999)))

    def _request(self) -> PreviewRequest:
        states = [self.lb_states.get(i) for i in self.lb_states.curselection()]
        if not states:
            raise ValueError("Select at least one defect state.")
        try:
            seed = int(self.var_seed.get().strip())
            variants = max(1, min(MAX_PER_STATE, int(self.var_variants.get())))
            samples = max(1, int(self.var_samples.get()))
        except (ValueError, tk.TclError) as e:
            raise ValueError("Seed, variants and samples must be integers.") from e
        filters = None
        if self.var_filters.get():
            from simple_sim.filter_profiles import load_shared_filter_profiles, shared_settings_path

            if self.state.settings_store is not None:
                self.state.settings_store.save()
            _p, _a, filters = load_shared_filter_profiles(shared_settings_path(self.sim_root))
        return PreviewRequest(
            profile_id=self.var_profile.get(),
            backend=self.var_backend.get(),
            config_name=self.var_config.get(),
            states=states,
            variants=variants,
            seed=seed,
            style=self.var_style.get(),
            supersampling=int(self.var_ss.get()),
            camera=self.var_camera.get(),
            use_filters=bool(self.var_filters.get()),
            image_filters=filters,
            cardinal_rotation=bool(self.var_cardinal.get()),
            blender_samples=samples,
            blender_engine="" if self.var_engine.get() == FROM_CONFIG else self.var_engine.get(),
            blender_device="" if self.var_device.get() == FROM_CONFIG else self.var_device.get(),
            look="" if self.var_look.get() == LOOK_CHOICES[0] else self.var_look.get(),
            board_routing="" if self.var_board_routing.get() == BOARD_ROUTING_CHOICES[0] else self.var_board_routing.get(),
            board_drc="" if self.var_board_drc.get() == BOARD_DRC_CHOICES[0] else self.var_board_drc.get(),
            board_rules="" if self.var_board_rules.get() == BOARD_RULES_CHOICES[0] else self.var_board_rules.get(),
        )

    # ------------------------------------------------------------------ rendering

    def _start_render(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            req = self._request()
        except ValueError as e:
            dialogs.show_error(self.frame, "Profile Preview", str(e))
            return
        self._stop.clear()
        self._images = []
        self._relayout()
        self._expected = len(req.states) * req.variants
        self.progress.configure(maximum=self._expected, value=0)
        self.btn_render.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self._t_start = time.perf_counter()
        self._last_image_s = 0.0
        self.var_status.set(f"Rendering {self._expected} images...")
        self._log_clear()

        def run() -> None:
            try:
                self.logic.render(
                    req,
                    on_image=lambda im: self._queue.put(("image", im)),
                    on_log=lambda s: self._queue.put(("log", s)),
                    should_stop=self._stop.is_set,
                )
                self._queue.put(("done", None))
            except PreviewCancelled:
                self._queue.put(("cancelled", None))
            except Exception as e:  # noqa: BLE001 - shown in the GUI
                self._queue.put(("error", f"{type(e).__name__}: {e}"))

        self.bind_queue(self._queue, self._on_render_items)
        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()
        self._tick_elapsed()

    def _request_stop(self) -> None:
        """Stop takes effect between images; a running Blender render/board bake finishes first."""
        self._stop.set()
        self.btn_stop.configure(state="disabled")
        self.var_status.set("Stopping ... finishing the image/bake in progress")

    def _on_render_items(self, items: list[tuple[str, Any]]) -> None:
        """Worker results, in order, on the Tk thread (``DispatchQueue`` handler)."""
        for kind, payload in items:
            if kind == "image":
                self._images.append(payload)
                n_cells = len(self._cells)
                self._add_thumb(len(self._images) - 1, payload)
                if len(self._cells) == n_cells:
                    self._log(f"[warn] {payload.state} #{payload.variant}: image missing or unreadable "
                              f"({payload.path})")
                self.progress.configure(value=len(self._images))
                self._last_image_s = payload.seconds
            elif kind == "log":
                self._log(payload)
            else:
                self._finish(kind, payload)
                return
        if self._poll_id is not None:
            self._show_elapsed()

    def _show_elapsed(self) -> None:
        # Live elapsed time, so a slow first image (3D: board bake + Blender start) does not look frozen.
        elapsed = time.perf_counter() - self._t_start
        last = f" | last image {self._last_image_s:.1f}s" if self._last_image_s else ""
        if self._stop.is_set():
            self.var_status.set(f"Stopping ... finishing the image/bake in progress ({elapsed:.0f}s elapsed)")
        else:
            self.var_status.set(f"Rendering image {min(len(self._images) + 1, self._expected)}/{self._expected} "
                                f"... {elapsed:.0f}s elapsed{last}")

    def _tick_elapsed(self) -> None:
        """Clock only (no queue work): results arrive through ``_on_render_items``."""
        self._show_elapsed()
        self._poll_id = self.frame.after(_ELAPSED_TICK_MS, self._tick_elapsed)

    def _finish(self, kind: str, payload: Any) -> None:
        if self._poll_id is not None:
            with quiet():
                self.frame.after_cancel(self._poll_id)
        self._poll_id = None
        self.btn_render.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        elapsed = time.perf_counter() - self._t_start
        if kind == "done":
            missing = len(self._images) - len(self._cells)
            note = f" ({missing} missing/unreadable, see log)" if missing > 0 else ""
            self.var_status.set(f"Done: {len(self._cells)} images in {elapsed:.1f}s.{note}")
        elif kind == "cancelled":
            self.var_status.set(f"Stopped after {len(self._images)} images.")
        else:
            self.var_status.set("Render failed, see log.")
            self._log(str(payload))
            dialogs.show_error(self.frame, "Profile Preview", str(payload))

    # ------------------------------------------------------------------ gallery

    def _show_existing(self) -> None:
        """Show images of the last render of the selected profile/backend (if any)."""
        if self._worker is not None and self._worker.is_alive():
            return
        out_dir = self.logic.output_dir(self.var_profile.get(), self.var_backend.get())
        order = {s: i for i, s in enumerate(self.lb_states.get(0, "end"))}
        images = []
        for p in out_dir.glob("*_v*.png"):
            state, _, v = p.stem.rpartition("_v")
            if v.isdigit():
                side = read_sidecar(p)
                images.append(PreviewImage(state, int(v), p, 0.0, str(side.get("info") or ""),
                                           side.get("defect"), side.get("nominal")))
        images.sort(key=lambda im: (order.get(im.state, len(order)), im.state, im.variant))
        self._images = images
        self._relayout()

    def _columns(self) -> int:
        cell = int(self.var_thumb.get()) + 16
        return max(1, self.canvas.winfo_width() // cell)

    def _on_canvas_resize(self) -> None:
        cols = self._columns()
        if cols != self._layout_cols:
            # Only the column count changed: move the existing cells instead of rebuilding them.
            self._layout_cols = cols
            for i, cell in enumerate(self._cells):
                cell.grid_configure(row=i // cols, column=i % cols)

    def _relayout(self) -> None:
        self._layout_cols = self._columns()
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self._thumb_refs.clear()
        self._cells.clear()
        for i, im in enumerate(self._images):
            self._add_thumb(i, im)
        self.canvas.yview_moveto(0.0)

    def _load(self, im: PreviewImage) -> Image.Image | None:
        """Preview image as PIL RGB, with the defect overlay if enabled (cached per file version)."""
        overlay = bool(self.var_overlay.get() and im.defect and im.nominal)
        try:
            key = (str(im.path), im.path.stat().st_mtime_ns, overlay)
        except OSError:
            return None
        pil = self._pil_cache.get(key)
        if pil is not None:
            return pil
        img = cv2.imread(str(im.path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        if overlay:
            try:
                img = draw_defect_overlay(img, im.defect, im.nominal)
            except Exception:  # noqa: BLE001 - overlay is cosmetic
                pass
        pil = Image.fromarray(np.ascontiguousarray(img[..., ::-1]))
        if len(self._pil_cache) >= 4 * MAX_PER_STATE:
            self._pil_cache.clear()
        self._pil_cache[key] = pil
        return pil

    def _add_thumb(self, index: int, im: PreviewImage) -> None:
        size = int(self.var_thumb.get())
        pil = self._load(im)
        if pil is None:  # file removed meanwhile
            return
        resample = Image.Resampling.NEAREST if size >= pil.width * 2 else Image.Resampling.LANCZOS
        thumb = pil.resize((size, round(size * pil.height / pil.width)), resample)
        photo = ImageTk.PhotoImage(thumb)
        self._thumb_refs.append(photo)
        cols = self._layout_cols or self._columns()
        cell = ttk.Frame(self.grid_frame, padding=4)
        pos = len(self._cells)  # not *index*: skipped (unreadable) images leave no gap
        cell.grid(row=pos // cols, column=pos % cols, sticky="n")
        self._cells.append(cell)
        lbl = ttk.Label(cell, image=photo, cursor="hand2")
        lbl.pack()
        lbl.bind("<Button-1>", lambda _e, i=im: self._open_full(i))
        caption = f"{im.state}  #{im.variant}"
        if im.info:
            caption += f"\n{im.info}"
        ttk.Label(cell, text=caption, justify="center", wraplength=size).pack()

    def _open_full(self, im: PreviewImage) -> None:
        path = im.path
        pil = self._load(im)
        if pil is None:
            dialogs.show_error(self.frame, "Profile Preview", f"Cannot read {path}")
            return
        top = tk.Toplevel(self.frame)
        top.title(path.name)
        var_zoom = tk.IntVar(value=3 if pil.width <= 300 else 1)
        bar = ttk.Frame(top, padding=4)
        bar.pack(fill="x")
        ttk.Label(bar, text=f"{path.name}  ({pil.width}x{pil.height})  Zoom").pack(side="left")
        label = ttk.Label(top)
        label.pack()

        def show() -> None:
            z = int(var_zoom.get())
            photo = ImageTk.PhotoImage(pil.resize((pil.width * z, pil.height * z), Image.Resampling.NEAREST))
            label.configure(image=photo)
            label.image = photo  # keep a reference

        for z in (1, 2, 3, 4):
            ttk.Radiobutton(bar, text=f"{z}x", value=z, variable=var_zoom, command=show).pack(side="left", padx=2)
        show()

    def _on_wheel(self, event: tk.Event) -> None:
        # Bound application-wide: scroll only when the pointer is over the gallery
        # canvas or one of its descendants (not the log, lists or settings).
        if not self.initialized:
            return
        try:
            w = self.canvas.winfo_containing(event.x_root, event.y_root)
        except Exception:  # noqa: BLE001 - e.g. pointer over a destroyed/foreign window
            return
        while w is not None and w is not self.canvas:
            w = getattr(w, "master", None)
        if w is None:
            return
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")

    def _open_folder(self) -> None:
        out_dir = self.logic.output_dir(self.var_profile.get(), self.var_backend.get())
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            open_path(out_dir)
        except Exception as e:  # noqa: BLE001
            dialogs.show_error(self.frame, "Profile Preview", f"{e}\n{out_dir}")

    # ------------------------------------------------------------------ log

    def _log(self, s: str) -> None:
        self.log.append(s.rstrip() + "\n")

    def _log_clear(self) -> None:
        self.log.clear()

    # ------------------------------------------------------------------ lifecycle

    def on_close(self) -> None:
        self._stop.set()
        # The worker terminates a running Blender on its next stop poll (0.2 s); it is a daemon thread,
        # so without this short wait the interpreter exits first and Blender renders on as an orphan.
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self._poll_id is not None:
            with quiet():
                self.frame.after_cancel(self._poll_id)

"""Smoke test: build the monitor and every tab on a sandbox root (needs tkinter + a display)."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

tk = pytest.importorskip("tkinter")
pytest.importorskip("PIL")


def _display_available() -> bool:
    if not os.environ.get("DISPLAY"):
        return False
    try:
        r = tk.Tk()
        r.destroy()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def xdisplay():
    """Prefer a private headless Xvfb (no windows on the desktop); fall back to $DISPLAY."""
    xvfb = shutil.which("Xvfb")
    old = os.environ.get("DISPLAY")
    proc = None
    if xvfb:
        r, w = os.pipe()
        proc = subprocess.Popen([xvfb, "-displayfd", str(w), "-screen", "0", "1600x1000x24", "-nolisten", "tcp"],
                                pass_fds=(w,), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(w)
        import select

        num = b""
        if select.select([r], [], [], 10)[0]:
            num = os.read(r, 16).strip()
        os.close(r)
        if num:
            os.environ["DISPLAY"] = f":{num.decode()}"
        else:
            proc.terminate()
            proc = None
    try:
        if not _display_available():
            if old is not None:
                os.environ["DISPLAY"] = old
            if not _display_available():
                pytest.skip("no usable X display (install Xvfb or run inside a desktop session)")
        yield
    finally:
        if proc is not None:
            proc.terminate()
        if old is None:
            os.environ.pop("DISPLAY", None)
        else:
            os.environ["DISPLAY"] = old


def _all_widgets(w):
    yield w
    for c in w.winfo_children():
        yield from _all_widgets(c)


def test_monitor_builds_every_tab(xdisplay, tmp_path):
    from gui.devtools.sandbox import make_sandbox_root
    from gui.monitor import MonitorAppTabbed

    root_dir = make_sandbox_root(tmp_path / "sim")
    root = tk.Tk()
    try:
        app = MonitorAppTabbed(root, sim_root=root_dir)
        for key, tab in app.tabs.items():
            tab.on_activate()
            assert tab.initialized, key
            root.update_idletasks()
        root.update()
        # Every button must explain itself (tooltip with a help text).
        from tkinter import ttk

        missing = []
        # Tabs not yet migrated to the help system (see docs/refactoring/GUI_MODERNIZATION_PLAN.md).
        tooltip_todo = {"profile_preview"}
        for key, tab in app.tabs.items():
            if key in tooltip_todo:
                continue
            for w in _all_widgets(tab.frame):
                if isinstance(w, (ttk.Button, ttk.Menubutton)) and str(w.cget("state")) != "hidden":
                    tip = getattr(w, "_simsim_tooltip", None)
                    if tip is None or not tip.current_text():
                        missing.append(f"{key}: {w.cget('text')!r}")
        assert not missing, "buttons without tooltip:\n" + "\n".join(missing)
        # Theme switch must not raise.
        app._toggle_theme()
        app._toggle_theme()
        root.update()
        # Full shutdown path (tabs, tasks, stats sampler, hover sink, root.destroy()).
        app._on_close()
        from gui.utils import tooltip as _tooltip

        assert _tooltip._hint_sink is None
        assert not app.ctx.tasks.active
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass  # already destroyed by _on_close

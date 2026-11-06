#!/usr/bin/env python3
"""Simple-Sim monitor: multi-tab Tk GUI for the generate -> train -> eval pipeline.

The shell owns the window, theme, menu, keyboard shortcuts, status bar and the
``AppContext`` (event bus, task runner). Tabs are registered in
``gui/tabs/core/registry.py`` and built lazily on first activation.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import tkinter as tk
    from tkinter import ttk
except Exception as e:
    raise SystemExit(
        "Tkinter is not available. On Ubuntu/Debian install it with:\n"
        "  sudo apt-get install python3-tk\n"
        f"Original error: {e}"
    ) from e

try:
    import PIL  # noqa: F401  (dependency check: thumbnails need Pillow)
except Exception as e:
    raise SystemExit(
        "Pillow is required for thumbnails. Install dependencies:\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        f"Original error: {e}"
    ) from e

from gui.core.context import AppContext
from gui.core.events import Topics
from gui.help.texts import QUICK_START, SHORTCUTS
from gui.state import UiState
from gui.tabs import BaseTab
from gui.tabs.core.registry import TabRegistry
from gui.theme.style import apply_scaling, apply_theme, ui_scale_from_env
from gui.theme.tokens import SPACE
from gui.utils.settings_store import SettingsStore
from gui.utils.system_stats import SystemStatsSampler
from gui.utils.tooltip import set_hover_hint_sink
from gui.widgets.status import StatusBar

APP_TITLE = "Simple-Sim Monitor"
APP_VERSION = "1.0"
DEFAULT_GEOMETRY = "1440x900"
UI_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0)


class MonitorAppTabbed:
    """Multi-tab monitor application with lazy loading."""

    def __init__(self, root: tk.Tk, sim_root: Path) -> None:
        self.root = root
        self.sim_root = sim_root

        # Shared state across all tabs
        self.state = UiState()

        self.tabs: dict[str, BaseTab] = {}
        self._tab_order: list[str] = []
        self.current_tab: str = ""
        self._perf_tick_id: str | None = None
        self._stats = SystemStatsSampler()

        self._build_ui()

    # ------------------------------------------------------------------ setup
    def _build_ui(self) -> None:
        self.root.title(f"{APP_TITLE} {APP_VERSION}")
        settings_path = self.sim_root / "outputs" / "gui" / "settings.json"
        store = SettingsStore(settings_path)
        store.load()
        self.state.settings_store = store

        # Services
        self.ctx = AppContext(sim_root=self.sim_root, settings=store)
        self.state.ctx = self.ctx
        self.ctx.dispatcher.attach(self.root)

        # Scaling must happen before widgets/fonts are created.
        scale = ui_scale_from_env()
        if scale is None:
            saved = store.get("app.ui_scale")
            scale = float(saved) if isinstance(saved, (int, float)) and saved else None
        apply_scaling(self.root, scale)
        self.var_dark = tk.BooleanVar(value=store.get("app.theme") == "dark")
        apply_theme(self.root, "dark" if self.var_dark.get() else "light")

        geom = store.get("app.geometry")
        try:
            self.root.geometry(geom if isinstance(geom, str) and geom else DEFAULT_GEOMETRY)
        except Exception:
            self.root.geometry(DEFAULT_GEOMETRY)
        self.root.minsize(1100, 680)

        self._build_menu()

        # Status bar first (bottom), then the notebook fills the rest.
        self.statusbar = StatusBar(self.root)
        self.statusbar.pack(side="bottom", fill="x")
        ttk.Separator(self.root, orient="horizontal").pack(side="bottom", fill="x")
        set_hover_hint_sink(self.statusbar.hint)
        bus = self.ctx.bus
        bus.subscribe(Topics.STATUS, lambda text, level="info": self.statusbar.show(text, level))
        bus.subscribe(Topics.TASK_STARTED, lambda task: self.statusbar.task_started(task))
        bus.subscribe(Topics.TASK_PROGRESS, lambda task: self.statusbar.task_progress(task))
        bus.subscribe(Topics.TASK_FINISHED, lambda task: self.statusbar.task_finished(task))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True)
        self._build_sidebar(body)
        ttk.Separator(body, orient="vertical").pack(side="left", fill="y")
        content = ttk.Frame(body, padding=(SPACE.L, SPACE.M, SPACE.L, SPACE.S))
        content.pack(side="left", fill="both", expand=True)
        header = ttk.Frame(content)
        header.pack(fill="x", pady=(0, SPACE.M))
        self.var_page_title = tk.StringVar(value="")
        self.var_page_desc = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.var_page_title, style="Title.TLabel").pack(side="left")
        ttk.Label(header, textvariable=self.var_page_desc, style="PageDesc.TLabel").pack(
            side="left", padx=(SPACE.L, 0))
        # Pages: a notebook whose tab row is hidden; the sidebar selects pages.
        self.notebook = ttk.Notebook(content, style="Pages.TNotebook")
        self.notebook.pack(fill="both", expand=True)
        self.notebook.enable_traversal()

        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        # Legacy virtual events from tabs are forwarded to the bus and to the tabs.
        self.notebook.bind("<<DatasetChanged>>", lambda _e: self._notify_dataset_changed())
        self.notebook.bind("<<DatasetCatalogChanged>>", lambda _e: self._notify_dataset_catalog_changed())

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._bind_shortcuts()

        self._tab_order = TabRegistry.keys()
        for spec in TabRegistry.iter_specs():
            tab = spec.cls(self.notebook, self.sim_root, self.state)
            self.tabs[spec.key] = tab
            self.notebook.add(tab.frame, text=spec.title)

        last = store.get("app.last_tab")
        first = last if isinstance(last, str) and last in self.tabs else "pipeline"
        self.current_tab = first
        self.tabs[first].on_activate()
        self._sync_page_header(first)
        if first != self._tab_order[0]:
            self.notebook.select(self._tab_order.index(first))
        self._tick_global_performance()
        self.root.update_idletasks()

        if not store.get("app.welcome_shown"):
            store.set("app.welcome_shown", True)
            store.schedule_save(self.root)
            self.root.after(400, self.show_quick_start)
        else:
            self.statusbar.show("Ready – press F1 for help", "info")

    def _build_sidebar(self, parent: tk.Misc) -> None:
        from gui.utils.tooltip import ToolTip

        bar = ttk.Frame(parent, style="Sidebar.TFrame", padding=(SPACE.S, SPACE.M, SPACE.S, SPACE.M))
        bar.pack(side="left", fill="y")
        ttk.Label(bar, text="Simple-Sim", style="Brand.TLabel", padding=(SPACE.M, 0)).pack(anchor="w")
        self.var_page = tk.StringVar(value="")
        self._nav_buttons: dict[str, ttk.Radiobutton] = {}
        self._nav_order: list[str] = []
        keys = TabRegistry.keys()
        for group, specs in TabRegistry.grouped():
            ttk.Label(bar, text=group, style="SidebarHeading.TLabel").pack(fill="x")
            for spec in specs:
                idx = keys.index(spec.key)
                pos = len(self._nav_order)
                self._nav_order.append(spec.key)
                btn = ttk.Radiobutton(bar, text=spec.title, value=spec.key, variable=self.var_page,
                                      style="Nav.Toolbutton", command=lambda i=idx: self._select_tab(i))
                btn.pack(fill="x")
                shortcut = f"  (Ctrl+{pos + 1})" if pos < 9 else ""
                ToolTip(btn, text=f"{spec.description}{shortcut}")
                self._nav_buttons[spec.key] = btn
        bottom = ttk.Frame(bar, style="Sidebar.TFrame")
        bottom.pack(side="bottom", fill="x")
        theme_btn = ttk.Checkbutton(bottom, text="Dark Mode", variable=self.var_dark, style="Nav.Toolbutton",
                                    command=self._apply_theme_from_var)
        theme_btn.pack(fill="x")
        ToolTip(theme_btn, text="Switch between light and dark theme (Ctrl+Shift+D).")
        help_btn = ttk.Button(bottom, text="Help & Shortcuts", style="Nav.Toolbutton", command=self.show_quick_start)
        help_btn.pack(fill="x", pady=(1, 0))
        ToolTip(help_btn, text="Quick start and keyboard shortcuts (F1).")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Open Outputs Folder", command=lambda: self._open_path(self.sim_root / "outputs"))
        m_file.add_separator()
        m_file.add_command(label="Quit", accelerator="Ctrl+Q", command=self._on_close)
        menubar.add_cascade(label="File", menu=m_file)

        m_view = tk.Menu(menubar, tearoff=False)
        m_view.add_checkbutton(label="Dark Mode", variable=self.var_dark, accelerator="Ctrl+Shift+D",
                               command=self._apply_theme_from_var)
        m_scale = tk.Menu(m_view, tearoff=False)
        cur = self.state.settings_store.get("app.ui_scale") if self.state.settings_store else None
        self.var_scale = tk.DoubleVar(value=float(cur) if isinstance(cur, (int, float)) and cur else 0.0)
        m_scale.add_radiobutton(label="Automatic", value=0.0, variable=self.var_scale, command=self._save_scale)
        for s in UI_SCALES:
            m_scale.add_radiobutton(label=f"{int(s * 100)} %", value=s, variable=self.var_scale, command=self._save_scale)
        m_view.add_cascade(label="UI Scale (restart)", menu=m_scale)
        m_view.add_separator()
        m_view.add_command(label="Refresh Current Tab", accelerator="Ctrl+R", command=self._refresh_current)
        menubar.add_cascade(label="View", menu=m_view)

        m_help = tk.Menu(menubar, tearoff=False)
        m_help.add_command(label="Quick Start", accelerator="F1", command=self.show_quick_start)
        m_help.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        m_help.add_separator()
        m_help.add_command(label="Open README", command=lambda: self._open_path(self.sim_root / "README.md"))
        m_help.add_command(label="Open Documentation", command=lambda: self._open_path(self.sim_root / "docs"))
        m_help.add_separator()
        m_help.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=m_help)

        self.root.configure(menu=menubar)
        self._menubar = menubar

    def _bind_shortcuts(self) -> None:
        r = self.root
        r.bind_all("<F1>", lambda _e: self.show_quick_start())
        r.bind_all("<Control-r>", lambda _e: self._refresh_current())
        r.bind_all("<Control-q>", lambda _e: self._on_close())
        r.bind_all("<Control-D>", lambda _e: self._toggle_theme())  # Ctrl+Shift+D
        r.bind_all("<F5>", lambda _e: self._pipeline_button("btn_start"))
        r.bind_all("<Shift-F5>", lambda _e: self._pipeline_button("btn_stop"))
        for i in range(1, 10):
            r.bind_all(f"<Control-Key-{i}>", lambda _e, pos=i - 1: self._select_nav(pos))

    # ---------------------------------------------------------------- actions
    def _select_tab(self, idx: int) -> None:
        if 0 <= idx < len(self._tab_order):
            self.notebook.select(idx)

    def _select_nav(self, pos: int) -> None:
        """Select the page at sidebar position ``pos`` (Ctrl+1 = first sidebar entry)."""
        if 0 <= pos < len(self._nav_order):
            self._select_tab(self._tab_order.index(self._nav_order[pos]))

    def _refresh_current(self) -> None:
        tab = self.tabs.get(self.current_tab)
        if tab is not None and tab.initialized:
            try:
                tab.refresh()
                self.statusbar.show("Refreshed", "info", timeout_ms=2000)
            except Exception as exc:
                self.statusbar.show(f"Refresh failed: {exc}", "error")

    def _pipeline_button(self, name: str) -> None:
        if self.current_tab != "pipeline":
            return
        ui = getattr(self.tabs["pipeline"], "ui", None)
        btn = getattr(ui, name, None)
        if btn is not None:
            try:
                btn.invoke()  # no-op while disabled
            except Exception:
                pass

    def _toggle_theme(self) -> None:
        self.var_dark.set(not self.var_dark.get())
        self._apply_theme_from_var()

    def _apply_theme_from_var(self) -> None:
        mode = "dark" if self.var_dark.get() else "light"
        apply_theme(self.root, mode)
        self.statusbar.retheme()
        if self.state.settings_store is not None:
            self.state.settings_store.set("app.theme", mode)
            self.state.settings_store.schedule_save(self.root)
        self.ctx.bus.publish(Topics.THEME_CHANGED, mode=mode)

    def _save_scale(self) -> None:
        val = float(self.var_scale.get())
        if self.state.settings_store is not None:
            self.state.settings_store.set("app.ui_scale", val or None)
            self.state.settings_store.schedule_save(self.root)
        self.statusbar.show("UI scale is applied after a restart.", "info")

    def _open_path(self, path: Path) -> None:
        import subprocess

        opener = "xdg-open" if sys.platform.startswith("linux") else ("open" if sys.platform == "darwin" else None)
        try:
            if opener is None:
                import os

                os.startfile(str(path))  # type: ignore[attr-defined]
            else:
                subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            self.statusbar.show(f"Cannot open {path}: {exc}", "error")

    def show_quick_start(self) -> None:
        from gui.dialogs import show_text

        show_text(self.root, "Quick Start", QUICK_START, mono=False, size=(560, 380))

    def show_shortcuts(self) -> None:
        from gui.dialogs import show_text

        width = max(len(k) for k, _ in SHORTCUTS) + 4
        text = "\n".join(f"{k.ljust(width)}{v}" for k, v in SHORTCUTS)
        show_text(self.root, "Keyboard Shortcuts", text, mono=True, size=(560, 340))

    def show_about(self) -> None:
        from gui.dialogs import show_info

        show_info(self.root, "About", f"{APP_TITLE} {APP_VERSION}\n\nSynthetic PCB-defect data generation, "
                                      f"training and evaluation.\n\nRoot: {self.sim_root}")

    # -------------------------------------------------------------- lifecycle
    def _on_close(self) -> None:
        try:
            self._close_tabs()
            try:
                self.ctx.tasks.cancel_all()
                self.ctx.dispatcher.detach()
            except Exception:
                pass
            try:
                self._stats.stop()
            except Exception:
                pass
            if self._perf_tick_id is not None:
                try:
                    self.root.after_cancel(self._perf_tick_id)
                except Exception:
                    pass
                self._perf_tick_id = None
            if self.state.settings_store is not None:
                try:
                    self.state.settings_store.set("app.geometry", self.root.winfo_geometry())
                    self.state.settings_store.set("app.last_tab", self.current_tab)
                except Exception:
                    pass
                self.state.settings_store.save()
        finally:
            set_hover_hint_sink(None)
            try:
                self.root.destroy()
            except Exception:
                pass

    def _close_tabs(self) -> None:
        """Give every instantiated tab a chance to stop its background work.

        Tabs that override ``BaseTab.on_close`` handle this themselves. For tabs
        that don't (e.g. the pipeline tab), fall back to a stop method on their
        ``logic`` object so subprocesses started with ``start_new_session=True``
        don't outlive the window.
        """
        for tab in self.tabs.values():
            try:
                tab.on_close()
            except Exception:
                pass
            if type(tab).on_close is not BaseTab.on_close:
                continue
            logic = getattr(tab, "logic", None)
            if logic is None:
                continue
            for name in ("stop_pipeline", "stop"):
                stop = getattr(logic, name, None)
                if callable(stop):
                    try:
                        stop()
                    except Exception:
                        pass
                    break

    def _tick_global_performance(self) -> None:
        parts = []
        cpu_pct = self._stats.cpu_percent()
        gpu_pct = self._stats.gpu_percent()
        ram_info = self._stats.ram_info()
        parts.append(f"CPU {cpu_pct:3.0f}%" if cpu_pct is not None else "CPU –")
        if gpu_pct is not None:
            parts.append(f"GPU {gpu_pct:3.0f}%")
        parts.append(f"RAM {ram_info[0]:3.0f}%" if ram_info is not None else "RAM n/a")
        self.statusbar.set_stats("   ".join(parts))
        self._perf_tick_id = self.root.after(1000, self._tick_global_performance)

    def _on_tab_changed(self, _event) -> None:
        selected_idx = self.notebook.index(self.notebook.select())
        if selected_idx < len(self._tab_order):
            tab_name = self._tab_order[selected_idx]
            self.current_tab = tab_name
            self._sync_page_header(tab_name)
            if tab_name in self.tabs:
                self.tabs[tab_name].on_activate()

    def _sync_page_header(self, key: str) -> None:
        spec = next((s for s in TabRegistry.iter_specs() if s.key == key), None)
        if spec is None:
            return
        self.var_page.set(key)
        self.var_page_title.set(spec.title)
        self.var_page_desc.set(spec.description)
        self.root.title(f"{spec.title} \u2013 {APP_TITLE}")

    def _notify_dataset_changed(self) -> None:
        """Notify all tabs (and bus subscribers) that the dataset has changed."""
        for tab in self.tabs.values():
            if tab.initialized:
                try:
                    tab.on_dataset_changed()
                except Exception:
                    pass
        self.ctx.bus.publish(Topics.DATASET_CHANGED, path=self.state.dataset_dir)

    def _notify_dataset_catalog_changed(self) -> None:
        """Notify initialized tabs that dataset catalog metadata changed."""
        for tab in self.tabs.values():
            if tab.initialized:
                try:
                    tab.refresh()
                except Exception:
                    pass
        self.ctx.bus.publish(Topics.DATASET_CATALOG_CHANGED)


def main() -> None:
    """Main entry point for tabbed monitor."""
    sim_root = Path(__file__).resolve().parent.parent
    root = tk.Tk(className="SimpleSim")
    MonitorAppTabbed(root, sim_root=sim_root)
    root.mainloop()


if __name__ == "__main__":
    main()

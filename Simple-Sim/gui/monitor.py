#!/usr/bin/env python3
"""Multi-tab Tkinter GUI for Simple-Sim pipeline monitoring and analysis.

Professional GUI with:
- Tab 1: Pipeline Control - Enhanced pipeline monitoring
- Tab 2: Analysis - Interactive image browser with defect overlays
- Tab 3: Predictions - Batch predict/evaluate (predict.sh)
- Tab 4: Weights - Model versioning/backup/export + comparisons
- Tab 5: Validation - Advanced dataset testing with automated flagging
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

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
    )

try:
    from PIL import Image, ImageTk
except Exception as e:
    raise SystemExit(
        "Pillow is required for thumbnails. Install dependencies:\n"
        "  .venv/bin/python -m pip install -r requirements.txt\n"
        f"Original error: {e}"
    )

from gui.state import UiState
from gui.tabs import PipelineControlTab, AnalysisTab, PredictionsTab, WeightsTab, ValidationTab, BaseTab
from gui.utils.settings_store import SettingsStore


class MonitorAppTabbed:
    """Multi-tab monitor application with lazy loading."""

    def __init__(self, root: tk.Tk, sim_root: Path) -> None:
        """Initialize tabbed monitor application.

        Args:
            root: Tkinter root window
            sim_root: Simple-Sim root directory
        """
        self.root = root
        self.sim_root = sim_root

        # Shared state across all tabs
        self.state = UiState()

        # Tab instances (lazy loaded)
        self.tabs: Dict[str, BaseTab] = {}
        self.current_tab: str = ""

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the main UI with notebook tabs."""
        self.root.title("A-AIO-Simple-Sim-v1.0")
        # Settings persistence
        settings_path = self.sim_root / "outputs" / "gui" / "settings.json"
        store = SettingsStore(settings_path)
        store.load()
        self.state.settings_store = store
        self.state.settings = dict(store.data)

        geom = store.get("app.geometry")
        if isinstance(geom, str) and geom:
            try:
                self.root.geometry(geom)
            except Exception:
                self.root.geometry("1400x900")
        else:
            self.root.geometry("1400x900")

        # Main container
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # Create notebook (tab container)
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True)

        # Bind tab selection event for lazy loading
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        # Custom app event for dataset selection changes (emitted by PipelineControlTab).
        self.notebook.bind("<<DatasetChanged>>", lambda _e: self._notify_dataset_changed())

        def on_close() -> None:
            try:
                if self.state.settings_store is not None:
                    try:
                        self.state.settings_store.set("app.geometry", self.root.winfo_geometry())
                    except Exception:
                        pass
                    self.state.settings_store.save()
            finally:
                try:
                    self.root.destroy()
                except Exception:
                    pass

        self.root.protocol("WM_DELETE_WINDOW", on_close)

        # Create tab frames (but don't build UI yet - lazy loading)
        self.tabs["pipeline"] = PipelineControlTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["pipeline"].frame, text="Pipeline Control")

        self.tabs["analysis"] = AnalysisTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["analysis"].frame, text="Analysis")

        self.tabs["predictions"] = PredictionsTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["predictions"].frame, text="Predictions")

        self.tabs["weights"] = WeightsTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["weights"].frame, text="Weights")

        self.tabs["validation"] = ValidationTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["validation"].frame, text="Validation")

        # Activate first tab immediately (not lazy for first tab)
        self.current_tab = "pipeline"
        self.tabs["pipeline"].on_activate()

        # Force update to ensure UI is rendered
        self.root.update_idletasks()

    def _on_tab_changed(self, event) -> None:
        """Handle tab change event (lazy loading trigger).

        Args:
            event: Tkinter event
        """
        # Get selected tab index
        selected_idx = self.notebook.index(self.notebook.select())

        # Map index to tab name
        tab_names = ["pipeline", "analysis", "predictions", "weights", "validation"]
        if selected_idx < len(tab_names):
            tab_name = tab_names[selected_idx]
            self.current_tab = tab_name

            # Trigger lazy loading
            if tab_name in self.tabs:
                self.tabs[tab_name].on_activate()

    def _notify_dataset_changed(self) -> None:
        """Notify all tabs that dataset has changed."""
        for tab in self.tabs.values():
            if tab.initialized:  # Only notify initialized tabs
                tab.on_dataset_changed()


def main() -> None:
    """Main entry point for tabbed monitor."""
    sim_root = Path(__file__).resolve().parent.parent
    root = tk.Tk()
    app = MonitorAppTabbed(root, sim_root=sim_root)
    root.mainloop()


if __name__ == "__main__":
    main()

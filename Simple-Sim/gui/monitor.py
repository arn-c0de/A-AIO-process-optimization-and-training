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
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

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
from gui.tabs import PipelineControlTab, AnalysisTab, PredictionsTab, WeightsTab, ValidationTab, MergeTab, BoardDetectionTab, DatasetsTab, BaseTab
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
        self._perf_tick_id: Optional[str] = None
        self._cpu_prev_total: Optional[int] = None
        self._cpu_prev_idle: Optional[int] = None

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

        # Compact global performance indicator overlaid into the tab row (no extra vertical space).
        perf_box = ttk.Frame(self.notebook, padding=(6, 1))
        perf_box.columnconfigure(0, weight=1)
        perf_box.columnconfigure(1, weight=1)
        perf_box.columnconfigure(2, weight=1)

        self.var_cpu = tk.StringVar(value="CPU: -")
        self.var_gpu = tk.StringVar(value="GPU: -")
        self.var_ram = tk.StringVar(value="RAM: -")

        ttk.Label(perf_box, textvariable=self.var_cpu, width=10).grid(row=0, column=0, sticky="w")
        ttk.Label(perf_box, textvariable=self.var_gpu, width=10).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(perf_box, textvariable=self.var_ram, width=10).grid(row=0, column=2, sticky="w", padx=(4, 0))
        perf_box.place(relx=1.0, x=-8, y=1, anchor="ne")

        # Bind tab selection event for lazy loading
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        # Custom app event for dataset selection changes (emitted by PipelineControlTab).
        self.notebook.bind("<<DatasetChanged>>", lambda _e: self._notify_dataset_changed())
        self.notebook.bind("<<DatasetCatalogChanged>>", lambda _e: self._notify_dataset_catalog_changed())

        def on_close() -> None:
            try:
                if self._perf_tick_id is not None:
                    try:
                        self.root.after_cancel(self._perf_tick_id)
                    except Exception:
                        pass
                    self._perf_tick_id = None
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

        self.tabs["datasets"] = DatasetsTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["datasets"].frame, text="Datasets")

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

        self.tabs["merge"] = MergeTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["merge"].frame, text="Merge")

        self.tabs["board_detection"] = BoardDetectionTab(
            self.notebook,
            self.sim_root,
            self.state
        )
        self.notebook.add(self.tabs["board_detection"].frame, text="Board Detection")

        # Activate first tab immediately (not lazy for first tab)
        self.current_tab = "pipeline"
        self.tabs["pipeline"].on_activate()
        self._tick_global_performance()

        # Force update to ensure UI is rendered
        self.root.update_idletasks()

    def _read_cpu_percent(self) -> Optional[float]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                line = f.readline().strip()
            parts = line.split()
            if not parts or parts[0] != "cpu" or len(parts) < 5:
                return None
            nums = [int(x) for x in parts[1:]]
            total = sum(nums)
            idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
            if self._cpu_prev_total is None or self._cpu_prev_idle is None:
                self._cpu_prev_total, self._cpu_prev_idle = total, idle
                return None
            dt = total - self._cpu_prev_total
            di = idle - self._cpu_prev_idle
            self._cpu_prev_total, self._cpu_prev_idle = total, idle
            return 100.0 * (1.0 - (di / dt)) if dt > 0 else None
        except Exception:
            return None

    def _read_ram_percent(self) -> Optional[Tuple[float, int, int]]:
        try:
            total_kb, avail_kb = None, None
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                    if total_kb is not None and avail_kb is not None:
                        break
            if total_kb is None or avail_kb is None or total_kb <= 0:
                return None
            used_kb = total_kb - avail_kb
            return 100.0 * (used_kb / total_kb), used_kb * 1024, total_kb * 1024
        except Exception:
            return None

    def _read_gpu_percent(self) -> Optional[float]:
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=0.5,
                check=False,
            )
            out = (proc.stdout or "").strip()
            if not out:
                return None
            return float(out.splitlines()[0].strip())
        except Exception:
            return None

    def _tick_global_performance(self) -> None:
        cpu_pct = self._read_cpu_percent()
        ram_info = self._read_ram_percent()
        gpu_pct = self._read_gpu_percent()

        if cpu_pct is not None:
            self.var_cpu.set(f"CPU: {cpu_pct:3.0f}%")
        if ram_info is None:
            self.var_ram.set("RAM: n/a")
        else:
            ram_pct, _used_b, _total_b = ram_info
            self.var_ram.set(f"RAM: {ram_pct:3.0f}%")
        if gpu_pct is not None:
            self.var_gpu.set(f"GPU: {gpu_pct:3.0f}%")

        self._perf_tick_id = self.root.after(1000, self._tick_global_performance)

    def _on_tab_changed(self, event) -> None:
        """Handle tab change event (lazy loading trigger).

        Args:
            event: Tkinter event
        """
        # Get selected tab index
        selected_idx = self.notebook.index(self.notebook.select())

        # Map index to tab name
        tab_names = ["pipeline", "datasets", "analysis", "predictions", "weights", "validation", "merge", "board_detection"]
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

    def _notify_dataset_catalog_changed(self) -> None:
        """Notify initialized tabs that dataset catalog metadata changed."""
        for tab in self.tabs.values():
            if tab.initialized:
                try:
                    tab.refresh()
                except Exception:
                    pass


def main() -> None:
    """Main entry point for tabbed monitor."""
    sim_root = Path(__file__).resolve().parent.parent
    root = tk.Tk()
    app = MonitorAppTabbed(root, sim_root=sim_root)
    root.mainloop()


if __name__ == "__main__":
    main()

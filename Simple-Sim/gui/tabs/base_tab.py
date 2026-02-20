"""Base class for all tabs in the monitor GUI."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from gui.state import UiState


@dataclass
class PerformanceWidgets:
    """Shared performance indicator widgets used by tab UIs."""

    frame: ttk.Frame
    var_cpu: tk.StringVar
    var_gpu: tk.StringVar
    var_ram: tk.StringVar
    pb_cpu: Optional[ttk.Progressbar]
    pb_gpu: Optional[ttk.Progressbar]
    pb_ram: Optional[ttk.Progressbar]


class BaseTab:
    """Base class for all tabs with shared interface."""

    def __init__(self, parent: ttk.Frame, sim_root: Path, state: UiState) -> None:
        """Initialize base tab.

        Args:
            parent: Parent ttk.Frame (the tab container)
            sim_root: Simple-Sim root directory
            state: Shared UI state object
        """
        self.parent = parent
        self.sim_root = sim_root
        self.state = state
        self.initialized = False

        # Create main frame for this tab
        self.frame = ttk.Frame(parent, padding=10)

    def build_ui(self) -> None:
        """Build the tab UI. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement build_ui()")

    def on_activate(self) -> None:
        """Called when tab is activated/selected by user.

        Lazy loading: Build UI on first activation.
        """
        if not self.initialized:
            self.build_ui()
            self.initialized = True

    def on_dataset_changed(self) -> None:
        """Called when dataset selection changes.

        Subclasses can override to react to dataset changes.
        """
        pass

    def refresh(self) -> None:
        """Refresh tab contents.

        Subclasses can override to implement refresh functionality.
        """
        pass

    def build_performance_widgets(
        self,
        parent: ttk.Frame,
        *,
        with_bars: bool = True,
        label_width: int = 12,
        bar_length: int = 100,
    ) -> PerformanceWidgets:
        """Create a consistent CPU/GPU/RAM performance UI block."""
        perf_frame = ttk.Frame(parent)
        perf_frame.columnconfigure(0, weight=1)
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.columnconfigure(2, weight=1)

        var_cpu = tk.StringVar(value="CPU: -")
        var_gpu = tk.StringVar(value="GPU: -")
        var_ram = tk.StringVar(value="RAM: -")

        ttk.Label(perf_frame, textvariable=var_cpu, width=label_width).grid(row=0, column=0, sticky="w")
        ttk.Label(perf_frame, textvariable=var_gpu, width=label_width).grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(perf_frame, textvariable=var_ram, width=label_width).grid(row=0, column=2, sticky="w", padx=(6, 0))

        pb_cpu: Optional[ttk.Progressbar] = None
        pb_gpu: Optional[ttk.Progressbar] = None
        pb_ram: Optional[ttk.Progressbar] = None

        if with_bars:
            pb_cpu = ttk.Progressbar(perf_frame, orient="horizontal", mode="determinate", maximum=100, length=bar_length)
            pb_cpu.grid(row=1, column=0, sticky="ew")
            pb_gpu = ttk.Progressbar(perf_frame, orient="horizontal", mode="determinate", maximum=100, length=bar_length)
            pb_gpu.grid(row=1, column=1, sticky="ew", padx=(6, 0))
            pb_ram = ttk.Progressbar(perf_frame, orient="horizontal", mode="determinate", maximum=100, length=bar_length)
            pb_ram.grid(row=1, column=2, sticky="ew", padx=(6, 0))

        return PerformanceWidgets(
            frame=perf_frame,
            var_cpu=var_cpu,
            var_gpu=var_gpu,
            var_ram=var_ram,
            pb_cpu=pb_cpu,
            pb_gpu=pb_gpu,
            pb_ram=pb_ram,
        )

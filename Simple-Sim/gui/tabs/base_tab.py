"""Base class for all tabs in the monitor GUI."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gui.state import UiState


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

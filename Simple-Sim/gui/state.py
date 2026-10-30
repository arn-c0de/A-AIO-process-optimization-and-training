"""Shared UI state management for all tabs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gui.utils.settings_store import SettingsStore


@dataclass
class UiState:
    """Extended UI state shared across all tabs."""
    # Existing fields (from current monitor)
    phase: str = "idle"
    epoch: str = "-"
    img_per_s: str = "-"
    last_id: str = "-"
    last_img: str = "-"
    dataset_dir: Path | None = None

    # New fields for enhanced functionality
    selected_image_id: str | None = None
    current_model_path: Path | None = None
    analysis_results: dict[str, Any] | None = None
    flags: dict[str, list[str]] = field(default_factory=dict)  # id -> list of flags

    # Settings persistence
    settings: dict[str, Any] = field(default_factory=dict)
    settings_store: SettingsStore | None = None

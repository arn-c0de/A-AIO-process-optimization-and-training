"""Shared UI state management for all tabs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Any

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
    dataset_dir: Optional[Path] = None

    # New fields for enhanced functionality
    selected_image_id: Optional[str] = None
    current_model_path: Optional[Path] = None
    analysis_results: Optional[Dict[str, Any]] = None
    flags: Dict[str, List[str]] = field(default_factory=dict)  # id -> list of flags

    # Settings persistence
    settings: Dict[str, Any] = field(default_factory=dict)
    settings_store: Optional[SettingsStore] = None

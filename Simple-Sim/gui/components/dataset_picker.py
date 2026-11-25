"""Multi-dataset picker shared by the Pipeline and Predictions tabs."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Iterable, Mapping
from pathlib import Path

from gui import dialogs
from gui.utils.dataset_catalog import same_path


def pick_datasets(parent: tk.Misc, labels: Mapping[str, Path], selected: Iterable[Path], *, prompt: str,
                  size: tuple[int, int]) -> list[Path] | None:
    """Let the user tick several of *labels* (label -> dataset dir); *selected* are pre-ticked.

    Returns the picked dataset dirs in list order, or ``None`` when the dialog was cancelled.
    """
    selected = list(selected)
    preselected = [label for label, p in labels.items() if any(same_path(p, s) for s in selected)]
    picked = dialogs.pick_from_list(parent, "Select Datasets", list(labels), multi=True, selected=preselected,
                                    prompt=prompt, size=size)
    if picked is None:
        return None
    return [labels[label] for label in picked if label in labels]

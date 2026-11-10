"""Matplotlib chart widgets for embedding in tkinter (theme-aware)."""

from __future__ import annotations
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from gui.theme.tokens import Palette, class_color
import contextlib


def _palette(parent: tk.Widget) -> Palette:
    from gui.theme.style import palette

    return palette(parent)


def _style_axes(fig: Figure, ax, pal: Palette) -> None:
    """Apply theme colours to a figure/axes (background, text, spines)."""
    fig.patch.set_facecolor(pal.surface)
    ax.set_facecolor(pal.surface)
    for spine in ax.spines.values():
        spine.set_color(pal.border)
    ax.tick_params(colors=pal.text_muted, labelcolor=pal.text)
    ax.xaxis.label.set_color(pal.text_muted)
    ax.yaxis.label.set_color(pal.text_muted)
    ax.title.set_color(pal.text)


def _embed(fig: Figure, parent: tk.Widget, pal: Palette) -> FigureCanvasTkAgg:
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    with contextlib.suppress(Exception):
        canvas.get_tk_widget().configure(background=pal.surface, highlightthickness=0)
    return canvas


def create_confusion_matrix_widget(parent: tk.Widget, confusion_matrix: np.ndarray,
                                   class_names: list[str]) -> FigureCanvasTkAgg:
    """Create a confusion matrix heatmap widget (rows = true, columns = predicted)."""
    pal = _palette(parent)
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)
    _style_axes(fig, ax, pal)

    cm = np.asarray(confusion_matrix)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors=pal.text_muted, labelcolor=pal.text)
    cbar.outline.set_edgecolor(pal.border)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if cm[i, j] > thresh else "#1b1f24"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=text_color, fontsize=10,
                    fontweight="bold" if i == j else "normal")

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    return _embed(fig, parent, pal)


def create_class_distribution_widget(parent: tk.Widget,
                                     class_counts: dict[str, int],
                                     title: str = "Class Distribution") -> FigureCanvasTkAgg:
    """Create a class distribution bar chart widget (fixed colour per defect class)."""
    pal = _palette(parent)
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    _style_axes(fig, ax, pal)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = [class_color(c) for c in classes]

    ax.bar(classes, counts, color=colors, alpha=0.9, width=0.6)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.25, color=pal.border)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='x', rotation=30)

    for i, count in enumerate(counts):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10, color=pal.text)

    fig.tight_layout()
    return _embed(fig, parent, pal)

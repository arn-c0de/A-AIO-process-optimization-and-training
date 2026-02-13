"""Matplotlib chart widgets for embedding in tkinter."""

from __future__ import annotations
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from typing import Dict, List, Optional


def create_confusion_matrix_widget(parent: tk.Widget, confusion_matrix: np.ndarray,
                                   class_names: List[str]) -> FigureCanvasTkAgg:
    """Create confusion matrix heatmap widget.

    Args:
        parent: Parent tkinter widget
        confusion_matrix: Confusion matrix as 2D numpy array
        class_names: List of class names

    Returns:
        FigureCanvasTkAgg widget
    """
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Plot heatmap
    im = ax.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Add text annotations
    thresh = confusion_matrix.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text_color = "white" if confusion_matrix[i, j] > thresh else "black"
            ax.text(j, i, int(confusion_matrix[i, j]),
                   ha="center", va="center", color=text_color, fontsize=10)

    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')

    fig.tight_layout()

    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()

    return canvas


def create_class_distribution_widget(parent: tk.Widget,
                                     class_counts: Dict[str, int],
                                     title: str = "Class Distribution") -> FigureCanvasTkAgg:
    """Create class distribution bar chart widget.

    Args:
        parent: Parent tkinter widget
        class_counts: Dictionary mapping class names to counts
        title: Chart title

    Returns:
        FigureCanvasTkAgg widget
    """
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    # Color map for classes
    color_map = {
        'OK': '#2ecc71',         # Green
        'MISSING': '#e74c3c',    # Red
        'MISALIGNED': '#f39c12', # Orange
        'TOMBSTONE': '#9b59b6'   # Purple
    }
    colors = [color_map.get(c, '#95a5a6') for c in classes]

    ax.bar(classes, counts, color=colors, alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    # Add count labels on bars
    for i, (c, count) in enumerate(zip(classes, counts)):
        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=10)

    fig.tight_layout()

    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()

    return canvas


def create_histogram_widget(parent: tk.Widget, data: np.ndarray,
                            title: str, xlabel: str,
                            bins: int = 30) -> FigureCanvasTkAgg:
    """Create histogram widget.

    Args:
        parent: Parent tkinter widget
        data: Data to plot
        title: Chart title
        xlabel: X-axis label
        bins: Number of histogram bins

    Returns:
        FigureCanvasTkAgg widget
    """
    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)

    ax.hist(data, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    # Add statistics
    mean = np.mean(data)
    std = np.std(data)
    stats_text = f'μ={mean:.2f}, σ={std:.2f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()

    return canvas


def create_scatter_plot_widget(parent: tk.Widget,
                               x_data: np.ndarray, y_data: np.ndarray,
                               labels: np.ndarray,
                               class_names: List[str],
                               xlabel: str, ylabel: str,
                               title: str) -> FigureCanvasTkAgg:
    """Create scatter plot widget with class-colored points.

    Args:
        parent: Parent tkinter widget
        x_data: X coordinates
        y_data: Y coordinates
        labels: Class labels (indices)
        class_names: List of class names
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Chart title

    Returns:
        FigureCanvasTkAgg widget
    """
    fig = Figure(figsize=(6, 5), dpi=100)
    ax = fig.add_subplot(111)

    # Color map for classes
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    # Plot each class separately for legend
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):
            ax.scatter(x_data[mask], y_data[mask],
                      c=colors[i % len(colors)], label=class_name,
                      alpha=0.6, s=30)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    # Embed in tkinter
    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()

    return canvas

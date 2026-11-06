"""Atomic dataset writing with validation."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from simple_sim.schema import LabelRow, MetaRow, validate_jsonl_pair, write_jsonl


def validate_dataset_files(data_dir: Path) -> None:
    """Validate dataset structure and files before committing.

    Raises:
        ValueError: If a required file/dir is missing or meta/labels are inconsistent.
    """
    data_dir = Path(data_dir)
    for filename in ("meta.jsonl", "labels.jsonl", "config.yaml"):
        if not (data_dir / filename).exists():
            raise ValueError(f"Missing required file: {filename}")

    validate_jsonl_pair(data_dir / "meta.jsonl", data_dir / "labels.jsonl")

    images_dir = data_dir / "images"
    if not images_dir.is_dir():
        raise ValueError("Missing images directory")


@contextmanager
def atomic_dataset_dir(output_dir: Path) -> Iterator[Path]:
    """Yield a fresh ``<output_dir>.tmp`` dir (with ``images/``); on success it replaces *output_dir*.

    On any exception the temp dir is removed and *output_dir* is left untouched.
    """
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.with_name(output_dir.name + ".tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    (temp_dir / "images").mkdir(parents=True)
    try:
        yield temp_dir
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.rename(output_dir)
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def write_dataset_files(
    data_dir: Path, meta_rows: list[MetaRow], label_rows: list[LabelRow], config: dict[str, Any]
) -> None:
    """Write ``meta.jsonl``, ``labels.jsonl`` and ``config.yaml`` into *data_dir*, then validate it."""
    data_dir = Path(data_dir)
    write_jsonl(data_dir / "meta.jsonl", meta_rows)
    write_jsonl(data_dir / "labels.jsonl", label_rows)
    with open(data_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    validate_dataset_files(data_dir)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write *image* with OpenCV, creating parent dirs; raise ``OSError`` on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write image: {path}")


def write_dataset(
    output_dir: Path,
    images: dict[str, np.ndarray],
    meta_rows: list[MetaRow],
    label_rows: list[LabelRow],
    config: dict[str, Any],
) -> None:
    """Write a complete dataset (images keyed by relative path + rows + config) atomically."""
    with atomic_dataset_dir(output_dir) as temp_dir:
        for image_path, image_array in images.items():
            write_image(temp_dir / image_path, image_array)
        write_dataset_files(temp_dir, meta_rows, label_rows, config)

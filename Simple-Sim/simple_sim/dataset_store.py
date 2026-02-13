"""Atomic dataset writing with validation."""

import shutil
import yaml
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import cv2

from simple_sim.schema import MetaRow, LabelRow, write_jsonl, validate_jsonl_pair


def validate_dataset_files(data_dir: Path) -> None:
    """Validate dataset structure and files before committing.

    Args:
        data_dir: Dataset directory to validate

    Raises:
        ValueError: If validation fails
    """
    data_dir = Path(data_dir)

    # Check required files exist
    required_files = ['meta.jsonl', 'labels.jsonl', 'config.yaml']
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            raise ValueError(f"Missing required file: {filename}")

    # Validate JSONL consistency
    meta_path = data_dir / 'meta.jsonl'
    labels_path = data_dir / 'labels.jsonl'
    validate_jsonl_pair(meta_path, labels_path)

    # Check images directory exists
    images_dir = data_dir / 'images'
    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError("Missing images directory")


def write_dataset(
    output_dir: Path,
    images: Dict[str, np.ndarray],
    meta_rows: List[MetaRow],
    label_rows: List[LabelRow],
    config: Dict[str, Any]
) -> None:
    """Write complete dataset with atomic operation.

    Uses temporary directory and atomic rename to ensure consistency.

    Args:
        output_dir: Target output directory
        images: Dictionary mapping image_path to image array
        meta_rows: List of metadata rows
        label_rows: List of label rows
        config: Configuration dictionary
    """
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary directory
    temp_dir = output_dir.with_name(output_dir.name + '.tmp')
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    try:
        # Write images
        images_dir = temp_dir / 'images'
        images_dir.mkdir()

        for image_path, image_array in images.items():
            full_path = temp_dir / image_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(full_path), image_array)
            if not success:
                raise IOError(f"Failed to write image: {image_path}")

        # Write JSONL files
        write_jsonl(temp_dir / 'meta.jsonl', meta_rows)
        write_jsonl(temp_dir / 'labels.jsonl', label_rows)

        # Write config
        with open(temp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # Validate before committing
        validate_dataset_files(temp_dir)

        # Atomic rename
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.rename(output_dir)

    except Exception as e:
        # Cleanup on failure
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise e

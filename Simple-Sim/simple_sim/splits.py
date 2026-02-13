"""Stratified dataset splitting."""

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
from simple_sim.schema import MetaRow, LabelRow


def generate_splits(
    meta_rows: List[MetaRow],
    label_rows: List[LabelRow],
    config: Dict,
    run_seed: int
) -> Dict[str, List[str]]:
    """Generate stratified train/val/test splits.

    Stratifies by (domain, class) to ensure balanced distribution.

    Args:
        meta_rows: List of metadata rows
        label_rows: List of label rows
        config: Configuration dictionary
        run_seed: Master random seed

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to lists of sample IDs
    """
    # Create ID to label mapping
    id_to_label = {row.id: row.class_name for row in label_rows}

    # Create ID to domain mapping
    id_to_domain = {row.id: row.domain for row in meta_rows}

    # Group samples by (domain, class)
    groups = defaultdict(list)
    for meta_row in meta_rows:
        sample_id = meta_row.id
        domain = id_to_domain[sample_id]
        class_name = id_to_label[sample_id]
        groups[(domain, class_name)].append(sample_id)

    # Extract split fractions
    train_frac = config['splits']['train_frac']
    val_frac = config['splits']['val_frac']
    test_frac = config['splits']['test_frac']

    # Initialize splits
    splits = {
        'train': [],
        'val': [],
        'test': []
    }

    # Create RNG for splitting
    rng = random.Random(run_seed)

    # Split each group stratified
    for (domain, class_name), sample_ids in groups.items():
        # Shuffle group
        sample_ids = sample_ids.copy()
        rng.shuffle(sample_ids)

        n = len(sample_ids)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        # Remaining goes to test to ensure all samples are used

        train_ids = sample_ids[:n_train]
        val_ids = sample_ids[n_train:n_train + n_val]
        test_ids = sample_ids[n_train + n_val:]

        splits['train'].extend(train_ids)
        splits['val'].extend(val_ids)
        splits['test'].extend(test_ids)

    # Sort splits for determinism
    for split_name in splits:
        splits[split_name].sort()

    return splits


def assert_no_overlap(splits: Dict[str, List[str]]) -> None:
    """Verify that splits have no overlapping sample IDs.

    Args:
        splits: Dictionary of split name to sample ID lists

    Raises:
        ValueError: If any overlap is detected
    """
    train_set = set(splits['train'])
    val_set = set(splits['val'])
    test_set = set(splits['test'])

    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    errors = []
    if train_val:
        errors.append(f"Train/val overlap: {len(train_val)} samples")
    if train_test:
        errors.append(f"Train/test overlap: {len(train_test)} samples")
    if val_test:
        errors.append(f"Val/test overlap: {len(val_test)} samples")

    if errors:
        raise ValueError("Split overlap detected: " + ", ".join(errors))


def write_splits(output_dir: Path, splits: Dict[str, List[str]]) -> None:
    """Write split files to disk.

    Args:
        output_dir: Output directory for split files
        splits: Dictionary of split name to sample ID lists
    """
    output_dir = Path(output_dir)
    splits_dir = output_dir / 'splits'
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sample_ids in splits.items():
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for sample_id in sample_ids:
                f.write(sample_id + '\n')


def read_split(split_file: Path) -> List[str]:
    """Read sample IDs from split file.

    Args:
        split_file: Path to split file

    Returns:
        List of sample IDs
    """
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def check_class_coverage(splits: Dict[str, List[str]], label_rows: List[LabelRow]) -> List[str]:
    """Check if all splits contain all classes.

    Args:
        splits: Dictionary of split name to sample ID lists
        label_rows: List of label rows

    Returns:
        List of warning messages (empty if all checks pass)
    """
    id_to_label = {row.id: row.class_name for row in label_rows}
    all_classes = set(row.class_name for row in label_rows)

    warnings = []
    for split_name, sample_ids in splits.items():
        split_classes = set(id_to_label[sid] for sid in sample_ids)
        missing_classes = all_classes - split_classes
        if missing_classes:
            warnings.append(f"{split_name} split missing classes: {missing_classes}")

    return warnings

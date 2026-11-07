"""Stratified dataset splitting."""

from __future__ import annotations

import itertools
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Sequence

from simple_sim.classes import SPLIT_NAMES
from simple_sim.dataset_files import SPLITS_DIR
from simple_sim.rows import MetaRow, LabelRow



def generate_splits(
    meta_rows: list[MetaRow],
    label_rows: list[LabelRow],
    config: dict[str, Any],
    run_seed: int
) -> dict[str, list[str]]:
    """Generate stratified train/val/test splits from a run config's ``splits`` section.

    Thin wrapper over :func:`stratified_splits` (plan A6): reads ``splits.{train,val,test}_frac``
    (``KeyError`` when missing, as before) and ``splits.{train,val,test}_domains``.

    Args:
        meta_rows: List of metadata rows
        label_rows: List of label rows
        config: Configuration dictionary
        run_seed: Master random seed

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to lists of sample IDs
    """
    sc = config['splits']
    return stratified_splits(
        meta_rows, label_rows,
        train_frac=sc['train_frac'], val_frac=sc['val_frac'], test_frac=sc['test_frac'],
        seed=run_seed,
        split_domains={name: sc.get(f'{name}_domains') or [] for name in SPLIT_NAMES},
    )


def stratified_splits(
    meta_rows: Iterable[MetaRow],
    label_rows: Iterable[LabelRow],
    *,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    split_domains: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, list[str]]:
    """Stratified train/val/test splits from the three fractions.

    Stratifies by (domain, profile_id, class) to ensure balanced distribution.
    profile_id comes from v2 label rows ("" for single-profile datasets).
    Per-group counts use largest-remainder rounding; when a group has at
    least 3 samples, every split with frac > 0 gets at least 1 (taken from train).

    *split_domains* (``{"train": [...], "val": [...], "test": [...]}``, the config's
    ``splits.<name>_domains``) restricts which splits a domain's samples may land in: a domain
    listed only under ``val`` and ``test`` is split between val and test with the fractions
    renormalised over those two (a held-out domain measures generalisation). A domain listed
    everywhere, or not listed at all, is split over all three.

    Returns:
        ``{"train", "val", "test"}`` -> sorted sample ids. Deterministic for a given *seed*.
    """
    label_rows = list(label_rows)
    # Create ID to label mapping
    id_to_label = {row.id: row.class_name for row in label_rows}

    # Create ID to profile mapping (empty for single-profile / v1 labels)
    id_to_profile = {row.id: getattr(row, 'profile_id', '') or '' for row in label_rows}

    # Group samples by (domain, profile, class)
    groups: defaultdict[tuple[str, str, str], list[str]] = defaultdict(list)
    for meta_row in meta_rows:
        sample_id = meta_row.id
        profile_id = id_to_profile.get(sample_id, '')
        class_name = id_to_label[sample_id]
        groups[(meta_row.domain, profile_id, class_name)].append(sample_id)

    # Initialize splits
    splits: dict[str, list[str]] = {'train': [], 'val': [], 'test': []}

    # Create RNG for splitting
    rng = random.Random(seed)

    split_domains = split_domains or {}
    allowed = {name: set(split_domains.get(name) or []) for name in SPLIT_NAMES}

    # Split each group stratified
    for (domain, _profile, _class), sample_ids in groups.items():
        # Shuffle group
        sample_ids = sample_ids.copy()
        rng.shuffle(sample_ids)

        n = len(sample_ids)
        fracs = [train_frac, val_frac, test_frac]
        if any(domain in names for names in allowed.values()):  # unlisted domains (minimal configs): no restriction
            fracs = [f if domain in allowed[name] else 0.0 for name, f in zip(SPLIT_NAMES, fracs)]
        n_train, n_val, _n_test = _split_counts(n, fracs)
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


def _split_counts(n: int, fracs: Sequence[float]) -> list[int]:
    """Largest-remainder allocation of n samples over split fractions.

    Fractions are normalised so the counts always sum to n. For n >= 3 every
    split with frac > 0 gets at least 1 sample, borrowed from the largest
    split (train in practice).
    """
    total = float(sum(fracs))
    if n <= 0 or total <= 0:
        return [n] + [0] * (len(fracs) - 1)

    quotas = [n * f / total for f in fracs]
    counts = [int(q) for q in quotas]
    # Hand out the leftover samples by largest fractional part (ties: earlier split)
    order = sorted(range(len(fracs)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[:n - sum(counts)]:
        counts[i] += 1

    if n >= 3:
        for i, f in enumerate(fracs):
            if f > 0 and counts[i] == 0:
                donor = max(range(len(counts)), key=lambda j: (counts[j], -j))
                counts[donor] -= 1
                counts[i] += 1

    return counts


def assert_no_overlap(splits: dict[str, list[str]]) -> None:
    """Verify that splits have no overlapping sample IDs.

    Works for any number of named splits.

    Args:
        splits: Dictionary of split name to sample ID lists

    Raises:
        ValueError: If any overlap is detected
    """
    split_sets = {name: set(ids) for name, ids in splits.items()}
    errors = []

    for (name_a, set_a), (name_b, set_b) in itertools.combinations(split_sets.items(), 2):
        overlap = set_a & set_b
        if overlap:
            errors.append(f"{name_a}/{name_b} overlap: {len(overlap)} samples")

    if errors:
        raise ValueError("Split overlap detected: " + ", ".join(errors))


def write_splits(output_dir: Path, splits: dict[str, list[str]]) -> None:
    """Write split files to disk.

    Args:
        output_dir: Output directory for split files
        splits: Dictionary of split name to sample ID lists
    """
    output_dir = Path(output_dir)
    splits_dir = output_dir / SPLITS_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sample_ids in splits.items():
        split_file = splits_dir / f"{split_name}.txt"
        with open(split_file, 'w', encoding='utf-8') as f:
            for sample_id in sample_ids:
                f.write(sample_id + '\n')


def read_split(split_file: Path) -> list[str]:
    """Read sample IDs from split file.

    Args:
        split_file: Path to split file

    Returns:
        List of sample IDs
    """
    split_file = Path(split_file)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def check_class_coverage(splits: dict[str, list[str]], label_rows: list[LabelRow]) -> list[str]:
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
        split_classes = set()
        missing_ids: list[str] = []
        for sid in sample_ids:
            lbl = id_to_label.get(sid)
            if lbl is None:
                missing_ids.append(sid)
                continue
            split_classes.add(lbl)

        if missing_ids:
            examples = ", ".join(missing_ids[:3])
            suffix = "..." if len(missing_ids) > 3 else ""
            warnings.append(
                f"{split_name} split contains {len(missing_ids)} sample IDs not present in labels.jsonl "
                f"(e.g. {examples}{suffix})"
            )

        missing_classes = all_classes - split_classes
        if missing_classes:
            warnings.append(f"{split_name} split missing classes: {missing_classes}")

    return warnings

#!/usr/bin/env python3
"""Validate dataset quality and consistency."""

import argparse
import sys
from pathlib import Path
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.schema import read_jsonl, MetaRow, LabelRow, validate_jsonl_pair
from simple_sim.splits import read_split, assert_no_overlap
from simple_sim.rng import derive_sample_seed, parse_sample_id
from simple_sim.telemetry import emit


def validate_dataset(data_dir: Path) -> bool:
    """Validate complete dataset.

    Args:
        data_dir: Dataset directory

    Returns:
        True if all validations pass, False otherwise
    """
    data_dir = Path(data_dir)
    all_checks_passed = True
    emit("validate_start", dataset_dir=str(data_dir))

    print("="*60)
    print("DATASET VALIDATION")
    print("="*60)
    print(f"Dataset: {data_dir}\n")

    # Check 1: Required files exist
    print("1. Checking required files...")
    required_files = ['meta.jsonl', 'labels.jsonl', 'config.yaml']
    for filename in required_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ✗ FAIL: Missing {filename}")
            all_checks_passed = False
        else:
            print(f"  ✓ {filename} exists")

    # Check splits directory
    splits_dir = data_dir / 'splits'
    if not splits_dir.exists():
        print(f"  ✗ FAIL: Missing splits directory")
        all_checks_passed = False
    else:
        print(f"  ✓ splits/ directory exists")

    # Check images directory
    images_dir = data_dir / 'images'
    if not images_dir.exists():
        print(f"  ✗ FAIL: Missing images directory")
        all_checks_passed = False
        return False  # Can't continue without images
    else:
        print(f"  ✓ images/ directory exists")

    # Check 2: Schema validation
    print("\n2. Validating JSONL schemas...")
    try:
        meta_rows = read_jsonl(data_dir / 'meta.jsonl', MetaRow)
        print(f"  ✓ meta.jsonl valid ({len(meta_rows)} rows)")
    except Exception as e:
        print(f"  ✗ FAIL: meta.jsonl invalid - {e}")
        all_checks_passed = False
        return False

    try:
        label_rows = read_jsonl(data_dir / 'labels.jsonl', LabelRow)
        print(f"  ✓ labels.jsonl valid ({len(label_rows)} rows)")
    except Exception as e:
        print(f"  ✗ FAIL: labels.jsonl invalid - {e}")
        all_checks_passed = False
        return False

    # Check 3: Count matching
    print("\n3. Checking row counts...")
    if len(meta_rows) == len(label_rows):
        print(f"  ✓ Row counts match ({len(meta_rows)} samples)")
    else:
        print(f"  ✗ FAIL: Row count mismatch (meta={len(meta_rows)}, labels={len(label_rows)})")
        all_checks_passed = False

    # Check 4: ID consistency
    print("\n4. Checking ID consistency...")
    try:
        validate_jsonl_pair(data_dir / 'meta.jsonl', data_dir / 'labels.jsonl')
        print(f"  ✓ All IDs match between meta and labels")
    except Exception as e:
        print(f"  ✗ FAIL: ID mismatch - {e}")
        all_checks_passed = False

    # Check 5: Image files exist and readable
    print("\n5. Checking image files...")
    missing_images = []
    corrupt_images = []

    for meta_row in meta_rows:
        image_path = data_dir / meta_row.image_path

        if not image_path.exists():
            missing_images.append(meta_row.image_path)
            continue

        # Try to read image
        img = cv2.imread(str(image_path))
        if img is None:
            corrupt_images.append(meta_row.image_path)

    if missing_images:
        print(f"  ✗ FAIL: {len(missing_images)} missing images")
        for img_path in missing_images[:5]:  # Show first 5
            print(f"    - {img_path}")
        if len(missing_images) > 5:
            print(f"    ... and {len(missing_images) - 5} more")
        all_checks_passed = False
    else:
        print(f"  ✓ All {len(meta_rows)} images exist")

    if corrupt_images:
        print(f"  ✗ FAIL: {len(corrupt_images)} corrupt images")
        for img_path in corrupt_images[:5]:
            print(f"    - {img_path}")
        if len(corrupt_images) > 5:
            print(f"    ... and {len(corrupt_images) - 5} more")
        all_checks_passed = False
    else:
        print(f"  ✓ All images readable")

    # Check 6: Split overlap
    print("\n6. Checking split overlap...")
    try:
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_file = splits_dir / f'{split_name}.txt'
            if split_file.exists():
                splits[split_name] = read_split(split_file)
            else:
                print(f"  ✗ FAIL: Missing {split_name}.txt")
                all_checks_passed = False

        if len(splits) == 3:
            try:
                assert_no_overlap(splits)
                print(f"  ✓ No overlap between splits")
                print(f"    - train: {len(splits['train'])} samples")
                print(f"    - val: {len(splits['val'])} samples")
                print(f"    - test: {len(splits['test'])} samples")
            except Exception as e:
                print(f"  ✗ FAIL: Split overlap detected - {e}")
                all_checks_passed = False

    except Exception as e:
        print(f"  ✗ FAIL: Error reading splits - {e}")
        all_checks_passed = False

    # Check 7: Class distribution
    print("\n7. Checking class distribution...")
    id_to_label = {row.id: row.class_name for row in label_rows}

    for split_name, sample_ids in splits.items():
        split_classes = set(id_to_label[sid] for sid in sample_ids if sid in id_to_label)
        all_classes = set(row.class_name for row in label_rows)

        missing_classes = all_classes - split_classes
        if missing_classes:
            print(f"  ⚠ WARNING: {split_name} missing classes: {missing_classes}")
        else:
            print(f"  ✓ {split_name} contains all classes")

    # Check 8: Determinism check (sample a few)
    print("\n8. Checking determinism (sample verification)...")
    import yaml

    config_path = data_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_seed = config['run']['seed']
    checked = 0
    mismatches = 0

    for meta_row in meta_rows[:20]:  # Check first 20 samples
        try:
            run_id, domain, split, index = parse_sample_id(meta_row.id)
            expected_seed = derive_sample_seed(run_seed, domain, index)

            if meta_row.seed != expected_seed:
                mismatches += 1
                print(f"  ✗ Seed mismatch for {meta_row.id}")

            checked += 1
        except Exception as e:
            print(f"  ✗ Error checking {meta_row.id}: {e}")
            all_checks_passed = False

    if mismatches == 0:
        print(f"  ✓ Determinism verified ({checked} samples checked)")
    else:
        print(f"  ✗ FAIL: {mismatches}/{checked} samples have seed mismatches")
        all_checks_passed = False

    # Final summary
    print("\n" + "="*60)
    if all_checks_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("="*60)
        emit("validate_done", dataset_dir=str(data_dir), ok=True)
        return True
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("="*60)
        emit("validate_done", dataset_dir=str(data_dir), ok=False)
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate dataset quality')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset directory')

    args = parser.parse_args()
    data_dir = Path(args.data)

    success = validate_dataset(data_dir)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

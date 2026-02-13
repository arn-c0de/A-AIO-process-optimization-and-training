#!/usr/bin/env python3
"""Generate synthetic PCB defect dataset."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.config import load_config, validate_config
from simple_sim.schema import MetaRow, LabelRow
from simple_sim.rng import derive_sample_seed, make_sample_id
from simple_sim.defects import sample_defect_params, classify_defect
from simple_sim.generator_2d import (
    sample_nominal_geometry,
    sample_augment_params,
    render_roi
)
from simple_sim.dataset_store import write_dataset
from simple_sim.splits import generate_splits, assert_no_overlap, write_splits, check_class_coverage
from simple_sim.telemetry import emit


def generate_dataset(config_path: Path, output_dir: Path):
    """Generate complete dataset from configuration.

    Args:
        config_path: Path to YAML configuration file
        output_dir: Output directory for dataset
    """
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    run_id = config['run']['run_id']
    run_seed = config['run']['seed']
    roi_width = config['roi']['width_px']
    roi_height = config['roi']['height_px']
    classes = config['classes']
    footprint = '0603'  # Fixed for MVP
    tolerances = config['tolerances'][footprint]
    domain_name = 'domain_A'  # Single domain for MVP
    domain_config = config['domains'][domain_name]

    print(f"\nGenerating dataset: {run_id}")
    print(f"Classes: {classes}")
    total_target = sum(classes.values())
    print(f"Total samples: {total_target}")
    emit(
        "gen_start",
        run_id=run_id,
        run_seed=run_seed,
        output_dir=str(output_dir),
        total_target=int(total_target),
        classes=classes,
        roi_width=int(roi_width),
        roi_height=int(roi_height),
    )

    # Storage for all data
    all_images = {}
    # Store intermediate records keyed by stable index; split gets assigned after stratification.
    records = []

    # Generate samples for each class
    sample_index = 0
    t_total0 = time.perf_counter()
    for class_name, count in classes.items():
        print(f"\nGenerating {count} samples for class {class_name}")

        t0 = time.perf_counter()
        progress = tqdm(range(count), desc=class_name)
        for i in progress:
            # Derive deterministic seed
            seed = derive_sample_seed(run_seed, domain_name, sample_index)
            rng = np.random.default_rng(seed)

            # Sample geometry
            nominal = sample_nominal_geometry(config['roi'], rng, geometry_ranges=config.get('geometry_ranges'))

            # Sample defect parameters based on target class
            defect_params = sample_defect_params(class_name, rng, tolerances=tolerances)

            # Verify label matches target (sanity check)
            derived_label = classify_defect(nominal, defect_params, tolerances)
            if derived_label != class_name:
                print(f"\nWARNING: Label mismatch for sample {sample_index}: target={class_name}, derived={derived_label}")
                # Use derived label to ensure consistency
                actual_class = derived_label
            else:
                actual_class = class_name

            # Sample augmentation
            augment = sample_augment_params(config['augment'], domain_config, rng)

            # Render image
            img = render_roi(
                nominal=nominal,
                defect_params=defect_params,
                augment=augment,
                roi_size=(roi_width, roi_height),
                config=config['render'],
                tolerances=tolerances,
                rng=rng
            )

            image_path = f"images/{sample_index:06d}.png"

            # Store image
            all_images[image_path] = img

            records.append({
                'index': sample_index,
                'seed': seed,
                'image_path': image_path,
                'nominal': nominal,
                'defect': defect_params,
                'augment': augment,
                'class_name': actual_class,
            })

            sample_index += 1
            elapsed = max(1e-9, time.perf_counter() - t0)
            rate = (i + 1) / elapsed
            progress.set_postfix({'idx': sample_index - 1, 'samp/s': f'{rate:.1f}'})
            if (i + 1) % 10 == 0 or (i + 1) == count:
                emit(
                    "gen_progress",
                    class_name=class_name,
                    class_i=int(i + 1),
                    class_total=int(count),
                    global_idx=int(sample_index - 1),
                    samp_per_s=float(rate),
                    last_image_path=image_path,
                )

    # Generate splits
    print("\nGenerating train/val/test splits")
    # Use stable temporary IDs (domain + index) for splitting; final IDs include split.
    tmp_meta_rows = []
    tmp_label_rows = []
    for r in records:
        tmp_id = f"{domain_name}/{r['index']:06d}"
        tmp_meta_rows.append(MetaRow(
            schema_version=1,
            id=tmp_id,
            run_id=run_id,
            domain=domain_name,
            split='train',  # placeholder; only used for grouping, not persisted
            seed=r['seed'],
            image_path=r['image_path'],
            render_backend='opencv_2d',
            footprint=footprint,
            nominal=r['nominal'],
            defect=r['defect'],
            augment=r['augment'],
        ))
        tmp_label_rows.append(LabelRow(schema_version=1, id=tmp_id, class_name=r['class_name']))

    splits = generate_splits(tmp_meta_rows, tmp_label_rows, config, run_seed)

    # Map tmp_id -> split
    id_to_split = {}
    for split_name, sample_ids in splits.items():
        for sample_id in sample_ids:
            id_to_split[sample_id] = split_name

    # Build final rows and final split ID lists.
    meta_rows = []
    label_rows = []
    updated_splits = {'train': [], 'val': [], 'test': []}

    for r in records:
        tmp_id = f"{domain_name}/{r['index']:06d}"
        split_name = id_to_split[tmp_id]
        sample_id = make_sample_id(run_id, domain_name, split_name, r['index'])

        meta_rows.append(MetaRow(
            schema_version=1,
            id=sample_id,
            run_id=run_id,
            domain=domain_name,
            split=split_name,
            seed=r['seed'],
            image_path=r['image_path'],
            render_backend='opencv_2d',
            footprint=footprint,
            nominal=r['nominal'],
            defect=r['defect'],
            augment=r['augment'],
        ))
        label_rows.append(LabelRow(schema_version=1, id=sample_id, class_name=r['class_name']))
        updated_splits[split_name].append(sample_id)

    # Sort split files for determinism.
    for k in updated_splits:
        updated_splits[k].sort()

    # Write dataset atomically (final form only).
    print(f"\nWriting dataset to {output_dir}")
    write_dataset(output_dir, all_images, meta_rows, label_rows, config)

    # Validate splits
    assert_no_overlap(updated_splits)
    print("✓ No split overlap detected")

    # Check class coverage
    warnings = check_class_coverage(updated_splits, label_rows)
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("✓ All splits contain all classes")

    # Write split files
    write_splits(output_dir, updated_splits)

    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(meta_rows)}")
    total_s = max(1e-9, time.perf_counter() - t_total0)
    print(f"Generation rate (overall): {len(meta_rows) / total_s:.1f} samples/s  (time: {total_s:.1f}s)")
    if meta_rows:
        print(f"Last sample: {meta_rows[-1].id}")
        print(f"Last image:  {meta_rows[-1].image_path}")
        emit(
            "gen_done",
            output_dir=str(output_dir),
            total_samples=int(len(meta_rows)),
            samp_per_s=float(len(meta_rows) / total_s),
            elapsed_s=float(total_s),
            last_id=meta_rows[-1].id,
            last_image_path=meta_rows[-1].image_path,
            splits={k: int(len(v)) for k, v in updated_splits.items()},
        )
    for split_name, sample_ids in updated_splits.items():
        print(f"  {split_name}: {len(sample_ids)} samples")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic PCB defect dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--out', type=str, required=True, help='Output directory for dataset')

    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.out)

    generate_dataset(config_path, output_dir)


if __name__ == '__main__':
    main()

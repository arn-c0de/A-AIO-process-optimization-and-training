#!/usr/bin/env python3
"""Generate synthetic PCB defect dataset."""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import time
import json
import random
from dataclasses import asdict

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
from simple_sim.profile_hash import load_profile, hash_profile
from simple_sim.manifest import write_dataset_manifest, read_dataset_manifest


def _append_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            if isinstance(row, (MetaRow, LabelRow)):
                row.__post_init__()
            f.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def _read_existing_max_index(meta_rows: list[MetaRow]) -> int:
    mx = -1
    for r in meta_rows:
        try:
            # sample_id ends with /000042
            idx = int(str(r.id).split("/")[-1])
        except Exception:
            idx = None
        if idx is None:
            # fallback: image_path images/000042.png
            try:
                idx = int(Path(r.image_path).stem)
            except Exception:
                idx = None
        if idx is not None and idx > mx:
            mx = idx
    return mx


def _assign_splits_for_new_samples(tmp_meta_rows: list[MetaRow], tmp_label_rows: list[LabelRow], config: dict, run_seed: int, salt: int) -> dict[str, list[str]]:
    """Return splits for tmp ids; uses a salted seed to remain deterministic across extend calls."""
    try:
        seed = int(run_seed) + int(salt)
    except Exception:
        seed = int(run_seed)
    return generate_splits(tmp_meta_rows, tmp_label_rows, config, seed)


def generate_dataset(config_path: Path, output_dir: Path, *, extend: bool = False):
    """Generate complete dataset from configuration.

    Args:
        config_path: Path to YAML configuration file
        output_dir: Output directory for dataset
        extend: If True and output_dir exists, append new samples instead of overwriting
    """
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    run_id = config['run']['run_id']
    run_seed = config['run']['seed']
    schema_version = config['run']['schema_version']
    roi_width = config['roi']['width_px']
    roi_height = config['roi']['height_px']
    classes = config['classes']

    # Load component profile
    profile_id = config['run'].get('component_profile', 'chip_0603_resistor@1')
    profiles_dir = config_path.parent / "profiles"
    profile = load_profile(profile_id, profiles_dir)
    profile_path = profiles_dir / f"{profile_id}.yaml"
    profile_hash_str = hash_profile(profile_path)

    print(f"Component Profile: {profile_id}")
    print(f"Profile Hash: {profile_hash_str[:72]}...")

    # Extract component-specific parameters from profile
    footprint = profile['component']['footprint']
    geometry_ranges = profile.get('geometry_ranges')
    tolerances = profile.get('tolerances')

    # Backward compatibility: v1 configs can override from config
    if schema_version == 1:
        if 'tolerances' in config:
            tolerances = config['tolerances'].get(profile['component']['package'], tolerances)
        if 'geometry_ranges' in config:
            geometry_ranges = config['geometry_ranges']

    # Merge component_color from profile into render config if not present
    if 'component_color' not in config['render'] and 'render' in profile:
        config['render']['component_color'] = profile['render']['component_color_bgr']

    domain_name = 'domain_A'  # Single domain for MVP
    domain_config = config['domains'][domain_name]

    # Extend mode: load existing dataset and append.
    existing_meta_rows: list[MetaRow] = []
    existing_label_rows: list[LabelRow] = []
    existing_splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    start_index = 0
    if extend and output_dir.exists():
        meta_path = output_dir / "meta.jsonl"
        labels_path = output_dir / "labels.jsonl"
        cfg_path = output_dir / "config.yaml"
        manifest_path = output_dir / "dataset_manifest.json"

        if not (meta_path.exists() and labels_path.exists() and cfg_path.exists()):
            raise ValueError(f"--extend requires an existing valid dataset dir with meta.jsonl/labels.jsonl/config.yaml: {output_dir}")

        # GUARD: Validate dataset manifest exists
        if not manifest_path.exists():
            raise ValueError(
                f"--extend requires dataset with manifest: {manifest_path}\n"
                f"Legacy datasets must be regenerated or use backfill tool:\n"
                f"  .venv/bin/python tools/backfill_manifest.py --data {output_dir}"
            )

        # GUARD: Load and validate component profile match
        manifest = read_dataset_manifest(manifest_path)
        existing_profile_id = manifest['component_profile']['profile_id']
        existing_profile_hash = manifest['component_profile']['profile_hash']

        # HARD FAIL: Profile ID mismatch
        if existing_profile_id != profile_id:
            raise ValueError(
                f"Profile ID mismatch for --extend:\n"
                f"  Dataset profile: {existing_profile_id}\n"
                f"  Config profile:  {profile_id}\n"
                f"Cannot extend dataset with different component type."
            )

        # HARD FAIL: Profile hash mismatch
        if existing_profile_hash != profile_hash_str:
            raise ValueError(
                f"Profile hash mismatch for --extend:\n"
                f"  Dataset hash: {existing_profile_hash[:72]}...\n"
                f"  Config hash:  {profile_hash_str[:72]}...\n"
                f"Profile '{profile_id}' has changed.\n"
                f"Create a new profile version (e.g., @2) or regenerate dataset."
            )

        print("✓ Profile validation passed for extend mode")

        from simple_sim.schema import read_jsonl, validate_jsonl_pair
        from simple_sim.splits import read_split

        validate_jsonl_pair(meta_path, labels_path)
        existing_meta_rows = read_jsonl(meta_path, MetaRow)
        existing_label_rows = read_jsonl(labels_path, LabelRow)
        if existing_meta_rows and existing_meta_rows[0].run_id != run_id:
            raise ValueError(f"Run ID mismatch for --extend.\n  dataset run_id: {existing_meta_rows[0].run_id}\n  config run_id:   {run_id}")

        start_index = _read_existing_max_index(existing_meta_rows) + 1
        print(f"\nExtending dataset: {output_dir}")
        print(f"Existing samples: {len(existing_meta_rows)} (next index: {start_index})")

        for s in ["train", "val", "test"]:
            p = output_dir / "splits" / f"{s}.txt"
            if p.exists():
                existing_splits[s] = read_split(p)

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
        extend=bool(extend),
        start_index=int(start_index),
    )

    # Storage for all data
    all_images = {}
    # Store intermediate records keyed by stable index; split gets assigned after stratification.
    records = []

    # Generate samples for each class
    sample_index = int(start_index)
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
            nominal = sample_nominal_geometry(config['roi'], rng, geometry_ranges=geometry_ranges)

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

    # Generate splits for the new samples.
    print("\nGenerating train/val/test splits")
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

    splits = _assign_splits_for_new_samples(tmp_meta_rows, tmp_label_rows, config, run_seed, salt=start_index)

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
    if extend and output_dir.exists() and existing_meta_rows:
        # Append mode: keep existing dataset, add new images + rows + update split files.
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        for image_path, image_array in all_images.items():
            full_path = output_dir / image_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(full_path), image_array)
            if not ok:
                raise IOError(f"Failed to write image: {image_path}")

        _append_jsonl(output_dir / "meta.jsonl", meta_rows)
        _append_jsonl(output_dir / "labels.jsonl", label_rows)

        # Merge splits and rewrite sorted for determinism.
        merged = {k: list(existing_splits.get(k, [])) for k in ["train", "val", "test"]}
        for k, ids in updated_splits.items():
            merged[k].extend(ids)
            merged[k] = sorted(set(merged[k]))
        assert_no_overlap(merged)
        write_splits(output_dir, merged)
    else:
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

    # Merge splits for manifest (include existing samples in extend mode)
    if extend and existing_meta_rows:
        all_meta_rows = existing_meta_rows + meta_rows
        all_label_rows = existing_label_rows + label_rows
        final_splits = updated_splits  # Already merged above
    else:
        all_meta_rows = meta_rows
        all_label_rows = label_rows
        final_splits = updated_splits

    # Write dataset manifest
    # Store profile path relative to Simple-Sim root
    project_root = Path(__file__).parent.parent
    try:
        relative_profile_path = str(profile_path.relative_to(project_root))
    except ValueError:
        # If profile_path is not under project_root, use absolute path
        relative_profile_path = str(profile_path.absolute())

    write_dataset_manifest(
        output_dir=output_dir,
        run_id=run_id,
        profile_id=profile_id,
        profile_hash=profile_hash_str,
        profile_path=relative_profile_path,
        meta_rows=meta_rows,  # Only new samples for extend history
        label_rows=label_rows,
        splits=final_splits,
        extend=extend
    )

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
    parser.add_argument('--extend', action='store_true', help='Append new samples to an existing dataset directory instead of overwriting')

    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.out)

    generate_dataset(config_path, output_dir, extend=bool(args.extend))


if __name__ == '__main__':
    main()

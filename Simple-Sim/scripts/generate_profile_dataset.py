#!/usr/bin/env python3
"""Generate a mixed-profile dataset for training a profile (component type) classifier.

Iterates over multiple component profiles from the config, generates samples for each,
and writes v2 labels with profile_id set.

Usage:
    .venv/bin/python scripts/generate_profile_dataset.py --config configs/run_profile_cls.yaml --out outputs/sim_data/runs/run_profile_cls
"""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.config import load_config, validate_config
from simple_sim.schema import MetaRow, LabelRow
from simple_sim.rng import derive_sample_seed, make_sample_id
from simple_sim.defects import sample_defect_params, classify_defect
from simple_sim.generator_2d import sample_nominal_geometry, sample_augment_params, render_roi
from simple_sim.dataset_store import write_dataset
from simple_sim.splits import generate_splits, assert_no_overlap, write_splits, check_class_coverage
from simple_sim.telemetry import emit
from simple_sim.profile_hash import load_profile, hash_profile
from simple_sim.manifest import write_multi_profile_manifest


def generate_profile_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    enable_cardinal_rotation_90: bool = True,
):
    """Generate mixed-profile dataset from configuration."""
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    run_id = config['run']['run_id']
    run_seed = config['run']['seed']
    roi_width = config['roi']['width_px']
    roi_height = config['roi']['height_px']
    classes = config['classes']
    profile_ids = config['run']['component_profiles']

    project_root = Path(__file__).parent.parent
    profiles_dir = project_root / "configs" / "profiles"

    domain_name = 'domain_A'
    domain_config = config['domains'][domain_name]
    backend = config.get("render", {}).get("backend", "opencv_2d")
    if backend != "opencv_2d":
        raise ValueError("generate_profile_dataset.py currently supports only render.backend='opencv_2d'")

    # Load all profiles
    profile_data = []
    for pid in profile_ids:
        profile = load_profile(pid, profiles_dir)
        profile_path = profiles_dir / f"{pid}.yaml"
        profile_hash_str = hash_profile(profile_path)
        profile_data.append({
            'profile_id': pid,
            'profile': profile,
            'profile_hash': profile_hash_str,
            'profile_path': str(profile_path.relative_to(project_root)),
        })
        print(f"Loaded profile: {pid} (hash: {profile_hash_str[:40]}...)")

    total_per_class = sum(classes.values())
    total_samples = total_per_class * len(profile_ids)
    print(f"\nProfiles: {profile_ids}")
    print(f"Samples per profile: {total_per_class}")
    print(f"Total samples: {total_samples}")

    all_images = {}
    records = []
    global_index = 0
    t_total0 = time.perf_counter()

    for pinfo in profile_data:
        pid = pinfo['profile_id']
        profile = pinfo['profile']
        footprint = profile['component']['footprint']
        geometry_ranges = profile.get('geometry_ranges')
        tolerances = profile.get('tolerances')

        # Merge component_color from profile into render config
        render_config = dict(config['render'])
        if 'component_color' not in render_config and 'render' in profile:
            render_config['component_color'] = profile['render']['component_color_bgr']

        print(f"\nGenerating samples for profile: {pid}")

        for class_name, count in classes.items():
            progress = tqdm(range(count), desc=f"{pid}/{class_name}")
            for i in progress:
                seed = derive_sample_seed(run_seed, domain_name, global_index)
                rng = np.random.default_rng(seed)

                nominal = sample_nominal_geometry(config['roi'], rng, geometry_ranges=geometry_ranges)
                defect_params = sample_defect_params(class_name, rng, tolerances=tolerances)
                derived_label = classify_defect(nominal, defect_params, tolerances)
                actual_class = derived_label if derived_label != class_name else class_name

                augment = sample_augment_params(
                    config['augment'],
                    domain_config,
                    rng,
                    enable_cardinal_rotation_90=enable_cardinal_rotation_90,
                )

                img = render_roi(
                    nominal=nominal,
                    defect_params=defect_params,
                    augment=augment,
                    roi_size=(roi_width, roi_height),
                    config=render_config,
                    tolerances=tolerances,
                    rng=rng,
                    footprint=footprint,
                )

                image_path = f"images/{global_index:06d}.png"
                all_images[image_path] = img

                records.append({
                    'index': global_index,
                    'seed': seed,
                    'image_path': image_path,
                    'nominal': nominal,
                    'defect': defect_params,
                    'augment': augment,
                    'class_name': actual_class,
                    'profile_id': pid,
                    'footprint': footprint,
                })
                global_index += 1

    # Build temporary rows for split assignment
    print("\nGenerating train/val/test splits")
    tmp_meta_rows = []
    tmp_label_rows = []
    for r in records:
        tmp_id = f"{domain_name}/{r['index']:06d}"
        tmp_meta_rows.append(MetaRow(
            schema_version=2,
            id=tmp_id,
            run_id=run_id,
            domain=domain_name,
            split='train',
            seed=r['seed'],
            image_path=r['image_path'],
            render_backend=str(backend),
            footprint=r['footprint'],
            nominal=r['nominal'],
            defect=r['defect'],
            augment=r['augment'],
            render_meta={},
        ))
        tmp_label_rows.append(LabelRow(schema_version=2, id=tmp_id, class_name=r['class_name'], profile_id=r['profile_id']))

    splits = generate_splits(tmp_meta_rows, tmp_label_rows, config, run_seed)

    id_to_split = {}
    for split_name, sample_ids in splits.items():
        for sample_id in sample_ids:
            id_to_split[sample_id] = split_name

    # Build final rows
    meta_rows = []
    label_rows = []
    updated_splits = {'train': [], 'val': [], 'test': []}

    for r in records:
        tmp_id = f"{domain_name}/{r['index']:06d}"
        split_name = id_to_split[tmp_id]
        sample_id = make_sample_id(run_id, domain_name, split_name, r['index'])

        meta_rows.append(MetaRow(
            schema_version=2,
            id=sample_id,
            run_id=run_id,
            domain=domain_name,
            split=split_name,
            seed=r['seed'],
            image_path=r['image_path'],
            render_backend=str(backend),
            footprint=r['footprint'],
            nominal=r['nominal'],
            defect=r['defect'],
            augment=r['augment'],
            render_meta={},
        ))
        label_rows.append(LabelRow(
            schema_version=2,
            id=sample_id,
            class_name=r['class_name'],
            profile_id=r['profile_id'],
        ))
        updated_splits[split_name].append(sample_id)

    for k in updated_splits:
        updated_splits[k].sort()

    # Write dataset atomically
    print(f"\nWriting dataset to {output_dir}")
    write_dataset(output_dir, all_images, meta_rows, label_rows, config)

    assert_no_overlap(updated_splits)
    print("No split overlap detected")

    warnings = check_class_coverage(updated_splits, label_rows)
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("All splits contain all classes")

    write_splits(output_dir, updated_splits)

    # Write multi-profile manifest
    manifest_profiles = [
        {'profile_id': p['profile_id'], 'profile_hash': p['profile_hash'], 'profile_path': p['profile_path']}
        for p in profile_data
    ]
    write_multi_profile_manifest(
        output_dir=output_dir,
        run_id=run_id,
        profiles=manifest_profiles,
        meta_rows=meta_rows,
        label_rows=label_rows,
        splits=updated_splits,
    )

    total_s = max(1e-9, time.perf_counter() - t_total0)
    print("\n" + "=" * 60)
    print("PROFILE DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(meta_rows)}")
    print(f"Profiles: {profile_ids}")
    print(f"Generation rate: {len(meta_rows) / total_s:.1f} samples/s  (time: {total_s:.1f}s)")
    for split_name, sample_ids in updated_splits.items():
        print(f"  {split_name}: {len(sample_ids)} samples")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate mixed-profile dataset for profile classifier')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML configuration file')
    parser.add_argument('--out', type=str, required=True, help='Output directory for dataset')
    parser.add_argument(
        '--disable-cardinal-rotation-90',
        action='store_true',
        help='Disable 0/90/180/270 base orientation randomization (keep only augment.rotation_deg_range jitter)',
    )
    args = parser.parse_args()

    generate_profile_dataset(
        Path(args.config),
        Path(args.out),
        enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
    )


if __name__ == '__main__':
    main()

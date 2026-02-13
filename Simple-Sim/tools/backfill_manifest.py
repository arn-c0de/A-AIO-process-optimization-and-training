#!/usr/bin/env python3
"""Backfill dataset_manifest.json for legacy datasets.

This tool adds manifests to datasets created before the profile system was implemented.
It infers the profile from the dataset config and computes the profile hash.

Usage:
    # Single dataset
    .venv/bin/python tools/backfill_manifest.py --data outputs/sim_data/runs/run_0001

    # Batch process all datasets in a directory
    .venv/bin/python tools/backfill_manifest.py --data-root outputs/sim_data/runs

    # Use a specific profile (default: chip_0603_resistor@1)
    .venv/bin/python tools/backfill_manifest.py --data outputs/sim_data/runs/run_0001 --profile chip_0603_resistor@1
"""

import argparse
import json
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.profile_hash import hash_profile
from simple_sim.schema import read_jsonl, MetaRow, LabelRow


def backfill_dataset(
    data_dir: Path,
    default_profile_id: str = "chip_0603_resistor@1",
    profiles_dir: Path = None
) -> bool:
    """Add manifest to legacy dataset.

    Args:
        data_dir: Dataset directory
        default_profile_id: Profile ID to use if not in config
        profiles_dir: Directory containing profile YAML files

    Returns:
        True if successful, False if skipped or failed
    """
    print(f"\nProcessing: {data_dir}")

    # Check if manifest exists
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        print(f"  ✓ Manifest already exists, skipping")
        return False

    # Validate dataset structure
    config_path = data_dir / "config.yaml"
    meta_path = data_dir / "meta.jsonl"
    labels_path = data_dir / "labels.jsonl"

    if not all([config_path.exists(), meta_path.exists(), labels_path.exists()]):
        print(f"  ✗ Invalid dataset structure (missing config.yaml, meta.jsonl, or labels.jsonl)")
        return False

    # Load config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False

    run_id = config.get('run', {}).get('run_id', 'unknown')

    # Determine profile ID
    profile_id = config.get('run', {}).get('component_profile', default_profile_id)

    # Locate profiles directory if not provided
    if profiles_dir is None:
        # Try to find it relative to the script
        profiles_dir = Path(__file__).parent.parent / "configs" / "profiles"

    profile_path = profiles_dir / f"{profile_id}.yaml"

    if not profile_path.exists():
        print(f"  ✗ Profile not found: {profile_path}")
        print(f"     Available profiles: {list(profiles_dir.glob('*.yaml'))}")
        return False

    # Load profile and compute hash
    try:
        profile_hash_str = hash_profile(profile_path)
    except Exception as e:
        print(f"  ✗ Failed to hash profile: {e}")
        return False

    # Load dataset stats
    try:
        meta_rows = read_jsonl(meta_path, MetaRow)
        label_rows = read_jsonl(labels_path, LabelRow)
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return False

    # Compute stats
    splits = {'train': [], 'val': [], 'test': []}
    for meta in meta_rows:
        splits[meta.split].append(meta.id)

    class_counts = {}
    for label in label_rows:
        class_counts[label.class_name] = class_counts.get(label.class_name, 0) + 1

    # Get file modification time as a proxy for creation time
    creation_time = datetime.fromtimestamp(meta_path.stat().st_mtime)

    # Create manifest
    manifest = {
        'manifest_version': 1,
        'created_at': creation_time.isoformat(),
        'run_id': run_id,

        'component_profile': {
            'profile_id': profile_id,
            'profile_hash': profile_hash_str,
            'profile_path': f"configs/profiles/{profile_id}.yaml"
        },

        'generator': {
            'version': '1.0.0',  # Legacy
            'git_commit': None,
            'script': 'scripts/generate.py'
        },

        'dataset_stats': {
            'total_samples': len(meta_rows),
            'splits': {k: len(v) for k, v in splits.items()},
            'classes': class_counts
        },

        'extend_history': [{
            'timestamp': creation_time.isoformat(),
            'samples_added': len(meta_rows),
            'git_commit': None,
            'note': 'Backfilled from legacy dataset'
        }]
    }

    # Write manifest
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"  ✗ Failed to write manifest: {e}")
        return False

    print(f"  ✓ Manifest created")
    print(f"     Profile: {profile_id}")
    print(f"     Hash: {profile_hash_str[:48]}...")
    print(f"     Samples: {len(meta_rows)}")
    print(f"     Classes: {class_counts}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Backfill dataset manifests for legacy datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backfill a single dataset
  .venv/bin/python tools/backfill_manifest.py --data outputs/sim_data/runs/run_0001

  # Batch process all datasets in a directory
  .venv/bin/python tools/backfill_manifest.py --data-root outputs/sim_data/runs

  # Use a specific profile
  .venv/bin/python tools/backfill_manifest.py --data outputs/sim_data/runs/run_0001 --profile chip_0805_resistor@1
        """
    )

    parser.add_argument('--data', type=str, help='Single dataset directory')
    parser.add_argument('--data-root', type=str, help='Batch process all datasets in this directory')
    parser.add_argument('--profile', type=str, default='chip_0603_resistor@1',
                       help='Default profile ID (default: chip_0603_resistor@1)')
    parser.add_argument('--profiles-dir', type=str, default=None,
                       help='Directory containing profile YAML files (default: configs/profiles)')

    args = parser.parse_args()

    if not args.data and not args.data_root:
        parser.error("Provide --data or --data-root")

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else None

    success_count = 0
    skip_count = 0
    fail_count = 0

    if args.data:
        # Single dataset
        result = backfill_dataset(Path(args.data), args.profile, profiles_dir)
        if result:
            success_count += 1
        else:
            skip_count += 1

    elif args.data_root:
        # Batch process
        root_dir = Path(args.data_root)
        if not root_dir.exists():
            print(f"Error: Directory not found: {root_dir}")
            return 1

        datasets = sorted([d for d in root_dir.iterdir() if d.is_dir()])
        if not datasets:
            print(f"No datasets found in: {root_dir}")
            return 1

        print(f"Found {len(datasets)} dataset(s) in: {root_dir}")

        for data_dir in datasets:
            result = backfill_dataset(data_dir, args.profile, profiles_dir)
            if result:
                success_count += 1
            elif result is False:
                skip_count += 1
            else:
                fail_count += 1

    # Summary
    print("\n" + "="*60)
    print("BACKFILL SUMMARY")
    print("="*60)
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed:  {fail_count}")
    print("="*60)

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

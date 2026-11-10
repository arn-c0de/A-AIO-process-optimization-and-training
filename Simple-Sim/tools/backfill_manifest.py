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

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_sim.cli_utils import add_json_flag, configure_cli_logging, emit_result  # noqa: E402
from simple_sim.io_utils import atomic_write_json, load_yaml  # noqa: E402
from simple_sim.manifest import dataset_stats  # noqa: E402
from simple_sim.profile_hash import hash_profile  # noqa: E402
from simple_sim.schema import LabelRow, MetaRow, read_jsonl  # noqa: E402

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


def backfill_dataset(
    data_dir: Path,
    default_profile_id: str = "chip_0603_resistor@1",
    profiles_dir: Path | None = None
) -> str:
    """Add manifest to legacy dataset.

    Args:
        data_dir: Dataset directory
        default_profile_id: Profile ID to use if not in config
        profiles_dir: Directory containing profile YAML files

    Returns:
        STATUS_OK if a manifest was written, STATUS_SKIPPED if one already
        existed, STATUS_FAILED otherwise
    """
    print(f"\nProcessing: {data_dir}")

    # Check if manifest exists
    manifest_path = data_dir / "dataset_manifest.json"
    if manifest_path.exists():
        print("  ✓ Manifest already exists, skipping")
        return STATUS_SKIPPED

    # Validate dataset structure
    config_path = data_dir / "config.yaml"
    meta_path = data_dir / "meta.jsonl"
    labels_path = data_dir / "labels.jsonl"

    if not all([config_path.exists(), meta_path.exists(), labels_path.exists()]):
        print("  ✗ Invalid dataset structure (missing config.yaml, meta.jsonl, or labels.jsonl)")
        return STATUS_FAILED

    # Load config
    try:
        config = load_yaml(config_path) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"  ✗ Failed to load config: {e}")
        return STATUS_FAILED

    # config.yaml with an empty `run:` (None) or a non-mapping top level must not crash the batch.
    run_cfg = config.get('run') if isinstance(config, dict) else None
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    run_id = run_cfg.get('run_id', 'unknown')

    # Determine profile ID
    profile_id = run_cfg.get('component_profile', default_profile_id)

    # Locate profiles directory if not provided
    if profiles_dir is None:
        profiles_dir = ROOT / "configs" / "profiles"

    profile_path = profiles_dir / f"{profile_id}.yaml"

    if not profile_path.exists():
        print(f"  ✗ Profile not found: {profile_path}")
        print(f"     Available profiles: {list(profiles_dir.glob('*.yaml'))}")
        return STATUS_FAILED

    # Load profile and compute hash
    try:
        profile_hash_str = hash_profile(profile_path)
    except (OSError, yaml.YAMLError) as e:
        print(f"  ✗ Failed to hash profile: {e}")
        return STATUS_FAILED

    # Load dataset stats
    try:
        meta_rows = read_jsonl(meta_path, MetaRow)
        label_rows = read_jsonl(labels_path, LabelRow)
    except Exception as e:  # noqa: BLE001 - any parse error means the dataset is unusable
        print(f"  ✗ Failed to load dataset: {e}")
        return STATUS_FAILED

    splits: dict[str, list[str]] = {'train': [], 'val': [], 'test': []}
    for meta in meta_rows:
        splits.setdefault(meta.split, []).append(meta.id)
    stats = dataset_stats(meta_rows, label_rows, splits)

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

        'dataset_stats': stats,

        'extend_history': [{
            'timestamp': creation_time.isoformat(),
            'samples_added': len(meta_rows),
            'git_commit': None,
            'note': 'Backfilled from legacy dataset'
        }]
    }

    # Write manifest
    try:
        atomic_write_json(manifest_path, manifest)
    except OSError as e:
        print(f"  ✗ Failed to write manifest: {e}")
        return STATUS_FAILED

    print("  ✓ Manifest created")
    print(f"     Profile: {profile_id}")
    print(f"     Hash: {profile_hash_str[:48]}...")
    print(f"     Samples: {len(meta_rows)}")
    print(f"     Classes: {stats['classes']}")

    return STATUS_OK


def main() -> int:
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
    add_json_flag(parser)

    args = parser.parse_args()
    configure_cli_logging()

    if not args.data and not args.data_root:
        parser.error("Provide --data or --data-root")

    profiles_dir = Path(args.profiles_dir) if args.profiles_dir else None

    def fail(msg: str) -> int:
        print(msg)
        if args.json:
            emit_result("backfill-manifest", ok=False, error=msg)
        return 1

    if args.data:
        datasets = [Path(args.data)]
    else:
        root_dir = Path(args.data_root)
        if not root_dir.exists():
            return fail(f"Error: Directory not found: {root_dir}")
        # Only dirs that look like a dataset: a stray folder (scratch dir, partial output) is not a failure.
        datasets = sorted(
            d for d in root_dir.iterdir()
            if d.is_dir() and any((d / f).exists() for f in ("meta.jsonl", "labels.jsonl", "config.yaml"))
        )
        if not datasets:
            return fail(f"No datasets found in: {root_dir}")
        print(f"Found {len(datasets)} dataset(s) in: {root_dir}")

    counts = {STATUS_OK: 0, STATUS_SKIPPED: 0, STATUS_FAILED: 0}
    results: dict[str, str] = {}
    for data_dir in datasets:
        try:
            status = backfill_dataset(data_dir, args.profile, profiles_dir)
        except Exception as e:  # noqa: BLE001 - one broken dataset must not abort the batch or the --json line
            print(f"  ✗ {type(e).__name__}: {e}")
            status = STATUS_FAILED
        counts[status] += 1
        results[str(data_dir)] = status

    # Summary
    print("\n" + "="*60)
    print("BACKFILL SUMMARY")
    print("="*60)
    print(f"Success: {counts[STATUS_OK]}")
    print(f"Skipped: {counts[STATUS_SKIPPED]}")
    print(f"Failed:  {counts[STATUS_FAILED]}")
    print("="*60)

    ok = counts[STATUS_FAILED] == 0
    if args.json:
        emit_result(
            "backfill-manifest",
            {"counts": counts, "datasets": results},
            ok=ok,
            error=None if ok else f"{counts[STATUS_FAILED]} dataset(s) failed",
        )
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

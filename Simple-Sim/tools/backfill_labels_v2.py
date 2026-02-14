#!/usr/bin/env python3
"""Backfill labels.jsonl from v1 to v2 by adding profile_id from dataset manifest.

Usage:
    .venv/bin/python tools/backfill_labels_v2.py --data outputs/sim_data/runs/run_0001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.schema import LabelRow, read_jsonl, write_jsonl
from simple_sim.manifest import read_dataset_manifest


def main():
    parser = argparse.ArgumentParser(description="Backfill labels.jsonl to v2 with profile_id")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    args = parser.parse_args()

    data_dir = Path(args.data)
    labels_path = data_dir / "labels.jsonl"
    manifest_path = data_dir / "dataset_manifest.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.jsonl not found: {labels_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dataset_manifest.json not found: {manifest_path}\n"
            f"Run tools/backfill_manifest.py first."
        )

    manifest = read_dataset_manifest(manifest_path)
    profile_id = manifest["component_profile"]["profile_id"]
    print(f"Profile ID from manifest: {profile_id}")

    old_rows = read_jsonl(labels_path, LabelRow)
    print(f"Read {len(old_rows)} label rows")

    already_v2 = all(r.schema_version == 2 and r.profile_id for r in old_rows)
    if already_v2:
        print("Labels are already v2. Nothing to do.")
        return

    new_rows = []
    for row in old_rows:
        new_rows.append(LabelRow(
            schema_version=2,
            id=row.id,
            class_name=row.class_name,
            profile_id=profile_id,
        ))

    write_jsonl(labels_path, new_rows)
    print(f"Wrote {len(new_rows)} v2 label rows to {labels_path}")


if __name__ == "__main__":
    main()

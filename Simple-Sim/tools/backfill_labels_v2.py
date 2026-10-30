#!/usr/bin/env python3
"""Backfill labels.jsonl from v1 to v2 by adding profile_id from dataset manifest.

Only single-profile (manifest_version 1) datasets can be backfilled: their one
component profile is assigned to every label row that has no profile_id yet.
Rows that already carry a non-empty profile_id are kept unchanged.

Usage:
    .venv/bin/python tools/backfill_labels_v2.py --data outputs/sim_data/runs/run_0001
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_sim.schema import LabelRow, read_jsonl, write_jsonl
from simple_sim.manifest import read_dataset_manifest, resolve_dataset_profile


def backfill_labels(data_dir: Path) -> int:
    """Upgrade labels.jsonl in *data_dir* to schema v2.

    Returns:
        Number of label rows that received a profile_id (0 if nothing to do).

    Raises:
        FileNotFoundError: labels.jsonl or dataset_manifest.json missing
        ValueError: manifest is multi-profile (v2) and rows lack profile_id
    """
    data_dir = Path(data_dir)
    labels_path = data_dir / "labels.jsonl"
    manifest_path = data_dir / "dataset_manifest.json"

    if not labels_path.exists():
        raise FileNotFoundError(f"labels.jsonl not found: {labels_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dataset_manifest.json not found: {manifest_path}\n"
            f"Run tools/backfill_manifest.py first."
        )

    old_rows = read_jsonl(labels_path, LabelRow)
    print(f"Read {len(old_rows)} label rows")

    already_v2 = all(r.schema_version == 2 and r.profile_id for r in old_rows)
    if already_v2:
        print("Labels are already v2. Nothing to do.")
        return 0

    manifest = read_dataset_manifest(manifest_path)
    mver = int(manifest.get("manifest_version", 1) or 1)
    if mver != 1:
        raise ValueError(
            f"{manifest_path} is a multi-profile manifest (manifest_version={mver}); "
            f"the per-sample profile_id cannot be inferred from it. "
            f"Regenerate/merge the dataset so every label row carries its profile_id."
        )
    profile_id, _profile_hash = resolve_dataset_profile(manifest)
    print(f"Profile ID from manifest: {profile_id}")

    new_rows = []
    filled = 0
    for row in old_rows:
        pid = str(row.profile_id or "").strip()
        if not pid:
            pid = profile_id
            filled += 1
        new_rows.append(LabelRow(
            schema_version=2,
            id=row.id,
            class_name=row.class_name,
            profile_id=pid,
        ))

    write_jsonl(labels_path, new_rows)
    print(f"Wrote {len(new_rows)} v2 label rows to {labels_path} ({filled} profile_id filled)")
    return filled


def main():
    parser = argparse.ArgumentParser(description="Backfill labels.jsonl to v2 with profile_id")
    parser.add_argument("--data", required=True, help="Path to dataset directory")
    args = parser.parse_args()

    try:
        backfill_labels(Path(args.data))
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()

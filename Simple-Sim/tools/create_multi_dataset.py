#!/usr/bin/env python3
"""Create a merged dataset from multiple profile-specific runs.

This aggregates several existing `outputs/sim_data/runs/<name>` folders into
a single dataset folder so you can train a multi-profile checkpoint.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from simple_sim.config import load_config
from simple_sim.manifest import write_dataset_manifest
from simple_sim.profile_hash import hash_profile
from simple_sim.schema import MetaRow, LabelRow, read_jsonl, write_jsonl


def gather_rows(
    ds: Path,
    run_id: str,
    split_counters: Dict[str, int],
    images_dir: Path,
) -> Tuple[List[MetaRow], Dict[str, str]]:
    meta_rows = read_jsonl(ds / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(ds / "labels.jsonl", LabelRow)
    label_map = {row.id: row.class_name for row in label_rows}

    new_meta: List[MetaRow] = []
    new_label_map: Dict[str, str] = {}
    for row in meta_rows:
        class_name = label_map.get(row.id)
        if class_name is None:
            continue

        split = row.split
        idx = split_counters.setdefault(split, 0)
        new_id = f"{run_id}/{row.domain}/{split}/{idx:06d}"
        split_counters[split] = idx + 1

        new_image_name = f"{split}_{idx:06d}.png"
        dst_image = images_dir / new_image_name
        src_image = ds / row.image_path
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_image, dst_image)

        new_meta.append(
            MetaRow(
                schema_version=1,
                id=new_id,
                run_id=run_id,
                domain=row.domain,
                split=split,
                seed=row.seed,
                image_path=str(Path("images") / new_image_name),
                render_backend=row.render_backend,
                footprint=row.footprint,
                nominal=row.nominal,
                defect=row.defect,
                augment=row.augment,
            )
        )
        new_label_map[new_id] = class_name

    return new_meta, new_label_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-profile dataset.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Existing dataset folders to merge (e.g. outputs/sim_data/runs/run_0001)",
    )
    parser.add_argument("--out", required=True, help="Output dataset directory.")
    parser.add_argument(
        "--config",
        default="configs/run_multi.yaml",
        help="Config that defines the multi-profile training pipeline.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    images_dir = out_dir / "images"
    images_dir.mkdir()
    (out_dir / "splits").mkdir()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    run_id = cfg["run"]["run_id"]
    profile_id = cfg["run"]["component_profile"]
    profile_path = config_path.parent / "profiles" / f"{profile_id}.yaml"
    profile_hash = hash_profile(profile_path)
    classes = cfg["classes"]

    meta_rows: List[MetaRow] = []
    label_rows: List[LabelRow] = []
    split_counters: Dict[str, int] = {}
    class_counts: Dict[str, int] = {cls: 0 for cls in classes}
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for ds_path in args.datasets:
        ds = Path(ds_path)
        if not ds.exists():
            raise SystemExit(f"Dataset not found: {ds}")

        meta_batch, label_map = gather_rows(ds, run_id, split_counters, images_dir)
        meta_rows.extend(meta_batch)

        for row in meta_batch:
            class_name = label_map.get(row.id)
            if class_name is None:
                continue
            if class_name not in classes:
                raise SystemExit(f"Class {class_name} not declared in {config_path}")
            label_rows.append(LabelRow(schema_version=1, id=row.id, class_name=class_name))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            splits[row.split].append(row.id)

    meta_path = out_dir / "meta.jsonl"
    label_path = out_dir / "labels.jsonl"
    write_jsonl(meta_path, meta_rows)
    write_jsonl(label_path, label_rows)

    for split in ["train", "val", "test"]:
        with open(out_dir / "splits" / f"{split}.txt", "w") as f_split:
            for sample_id in splits[split]:
                f_split.write(sample_id + "\n")

    shutil.copy2(config_path, out_dir / "config.yaml")

    write_dataset_manifest(
        out_dir,
        run_id=run_id,
        profile_id=profile_id,
        profile_hash=profile_hash,
        profile_path=str(profile_path),
        meta_rows=meta_rows,
        label_rows=label_rows,
        splits=splits,
    )



if __name__ == "__main__":
    main()

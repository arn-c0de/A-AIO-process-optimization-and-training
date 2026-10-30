#!/usr/bin/env python3
"""Create a merged dataset from multiple profile-specific runs.

This aggregates several existing `outputs/sim_data/runs/<name>` folders into
a single dataset folder so you can train a multi-profile checkpoint.

Every label row is written as schema v2 carrying the profile_id of the source
dataset it came from. If the merged sources span 2+ profiles a multi-profile
(manifest_version 2) manifest is written, otherwise a regular v1 manifest.

Usage:
    .venv/bin/python tools/create_multi_dataset.py \
        --datasets outputs/sim_data/runs/run_0001 outputs/sim_data/runs/run_sot23 \
        --out outputs/sim_data/runs/run_multi
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_sim.config import load_config  # noqa: E402
from simple_sim.manifest import (  # noqa: E402
    read_dataset_manifest,
    write_dataset_manifest,
    write_multi_profile_manifest,
)
from simple_sim.profile_hash import hash_profile  # noqa: E402
from simple_sim.schema import LabelRow, MetaRow, read_jsonl, write_jsonl  # noqa: E402

SCRIPT_NAME = "tools/create_multi_dataset.py"


def _profile_entry(profile_id: str, profile_hash: str = "", profile_path: str = "") -> dict[str, str]:
    return {"profile_id": profile_id, "profile_hash": profile_hash, "profile_path": profile_path}


def _profile_from_config(ds: Path, profiles_dir: Path) -> dict[str, str] | None:
    """Fallback for sources without a manifest: use config.yaml run.component_profile."""
    cfg_path = ds / "config.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    run = cfg.get("run") if isinstance(cfg, dict) else None
    pid = str((run or {}).get("component_profile") or "").strip() if isinstance(run, dict) else ""
    if not pid:
        return None
    profile_path = profiles_dir / f"{pid}.yaml"
    phash = hash_profile(profile_path) if profile_path.exists() else ""
    return _profile_entry(pid, phash, str(profile_path) if profile_path.exists() else "")


def source_profiles(ds: Path, profiles_dir: Path) -> tuple[dict[str, str] | None, list[dict[str, str]]]:
    """Return (default_profile, all_profiles) declared by a source dataset.

    default_profile is used for label rows that carry no profile_id themselves
    (v1 labels); it is None for multi-profile (v2) sources.
    """
    manifest_path = ds / "dataset_manifest.json"
    if manifest_path.exists():
        manifest = read_dataset_manifest(manifest_path)
        if int(manifest.get("manifest_version", 1) or 1) == 1:
            cp = manifest["component_profile"]
            prof = _profile_entry(
                str(cp["profile_id"]), str(cp.get("profile_hash") or ""), str(cp.get("profile_path") or "")
            )
            return prof, [prof]
        profs = [
            _profile_entry(
                str(p["profile_id"]), str(p.get("profile_hash") or ""), str(p.get("profile_path") or "")
            )
            for p in manifest["component_profiles"]
        ]
        return None, profs

    prof = _profile_from_config(ds, profiles_dir)
    if prof is None:
        raise SystemExit(
            f"Cannot determine component profile of {ds}: no dataset_manifest.json and no "
            f"run.component_profile in config.yaml (run tools/backfill_manifest.py first)."
        )
    return prof, [prof]


def gather_rows(
    ds: Path,
    run_id: str,
    split_counters: dict[str, int],
    images_dir: Path,
    default_profile_id: str = "",
) -> tuple[list[MetaRow], dict[str, LabelRow]]:
    meta_rows = read_jsonl(ds / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(ds / "labels.jsonl", LabelRow)
    label_map = {row.id: row for row in label_rows}

    new_meta: list[MetaRow] = []
    new_label_map: dict[str, LabelRow] = {}
    for row in meta_rows:
        label = label_map.get(row.id)
        if label is None:
            continue
        profile_id = str(label.profile_id or default_profile_id).strip()
        if not profile_id:
            raise SystemExit(f"Sample {row.id} in {ds} has no profile_id and the source has no default profile")

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
                schema_version=int(row.schema_version),
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
                render_meta=dict(row.render_meta or {}),
            )
        )
        new_label_map[new_id] = LabelRow(
            schema_version=2, id=new_id, class_name=label.class_name, profile_id=profile_id
        )

    return new_meta, new_label_map


def _load_source_config(ds: Path) -> dict[str, Any]:
    cfg_path = ds / "config.yaml"
    if not cfg_path.exists():
        return {}
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def create_multi_dataset(
    datasets: list[Path],
    out_dir: Path,
    *,
    config_path: Path | None = None,
    run_id: str | None = None,
    profiles_dir: Path | None = None,
) -> None:
    datasets = [Path(d) for d in datasets]
    out_dir = Path(out_dir)
    profiles_dir = Path(profiles_dir) if profiles_dir is not None else ROOT / "configs" / "profiles"

    if not datasets:
        raise SystemExit("No source datasets given")
    for ds in datasets:
        if not ds.exists():
            raise SystemExit(f"Dataset not found: {ds}")

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise SystemExit(f"Config not found: {config_path}")
        cfg = load_config(config_path)
    else:
        # Without an explicit config, inherit classes/settings from the first source.
        cfg = _load_source_config(datasets[0])
    classes = cfg.get("classes") or {}
    if not classes:
        src = config_path or (datasets[0] / "config.yaml")
        raise SystemExit(f"No 'classes' defined in {src}; pass --config with a classes section")
    if not run_id and config_path is not None:
        run_id = str((cfg.get("run") or {}).get("run_id") or "")
    run_id = run_id or out_dir.name

    if out_dir.exists():
        raise SystemExit(f"Output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    images_dir = out_dir / "images"
    images_dir.mkdir()
    (out_dir / "splits").mkdir()

    meta_rows: list[MetaRow] = []
    label_rows: list[LabelRow] = []
    split_counters: dict[str, int] = {}
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    profiles: dict[str, dict[str, str]] = {}

    for ds in datasets:
        default_prof, src_profiles = source_profiles(ds, profiles_dir)
        for prof in src_profiles:
            profiles.setdefault(prof["profile_id"], prof)
        default_pid = default_prof["profile_id"] if default_prof else ""

        meta_batch, label_map = gather_rows(ds, run_id, split_counters, images_dir, default_pid)
        meta_rows.extend(meta_batch)

        for row in meta_batch:
            label = label_map[row.id]
            if label.class_name not in classes:
                raise SystemExit(f"Class {label.class_name} not declared in config classes")
            profiles.setdefault(label.profile_id, _profile_entry(label.profile_id))
            label_rows.append(label)
            splits[row.split].append(row.id)

    write_jsonl(out_dir / "meta.jsonl", meta_rows)
    write_jsonl(out_dir / "labels.jsonl", label_rows)

    for split in ["train", "val", "test"]:
        with open(out_dir / "splits" / f"{split}.txt", "w") as f_split:
            for sample_id in splits[split]:
                f_split.write(sample_id + "\n")

    if config_path is not None:
        shutil.copy2(config_path, out_dir / "config.yaml")
    elif (datasets[0] / "config.yaml").exists():
        shutil.copy2(datasets[0] / "config.yaml", out_dir / "config.yaml")

    used_pids = {row.profile_id for row in label_rows}
    used_profiles = [profiles[pid] for pid in sorted(used_pids)]
    if len(used_profiles) >= 2:
        write_multi_profile_manifest(
            out_dir,
            run_id=run_id,
            profiles=used_profiles,
            meta_rows=meta_rows,
            label_rows=label_rows,
            splits=splits,
            script_name=SCRIPT_NAME,
        )
    elif used_profiles:
        only = used_profiles[0]
        write_dataset_manifest(
            out_dir,
            run_id=run_id,
            profile_id=only["profile_id"],
            profile_hash=only["profile_hash"],
            profile_path=only["profile_path"],
            meta_rows=meta_rows,
            label_rows=label_rows,
            splits=splits,
        )
    else:
        print("WARNING: no samples merged; no dataset_manifest.json written")


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
        default=None,
        help="Optional run config for the merged dataset (classes, run_id). "
        "Defaults to the first source dataset's config.yaml.",
    )
    parser.add_argument("--run-id", default=None, help="Run ID for merged samples (default: --config run_id, else the --out folder name).")
    args = parser.parse_args()

    create_multi_dataset(
        [Path(d) for d in args.datasets],
        Path(args.out),
        config_path=Path(args.config) if args.config else None,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

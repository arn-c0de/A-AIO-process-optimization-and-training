"""Single-profile synthetic dataset generation (``scripts/generate.py``)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from simple_sim.config import DEFAULT_COMPONENT_PROFILE, load_config, merge_profile_render, validate_config
from simple_sim.dataset_store import atomic_dataset_dir, write_dataset, write_dataset_files, write_image
from simple_sim.generation.blender import blender_settings, render_blender_records
from simple_sim.generation.common import (
    SPLIT_NAMES,
    build_rows,
    finalize_splits,
    project_root as default_project_root,
    rel_profile_path,
    sample_record,
    summarize,
)
from simple_sim.generators.filter_settings import load_image_filters_env
from simple_sim.generators.opencv_2d import render_roi
from simple_sim.manifest import read_dataset_manifest, write_dataset_manifest, write_multi_profile_manifest
from simple_sim.profile_hash import hash_profile, load_profile
from simple_sim.schema import LabelRow, MetaRow, append_jsonl_rows, read_jsonl, validate_jsonl_pair
from simple_sim.splits import assert_no_overlap, read_split, write_splits
from simple_sim.telemetry import emit


@dataclass
class ExtendState:
    """What ``--extend`` needs from the existing dataset."""

    meta_rows: list[MetaRow] = field(default_factory=list)
    label_rows: list[LabelRow] = field(default_factory=list)
    splits: dict[str, list[str]] = field(default_factory=lambda: {k: [] for k in SPLIT_NAMES})
    manifest: dict[str, Any] | None = None
    manifest_version: int = 1
    profiles: list[dict[str, str]] = field(default_factory=list)
    run_id: str = ""
    start_index: int = 0


def read_existing_max_index(meta_rows: list[MetaRow]) -> int:
    """Highest sample index in *meta_rows* (from ``.../000042`` ids or ``images/000042.png``), or -1."""
    mx = -1
    for r in meta_rows:
        idx: int | None
        try:
            idx = int(str(r.id).split("/")[-1])
        except ValueError:
            try:
                idx = int(Path(r.image_path).stem)
            except ValueError:
                idx = None
        if idx is not None and idx > mx:
            mx = idx
    return mx


def _existing_profile(
    manifest: dict[str, Any], *, profile_id: str, profile_hash: str, profile_rel_path: str
) -> tuple[str, str, list[dict[str, str]]]:
    """Return ``(existing_profile_id, existing_profile_hash, target_profiles)`` from a dataset manifest.

    A multi-profile (v2) dataset may gain a new profile, which is appended to target_profiles.
    """
    version = int(manifest.get("manifest_version", 1))
    if "component_profile" in manifest:
        cp = manifest.get("component_profile") or {}
        return str(cp.get("profile_id") or "").strip(), str(cp.get("profile_hash") or "").strip(), []
    if "component_profiles" not in manifest:
        raise ValueError(
            f"Invalid dataset manifest (manifest_version={version}): missing component_profile/component_profiles."
        )
    profiles = manifest.get("component_profiles") or []
    if not isinstance(profiles, list) or not profiles:
        raise ValueError(
            f"Invalid dataset manifest (manifest_version={version}): missing usable component profile information."
        )
    target: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}
    for p in profiles:
        pobj = p or {}
        row = {
            "profile_id": str(pobj.get("profile_id") or "").strip(),
            "profile_hash": str(pobj.get("profile_hash") or "").strip(),
            "profile_path": str(pobj.get("profile_path") or "").strip(),
        }
        if row["profile_id"] and row["profile_hash"]:
            by_id[row["profile_id"]] = row
            target.append(row)
    if not target:
        raise ValueError(
            f"Invalid dataset manifest (manifest_version={version}): "
            "component_profiles has no valid profile_id/profile_hash entries."
        )
    if profile_id in by_id:
        return profile_id, by_id[profile_id]["profile_hash"], target
    # Allow adding a new single profile to an existing multi-profile dataset.
    target.append({"profile_id": profile_id, "profile_hash": profile_hash, "profile_path": profile_rel_path})
    print(f"✓ Extending multi-profile dataset with new profile: {profile_id}")
    return profile_id, profile_hash, target


def load_extend_state(
    output_dir: Path, *, profile_id: str, profile_hash: str, profile_rel_path: str, run_id: str
) -> ExtendState:
    """Validate an existing dataset for ``--extend`` (files, manifest, profile id/hash) and load it."""
    meta_path = output_dir / "meta.jsonl"
    labels_path = output_dir / "labels.jsonl"
    manifest_path = output_dir / "dataset_manifest.json"
    if not (meta_path.exists() and labels_path.exists() and (output_dir / "config.yaml").exists()):
        raise ValueError(f"--extend requires an existing valid dataset dir with meta.jsonl/labels.jsonl/config.yaml: {output_dir}")
    if not manifest_path.exists():
        raise ValueError(
            f"--extend requires dataset with manifest: {manifest_path}\n"
            f"Legacy datasets must be regenerated or use backfill tool:\n"
            f"  .venv/bin/python tools/backfill_manifest.py --data {output_dir}"
        )

    manifest = read_dataset_manifest(manifest_path)
    existing_id, existing_hash, target_profiles = _existing_profile(
        manifest, profile_id=profile_id, profile_hash=profile_hash, profile_rel_path=profile_rel_path
    )
    if not existing_id or not existing_hash:
        raise ValueError(
            "Invalid dataset manifest: component profile metadata is incomplete "
            "(profile_id/profile_hash required)."
        )
    if existing_id != profile_id:
        raise ValueError(
            f"Profile ID mismatch for --extend:\n"
            f"  Dataset profile: {existing_id}\n"
            f"  Config profile:  {profile_id}\n"
            f"Cannot extend dataset with different component type."
        )
    if existing_hash != profile_hash:
        raise ValueError(
            f"Profile hash mismatch for --extend:\n"
            f"  Dataset hash: {existing_hash[:72]}...\n"
            f"  Config hash:  {profile_hash[:72]}...\n"
            f"Profile '{profile_id}' has changed.\n"
            f"Create a new profile version (e.g., @2) or regenerate dataset."
        )
    print("✓ Profile validation passed for extend mode")

    validate_jsonl_pair(meta_path, labels_path)
    state = ExtendState(
        meta_rows=read_jsonl(meta_path, MetaRow),
        label_rows=read_jsonl(labels_path, LabelRow),
        manifest=manifest,
        manifest_version=int(manifest.get("manifest_version", 1)),
        profiles=target_profiles,
        run_id=run_id,
    )
    dataset_run_id = str(manifest.get("run_id") or "").strip()
    if not dataset_run_id and state.meta_rows:
        dataset_run_id = str(state.meta_rows[0].run_id or "").strip()
    if dataset_run_id and dataset_run_id != run_id:
        print(f"✓ Extend mode: using existing dataset run_id '{dataset_run_id}' (config had '{run_id}')")
        state.run_id = dataset_run_id

    state.start_index = read_existing_max_index(state.meta_rows) + 1
    print(f"\nExtending dataset: {output_dir}")
    print(f"Existing samples: {len(state.meta_rows)} (next index: {state.start_index})")
    for s in SPLIT_NAMES:
        p = output_dir / "splits" / f"{s}.txt"
        if p.exists():
            state.splits[s] = read_split(p)
    return state


def _dedup_profiles(profiles: list[dict[str, str]]) -> list[dict[str, str]]:
    """Unique (by profile_id), sorted, stripped profile entries for a multi-profile manifest."""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for p in sorted(profiles, key=lambda x: str(x.get("profile_id") or "")):
        pid = str(p.get("profile_id") or "").strip()
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append(
            {
                "profile_id": pid,
                "profile_hash": str(p.get("profile_hash") or "").strip(),
                "profile_path": str(p.get("profile_path") or "").strip(),
            }
        )
    return out


def generate_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    extend: bool = False,
    enable_cardinal_rotation_90: bool = True,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Generate a complete dataset from a YAML config into *output_dir*.

    With ``extend=True`` and an existing dataset, new samples are appended (profile id/hash
    must match). Profiles are looked up under ``<project_root>/configs/profiles``.
    Returns a summary (see :func:`simple_sim.generation.common.summarize`).
    """
    output_dir = Path(output_dir)
    root = Path(project_root) if project_root else default_project_root()
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    image_filters = load_image_filters_env()

    run_id = config["run"]["run_id"]
    run_seed = config["run"]["seed"]
    schema_version = config["run"]["schema_version"]
    roi_width = config["roi"]["width_px"]
    roi_height = config["roi"]["height_px"]
    classes = config["classes"]
    backend = config.get("render", {}).get("backend", "opencv_2d")

    profile_id = config["run"].get("component_profile", DEFAULT_COMPONENT_PROFILE)
    profiles_dir = root / "configs" / "profiles"
    profile = load_profile(profile_id, profiles_dir)
    profile_path = profiles_dir / f"{profile_id}.yaml"
    profile_hash_str = hash_profile(profile_path)
    relative_profile_path = rel_profile_path(profile_path, root)

    print(f"Component Profile: {profile_id}")
    print(f"Profile Hash: {profile_hash_str[:72]}...")

    footprint = profile["component"]["footprint"]
    geometry_ranges = profile.get("geometry_ranges")
    tolerances = profile.get("tolerances")
    component_height_mm = float(profile.get("component", {}).get("nominal_dims_mm", {}).get("height", 0.45) or 0.45)
    profile_render_3d = profile.get("render_3d") or {}

    # Backward compatibility: v1 configs can override from config
    if schema_version == 1:
        if "tolerances" in config:
            tolerances = config["tolerances"].get(profile["component"]["package"], tolerances)
        if "geometry_ranges" in config:
            geometry_ranges = config["geometry_ranges"]

    # Fill component_color/component_type from the profile (also persisted in config.yaml).
    config["render"] = merge_profile_render(config["render"], profile)

    ext: ExtendState | None = None
    if extend and output_dir.exists():
        ext = load_extend_state(
            output_dir,
            profile_id=profile_id,
            profile_hash=profile_hash_str,
            profile_rel_path=relative_profile_path,
            run_id=run_id,
        )
        run_id = ext.run_id
    start_index = ext.start_index if ext else 0
    # v2 (multi-profile) datasets need profile ids on labels.
    labels_v2 = ext is not None and ext.manifest_version == 2
    labels_schema_version = 2 if labels_v2 else 1
    labels_profile_id = profile_id if labels_v2 else ""

    print(f"\nGenerating dataset: {run_id}")
    print(f"Image filters: {json.dumps(image_filters, ensure_ascii=True)}")
    print(f"Classes: {classes}")
    total_target = sum(classes.values())
    print(f"Total samples: {total_target}")
    print(f"Render backend: {backend}")
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

    # Intermediate records keyed by stable index; splits are assigned after stratification.
    records: list[dict[str, Any]] = []
    sample_index = int(start_index)
    t_total0 = time.perf_counter()
    for class_name, count in classes.items():
        print(f"\nGenerating {count} samples for class {class_name}")
        t0 = time.perf_counter()
        progress = tqdm(range(count), desc=class_name)
        for i in progress:
            rec, _rng = sample_record(
                sample_index,
                class_name,
                config=config,
                run_seed=run_seed,
                geometry_ranges=geometry_ranges,
                tolerances=tolerances,
                image_filters=image_filters,
                enable_cardinal_rotation_90=enable_cardinal_rotation_90,
                footprint=footprint,
                profile_id=labels_profile_id,
            )
            if rec["class_name"] != class_name:
                # The derived label wins to keep labels consistent with geometry.
                print(f"\nWARNING: Label mismatch for sample {sample_index}: target={class_name}, derived={rec['class_name']}")
            records.append(rec)

            sample_index += 1
            rate = (i + 1) / max(1e-9, time.perf_counter() - t0)
            progress.set_postfix({"idx": sample_index - 1, "samp/s": f"{rate:.1f}"})
            if (i + 1) % 10 == 0 or (i + 1) == count:
                emit(
                    "gen_progress",
                    class_name=class_name,
                    class_i=int(i + 1),
                    class_total=int(count),
                    global_idx=int(sample_index - 1),
                    samp_per_s=float(rate),
                    last_image_path=rec["image_path"],
                )

    print("\nGenerating train/val/test splits")
    extra_render_meta = None
    if backend == "blender_3d":
        extra_render_meta = {
            "backend": "blender_3d",
            "mm_per_px": float(config["roi"]["mm_per_px"]),
            "cycles_samples": int(((config.get("render") or {}).get("blender") or {}).get("samples", 0) or 0),
            "device": blender_settings(config)["device"],
            "profile_render_3d": profile_render_3d,
        }
    meta_rows, label_rows, new_splits = build_rows(
        records,
        config=config,
        run_id=run_id,
        backend=backend,
        image_filters=image_filters,
        labels_schema_version=labels_schema_version,
        # Salted seed keeps split assignment deterministic across extend calls.
        split_seed=int(run_seed) + int(start_index),
        extra_render_meta=extra_render_meta,
    )
    if backend not in ("opencv_2d", "blender_3d"):
        raise ValueError(f"Unsupported render backend: {backend}")

    def render_2d(r: dict[str, Any]) -> np.ndarray:
        return render_roi(
            nominal=r["nominal"],
            defect_params=r["defect"],
            augment=r["augment"],
            roi_size=(roi_width, roi_height),
            config=config["render"],
            tolerances=tolerances,
            rng=np.random.default_rng(int(r["seed"])),
            footprint=footprint,
        )

    def render_3d(out_root: Path, jobs_filename: str, at_least: bool) -> None:
        render_blender_records(
            records,
            config=config,
            roi_width=roi_width,
            roi_height=roi_height,
            footprint=footprint,
            component_height_mm=component_height_mm,
            profile_render_3d=profile_render_3d,
            tolerances=tolerances,
            project_root=root,
            output_root=out_root,
            jobs_filename=jobs_filename,
            expect_at_least=at_least,
        )

    print(f"\nWriting dataset to {output_dir}")
    if ext is not None and ext.meta_rows:
        # Append mode: keep existing dataset, add new images + rows + update split files.
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        if backend == "opencv_2d":
            for r in records:
                write_image(output_dir / r["image_path"], render_2d(r))
        else:
            render_3d(output_dir, "blender_jobs_extend.jsonl", True)
        append_jsonl_rows(output_dir / "meta.jsonl", meta_rows)
        append_jsonl_rows(output_dir / "labels.jsonl", label_rows)

        final_splits = {k: sorted(set(ext.splits.get(k, [])) | set(new_splits[k])) for k in SPLIT_NAMES}
        assert_no_overlap(final_splits)
        write_splits(output_dir, final_splits)
        all_meta_rows = ext.meta_rows + meta_rows
        all_label_rows = ext.label_rows + label_rows
    else:
        if backend == "opencv_2d":
            write_dataset(output_dir, {r["image_path"]: render_2d(r) for r in records}, meta_rows, label_rows, config)
        else:
            with atomic_dataset_dir(output_dir) as temp_dir:
                render_3d(temp_dir, "blender_jobs.jsonl", False)
                write_dataset_files(temp_dir, meta_rows, label_rows, config)
        final_splits = new_splits
        all_meta_rows = meta_rows
        all_label_rows = label_rows

    finalize_splits(output_dir, final_splits, all_label_rows, mark="✓ ")

    if ext is not None and (ext.manifest_version == 2 or (ext.manifest and "component_profiles" in ext.manifest)):
        profiles = ext.profiles or [
            {"profile_id": profile_id, "profile_hash": profile_hash_str, "profile_path": relative_profile_path}
        ]
        write_multi_profile_manifest(
            output_dir=output_dir,
            run_id=run_id,
            profiles=_dedup_profiles(profiles),
            meta_rows=all_meta_rows,
            label_rows=all_label_rows,
            splits=final_splits,
            script_name="scripts/generate.py",
            existing=ext.manifest,
        )
    else:
        write_dataset_manifest(
            output_dir=output_dir,
            run_id=run_id,
            profile_id=profile_id,
            profile_hash=profile_hash_str,
            profile_path=relative_profile_path,
            meta_rows=meta_rows,  # Only new samples for extend history
            label_rows=label_rows,
            splits=final_splits,
            extend=extend,
        )

    total_s = max(1e-9, time.perf_counter() - t_total0)
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(meta_rows)}")
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
            splits={k: int(len(v)) for k, v in final_splits.items()},
        )
    for split_name, sample_ids in final_splits.items():
        print(f"  {split_name}: {len(sample_ids)} samples")
    print("=" * 60)

    result = summarize(output_dir, all_label_rows, final_splits, new_samples=len(meta_rows), elapsed_s=total_s)
    result.update(run_id=run_id, profile_id=profile_id, backend=backend, extend=bool(ext is not None and ext.meta_rows))
    return result

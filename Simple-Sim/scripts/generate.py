#!/usr/bin/env python3
"""Generate synthetic PCB defect dataset."""

import argparse
import sys
import os
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
    apply_image_filter_overrides,
    normalize_image_filters,
    apply_blur,
    apply_noise,
    apply_brightness,
    apply_contrast,
    apply_saturation,
    apply_hue_shift,
    apply_color_temperature,
    apply_vignetting,
    apply_chromatic_aberration,
    apply_lens_distortion,
    apply_motion_blur,
    apply_sharpen,
    apply_shadow,
    apply_reflection,
    apply_dust_particles,
    apply_jpeg_compression,
    apply_perspective_transform,
    render_roi,
)
from simple_sim.dataset_store import write_dataset
from simple_sim.splits import generate_splits, assert_no_overlap, write_splits, check_class_coverage
from simple_sim.telemetry import emit
from simple_sim.profile_hash import load_profile, hash_profile
from simple_sim.manifest import write_dataset_manifest, read_dataset_manifest
from simple_sim.generator_3d import write_jobs_jsonl, render_blender_batch


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


def _load_image_filters_env() -> dict:
    raw = os.environ.get("IMAGE_FILTERS", "").strip()
    if not raw:
        return normalize_image_filters(None)
    try:
        parsed = json.loads(raw)
    except Exception:
        print("[warn] Invalid IMAGE_FILTERS JSON, using defaults.")
        return normalize_image_filters(None)
    return normalize_image_filters(parsed if isinstance(parsed, dict) else None)


def _postprocess_blender_images(records: list[dict], output_root: Path) -> None:
    import cv2

    for r in records:
        aug = r.get("augment", {})
        blur_sigma = float(aug.get("blur_sigma", 0.0) or 0.0)
        noise_stddev = float(aug.get("noise_stddev", 0.0) or 0.0)
        brightness_factor = float(aug.get("brightness_factor", 1.0) or 1.0)
        contrast_factor = float(aug.get("contrast_factor", 1.0) or 1.0)
        saturation_factor = float(aug.get("saturation_factor", 1.0) or 1.0)
        hue_shift_deg = float(aug.get("hue_shift_deg", 0.0) or 0.0)
        color_temperature_kelvin = int(aug.get("color_temperature_kelvin", 5500) or 5500)
        vignetting_strength = float(aug.get("vignetting_strength", 0.0) or 0.0)
        chromatic_strength = float(aug.get("chromatic_strength", 0.0) or 0.0)
        distortion_k1 = float(aug.get("distortion_k1", 0.0) or 0.0)
        distortion_k2 = float(aug.get("distortion_k2", 0.0) or 0.0)
        motion_blur_strength = float(aug.get("motion_blur_strength", 0.0) or 0.0)
        motion_blur_angle = float(aug.get("motion_blur_angle", 0.0) or 0.0)
        sharpen_strength = float(aug.get("sharpen_strength", 0.0) or 0.0)
        shadow_strength = float(aug.get("shadow_strength", 0.0) or 0.0)
        shadow_size = float(aug.get("shadow_size", 0.2) or 0.2)
        reflection_strength = float(aug.get("reflection_strength", 0.0) or 0.0)
        reflection_size = float(aug.get("reflection_size", 0.15) or 0.15)
        dust_density = float(aug.get("dust_density", 0.0) or 0.0)
        dust_size = float(aug.get("dust_size", 2.0) or 2.0)
        jpeg_quality = int(aug.get("jpeg_quality", 100) or 100)
        perspective_strength = float(aug.get("perspective_strength", 0.0) or 0.0)
        perspective_angle_x = float(aug.get("perspective_angle_x", 0.0) or 0.0)
        perspective_angle_y = float(aug.get("perspective_angle_y", 0.0) or 0.0)
        rotation_deg = float(aug.get("rotation_deg", 0.0) or 0.0)

        needs_filter = (
            blur_sigma > 1e-6
            or noise_stddev > 1e-6
            or abs(brightness_factor - 1.0) > 1e-6
            or abs(contrast_factor - 1.0) > 1e-6
            or abs(saturation_factor - 1.0) > 1e-6
            or abs(hue_shift_deg) > 1e-6
            or color_temperature_kelvin != 5500
            or vignetting_strength > 1e-6
            or chromatic_strength > 1e-6
            or abs(distortion_k1) > 1e-6
            or abs(distortion_k2) > 1e-6
            or motion_blur_strength > 1e-6
            or sharpen_strength > 1e-6
            or shadow_strength > 1e-6
            or reflection_strength > 1e-6
            or dust_density > 1e-6
            or jpeg_quality < 100
            or perspective_strength > 1e-6
            or abs(rotation_deg) > 1e-6
        )
        if not needs_filter:
            continue

        path = output_root / str(r["image_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Failed to read Blender output image for filter postprocess: {path}")
        rng = np.random.default_rng(int(r["seed"]))
        img = apply_blur(img, blur_sigma)
        img = apply_noise(img, noise_stddev, rng)
        img = apply_brightness(img, brightness_factor)
        img = apply_contrast(img, contrast_factor)
        img = apply_saturation(img, saturation_factor)
        img = apply_hue_shift(img, hue_shift_deg)
        img = apply_color_temperature(img, color_temperature_kelvin)
        img = apply_vignetting(img, vignetting_strength)
        img = apply_chromatic_aberration(img, chromatic_strength)
        img = apply_lens_distortion(img, distortion_k1, distortion_k2)
        img = apply_motion_blur(img, motion_blur_strength, motion_blur_angle)
        img = apply_sharpen(img, sharpen_strength)
        img = apply_shadow(img, shadow_strength, shadow_size, rng)
        img = apply_reflection(img, reflection_strength, reflection_size, rng)
        img = apply_dust_particles(img, dust_density, dust_size, rng)
        img = apply_jpeg_compression(img, jpeg_quality)
        img = apply_perspective_transform(img, perspective_strength, perspective_angle_x, perspective_angle_y)
        if abs(rotation_deg) > 0.1:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            m = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
            img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if not cv2.imwrite(str(path), img):
            raise IOError(f"Failed to write filtered Blender image: {path}")


def generate_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    extend: bool = False,
    enable_cardinal_rotation_90: bool = True,
):
    """Generate complete dataset from configuration.

    Args:
        config_path: Path to YAML configuration file
        output_dir: Output directory for dataset
        extend: If True and output_dir exists, append new samples instead of overwriting
    """
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    image_filters = _load_image_filters_env()

    run_id = config['run']['run_id']
    run_seed = config['run']['seed']
    schema_version = config['run']['schema_version']
    roi_width = config['roi']['width_px']
    roi_height = config['roi']['height_px']
    classes = config['classes']
    backend = config.get("render", {}).get("backend", "opencv_2d")

    # Load component profile
    # Use project root to find profiles, not config file location
    project_root = Path(__file__).parent.parent
    profile_id = config['run'].get('component_profile', 'chip_0603_resistor@1')
    profiles_dir = project_root / "configs" / "profiles"
    profile = load_profile(profile_id, profiles_dir)
    profile_path = profiles_dir / f"{profile_id}.yaml"
    profile_hash_str = hash_profile(profile_path)

    print(f"Component Profile: {profile_id}")
    print(f"Profile Hash: {profile_hash_str[:72]}...")

    # Extract component-specific parameters from profile
    footprint = profile['component']['footprint']
    geometry_ranges = profile.get('geometry_ranges')
    tolerances = profile.get('tolerances')
    component_height_mm = float(profile.get("component", {}).get("nominal_dims_mm", {}).get("height", 0.45) or 0.45)
    profile_render_3d = profile.get("render_3d") or {}

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
            augment = sample_augment_params(
                config['augment'],
                domain_config,
                rng,
                enable_cardinal_rotation_90=(enable_cardinal_rotation_90 and bool(image_filters.get("cardinal_rotation_90", True))),
            )
            augment = apply_image_filter_overrides(augment, image_filters)

            image_path = f"images/{sample_index:06d}.png"

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
            schema_version=2,
            id=tmp_id,
            run_id=run_id,
            domain=domain_name,
            split='train',  # placeholder; only used for grouping, not persisted
            seed=r['seed'],
            image_path=r['image_path'],
            render_backend=str(backend),
            footprint=footprint,
            nominal=r['nominal'],
            defect=r['defect'],
            augment=r['augment'],
            render_meta={},
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

        render_meta: dict = {}
        if backend == "blender_3d":
            blender_cfg = (config.get("render") or {}).get("blender") or {}
            render_meta = {
                "backend": "blender_3d",
                "mm_per_px": float(config["roi"]["mm_per_px"]),
                "cycles_samples": int(blender_cfg.get("samples", 0) or 0),
                "device": str(blender_cfg.get("device", "CPU")),
                "profile_render_3d": profile_render_3d,
            }

        meta_rows.append(MetaRow(
            schema_version=2,
            id=sample_id,
            run_id=run_id,
            domain=domain_name,
            split=split_name,
            seed=r['seed'],
            image_path=r['image_path'],
            render_backend=str(backend),
            footprint=footprint,
            nominal=r['nominal'],
            defect=r['defect'],
            augment=r['augment'],
            render_meta=render_meta,
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
        if backend == "opencv_2d":
            import cv2
            for r in records:
                seed = int(r["seed"])
                rng = np.random.default_rng(seed)
                img = render_roi(
                    nominal=r["nominal"],
                    defect_params=r["defect"],
                    augment=r["augment"],
                    roi_size=(roi_width, roi_height),
                    config=config["render"],
                    tolerances=tolerances,
                    rng=rng,
                    footprint=footprint,
                )
                full_path = output_dir / r["image_path"]
                full_path.parent.mkdir(parents=True, exist_ok=True)
                ok = cv2.imwrite(str(full_path), img)
                if not ok:
                    raise IOError(f"Failed to write image: {r['image_path']}")
        elif backend == "blender_3d":
            blender_cfg = (config.get("render") or {}).get("blender") or {}
            exe = str(blender_cfg.get("executable", "blender"))
            samples = int(blender_cfg.get("samples", 64))
            device = str(blender_cfg.get("device", "CPU"))
            jobs = []
            for r in records:
                jobs.append({
                    "image_path": r["image_path"],
                    "seed": int(r["seed"]),
                    "mm_per_px": float(config["roi"]["mm_per_px"]),
                    "roi_width_px": int(roi_width),
                    "roi_height_px": int(roi_height),
                    "footprint": str(footprint),
                    "component_height_mm": float(component_height_mm),
                    "nominal": r["nominal"],
                    "defect": r["defect"],
                    "augment": r["augment"],
                    "render_3d": profile_render_3d,
                })
            jobs_path = output_dir / "blender_jobs_extend.jsonl"
            write_jobs_jsonl(jobs_path, jobs)
            render_blender_batch(
                sim_root=project_root,
                jobs_path=jobs_path,
                output_root=output_dir,
                blender_executable=exe,
                cycles_samples=samples,
                device=device,
            )
            _postprocess_blender_images(records, output_dir)
            # Sanity check: ensure Blender produced the expected images.
            try:
                img_count = len(list((output_dir / "images").glob("*.png")))
            except Exception:
                img_count = 0
            if img_count < len(records):
                raise RuntimeError(f"Blender render incomplete (extend): expected >= {len(records)} images, found {img_count} under {output_dir / 'images'}")
        else:
            raise ValueError(f"Unsupported render backend: {backend}")

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
        if backend == "opencv_2d":
            # Keep existing atomic writer for 2D.
            all_images = {}
            for r in records:
                seed = int(r["seed"])
                rng = np.random.default_rng(seed)
                img = render_roi(
                    nominal=r["nominal"],
                    defect_params=r["defect"],
                    augment=r["augment"],
                    roi_size=(roi_width, roi_height),
                    config=config["render"],
                    tolerances=tolerances,
                    rng=rng,
                    footprint=footprint,
                )
                all_images[r["image_path"]] = img
            write_dataset(output_dir, all_images, meta_rows, label_rows, config)
        elif backend == "blender_3d":
            # Atomic write similar to dataset_store.write_dataset, but images come from Blender batch render.
            import shutil
            import yaml
            from simple_sim.dataset_store import validate_dataset_files
            from simple_sim.schema import write_jsonl

            output_dir = Path(output_dir)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_dir = output_dir.with_name(output_dir.name + ".tmp")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)

            try:
                (temp_dir / "images").mkdir(parents=True, exist_ok=True)

                blender_cfg = (config.get("render") or {}).get("blender") or {}
                exe = str(blender_cfg.get("executable", "blender"))
                samples = int(blender_cfg.get("samples", 64))
                device = str(blender_cfg.get("device", "CPU"))

                jobs = []
                for r in records:
                    jobs.append({
                        "image_path": r["image_path"],
                        "seed": int(r["seed"]),
                        "mm_per_px": float(config["roi"]["mm_per_px"]),
                        "roi_width_px": int(roi_width),
                        "roi_height_px": int(roi_height),
                        "footprint": str(footprint),
                        "component_height_mm": float(component_height_mm),
                        "nominal": r["nominal"],
                        "defect": r["defect"],
                        "augment": r["augment"],
                        "render_3d": profile_render_3d,
                    })

                jobs_path = temp_dir / "blender_jobs.jsonl"
                write_jobs_jsonl(jobs_path, jobs)
                render_blender_batch(
                    sim_root=project_root,
                    jobs_path=jobs_path,
                    output_root=temp_dir,
                    blender_executable=exe,
                    cycles_samples=samples,
                    device=device,
                )
                _postprocess_blender_images(records, temp_dir)

                # Sanity check: ensure Blender produced the expected images.
                try:
                    img_count = len(list((temp_dir / "images").glob("*.png")))
                except Exception:
                    img_count = 0
                if img_count != len(records):
                    raise RuntimeError(f"Blender render incomplete: expected {len(records)} images, found {img_count} under {temp_dir / 'images'}")

                write_jsonl(temp_dir / "meta.jsonl", meta_rows)
                write_jsonl(temp_dir / "labels.jsonl", label_rows)

                with open(temp_dir / "config.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

                validate_dataset_files(temp_dir)

                if output_dir.exists():
                    shutil.rmtree(output_dir)
                temp_dir.rename(output_dir)

            except Exception:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise
        else:
            raise ValueError(f"Unsupported render backend: {backend}")

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
    parser.add_argument(
        '--disable-cardinal-rotation-90',
        action='store_true',
        help='Disable 0/90/180/270 base orientation randomization (keep only augment.rotation_deg_range jitter)',
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.out)

    generate_dataset(
        config_path,
        output_dir,
        extend=bool(args.extend),
        enable_cardinal_rotation_90=not bool(args.disable_cardinal_rotation_90),
    )


if __name__ == '__main__':
    main()

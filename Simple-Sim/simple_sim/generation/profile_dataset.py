"""Mixed-profile dataset generation for the profile (component type) classifier."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from simple_sim.config import load_config, merge_profile_render, validate_config
from simple_sim.dataset_store import atomic_dataset_dir, write_dataset_files, write_image
from simple_sim.parallel import imap_ordered, render_workers
from simple_sim.generation.common import (
    build_rows,
    finalize_splits,
    project_root as default_project_root,
    sample_record,
    summarize,
)
from simple_sim.generators.filter_settings import load_image_filters_env
from simple_sim.generators.opencv_2d import render_roi
from simple_sim.manifest import write_multi_profile_manifest
from simple_sim.profile_hash import hash_profile, load_profile


def generate_profile_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    enable_cardinal_rotation_90: bool = True,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Generate a mixed-profile dataset (``run.component_profiles``) with v2 labels.

    Only ``render.backend='opencv_2d'`` is supported. Returns a summary
    (see :func:`simple_sim.generation.common.summarize`).
    """
    output_dir = Path(output_dir)
    root = Path(project_root) if project_root else default_project_root()
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    image_filters = load_image_filters_env()

    run_id = config["run"]["run_id"]
    run_seed = config["run"]["seed"]
    roi_width = config["roi"]["width_px"]
    roi_height = config["roi"]["height_px"]
    classes = config["classes"]
    profile_ids = config["run"]["component_profiles"]
    profiles_dir = root / "configs" / "profiles"

    backend = config.get("render", {}).get("backend", "opencv_2d")
    if backend != "opencv_2d":
        raise ValueError("generate_profile_dataset.py currently supports only render.backend='opencv_2d'")

    profile_data = []
    for pid in profile_ids:
        profile_path = profiles_dir / f"{pid}.yaml"
        profile_hash_str = hash_profile(profile_path)
        profile_data.append(
            {
                "profile_id": pid,
                "profile": load_profile(pid, profiles_dir),
                "profile_hash": profile_hash_str,
                "profile_path": str(profile_path.relative_to(root)),
            }
        )
        print(f"Loaded profile: {pid} (hash: {profile_hash_str[:40]}...)")

    total_per_class = sum(classes.values())
    print(f"\nProfiles: {profile_ids}")
    print(f"Samples per profile: {total_per_class}")
    print(f"Total samples: {total_per_class * len(profile_ids)}")
    print(f"Image filters: {json.dumps(image_filters, ensure_ascii=True)}")

    jobs: list[tuple[dict[str, Any], np.random.Generator, dict[str, Any], dict[str, Any] | None, str]] = []
    records: list[dict[str, Any]] = []
    global_index = 0
    t_total0 = time.perf_counter()

    for pinfo in profile_data:
        pid = pinfo["profile_id"]
        profile = pinfo["profile"]
        footprint = profile["component"]["footprint"]
        geometry_ranges = profile.get("geometry_ranges")
        tolerances = profile.get("tolerances")

        render_config = merge_profile_render(config["render"], profile, config.get("roi"))

        print(f"\nGenerating samples for profile: {pid}")
        for class_name, count in classes.items():
            for _i in tqdm(range(count), desc=f"{pid}/{class_name}"):
                rec, rng = sample_record(
                    global_index,
                    class_name,
                    config=config,
                    run_seed=run_seed,
                    geometry_ranges=geometry_ranges,
                    tolerances=tolerances,
                    image_filters=image_filters,
                    enable_cardinal_rotation_90=enable_cardinal_rotation_90,
                    footprint=footprint,
                    profile_id=pid,
                )
                # Rendering continues the sampling rng (unlike generate.py, which reseeds); each
                # record has its own rng, so rendering them later in threads gives the same images.
                jobs.append((rec, rng, render_config, tolerances, footprint))
                records.append(rec)
                global_index += 1

    print("\nGenerating train/val/test splits")
    meta_rows, label_rows, splits = build_rows(
        records,
        config=config,
        run_id=run_id,
        backend=backend,
        image_filters=image_filters,
        labels_schema_version=2,
        split_seed=run_seed,
    )

    print(f"\nWriting dataset to {output_dir}")
    workers = render_workers()

    def render_job(job: tuple[dict[str, Any], np.random.Generator, dict[str, Any], dict[str, Any] | None, str]) -> None:
        rec, rng, render_config, tolerances, footprint = job
        img = render_roi(
            nominal=rec["nominal"],
            defect_params=rec["defect"],
            augment=rec["augment"],
            roi_size=(roi_width, roi_height),
            config=render_config,
            tolerances=tolerances,
            rng=rng,
            footprint=footprint,
        )
        write_image(temp_dir / rec["image_path"], img)

    with atomic_dataset_dir(output_dir) as temp_dir:
        print(f"Rendering {len(jobs)} images with {workers} worker thread(s)")
        for _ in tqdm(imap_ordered(render_job, jobs, workers=workers), total=len(jobs), desc="render"):
            pass
        write_dataset_files(temp_dir, meta_rows, label_rows, config)
    finalize_splits(output_dir, splits, label_rows)

    write_multi_profile_manifest(
        output_dir=output_dir,
        run_id=run_id,
        profiles=[{k: p[k] for k in ("profile_id", "profile_hash", "profile_path")} for p in profile_data],
        meta_rows=meta_rows,
        label_rows=label_rows,
        splits=splits,
    )

    total_s = max(1e-9, time.perf_counter() - t_total0)
    print("\n" + "=" * 60)
    print("PROFILE DATASET GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Total samples: {len(meta_rows)}")
    print(f"Profiles: {profile_ids}")
    print(f"Generation rate: {len(meta_rows) / total_s:.1f} samples/s  (time: {total_s:.1f}s)")
    for split_name, sample_ids in splits.items():
        print(f"  {split_name}: {len(sample_ids)} samples")
    print("=" * 60)

    result = summarize(output_dir, label_rows, splits, new_samples=len(meta_rows), elapsed_s=total_s)
    result.update(run_id=run_id, profile_ids=list(profile_ids), backend=backend)
    return result

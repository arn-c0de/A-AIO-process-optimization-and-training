"""Mixed-profile dataset generation for the profile (component type) classifier."""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from simple_sim.config import load_config, merge_profile_render, validate_config
from simple_sim.dataset_store import atomic_dataset_dir, write_dataset_files
from simple_sim.generation.common import (
    add_render_meta,
    build_rows,
    finalize_splits,
    RenderTask,
    project_root as default_project_root,
    render_2d_records,
    sample_record,
    summarize,
)
from simple_sim.filter_settings import load_image_filters_env, normalize_image_filters
from simple_sim.generators.backends import backend_of, supporting
from simple_sim.generators.opencv2d.spec import RenderSpec
from simple_sim.generators.version import RENDERER_VERSION
from simple_sim.manifest import write_multi_profile_manifest
from simple_sim.profile_hash import hash_profile, load_profile


log = logging.getLogger(__name__)


def generate_profile_dataset(
    config_path: Path,
    output_dir: Path,
    *,
    enable_cardinal_rotation_90: bool = True,
    project_root: Path | None = None,
    image_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a mixed-profile dataset (``run.component_profiles``) with v2 labels.

    Only ``render.backend='opencv_2d'`` is supported. *image_filters* default to ``$IMAGE_FILTERS``
    (the GUI hand-off, read here as before) and are normalised like in
    :func:`simple_sim.generation.dataset.generate_dataset`. Returns a summary
    (see :func:`simple_sim.generation.common.summarize`).
    """
    output_dir = Path(output_dir)
    root = Path(project_root) if project_root else default_project_root()
    log.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    validate_config(config)

    image_filters = load_image_filters_env() if image_filters is None else normalize_image_filters(image_filters)

    run_id = config["run"]["run_id"]
    run_seed = config["run"]["seed"]
    roi_width = config["roi"]["width_px"]
    roi_height = config["roi"]["height_px"]
    classes = config["classes"]
    profile_ids = config["run"]["component_profiles"]
    profiles_dir = root / "configs" / "profiles"

    info = backend_of(config.get("render"))
    backend = info.backend.value
    if not info.profile_classifier:
        raise ValueError("generate_profile_dataset.py currently supports only render.backend="
                         + " | ".join(repr(b) for b in supporting("profile_classifier")))

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
        log.info(f"Loaded profile: {pid} (hash: {profile_hash_str[:40]}...)")

    total_per_class = sum(classes.values())
    log.info(f"\nProfiles: {profile_ids}")
    log.info(f"Samples per profile: {total_per_class}")
    log.info(f"Total samples: {total_per_class * len(profile_ids)}")
    log.info(f"Image filters: {json.dumps(image_filters, ensure_ascii=True)}")

    jobs: list[RenderTask] = []
    records: list[dict[str, Any]] = []
    global_index = 0
    t_total0 = time.perf_counter()

    for pinfo in profile_data:
        pid = pinfo["profile_id"]
        profile = pinfo["profile"]
        footprint = profile["component"]["footprint"]
        geometry_ranges = profile.get("geometry_ranges")
        tolerances = profile.get("tolerances")
        # Per-domain cameras do not apply here: every sample of this generator is in DOMAIN.
        spec = RenderSpec(roi_size=(roi_width, roi_height), footprint=footprint, tolerances=tolerances,
                          render=merge_profile_render(config["render"], profile, config.get("roi")))

        log.info(f"\nGenerating samples for profile: {pid}")
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
                jobs.append((spec, rec, rng))
                records.append(rec)
                global_index += 1

    log.info("\nGenerating train/val/test splits")
    meta_rows, label_rows, splits = build_rows(
        records,
        config=config,
        run_id=run_id,
        backend=backend,
        image_filters=image_filters,
        labels_schema_version=2,
        split_seed=run_seed,
    )

    log.info(f"\nWriting dataset to {output_dir}")
    with atomic_dataset_dir(output_dir) as temp_dir:
        bar = tqdm(total=len(jobs), desc="render")
        details = render_2d_records(jobs, temp_dir, progress=lambda done, _total: bar.update(1))
        bar.close()
        add_render_meta(meta_rows, details)  # e.g. board_drc
        write_dataset_files(temp_dir, meta_rows, label_rows, config)
    finalize_splits(output_dir, splits, label_rows)

    write_multi_profile_manifest(
        output_dir=output_dir,
        run_id=run_id,
        profiles=[{k: p[k] for k in ("profile_id", "profile_hash", "profile_path")} for p in profile_data],
        meta_rows=meta_rows,
        label_rows=label_rows,
        splits=splits,
        renderer_version=RENDERER_VERSION,
    )

    total_s = max(1e-9, time.perf_counter() - t_total0)
    log.info("\n" + "=" * 60)
    log.info("PROFILE DATASET GENERATION COMPLETE")
    log.info("=" * 60)
    log.info(f"Output directory: {output_dir}")
    log.info(f"Total samples: {len(meta_rows)}")
    log.info(f"Profiles: {profile_ids}")
    log.info(f"Generation rate: {len(meta_rows) / total_s:.1f} samples/s  (time: {total_s:.1f}s)")
    for split_name, sample_ids in splits.items():
        log.info(f"  {split_name}: {len(sample_ids)} samples")
    log.info("=" * 60)

    result = summarize(output_dir, label_rows, splits, new_samples=len(meta_rows), elapsed_s=total_s)
    result.update(run_id=run_id, profile_ids=list(profile_ids), backend=backend)
    return result

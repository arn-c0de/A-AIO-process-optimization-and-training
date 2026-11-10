"""Mixed-profile dataset generation for the profile (component type) classifier."""

from __future__ import annotations

import copy
import logging
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tqdm import tqdm

from simple_sim.config import load_config, validate_config
from simple_sim.dataset_store import atomic_dataset_dir, write_dataset_files
from simple_sim.generation.common import (
    DOMAIN,
    add_render_meta,
    build_rows,
    finalize_splits,
    project_root as default_project_root,
    summarize,
)
from simple_sim.generation.dataset import RunInputs, SampleProgress, plan_records, resolve_run
from simple_sim.filter_settings import load_image_filters_env, normalize_image_filters
from simple_sim.generators.backends import backend_of, supporting
from simple_sim.generators.version import RENDERER_VERSION
from simple_sim.manifest import write_multi_profile_manifest


log = logging.getLogger(__name__)


def _profile_progress(profile_id: str) -> SampleProgress:
    """tqdm bar per ``<profile>/<class>``."""
    state: dict[str, Any] = {}

    def progress(class_name: str, i: int, count: int, _rec: dict[str, Any]) -> None:
        if i == 1:
            state["bar"] = tqdm(total=count, desc=f"{profile_id}/{class_name}")
        state["bar"].update(1)
        if i == count:
            state["bar"].close()

    return progress


def generate_profile_dataset(
    config_path: Path | Mapping[str, Any],
    output_dir: Path,
    *,
    enable_cardinal_rotation_90: bool = True,
    project_root: Path | None = None,
    profiles_dir: Path | None = None,
    image_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a mixed-profile dataset (``run.component_profiles``) with v2 labels.

    Only backends with the ``profile_classifier`` capability (``opencv_2d``) are supported.
    *config_path*: a YAML file or an already loaded config mapping (not modified). Profiles come
    from *profiles_dir* (default ``<project_root>/configs/profiles``); *image_filters* default to
    ``$IMAGE_FILTERS`` (the GUI hand-off; callers that resolve it at their boundary pass it in) and
    are normalised like in :func:`simple_sim.generation.dataset.generate_dataset`.

    Built from the same steps as ``generate_dataset``: one :func:`resolve_run` per profile and
    :func:`plan_records` with consecutive indices. Unlike ``generate``, every sample is in
    ``domain_A`` (no per-domain cameras, no v1 config overrides of the profile) and its image
    continues the sampling generator (``continue_rng``). Returns a summary
    (see :func:`simple_sim.generation.common.summarize`).
    """
    output_dir = Path(output_dir)
    root = Path(project_root) if project_root else default_project_root()
    if isinstance(config_path, Mapping):
        config = copy.deepcopy(dict(config_path))
    else:
        log.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
    validate_config(config)

    image_filters = load_image_filters_env() if image_filters is None else normalize_image_filters(image_filters)

    run_id = config["run"]["run_id"]
    run_seed = config["run"]["seed"]
    classes = config["classes"]
    profile_ids = config["run"]["component_profiles"]
    profiles_dir = Path(profiles_dir) if profiles_dir else root / "configs" / "profiles"

    info = backend_of(config.get("render"))
    backend = info.backend.value
    if not info.profile_classifier:
        raise ValueError("generate_profile_dataset.py currently supports only render.backend="
                         + " | ".join(repr(b) for b in supporting("profile_classifier")))

    runs: list[RunInputs] = []
    for pid in profile_ids:
        # Per-domain cameras do not apply here: every sample of this generator is in DOMAIN.
        run = resolve_run(config, profiles_dir=profiles_dir, root=root, profile_id=pid, domain_cameras=False)
        runs.append(run)
        log.info(f"Loaded profile: {pid} (hash: {run.profile_hash[:40]}...)")

    total_per_class = sum(classes.values())
    log.info(f"\nProfiles: {profile_ids}")
    log.info(f"Samples per profile: {total_per_class}")
    log.info(f"Total samples: {total_per_class * len(profile_ids)}")
    log.info(f"Image filters: {json.dumps(image_filters, ensure_ascii=True)}")

    # Every sample is in one domain: DOMAIN when the config has it, else the config's first one.
    domain = DOMAIN if DOMAIN in config["domains"] else next(iter(config["domains"]))
    t_total0 = time.perf_counter()
    groups: list[tuple[RunInputs, list[dict[str, Any]]]] = []
    next_index = 0
    for run in runs:
        log.info(f"\nGenerating samples for profile: {run.profile_id}")
        # Rendering continues the sampling rng (unlike generate.py, which reseeds); each record has
        # its own rng, so rendering them later in parallel gives the same images.
        recs = plan_records(run, start_index=next_index, image_filters=image_filters,
                            enable_cardinal_rotation_90=enable_cardinal_rotation_90, labels_profile_id=run.profile_id,
                            progress=_profile_progress(run.profile_id), domain=domain, continue_rng=True)
        groups.append((run, recs))
        next_index += len(recs)
    records = [r for _run, recs in groups for r in recs]

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
        add_render_meta(meta_rows, info.render_records(groups, temp_dir))  # e.g. board_drc
        write_dataset_files(temp_dir, meta_rows, label_rows, config)
    finalize_splits(output_dir, splits, label_rows)

    write_multi_profile_manifest(
        output_dir=output_dir,
        run_id=run_id,
        profiles=[{"profile_id": r.profile_id, "profile_hash": r.profile_hash, "profile_path": r.profile_rel_path}
                  for r in runs],
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

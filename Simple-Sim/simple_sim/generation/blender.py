"""Blender 3D batch rendering helpers shared by dataset generation and debug previews."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from simple_sim.generators.blender3d.runner import render_prepared_batch
from simple_sim.generators.blender_3d import render_blender_batch, write_jobs_jsonl
from simple_sim.generators.filter_settings import NEUTRAL_AUGMENT
from simple_sim.generators.opencv_2d import apply_post_filters
from simple_sim.generators.blender3d.bake import bake_fields
from simple_sim.parallel import imap_ordered

# Augment keys that do not by themselves require the post-filter chain:
# rotation_deg is already applied by render_batch.py (global_rotation_root), and the
# angle/size parameters only matter when their strength/density is non-neutral.
_POST_FILTER_SECONDARY_KEYS = frozenset({
    "rotation_deg",
    "perspective_angle_x",
    "perspective_angle_y",
    "motion_blur_angle",
    "shadow_size",
    "reflection_size",
    "dust_size",
})


def blender_settings(config: dict[str, Any]) -> dict[str, Any]:
    """``render.blender`` settings with defaults: executable, samples, device, engine."""
    cfg = (config.get("render") or {}).get("blender") or {}
    return {
        "executable": str(cfg.get("executable", "blender")),
        "samples": int(cfg.get("samples", 64)),
        "device": str(cfg.get("device", "CPU")),
        "engine": str(cfg.get("engine", "BLENDER_EEVEE")),
    }


def blender_job(
    record: dict[str, Any],
    *,
    mm_per_px: float,
    roi_width: int,
    roi_height: int,
    footprint: str,
    component_height_mm: float,
    render_3d: dict[str, Any],
    bake: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One ``render_batch.py`` job from a record with image_path/seed/nominal/defect/augment.

    *bake*: fields for the shared board bake and camera model (``component_type``, ``tolerances``,
    ``texture_dir``, ``camera``), see ``simple_sim.generators.blender3d.bake``.
    """
    return {
        "image_path": record["image_path"],
        "seed": int(record["seed"]),
        "mm_per_px": float(mm_per_px),
        "roi_width_px": int(roi_width),
        "roi_height_px": int(roi_height),
        "footprint": str(footprint),
        "component_height_mm": float(component_height_mm),
        "nominal": record["nominal"],
        "defect": record["defect"],
        "augment": record["augment"],
        "render_3d": render_3d,
        **(bake or {}),
    }


def _needs_post_filter(aug: dict[str, Any]) -> bool:
    return any(
        abs(float(aug.get(key, neutral)) - float(neutral)) > 1e-6
        for key, neutral in NEUTRAL_AUGMENT.items()
        if key not in _POST_FILTER_SECONDARY_KEYS
    )


def postprocess_blender_images(records: Iterable[dict[str, Any]], output_root: Path) -> None:
    """Apply the 2D post-render filter chain to Blender outputs in place.

    Rotation is intentionally not applied: ``render_batch.py`` already rotates the Blender
    scene by ``augment.rotation_deg``. Images whose augment is neutral are left untouched.
    """
    import cv2

    def one(r: dict[str, Any]) -> None:
        aug = {k: v for k, v in (r.get("augment") or {}).items() if v is not None}
        if not _needs_post_filter(aug):
            return
        path = Path(output_root) / str(r["image_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Failed to read Blender output image for filter postprocess: {path}")
        img = apply_post_filters(img, aug, np.random.default_rng(int(r["seed"])))
        if not cv2.imwrite(str(path), img):
            raise OSError(f"Failed to write filtered Blender image: {path}")

    for _ in imap_ordered(one, list(records)):
        pass


def run_blender_jobs(
    jobs: list[dict[str, Any]],
    *,
    sim_root: Path,
    output_root: Path,
    jobs_path: Path,
    settings: dict[str, Any],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Write *jobs* to *jobs_path*, run one Blender batch and post-filter the results.

    *should_stop*: polled while Blender runs; True terminates it and raises
    ``simple_sim.generators.blender3d.runner.BlenderCancelled``. None: run to completion.
    """
    write_jobs_jsonl(jobs_path, jobs)
    render_blender_batch(
        sim_root=sim_root,
        jobs_path=jobs_path,
        output_root=output_root,
        blender_executable=settings["executable"],
        cycles_samples=settings["samples"],
        device=settings["device"],
        engine=settings["engine"],
        should_stop=should_stop,
    )
    postprocess_blender_images(jobs, output_root)


def run_prepared_blender_jobs(
    prepared: list[dict[str, Any]],
    *,
    sim_root: Path,
    output_root: Path,
    jobs_path: Path,
    settings: dict[str, Any],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Like :func:`run_blender_jobs` for jobs whose boards are already baked (``bake.prepare_job``)."""
    render_prepared_batch(
        sim_root=sim_root,
        prepared=prepared,
        prepared_path=jobs_path,
        output_root=output_root,
        blender_executable=settings["executable"],
        cycles_samples=settings["samples"],
        device=settings["device"],
        engine=settings["engine"],
        should_stop=should_stop,
    )
    postprocess_blender_images(prepared, output_root)


def render_blender_records(
    records: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    roi_width: int,
    roi_height: int,
    footprint: str,
    component_height_mm: float,
    profile_render_3d: dict[str, Any],
    tolerances: dict[str, Any] | None,
    project_root: Path,
    output_root: Path,
    jobs_filename: str,
    expect_at_least: bool = False,
) -> None:
    """Render dataset *records* with Blender into ``output_root/images`` and verify the image count."""
    render_cfg = config.get("render") or {}
    bake = {"component_type": str(render_cfg.get("component_type", "")), "tolerances": tolerances,
            **bake_fields(render_cfg, config.get("roi"))}
    jobs = [
        blender_job(
            r,
            mm_per_px=float(config["roi"]["mm_per_px"]),
            roi_width=roi_width,
            roi_height=roi_height,
            footprint=footprint,
            component_height_mm=component_height_mm,
            render_3d=profile_render_3d,
            bake=bake,
        )
        for r in records
    ]
    run_blender_jobs(
        jobs,
        sim_root=project_root,
        output_root=output_root,
        jobs_path=output_root / jobs_filename,
        settings=blender_settings(config),
    )

    images_dir = output_root / "images"
    img_count = len(list(images_dir.glob("*.png"))) if images_dir.is_dir() else 0
    expected = len(records)
    if expect_at_least and img_count < expected:
        raise RuntimeError(
            f"Blender render incomplete (extend): expected >= {expected} images, found {img_count} under {images_dir}"
        )
    if not expect_at_least and img_count != expected:
        raise RuntimeError(f"Blender render incomplete: expected {expected} images, found {img_count} under {images_dir}")

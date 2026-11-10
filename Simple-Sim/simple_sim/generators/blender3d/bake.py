"""Shared 2D/3D look: bake the photoreal board for Blender jobs, apply the camera model to the renders.

Blender only models what needs real 3D (solder fillets, part bodies, leads, lighting). The board
itself (solder mask, traces, vias, silkscreen, pad finish) is the same procedural generator as the
2D backend, baked to a texture, so both backends share one look and one pad layout.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.blender.job_schema import (BOARD_SCALE, DEFAULT_COMPONENT_HEIGHT_MM, BakeOptions, BlenderJob,  # noqa: F401 (BOARD_SCALE re-exported)
                                           PreparedJob)
from simple_sim.board.footprints import DEFAULT_FOOTPRINT
from simple_sim.parallel import imap_ordered
from simple_sim.rng import STREAM_BAKE, render_seed
from simple_sim.generators.opencv2d.photoreal.look import LEGACY_LOOK
from simple_sim.generators.opencv2d.photoreal import (DEFAULT_SUPERSAMPLING, camera_config, camera_model, camera_settings, mask_gloss, render_board,
                                                     rework_probability, sample_camera_pose)

# Supersampling above the default 2 is capped so the supersampled bake stays <= this many px per side
# (a 1024 px ROI at 4x would bake 8192^2 px: ~20 s per image for no visible gain in a 2048 px texture).
BAKE_MAX_SUPERSAMPLED_PX = 4096

#: Job fields of the bake that become the render config of ``render_board`` in :func:`prepare_job`, with
#: the value a job without the field gets (the renderer's defaults; ``camera`` is not baked: its pose
#: is sampled for Blender and the camera model runs on the finished render).
BAKE_CONFIG_DEFAULTS: dict[str, Any] = {
    "texture_dir": None, "rework_artifacts": LEGACY_LOOK.rework, "mask_gloss": LEGACY_LOOK.mask_gloss,
    "supersampling": None, "board": None, "solder": None, "look": None,
}


def profile_job_fields(profile: Mapping[str, Any]) -> BakeOptions:
    """Job fields the board bake needs from a component profile (part type, tombstone threshold)."""
    return {"component_type": str((profile.get("component") or {}).get("type", "")),
            "tolerances": profile.get("tolerances")}


def profile_component_height_mm(profile: Mapping[str, Any]) -> float:
    """``component.nominal_dims_mm.height`` of a profile (:data:`DEFAULT_COMPONENT_HEIGHT_MM` when unset or 0)."""
    dims = (profile.get("component") or {}).get("nominal_dims_mm") or {}
    return float(dims.get("height", DEFAULT_COMPONENT_HEIGHT_MM) or DEFAULT_COMPONENT_HEIGHT_MM)


def bake_fields(render_cfg: Mapping[str, Any], roi: Mapping[str, Any] | None = None, **overrides: Any) -> BakeOptions:
    """Job fields of the board bake from a run config's ``render`` section.

    ``texture_dir``, ``camera``, ``rework_artifacts``, ``mask_gloss``, ``supersampling``,
    ``board``, ``solder`` and ``look`` (whose camera preset and mask gloss fill unset ``camera`` /
    ``mask_gloss``); *overrides* replace any of them (e.g. a preview's camera or supersampling).
    ``mm_per_px`` is deliberately not a bake field: the job's top-level ``mm_per_px`` (always
    ``roi.mm_per_px``, see ``blender_job``) is the one scale of part, scene and board, so a stale
    ``render.mm_per_px`` (e.g. persisted in a dataset's ``config.yaml``) can never override it.
    *roi* is accepted for call-site symmetry and ignored.
    """
    del roi
    fields = {"texture_dir": render_cfg.get("texture_dir"), "camera": camera_config(render_cfg),
              "rework_artifacts": rework_probability(render_cfg),
              "mask_gloss": mask_gloss(render_cfg),
              "supersampling": render_cfg.get("supersampling"), "board": render_cfg.get("board"),
              "solder": render_cfg.get("solder"), "look": render_cfg.get("look")}
    return {**fields, **overrides}


def bake_supersampling(requested: int, board_px: int) -> int:
    """Supersampling for a *board_px* wide bake: *requested*, but above 2 only within the pixel budget."""
    requested = int(requested)
    if requested <= 2:
        return requested
    return max(2, min(requested, BAKE_MAX_SUPERSAMPLED_PX // max(1, int(board_px))))


def prepare_job(job: BlenderJob, work_dir: Path, index: int) -> PreparedJob:
    """Bake the board texture of one job to ``work_dir/board_<index>.png``; return the job with ``board``.

    The camera view (``render.camera``: tilt, zoom) is sampled here, so Blender can place its camera
    and the baked board is large enough for a zoomed-out, oblique, rotated view. Deterministic per
    job seed and thread-safe, so jobs can be baked in parallel or ahead of the Blender run.
    """
    w, h = int(job["roi_width_px"]), int(job["roi_height_px"])
    config: dict[str, Any] = {"component_type": job.get("component_type", "")}
    config.update((key, job.get(key, default)) for key, default in BAKE_CONFIG_DEFAULTS.items())
    config["supersampling"] = config["supersampling"] or DEFAULT_SUPERSAMPLING
    if job.get("mm_per_px"):
        config["mm_per_px"] = float(job["mm_per_px"])
    seed = int(job.get("seed", 0))
    pose = sample_camera_pose(camera_settings(job.get("camera")), np.random.default_rng([seed, 2]))
    scale = max(float(BOARD_SCALE), 1.5 * pose["zoom"] / max(np.cos(np.radians(pose["tilt"])), 0.5))
    config["supersampling"] = bake_supersampling(config["supersampling"], int(np.ceil(max(w, h) * scale)))
    texture, scene = render_board(job["nominal"], job["defect"], config, job.get("tolerances"),
                                  np.random.default_rng(render_seed(seed, STREAM_BAKE)), str(job.get("footprint", DEFAULT_FOOTPRINT)),
                                  (int(np.ceil(w * scale)), int(np.ceil(h * scale))), crowd=scale * scale)
    work_dir.mkdir(parents=True, exist_ok=True)
    path = (work_dir / f"board_{index:06d}.png").resolve()
    if not cv2.imwrite(str(path), texture):
        raise OSError(f"Failed to write board texture: {path}")
    return {**job, "board": {"texture": str(path), "scale": scale, "pose": pose, **scene}}


def prepare_jobs(jobs: list[BlenderJob], work_dir: Path, *, workers: int | None = None) -> list[PreparedJob]:
    """Bake one board texture per job into *work_dir* (in parallel threads) and add the ``board`` scene."""
    work_dir.mkdir(parents=True, exist_ok=True)
    return list(imap_ordered(lambda item: prepare_job(item[1], work_dir, item[0]), list(enumerate(jobs)), workers=workers))


def apply_camera(jobs: list[PreparedJob], output_root: Path, *, workers: int | None = None) -> None:
    """Run every rendered image through the same camera model as the 2D backend (seeded per job, in parallel)."""

    def one(job: PreparedJob) -> None:
        path = output_root / str(job["image_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Missing Blender output image: {path}")
        rng = np.random.default_rng([int(job.get("seed", 0)), 1])  # own stream, independent of the bake
        pose = (job.get("board") or {}).get("pose")
        # Blender renders ortho_scale = roi_mm * zoom, so one pixel covers mm_per_px * zoom (as in the 2D scene).
        mm_per_px = job.get("mm_per_px")
        if mm_per_px is not None:
            mm_per_px = float(mm_per_px) * float((pose or {}).get("zoom", 1.0))
        if not cv2.imwrite(str(path), camera_model(img, rng, job.get("camera"), pose, mm_per_px=mm_per_px)):
            raise OSError(f"Failed to write camera-processed image: {path}")

    for _ in imap_ordered(one, jobs, workers=workers):
        pass

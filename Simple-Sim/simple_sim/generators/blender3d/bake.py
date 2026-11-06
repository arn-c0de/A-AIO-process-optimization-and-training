"""Shared 2D/3D look: bake the photoreal board for Blender jobs, apply the camera model to the renders.

Blender only models what needs real 3D (solder fillets, part bodies, leads, lighting). The board
itself (solder mask, traces, vias, silkscreen, pad finish) is the same procedural generator as the
2D backend, baked to a texture, so both backends share one look and one pad layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.generators.opencv2d.photoreal import camera_model, render_board

BOARD_SCALE = 2  # baked board covers 2x the ROI, so global rotations never reveal its edge


def profile_job_fields(profile: dict[str, Any]) -> dict[str, Any]:
    """Job fields the board bake needs from a component profile (part type, tombstone threshold)."""
    return {"component_type": str((profile.get("component") or {}).get("type", "")),
            "tolerances": profile.get("tolerances")}


def prepare_jobs(jobs: list[dict[str, Any]], work_dir: Path) -> list[dict[str, Any]]:
    """Bake one board texture per job into *work_dir* and add the ``board`` scene description."""
    work_dir.mkdir(parents=True, exist_ok=True)
    prepared = []
    for i, job in enumerate(jobs):
        w, h = int(job["roi_width_px"]), int(job["roi_height_px"])
        config = {"component_type": job.get("component_type", ""), "texture_dir": job.get("texture_dir")}
        rng = np.random.default_rng(int(job.get("seed", 0)))
        texture, scene = render_board(job["nominal"], job["defect"], config, job.get("tolerances"), rng,
                                      str(job.get("footprint", "chip_2pad")), (w * BOARD_SCALE, h * BOARD_SCALE))
        path = (work_dir / f"board_{i:06d}.png").resolve()
        if not cv2.imwrite(str(path), texture):
            raise OSError(f"Failed to write board texture: {path}")
        prepared.append({**job, "board": {"texture": str(path), "scale": BOARD_SCALE, **scene}})
    return prepared


def apply_camera(jobs: list[dict[str, Any]], output_root: Path) -> None:
    """Run every rendered image through the same camera model as the 2D backend (seeded per job)."""
    for job in jobs:
        path = output_root / str(job["image_path"])
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Missing Blender output image: {path}")
        rng = np.random.default_rng([int(job.get("seed", 0)), 1])  # own stream, independent of the bake
        if not cv2.imwrite(str(path), camera_model(img, rng)):
            raise OSError(f"Failed to write camera-processed image: {path}")

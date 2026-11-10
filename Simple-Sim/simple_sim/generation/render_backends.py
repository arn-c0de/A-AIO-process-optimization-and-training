"""Dataset renderers of the render backends (:attr:`simple_sim.generators.backends.BackendInfo.renderer`).

Each function is a :class:`~simple_sim.generators.backends.RecordRenderer`: it renders the records of
``(run, records)`` groups below ``out_root`` and returns one ``render_meta`` detail dict per record.
They are imported lazily by :meth:`BackendInfo.render_records`, so choosing a backend loads only its
renderer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tqdm import tqdm

from simple_sim.telemetry import emit

if TYPE_CHECKING:
    from simple_sim.generation.dataset import RunInputs

RenderGroup = tuple["RunInputs", Sequence[Mapping[str, Any]]]


def render_opencv_2d(groups: Sequence[RenderGroup], out_root: Path, *, jobs_filename: str = "",
                     expect_at_least: bool = False) -> list[dict[str, Any]]:
    """Render every record with the OpenCV 2D renderer in one parallel, streamed pass
    (:func:`simple_sim.generation.common.render_2d_records`).

    Per run one :meth:`RunInputs.render_spec` (per-domain cameras unless ``run.domain_cameras`` is
    off). A record planned with ``continue_rng`` renders from its sampling generator
    (:data:`~simple_sim.generation.dataset.RENDER_RNG`), any other from a fresh one seeded with
    ``record["seed"]``. *jobs_filename* / *expect_at_least* are unused (no batch file).
    """
    from simple_sim.generation.common import render_2d_records
    from simple_sim.generation.dataset import RENDER_RNG

    tasks = []
    for run, records in groups:
        spec = run.render_spec()
        tasks.extend((spec, r, r.get(RENDER_RNG)) for r in records)
    bar = tqdm(total=len(tasks), desc="render")
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        # Telemetry comes from the render pass: planning writes no image, so events from there
        # named files that did not exist yet (or the previous dataset's) at a sampling rate.
        bar.update(1)
        if done % 10 == 0 or done == total:
            record = tasks[done - 1][1]  # results arrive in task order
            emit("gen_progress", class_name=str(record.get("class_name") or ""), done=int(done), total=int(total),
                 global_idx=int(record["index"]), samp_per_s=float(done / max(1e-9, time.perf_counter() - t0)),
                 last_image_path=record["image_path"], image_root=str(out_root))

    try:
        return render_2d_records(tasks, out_root, progress=progress)
    finally:
        bar.close()


def render_blender_3d(groups: Sequence[RenderGroup], out_root: Path, *, jobs_filename: str = "blender_jobs.jsonl",
                      expect_at_least: bool = False) -> list[dict[str, Any]]:
    """Render the records of one run with Blender (:func:`simple_sim.generation.blender.render_blender_records`:
    one jobs file *jobs_filename* below *out_root*, image count check). No per-record details."""
    from simple_sim.generation.blender import render_blender_records

    if len(groups) != 1:
        raise ValueError(f"The Blender backend renders one run per call, got {len(groups)}")
    run, records = groups[0]
    roi_width, roi_height = run.roi_size
    render_blender_records(
        list(records),
        config=run.config,
        roi_width=roi_width,
        roi_height=roi_height,
        footprint=run.footprint,
        component_height_mm=run.component_height_mm,
        profile_render_3d=run.profile_render_3d,
        tolerances=run.tolerances,
        output_root=Path(out_root),
        jobs_filename=jobs_filename,
        expect_at_least=expect_at_least,
    )
    return [{} for _ in records]

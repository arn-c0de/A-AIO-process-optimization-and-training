"""Blender process runner for 3D batch rendering."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .bake import apply_camera, prepare_jobs
from .io import write_jobs_jsonl


def render_blender_batch(
    *,
    sim_root: Path,
    jobs_path: Path,
    output_root: Path,
    blender_executable: str,
    cycles_samples: int,
    device: str = "CPU",
    engine: str = "BLENDER_EEVEE",
    extra_env: dict[str, str] | None = None,
) -> None:
    """Bake the boards of all jobs in *jobs_path* (in parallel), render them in one Blender run, apply the camera."""
    sim_root = Path(sim_root)
    jobs_path = Path(jobs_path)
    output_root = Path(output_root)
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_path}")
    _script_path(sim_root)  # fail before baking
    output_root.mkdir(parents=True, exist_ok=True)

    # Bake the shared board textures next to the output; removed again once Blender is done.
    work_dir = output_root / "_board_textures"
    jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        prepared = prepare_jobs(jobs, work_dir)
        _run_blender(sim_root=sim_root, prepared=prepared, prepared_path=work_dir / "jobs_prepared.jsonl",
                     output_root=output_root, blender_executable=blender_executable, cycles_samples=cycles_samples,
                     device=device, engine=engine, extra_env=extra_env)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    apply_camera(prepared, output_root)


def render_prepared_batch(
    *,
    sim_root: Path,
    prepared: list[dict[str, Any]],
    prepared_path: Path,
    output_root: Path,
    blender_executable: str,
    cycles_samples: int,
    device: str = "CPU",
    engine: str = "BLENDER_EEVEE",
    extra_env: dict[str, str] | None = None,
) -> None:
    """Render jobs already baked with :func:`~.bake.prepare_job` (the caller owns their textures), apply the camera.

    Lets a caller bake the next jobs while Blender renders the current ones.
    """
    Path(output_root).mkdir(parents=True, exist_ok=True)
    _run_blender(sim_root=Path(sim_root), prepared=prepared, prepared_path=Path(prepared_path),
                 output_root=Path(output_root), blender_executable=blender_executable, cycles_samples=cycles_samples,
                 device=device, engine=engine, extra_env=extra_env)
    apply_camera(prepared, Path(output_root))


def _script_path(sim_root: Path) -> Path:
    script_path = sim_root / "simple_sim" / "blender" / "render_batch.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing Blender batch script: {script_path}")
    return script_path


def _run_blender(
    *,
    sim_root: Path,
    prepared: list[dict[str, Any]],
    prepared_path: Path,
    output_root: Path,
    blender_executable: str,
    cycles_samples: int,
    device: str,
    engine: str,
    extra_env: dict[str, str] | None,
) -> None:
    script_path = _script_path(sim_root)
    write_jobs_jsonl(prepared_path, prepared)
    cmd = [
        blender_executable,
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "--jobs",
        str(prepared_path),
        "--out_root",
        str(output_root),
        "--samples",
        str(int(cycles_samples)),
        "--device",
        str(device),
        "--engine",
        str(engine),
    ]

    env = None
    if extra_env:
        import os

        env = os.environ.copy()
        env.update({k: str(v) for k, v in extra_env.items()})

    result = subprocess.run(cmd, cwd=str(sim_root), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Blender batch render failed (exit code {result.returncode}). Command: {cmd}")

"""Blender process runner for 3D batch rendering."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

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
    sim_root = Path(sim_root)
    jobs_path = Path(jobs_path)
    output_root = Path(output_root)

    script_path = sim_root / "simple_sim" / "blender" / "render_batch.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing Blender batch script: {script_path}")
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing jobs file: {jobs_path}")

    output_root.mkdir(parents=True, exist_ok=True)

    # Bake the shared board textures next to the output; removed again once Blender is done.
    work_dir = output_root / "_board_textures"
    jobs = [json.loads(line) for line in jobs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prepared_path = work_dir / "jobs_prepared.jsonl"
    write_jobs_jsonl(prepared_path, prepare_jobs(jobs, work_dir))

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

    try:
        result = subprocess.run(cmd, cwd=str(sim_root), env=env)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    if result.returncode != 0:
        raise RuntimeError(f"Blender batch render failed (exit code {result.returncode}). Command: {cmd}")
    apply_camera(jobs, output_root)

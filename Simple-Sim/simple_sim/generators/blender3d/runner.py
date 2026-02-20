"""Blender process runner for 3D batch rendering."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional


def render_blender_batch(
    *,
    sim_root: Path,
    jobs_path: Path,
    output_root: Path,
    blender_executable: str,
    cycles_samples: int,
    device: str = "CPU",
    extra_env: Optional[Dict[str, str]] = None,
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

    cmd = [
        blender_executable,
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "--jobs",
        str(jobs_path),
        "--out_root",
        str(output_root),
        "--samples",
        str(int(cycles_samples)),
        "--device",
        str(device),
    ]

    env = None
    if extra_env:
        import os

        env = os.environ.copy()
        env.update({k: str(v) for k, v in extra_env.items()})

    result = subprocess.run(cmd, cwd=str(sim_root), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Blender batch render failed (exit code {result.returncode}). Command: {cmd}")

"""3D Blender/Cycles rendering backend (batch).

This module is intentionally thin: it writes a job file and invokes Blender once
in headless mode to render all samples in a batch.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def write_jobs_jsonl(path: Path, jobs: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for j in jobs:
            f.write(json.dumps(j, ensure_ascii=True) + "\n")


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
    """Invoke Blender once to render all jobs listed in jobs_path.

    The Blender Python script writes images under output_root/<image_path>.
    """
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
        "--python", str(script_path),
        "--",
        "--jobs", str(jobs_path),
        "--out_root", str(output_root),
        "--samples", str(int(cycles_samples)),
        "--device", str(device),
    ]

    env = None
    if extra_env:
        import os
        env = os.environ.copy()
        env.update({k: str(v) for k, v in extra_env.items()})

    # Let Blender stream output; upstream caller usually captures/prints it.
    res = subprocess.run(cmd, cwd=str(sim_root), env=env)
    if res.returncode != 0:
        raise RuntimeError(f"Blender batch render failed (exit code {res.returncode}). Command: {cmd}")

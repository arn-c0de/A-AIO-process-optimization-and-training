"""Blender process runner for 3D batch rendering."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .bake import apply_camera, prepare_jobs
from .io import write_jobs_jsonl

StopCheck = Callable[[], bool]
STOP_POLL_S = 0.2        # how often a running Blender is checked against ``should_stop``
TERMINATE_GRACE_S = 5.0  # SIGTERM first, SIGKILL after this many seconds


class BlenderCancelled(RuntimeError):
    """A Blender batch was stopped through ``should_stop`` (the process was terminated)."""


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
    should_stop: StopCheck | None = None,
) -> None:
    """Bake the boards of all jobs in *jobs_path* (in parallel), render them in one Blender run, apply the camera.

    *should_stop*: polled while Blender runs (and before/after the bake); when it returns True the
    Blender process is terminated and :class:`BlenderCancelled` is raised. None: run to completion.
    """
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
        _check_stop(should_stop)
        prepared = prepare_jobs(jobs, work_dir)
        _check_stop(should_stop)
        _run_blender(sim_root=sim_root, prepared=prepared, prepared_path=work_dir / "jobs_prepared.jsonl",
                     output_root=output_root, blender_executable=blender_executable, cycles_samples=cycles_samples,
                     device=device, engine=engine, extra_env=extra_env, should_stop=should_stop)
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
    should_stop: StopCheck | None = None,
) -> None:
    """Render jobs already baked with :func:`~.bake.prepare_job` (the caller owns their textures), apply the camera.

    Lets a caller bake the next jobs while Blender renders the current ones. *should_stop*: see
    :func:`render_blender_batch`.
    """
    Path(output_root).mkdir(parents=True, exist_ok=True)
    _run_blender(sim_root=Path(sim_root), prepared=prepared, prepared_path=Path(prepared_path),
                 output_root=Path(output_root), blender_executable=blender_executable, cycles_samples=cycles_samples,
                 device=device, engine=engine, extra_env=extra_env, should_stop=should_stop)
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
    should_stop: StopCheck | None = None,
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

    if should_stop is None:
        returncode = subprocess.run(cmd, cwd=str(sim_root), env=env).returncode
    else:
        returncode = _run_stoppable(cmd, cwd=str(sim_root), env=env, should_stop=should_stop)
    if returncode != 0:
        raise RuntimeError(f"Blender batch render failed (exit code {returncode}). Command: {cmd}")


def _check_stop(should_stop: StopCheck | None) -> None:
    if should_stop is not None and should_stop():
        raise BlenderCancelled("Blender batch render cancelled")


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _run_stoppable(cmd: list[str], *, cwd: str, env: dict[str, str] | None, should_stop: StopCheck) -> int:
    """Run *cmd*, polling *should_stop*; terminate (then kill) the process on stop or on any exception."""
    proc = subprocess.Popen(cmd, cwd=cwd, env=env)
    try:
        while True:
            try:
                return proc.wait(timeout=STOP_POLL_S)
            except subprocess.TimeoutExpired:
                pass
            if should_stop():
                _terminate(proc)
                raise BlenderCancelled("Blender batch render cancelled (process terminated)")
    except BaseException:
        _terminate(proc)  # no orphaned Blender on errors / KeyboardInterrupt either
        raise

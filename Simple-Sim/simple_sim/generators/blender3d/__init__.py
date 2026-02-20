"""Blender 3D backend split into io/runner modules."""

from .io import write_jobs_jsonl
from .runner import render_blender_batch

__all__ = ["write_jobs_jsonl", "render_blender_batch"]

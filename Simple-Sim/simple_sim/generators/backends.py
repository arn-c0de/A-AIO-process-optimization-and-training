"""Render backends (``render.backend``) and what each supports, in one place (stdlib only).

Config validation, dataset generation and the preview ask :func:`backend_of` / :data:`BACKENDS`
instead of comparing ``"opencv_2d"`` strings at every site, so a new backend is one entry here plus
its renderer. Importing this module loads neither OpenCV nor Blender.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from simple_sim.classes import Backend

DEFAULT_BACKEND = Backend.OPENCV_2D


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Capabilities of a render backend."""

    backend: Backend
    label: str
    domain_camera: bool        # ``domains.<name>.camera`` overrides
    post_filters_camera: bool  # ``render.post_filters: camera``
    profile_classifier: bool   # ``mode: profile_classifier`` datasets (generate-profile-dataset)
    needs_blender: bool        # renders through an external Blender process


BACKENDS: dict[Backend, BackendInfo] = {
    Backend.OPENCV_2D: BackendInfo(Backend.OPENCV_2D, "OpenCV 2D", domain_camera=True, post_filters_camera=True,
                                   profile_classifier=True, needs_blender=False),
    Backend.BLENDER_3D: BackendInfo(Backend.BLENDER_3D, "Blender 3D", domain_camera=False, post_filters_camera=False,
                                    profile_classifier=False, needs_blender=True),
}


def parse_backend(name: Any) -> Backend:
    """``Backend`` of *name*; ``ValueError("Unsupported render backend: ...")`` for anything else."""
    try:
        return Backend(str(name))
    except ValueError:
        raise ValueError(f"Unsupported render backend: {name}") from None


def backend_of(render: Mapping[str, Any] | None) -> BackendInfo:
    """Backend of a ``render`` section (missing ``backend``: :data:`DEFAULT_BACKEND`)."""
    return BACKENDS[parse_backend((render or {}).get("backend", DEFAULT_BACKEND))]


def supporting(capability: str) -> list[str]:
    """Names of the backends with *capability* (a :class:`BackendInfo` flag), for error messages."""
    return [b.value for b, info in BACKENDS.items() if getattr(info, capability)]

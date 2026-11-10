"""Render backends (``render.backend``) and what each supports, in one place (stdlib only).

Config validation, dataset generation and the preview ask :func:`backend_of` / :data:`BACKENDS`
instead of comparing ``"opencv_2d"`` strings at every site, so a new backend is one entry here plus
its renderer. Importing this module loads neither OpenCV nor Blender: the renderers are named as
``"module:function"`` entry points and imported on the first call (:meth:`BackendInfo.render_records`,
:meth:`BackendInfo.render_preview`).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from simple_sim.classes import Backend

DEFAULT_BACKEND = Backend.OPENCV_2D


class RecordRenderer(Protocol):
    """Dataset renderer of a backend (:attr:`BackendInfo.renderer`).

    *groups*: ``(run, records)`` pairs (``run`` a :class:`simple_sim.generation.dataset.RunInputs`,
    records from ``plan_records``), rendered below *out_root* in that order. Returns one
    ``render_meta`` detail dict per record in the same order (empty dicts when there are none).
    *jobs_filename* / *expect_at_least* matter to batch backends (jobs file name, image count check
    of an extend run).
    """

    def __call__(self, groups: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]], out_root: Path, *,
                 jobs_filename: str, expect_at_least: bool) -> list[dict[str, Any]]: ...


def load_entry_point(spec: str) -> Any:
    """The object named by ``"package.module:attr"`` (imported now)."""
    module, sep, attr = spec.partition(":")
    if not sep or not module or not attr:
        raise ValueError(f"Entry point must be 'module:attr', got {spec!r}")
    return getattr(importlib.import_module(module), attr)


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Capabilities and (lazily imported) renderers of a render backend."""

    backend: Backend
    label: str
    domain_camera: bool        # ``domains.<name>.camera`` overrides
    post_filters_camera: bool  # ``render.post_filters: camera``
    profile_classifier: bool   # ``mode: profile_classifier`` datasets (generate-profile-dataset)
    needs_blender: bool        # renders through an external Blender process (previews are batched)
    renderer: str = ""         # ``module:function`` of a :class:`RecordRenderer` (datasets)
    previewer: str = ""        # ``module:function`` rendering one profile's previews now (preview viewer)

    def render_records(self, groups: Sequence[tuple[Any, Sequence[Mapping[str, Any]]]], out_root: Path, *,
                       jobs_filename: str = "blender_jobs.jsonl", expect_at_least: bool = False) -> list[dict[str, Any]]:
        """Render dataset records with this backend (see :class:`RecordRenderer`)."""
        if not self.renderer:
            raise ValueError(f"Render backend {self.backend.value!r} has no dataset renderer")
        render: RecordRenderer = load_entry_point(self.renderer)
        return render(groups, Path(out_root), jobs_filename=jobs_filename, expect_at_least=expect_at_least)

    def render_preview(self, request: Any) -> Any:
        """Render one profile's previews now and return their preview records (*request*: a
        :class:`simple_sim.preview.render.PreviewRequest`)."""
        if not self.previewer:
            raise ValueError(f"Render backend {self.backend.value!r} has no preview renderer")
        return load_entry_point(self.previewer)(request)


BACKENDS: dict[Backend, BackendInfo] = {
    Backend.OPENCV_2D: BackendInfo(Backend.OPENCV_2D, "OpenCV 2D", domain_camera=True, post_filters_camera=True,
                                   profile_classifier=True, needs_blender=False,
                                   renderer="simple_sim.generation.render_backends:render_opencv_2d",
                                   previewer="simple_sim.preview.render:preview_opencv_2d"),
    Backend.BLENDER_3D: BackendInfo(Backend.BLENDER_3D, "Blender 3D", domain_camera=False, post_filters_camera=False,
                                    profile_classifier=False, needs_blender=True,
                                    renderer="simple_sim.generation.render_backends:render_blender_3d",
                                    previewer="simple_sim.preview.render:preview_blender_3d"),
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


def backend_info(name: Any) -> BackendInfo:
    """:class:`BackendInfo` of the backend *name* (``ValueError`` like :func:`parse_backend`)."""
    return BACKENDS[parse_backend(name)]


def supporting(capability: str) -> list[str]:
    """Names of the backends with *capability* (a :class:`BackendInfo` flag), for error messages."""
    return [b.value for b, info in BACKENDS.items() if getattr(info, capability)]

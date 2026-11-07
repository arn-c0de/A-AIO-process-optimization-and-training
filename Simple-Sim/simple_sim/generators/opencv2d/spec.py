"""Render one 2D image from a run config without the dataset pipeline.

:class:`RenderSpec` is everything the 2D renderer needs besides the sampled sample (geometry, defect,
augmentation, seed): the merged ``render`` section per domain, the ROI size, the footprint and the
tolerances of the component profile. It is a small picklable value, so it can be sent to worker
processes, and it replaces the hand-merged config dicts of every caller::

    spec = RenderSpec.from_run_config(cfg, profile)
    img = render_sample(spec, nominal, defect, augment, np.random.default_rng(seed), domain="domain_A")
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from simple_sim.board import collect_drc
from simple_sim.board.footprints import DEFAULT_FOOTPRINT
from simple_sim.config import domain_camera, merge_profile_render

from .render import render_roi


@dataclass(frozen=True, slots=True)
class RenderSpec:
    """Inputs of :func:`render_sample` that are fixed for a whole dataset (or one profile of it).

    ``render``: the run config's ``render`` section with the profile's defaults merged in
    (:func:`simple_sim.config.merge_profile_render`). ``domain_render``: per-domain overrides of it
    (``domains.<name>.camera``); domains not listed render with ``render``.
    """

    roi_size: tuple[int, int]
    render: Mapping[str, Any]
    footprint: str = DEFAULT_FOOTPRINT
    tolerances: Mapping[str, Any] | None = None
    domain_render: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_run_config(cls, config: Mapping[str, Any], profile: Mapping[str, Any], *,
                        tolerances: Mapping[str, Any] | None = None) -> RenderSpec:
        """Spec of a run config (``roi``, ``render``, ``domains``) and its component profile.

        *tolerances* overrides the profile's (schema v1 configs carry their own).
        """
        render = merge_profile_render(dict(config["render"]), dict(profile), config.get("roi"))
        domain_render = {}
        for name, dcfg in (config.get("domains") or {}).items():
            camera = domain_camera(render, dcfg)
            if camera is not None:
                domain_render[str(name)] = {**render, "camera": camera}
        roi = config["roi"]
        return cls(roi_size=(int(roi["width_px"]), int(roi["height_px"])), render=render,
                   footprint=str(profile["component"]["footprint"]),
                   tolerances=tolerances if tolerances is not None else profile.get("tolerances"),
                   domain_render=domain_render)

    def render_config(self, domain: str | None = None) -> dict[str, Any]:
        """The render config of *domain* (a fresh dict)."""
        return dict(self.domain_render.get(str(domain), self.render) if domain is not None else self.render)


def render_sample(spec: RenderSpec, nominal: Mapping[str, float], defect: Mapping[str, Any], augment: Mapping[str, Any],
                  rng: np.random.Generator, *, domain: str | None = None) -> np.ndarray:
    """Render one ROI (BGR uint8, ``spec.roi_size``) of a sampled sample; see :func:`render_roi`."""
    return render_roi(nominal=dict(nominal), defect_params=dict(defect), augment=dict(augment), roi_size=spec.roi_size,
                      config=spec.render_config(domain), tolerances=None if spec.tolerances is None else dict(spec.tolerances),
                      rng=rng, footprint=spec.footprint)


def render_record(spec: RenderSpec, record: Mapping[str, Any], rng: np.random.Generator | None = None) -> np.ndarray:
    """Render a dataset record (``generation.common.sample_record``). Without *rng* the image is drawn
    from a fresh generator seeded with ``record['seed']`` (the contract of ``generate``)."""
    if rng is None:
        rng = np.random.default_rng(int(record["seed"]))
    return render_sample(spec, record["nominal"], record["defect"], record["augment"], rng, domain=record.get("domain"))


def render_record_with_meta(spec: RenderSpec, record: Mapping[str, Any],
                            rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """:func:`render_record` plus the per-sample render details for ``render_meta`` of ``meta.jsonl``.

    Same image and random draws as :func:`render_record`. The details hold ``board_drc``
    (:func:`simple_sim.board.drc_summary`: ``ok``, violation counts per kind, router, design-rule
    preset, DRC mode) when the board model checked the window (``render.board.routing: escape``
    with ``drc: warn | strict``), and nothing otherwise (``legacy`` / ``none`` copper, ``drc: off``).
    """
    with collect_drc() as reports:
        img = render_record(spec, record, rng)
    meta: dict[str, Any] = {}
    if reports:
        meta["board_drc"] = reports[0]  # one board per render window
    return img, meta

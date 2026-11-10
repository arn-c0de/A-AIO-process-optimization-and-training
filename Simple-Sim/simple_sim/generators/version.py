"""Renderer version: identifies the look of generated images, stored in ``dataset_manifest.json``.

Bump :data:`RENDERER_VERSION` whenever the same config and seed would produce different images
(new random draws, changed colours or constants, a changed camera model, ...). ``generate --extend``
refuses to append to a dataset rendered by another version, so a dataset never silently mixes two
looks (override: ``--allow-renderer-mismatch``).

History:
  (missing)  up to v1.0.3: datasets without ``generator.renderer_version``.
  2          photoreal realism pass (mask palette, copper, vias, rework, camera), ``render.board``,
             parallel rendering. ``render.board.routing: legacy`` means "no board model", it is
             *not* pixel-identical to v1.0.3.
  3          logic fixes that change pixels: body side walls / top gradient / edge highlight lit on the
             light side (were inverted), SOT-23 solder bridge routed against the upright body, motion
             blur kernel normalised and centred, sub-pixel chromatic aberration, perspective / motion
             blur angles from the filter settings applied.
  4          the render pass draws from its own random stream (``render_seed``; it used to replay the
             stream that sampled geometry and defect, so view angle / board brightness were functions
             of the geometry), corner-lift cap and SOT-23 pad gap scale with the supersampling, the
             realism sampler's tail is two-sided for signed ranges, flat style and Blender body keep
             the sub-pixel placement of the label.
"""

from __future__ import annotations

RENDERER_VERSION = "4"

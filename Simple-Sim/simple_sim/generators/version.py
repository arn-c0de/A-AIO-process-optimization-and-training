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
"""

from __future__ import annotations

RENDERER_VERSION = "2"

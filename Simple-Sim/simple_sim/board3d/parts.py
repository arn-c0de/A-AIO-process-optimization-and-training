"""Part catalog -> board3d: body heights (and the matched entry) from :mod:`simple_sim.parts`.

Applied when a board3d is loaded (:func:`simple_sim.board3d.load_board3d`) and on import, so a catalog edit shows up
without re-importing. A catalog height replaces an unknown height (``h: null``) and a family default
(``heightSource: "default"``) when the part is matched by MPN or package name; a measured height is never touched.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from simple_sim.parts import PartCatalog


#: catalog fields copied to ``body.catalog`` (and into the digest: a changed kind re-exports the viewer's GLB)
SHAPE_KEYS = ("kind", "orientation", "material", "totalHeightMm", "rows", "cell", "topCover", "variants")


def _texts(c: dict[str, Any]) -> list[str]:
    return [str(c[k]) for k in ("value", "part", "mpn") if c.get(k)]


def apply_part_catalog(board3d: dict[str, Any], catalog: PartCatalog | None = None) -> dict[str, Any]:
    """Set ``body.h`` / ``heightSource: "catalog"`` / ``body.catalog`` in place. Returns ``{"applied", "unmatched",
    "digest"}`` (digest of the applied heights: a changed catalog changes it)."""
    catalog = catalog or PartCatalog.load()
    applied, unmatched = {}, set()
    for c in board3d.get("components", []):
        body = c.get("body")
        if not body:
            continue
        src = body.get("heightSource")
        if src not in (None, "default", "catalog"):
            continue                                      # measured / edited: keep
        m = catalog.lookup(texts=_texts(c), package=str(c.get("package", "")))
        if m is None or m.entry.height_mm is None or (src == "default" and m.how == "regex"):
            if body.get("h") is None:
                unmatched.add(str(c.get("package")))
            continue
        body["h"] = m.entry.height_mm
        body["heightSource"] = "catalog"
        # kind / orientation / material / totalHeightMm: the board renderer builds connectors and other parts
        # without a family builder from them (``board3d.footprints.board_part``)
        shape = {k: m.entry.data[k] for k in SHAPE_KEYS if k in m.entry.data}
        body["catalog"] = {"id": m.entry.id, "how": m.how, "source": m.entry.source_kind, **shape}
        applied[c["refdes"]] = [m.entry.id, m.entry.height_mm, *[shape.get(k) for k in SHAPE_KEYS]]
    digest = hashlib.sha1(json.dumps(applied, sort_keys=True).encode()).hexdigest()[:12]
    return {"applied": applied, "unmatched": sorted(unmatched), "digest": digest}


__all__ = ["apply_part_catalog"]

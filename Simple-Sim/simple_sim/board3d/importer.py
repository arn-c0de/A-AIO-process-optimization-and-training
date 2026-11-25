"""T0: DeepVision product -> ``board3d.json`` (one board side) plus an inventory report.

Per placement of the side (``pnp.csv``):

* **body box** from the app's body polygon (``simple-aoi/<side>/parts.json``; the four corners are
  ``[-w/2,-h/2] [w/2,-h/2] [w/2,h/2] [-w/2,h/2]`` rotated, see ``package-library/src/shapes.js``, so edge 0->1 is
  the part X axis, its angle the final rotation incl. the project's rotation correction). Placements the app
  lists without a body fall back to the IPC-7351 name (:func:`~simple_sim.board3d.packages.ipc_dimensions`), at the
  PnP rotation; without both the part stays in the scene with ``body: null`` and is reported;
* **family** (3D builder) and **stage**: ``smd`` (on the SMD-soldered flatbed scan), ``tht`` (through-hole,
  only on photos of the fully populated board), ``mount`` (mounting holes: screws / bolts in the mounted stage);
* **dnp**: only SMD sites (DeepVision's list also names every THT part, being built from SMD-stage scans);
* **height**: typical value (``heightSource: "default"``), not measured.

Images are referenced by path + sha256 only; the scene carries no image data.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from simple_sim.board3d import SCHEMA
from simple_sim.board3d.dv_store import ProductStore, count_media, sha256
from simple_sim.board3d.packages import NEW_FAMILIES, default_height_mm, family_status, ipc_dimensions, package_family

BOARD_THICKNESS_MM = 1.6   # FR-4 standard; not measured on the board


def body_from_polygon(points: list[list[float]]) -> dict[str, float]:
    """Centre, size along part X (``l``) / Y (``w``) and rotation of a body polygon in board mm. Four points:
    the app's rotated rectangle; more points (round parts): bounding circle."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    if len(points) == 4:
        (x0, y0), (x1, y1), (x2, y2) = points[0], points[1], points[2]
        l, w = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
        rot = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 360
    else:
        l = w = 2 * max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys, strict=True))
        rot = 0.0
    return {"x": round(cx, 4), "y": round(cy, 4), "l": round(l, 4), "w": round(w, 4), "rotDeg": round(rot, 3)}


def _stage(mount: str | None, family: str) -> str:
    if family == "mounting_hole":
        return "mount"
    return "tht" if mount == "tht" else "smd"


def _component(pl, part: dict[str, Any] | None, dnp: bool) -> tuple[dict[str, Any], str | None]:
    """One scene component and the reason it has no body (or None)."""
    family = package_family(pl.package)
    reason = None
    if part and part.get("outline"):
        b = body_from_polygon(part["outline"])
        shape_source = "app"
    elif dims := ipc_dimensions(pl.package):
        b = {"x": pl.x_mm, "y": pl.y_mm, "l": dims["bodyW"], "w": dims["bodyH"], "rotDeg": pl.rotation_deg % 360}
        shape_source = "ipc-estimate" if dims.get("estimated") else "ipc-name"
    else:
        b, shape_source, reason = None, None, "no body in parts.json and no size in the package name"
    height = default_height_mm(pl.package, family, b["l"], b["w"]) if b else None
    comp: dict[str, Any] = {
        "refdes": pl.refdes, "id": (part or {}).get("id"), "package": pl.package, "part": pl.part, "value": pl.value,
        "family": family, "familyStatus": family_status(family), "stage": _stage((part or {}).get("mount"), family),
        "pnp": {"x": pl.x_mm, "y": pl.y_mm, "rotDeg": pl.rotation_deg},
        "body": None if b is None else {"l": b["l"], "w": b["w"], "h": height,
                                        "heightSource": "default" if height is not None else None},
        "pose": None if b is None else {"x": b["x"], "y": b["y"], "rotDeg": b["rotDeg"], "z": 0.0, "tiltDeg": [0.0, 0.0]},
        "shapeSource": shape_source, "dnp": dnp, "texture": None, "solder": {"volume": 1.0},
    }
    return comp, reason


def build_board3d(store: ProductStore, side: str, photo_dirs: list[Path] | None = None,
               hash_images: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """(scene, inventory) for one side of *store*'s product."""
    store.check()
    placements = [p for p in store.pnp() if p.side == side]
    parts_doc = store.parts(side)
    parts = {p["name"]: p for p in (parts_doc or {}).get("parts", [])}
    dnp_ids = store.dnp_ids(side)
    components, missing = [], []
    for pl in placements:
        part = parts.get(pl.refdes)
        # DeepVision's DNP list comes from the SMD-soldered good scans, where every through-hole part and
        # mounting hole is "empty" too. Only an SMD site is really not placed; THT / mount parts are simply
        # not there yet in the SMD stage (the stage says that already).
        flagged = bool(part and (part.get("dnp") or part.get("id") in dnp_ids))
        comp, reason = _component(pl, part, False)
        comp["dnp"] = flagged and comp["stage"] == "smd"
        components.append(comp)
        if reason:
            missing.append({"refdes": pl.refdes, "package": pl.package, "reason": reason})

    board = (parts_doc or {}).get("board") or {}
    if not board:
        xs, ys = [p.x_mm for p in placements], [p.y_mm for p in placements]
        board = {"widthMm": math.ceil(max(xs)), "heightMm": math.ceil(max(ys)), "estimated": True}

    def ref(path: Path | None, **extra: Any) -> dict[str, Any] | None:
        if path is None:
            return None
        return {"path": str(path), "sha256": sha256(path) if hash_images else None, **extra}

    scan = store.scan(side)
    sources: dict[str, Any] = {
        "dvRoot": str(store.dv_root), "product": store.product,
        "pnp": ref(store.dir / "pnp.csv"),
        "parts": ref(store.dir / "simple-aoi" / side / "parts.json") if parts_doc else None,
        "scan": None if scan is None else ref(scan[1], boardToImage=scan[0].get("boardToImage"), width=scan[0].get("width"),
                                              height=scan[0].get("height"), stage="smd", original=scan[0].get("source"),
                                              fitResidualMm=(scan[0].get("fit") or {}).get("residualMm")),
        "mountedRender": None if store.mounted_render(side) is None else ref(
            store.mounted_render(side), frame=(parts_doc or {}).get("frame"), stage="mounted",
            photos=(parts_doc or {}).get("photos")),
        "packageMap": ref(store.dir / "package-map.json") if store.package_map() is not None else None,
        "photoDirs": [{"path": str(d), **count_media(d)} for d in photo_dirs or [] if d.is_dir()],
        "mechanics": store.mechanics(side),
    }
    board3d = {
        "schema": SCHEMA, "product": store.product, "side": side,
        "createdAt": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "widthMm": board.get("widthMm"), "heightMm": board.get("heightMm"), "thicknessMm": BOARD_THICKNESS_MM,
        "coordinates": {"unit": "mm", "origin": "bottom-left", "yAxisUp": True},
        "stages": ["smd", "tht", "mount"],
        "sources": sources,
        "substrate": None,          # T1
        "components": components,
        "camera": None, "light": None,  # T4
    }
    from simple_sim.board3d.parts import apply_part_catalog
    apply_part_catalog(board3d)
    return board3d, inventory(board3d, missing)


def inventory(board3d: dict[str, Any], missing: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per package / family / stage / shape source and what T2 still has to build."""
    comps = board3d["components"]
    by_pkg: dict[str, dict[str, Any]] = {}
    for c in comps:
        e = by_pkg.setdefault(c["package"], {"package": c["package"], "count": 0, "family": c["family"],
                                             "familyStatus": c["familyStatus"], "stages": Counter(), "shapeSources": Counter(),
                                             "bodyMm": None, "heightMm": None, "dnp": 0})
        e["count"] += 1
        e["stages"][c["stage"]] += 1
        e["shapeSources"][c["shapeSource"] or "none"] += 1
        e["dnp"] += c["dnp"]
        if c["body"] and e["bodyMm"] is None:
            e["bodyMm"] = [c["body"]["l"], c["body"]["w"]]
            e["heightMm"] = c["body"]["h"]
    packages = sorted(by_pkg.values(), key=lambda e: (-e["count"], e["package"]))
    for e in packages:
        e["stages"], e["shapeSources"] = dict(e["stages"]), dict(e["shapeSources"])
    todo: dict[str, list[str]] = defaultdict(list)
    for e in packages:
        if e["familyStatus"] != "builder":
            todo[e["family"]].append(e["package"])
    no_height = sorted({c["package"] for c in comps if c["body"] and c["body"]["h"] is None})
    return {
        "product": board3d["product"], "side": board3d["side"], "components": len(comps),
        "byStage": dict(Counter(c["stage"] for c in comps)),
        "byFamily": dict(Counter(c["family"] for c in comps).most_common()),
        "byFamilyStatus": dict(Counter(c["familyStatus"] for c in comps)),
        "byShapeSource": dict(Counter(c["shapeSource"] or "none" for c in comps)),
        "dnp": sum(c["dnp"] for c in comps),
        "withoutBody": missing,
        "withoutHeight": no_height,
        "familiesToBuild": {f: {"what": NEW_FAMILIES.get(f, "generic box"), "packages": p,
                                "parts": sum(by_pkg[x]["count"] for x in p)} for f, p in todo.items()},
        "packages": packages,
        "sources": {k: (v if k in ("photoDirs", "mechanics") else v is not None) for k, v in board3d["sources"].items()
                    if k not in ("dvRoot", "product")},
    }

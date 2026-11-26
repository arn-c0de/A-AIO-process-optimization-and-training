"""3D-Board-Renderer -> Blender job for one window (``simple_sim/blender/render_board3d.py``), and defect diffs.

A defect diff (``defects.json``) is a list of operations applied to a copy of the scene before the job is
built; the nominal pose stays in the job, so the photo texture moves with the part:

=================  ===========================================================================================
``op``             fields
=================  ===========================================================================================
``move``           ``refdes``, ``dx``, ``dy`` (mm), ``dRot`` (deg)
``remove``         ``refdes`` (pads + reflowed paste stay)
``tombstone``      ``refdes``, ``tiltDeg`` (default 70), ``raisedSign`` (+1 / -1: which end lifts)
``excess``         ``refdes``, ``pads`` (indices; default all)
``lift``           ``refdes``, ``dz`` (mm; part floats above its pads)
``bridge``         ``refdes``, ``pad`` and ``to`` (refdes, default the same part), ``toPad``: a solder bar between the two
                   lands (pin to pin of one IC, or to the nearest land of a neighbour)
=================  ===========================================================================================
"""

from __future__ import annotations

import copy
import math
from typing import Any

from simple_sim.board3d.footprints import footprint_for
from simple_sim.board3d.substrate import BoardRaster

#: laminate thickness of the board3d's board (standard 1.6 mm FR4)
BOARD_THICKNESS_MM = 1.6

OPS = ("move", "remove", "tombstone", "excess", "lift", "bridge")


def pad_world(c: dict[str, Any], index: int, pose: dict[str, Any] | None = None) -> tuple[float, float, float, float]:
    """Land *index* of component *c* in board mm: ``(x, y, w, h)``, w/h in the part frame (lands never move:
    the nominal pose by default)."""
    pose = pose or c.get("nominalPose") or c["pose"]
    px, py, pw, ph = footprint_for(c)["pads"][index]
    a = math.radians(pose["rotDeg"])
    return (pose["x"] + px * math.cos(a) - py * math.sin(a), pose["y"] + px * math.sin(a) + py * math.cos(a), pw, ph)


def apply_fit(board3d: dict[str, Any]) -> dict[str, Any]:
    """*board3d* with every accepted measured fit (``component["fit"]``, :mod:`simple_sim.board3d.fit`) made the part's
    pose and body size, then the photo outline fit (``component["photoFit"]``, :func:`simple_sim.board3d.parttex.photo_fit`):
    the board3d then shows the board as it really is, defects are relative to that. Idempotent (``fitApplied``);
    returns *board3d* itself when there is nothing to apply, else a copy."""
    if board3d.get("fitApplied") or not any((c.get("fit") or {}).get("accepted") or c.get("photoFit") for c in board3d["components"]):
        return board3d
    out = copy.deepcopy(board3d)
    for c in out["components"]:
        if c.get("pose") is None or c.get("body") is None:
            continue
        f = c.get("fit") or {}
        if f.get("accepted"):
            c["pose"].update({"x": f["x"], "y": f["y"], "rotDeg": f["rotDeg"]})
            c["body"].update({"l": f["l"], "w": f["w"]})
        if pf := c.get("photoFit"):
            # the photos' outline (parttex.photo_fit): edges trimmed where the photos show board inside the body
            c["pose"].update({"x": pf["x"], "y": pf["y"]})
            c["body"].update({"l": pf["l"], "w": pf["w"]})
    out["fitApplied"] = True
    return out


def apply_defects(board3d: dict[str, Any], ops: list[dict[str, Any]]) -> dict[str, Any]:
    """Copy of *board3d* (measured fits applied first, :func:`apply_fit`) with every op applied; component entries get
    ``defects`` (the ops that hit them)."""
    out = copy.deepcopy(apply_fit(board3d))
    by = {c["refdes"]: c for c in out["components"]}
    for op in ops:
        kind = op.get("op")
        if kind not in OPS:
            raise ValueError(f"unknown defect op {kind!r} (known: {', '.join(OPS)})")
        c = by.get(op.get("refdes"))
        if c is None or c.get("pose") is None:
            raise ValueError(f"{kind}: no placed component {op.get('refdes')!r}")
        c.setdefault("nominalPose", copy.deepcopy(c["pose"]))
        c.setdefault("defects", []).append(dict(op))
        if kind == "move":
            c["pose"]["x"] += float(op.get("dx", 0.0))
            c["pose"]["y"] += float(op.get("dy", 0.0))
            c["pose"]["rotDeg"] = (c["pose"]["rotDeg"] + float(op.get("dRot", 0.0))) % 360
        elif kind == "lift":
            c["pose"]["z"] = float(op.get("dz", 0.1))
        elif kind == "bridge":
            other = by.get(op.get("to") or c["refdes"])
            if other is None or other.get("pose") is None:
                raise ValueError(f"bridge: no placed component {op.get('to')!r}")
            i, j = int(op.get("pad", 0)), int(op.get("toPad", 1 if other is c else 0))
            for comp, k in ((c, i), (other, j)):
                if not 0 <= k < len(footprint_for(comp)["pads"]):
                    raise ValueError(f"bridge: {comp['refdes']} has no land {k}")
            if other is c and i == j:
                raise ValueError("bridge: needs two different lands")
            out.setdefault("bridges", []).append({"from": [c["refdes"], i], "to": [other["refdes"], j],
                                                  "a": pad_world(c, i), "b": pad_world(other, j)})
    return out


def _in_window(c: dict[str, Any], win: BoardRaster, margin: float) -> bool:
    p, b = c.get("pose"), c.get("body")
    if not p or not b:
        return False
    r = 0.5 * max(b["l"], b["w"]) + margin
    return win.x0 - r <= p["x"] <= win.x0 + win.w + r and win.y0 - r <= p["y"] <= win.y0 + win.h + r


def job_component(c: dict[str, Any]) -> dict[str, Any]:
    fp = footprint_for(c)
    defects = {d["op"]: d for d in c.get("defects", [])}
    tomb = defects.get("tombstone")
    exc = defects.get("excess")
    return {
        "refdes": c["refdes"], "family": c["family"], "l": c["body"]["l"], "w": c["body"]["w"], "height": fp["height"],
        "heightGuess": fp["heightGuess"],
        "pads": [] if c["dnp"] else [list(p) for p in fp["pads"]],  # DNP: bare lands, no paste
        "thermal": fp["thermal"], "part": fp["part"],
        "pose": c["pose"], "nominal": c.get("nominalPose") or c["pose"],
        "missing": "remove" in defects or c["dnp"],
        "tombstone": None if tomb is None else {"tiltDeg": float(tomb.get("tiltDeg", 70.0)), "raisedSign": float(tomb.get("raisedSign", 1.0))},
        "excess": [] if exc is None else list(exc.get("pads") or range(len(fp["pads"]))),
    }


#: clearance (mm) of a bracket's cut-out around the connector body that passes through it
BRACKET_HOLE_MARGIN_MM = 0.3
BRACKET_EDGES = ("x-", "x+", "y-", "y+")


def bracket_job(bracket: dict[str, Any], board3d: dict[str, Any]) -> dict[str, Any]:
    """A front / edge bracket (``mechanics.json`` ``brackets[]``) for the Blender job: a plate perpendicular to the board
    along its ``edge`` (``y+``: the plane y = ``innerMm`` .. + ``thicknessMm``), from ``fromMm`` to ``toMm`` along the edge,
    ``bottomMm`` .. ``topMm`` in z (board top = 0), with a cut-out around the body of every connector in ``holesFor``
    (its extent along the edge, z from the board to its height, + ``holeMarginMm``)."""
    edge = str(bracket["edge"])
    if edge not in BRACKET_EDGES:
        raise ValueError(f"bracket edge must be one of {', '.join(BRACKET_EDGES)}, not {edge!r}")
    margin = float(bracket.get("holeMarginMm", BRACKET_HOLE_MARGIN_MM))
    along = 0 if edge[0] == "y" else 1
    by = {c["refdes"]: c for c in apply_fit(board3d)["components"]}
    holes = []
    for ref in bracket.get("holesFor", []):
        c = by.get(ref)
        if c is None or not c.get("pose") or not c.get("body"):
            raise ValueError(f"bracket: no placed component {ref!r} for a cut-out")
        jc = job_component(c)
        p, a = c["pose"], math.radians(c["pose"]["rotDeg"])
        ext = [(p["x"] + dx * math.cos(a) - dy * math.sin(a), p["y"] + dx * math.sin(a) + dy * math.cos(a))
               for dx in (-jc["l"] / 2, jc["l"] / 2) for dy in (-jc["w"] / 2, jc["w"] / 2)]
        vals = [e[along] for e in ext]
        holes.append([min(vals) - margin, max(vals) + margin, -margin, float(jc["height"]) + margin])
    return {"edge": edge, "innerMm": float(bracket["innerMm"]), "thicknessMm": float(bracket.get("thicknessMm", 1.0)),
            "fromMm": float(bracket["fromMm"]), "toMm": float(bracket["toMm"]), "bottomMm": float(bracket["bottomMm"]),
            "topMm": float(bracket["topMm"]), "holes": holes}


def build_job(board3d: dict[str, Any], window: BoardRaster, *, out: str, substrate: dict[str, Any], photo: dict[str, Any] | None,
              stages: set[str], light: dict[str, Any] | None = None, samples: int = 64, device: str = "GPU",
              engine: str = "CYCLES", seed: int = 0, margin_mm: float = 3.0) -> dict[str, Any]:
    """The JSON job of :mod:`simple_sim.blender.render_board3d` for *window* (components of *stages* inside it; measured
    fits applied, :func:`apply_fit`)."""
    board3d = apply_fit(board3d)
    comps = [job_component(c) for c in board3d["components"] if c["stage"] in stages and _in_window(c, window, margin_mm)]
    for jc in comps:
        spec = jc["part"].get("connector")
        if spec and spec.get("kind") == "battery_holder" and "cell" not in spec:
            # the coin cell goes in at final assembly: in the holder once the mount stage is shown (the SMD scan shows
            # it empty, the photos of the mounted board with the cell); the catalog's ``cell`` overrides
            jc["part"] = {**jc["part"], "connector": {**spec, "cell": "mount" in stages}}
    bridges = [b for b in board3d.get("bridges", []) if any(
        window.x0 - margin_mm <= p[0] <= window.x0 + window.w + margin_mm and window.y0 - margin_mm <= p[1] <= window.y0 + window.h + margin_mm
        for p in (b["a"], b["b"]))]
    return {"out": out, "window": window.meta(), "widthPx": window.cols, "heightPx": window.rows, "samples": samples,
            "device": device, "engine": engine, "seed": seed, "substrate": substrate, "photo": photo,
            "light": light or {"preset": "flatbed"}, "components": comps, "bridges": bridges,
            # mechanics are fitted at final assembly: only with the mount stage
            "brackets": [bracket_job(b, board3d) for b in board3d.get("brackets", [])] if "mount" in stages else [],
            "board": {"widthMm": float(board3d.get("widthMm", window.w)), "heightMm": float(board3d.get("heightMm", window.h)),
                      "thicknessMm": BOARD_THICKNESS_MM}}

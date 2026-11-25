"""Pad layout and Blender part spec of one board3d component, in the part's own frame (mm).

Frame: origin = body centre, x along the body's ``l`` (part X), y along ``w``; rotation and position come from
the component pose. Pads are ``(cx, cy, w, h)`` rectangles. The layouts follow IPC-7351 land patterns
**approximately** (pad sizes are rules of thumb, positions from body size, pin count and pitch in the package
name); exact lands would need the CAD footprints. Gull-wing rows run along the long body axis with the leads
leaving across the short axis to the span of the name (``SOT95P280``: pitch 0.95, span 2.80).

The part spec is the scene ``part`` of ``render_batch.py`` (:class:`~simple_sim.blender.job_schema.ScenePart`):
the family's row of :data:`~simple_sim.board.footprints.FAMILIES` when it has a builder, a plain moulded box
otherwise. ``height_ratio`` is set so that the body height is exactly the board3d's ``body.h`` and the marking
is off (the photo texture carries the real one).
"""

from __future__ import annotations

import re
from typing import Any

from simple_sim.board.footprints import FAMILIES, scene_part
from simple_sim.board3d.packages import ipc_dimensions

Pad = tuple[float, float, float, float]

#: body height used for parts whose height is still unknown (connectors before T3), flagged ``heightGuess``
GUESS_HEIGHT_MM = 2.5
_HEIGHT_RATIO_EXACT = 1000.0

_CHIP_FAMILIES = {"resistor", "capacitor", "inductor", "led", "diode", "filter", "power_inductor"}


def _pitch(package: str, default: float) -> float:
    m = re.search(r"(\d+)P\d", package.upper())
    return int(m[1]) / 100 if m else default


def _pins(package: str) -> int | None:
    m = re.search(r"-(\d+)(?:[_A-Z]|$)", package.upper())
    return int(m[1]) if m else None


def _row(n: int, pitch: float) -> list[float]:
    return [(i - (n - 1) / 2) * pitch for i in range(n)]


#: chip land: length / body length, reach past the body end / body length, width / body width. Read off the first product's
#: flatbed scan (the solder wings of a 1005 reach ~0.2 mm past the body and are ~1.3x as wide), not IPC-7351.
CHIP_PAD_LEN, CHIP_PAD_OUT, CHIP_PAD_WIDTH = 0.5, 0.3, 1.4


def chip_pads(l: float, w: float) -> list[Pad]:
    """Two end terminals under the end caps, reaching :data:`CHIP_PAD_OUT` past the body ends."""
    pl = CHIP_PAD_LEN * l
    c = l / 2 + CHIP_PAD_OUT * l - pl / 2
    return [(-c, 0.0, pl, CHIP_PAD_WIDTH * w), (c, 0.0, pl, CHIP_PAD_WIDTH * w)]


def gullwing_pads(l: float, w: float, package: str, family: str) -> list[Pad]:
    """Two pad rows along the long body axis; leads leave across the short axis to the name's span."""
    pins = _pins(package) or 3
    pitch = _pitch(package, 0.95 if family.startswith("sot") else 1.27)
    dims = ipc_dimensions(package)
    long_x = l >= w
    short = min(l, w)
    span = float(dims["compW"]) if dims else short + 1.2
    pad_len = max(0.5, 0.55 * (span - short) + 0.25)
    cross = (span - pad_len) / 2
    pad_w = 0.6 * pitch
    if family == "sot" and pins == 3:
        rows = [(-cross, _row(2, 2 * pitch)), (cross, [0.0])]
    else:
        a = (pins + 1) // 2
        rows = [(-cross, _row(a, pitch)), (cross, _row(pins - a, pitch))]
    pads = []
    for c, along in rows:
        for t in along:
            pads.append((t, c, pad_w, pad_len) if long_x else (c, t, pad_len, pad_w))
    return pads


def qfn_pads(l: float, w: float, package: str) -> tuple[list[Pad], int | None]:
    """Pads on all four edges (half under the body), plus the exposed thermal pad when the name has one."""
    pitch = _pitch(package, 0.5)
    m = re.search(r"-(\d+)(?:_(\d+))?", package.upper())
    pins = int(m[1]) if m else 4 * max(1, round(min(l, w) / pitch) - 2)
    per = max(1, pins // 4)
    plen, pw = 0.75, 0.55 * pitch
    pads: list[Pad] = []
    for t in _row(per, pitch):
        pads += [(-l / 2 + 0.15, t, plen, pw), (l / 2 - 0.15, t, plen, pw)]
        pads += [(t, -w / 2 + 0.15, pw, plen), (t, w / 2 - 0.15, pw, plen)]
    thermal = None
    if m and m[2]:
        thermal = len(pads)
        pads.append((0.0, 0.0, 0.62 * l, 0.62 * w))
    return pads, thermal


def array_pads(l: float, w: float, package: str) -> list[Pad]:
    """Resistor / capacitor arrays: terminals on both long sides, pitch from the name."""
    pins = _pins(package) or 8
    pitch = _pitch(package, 0.5)
    per = pins // 2
    long_x = l >= w
    short = min(l, w)
    pads = []
    for side in (-1, 1):
        for t in _row(per, pitch):
            c = side * short / 2
            pads.append((t, c, 0.6 * pitch, 0.45 * short) if long_x else (c, t, 0.45 * short, 0.6 * pitch))
    return pads


def crystal_pads(l: float, w: float, package: str) -> list[Pad]:
    if (_pins(package) or 4) == 2:
        return chip_pads(l, w)
    pl, pw = 0.35 * l, 0.3 * w
    return [(sx * (l / 2 - pl / 2), sy * (w / 2 - pw / 2), pl, pw) for sx in (-1, 1) for sy in (-1, 1)]


def _generic_part(standoff: float = 0.0) -> dict[str, Any]:
    return {"body": "ic", "height_ratio": _HEIGHT_RATIO_EXACT, "standoff": standoff, "body_material": "mold", "cap_len": None,
            "leads": False, "lead_width": [0.6, 0.1], "marking": None}


def footprint_for(c: dict[str, Any]) -> dict[str, Any]:
    """``{"pads": [...], "thermal": idx|None, "part": ScenePart, "height": mm, "heightGuess": bool}`` of a component."""
    fam, pkg = c["family"], str(c.get("package", ""))
    body = c.get("body") or {}
    l, w = float(body.get("l") or 0), float(body.get("w") or 0)
    h = body.get("h")
    guess = h is None
    height = GUESS_HEIGHT_MM if guess else float(h)
    thermal = None
    if fam in _CHIP_FAMILIES:
        pads = chip_pads(l, w)
    elif fam in ("sot", "sot6", "soic"):
        pads = gullwing_pads(l, w, pkg, fam)
    elif fam == "qfn":
        pads, thermal = qfn_pads(l, w, pkg)
    elif fam in ("array_r", "array_c"):
        pads = array_pads(l, w, pkg)
    elif fam == "crystal":
        pads = crystal_pads(l, w, pkg)
    elif fam == "electrolytic":
        # SMD can on a square plastic base: two long lands under the base, reaching past it on the terminal side
        pads = [(-0.36 * l, 0.0, 0.35 * l, 0.3 * w), (0.36 * l, 0.0, 0.35 * l, 0.3 * w)]
    elif fam in ("testpoint", "fiducial"):
        pads = [(0.0, 0.0, l, w)]
    else:
        pads = []                     # bga (hidden balls), connectors, holes, pads_only, generic: no SMD lands yet
    if fam in FAMILIES and fam != "tht_pin":
        part = scene_part(fam)
        part.update({"height_ratio": _HEIGHT_RATIO_EXACT, "marking": None})
    elif fam == "electrolytic":
        part = {**_generic_part(), "body": "can"}       # built by render_board3d (base + aluminium can), not render_batch
    else:
        part = _generic_part(standoff=0.3 / max(height, 0.3) if fam == "bga" else 0.0)
    return {"pads": pads, "thermal": thermal, "part": part, "height": height, "heightGuess": guess}

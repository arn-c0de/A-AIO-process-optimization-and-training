"""T7: labelled defect crops from the 3D-Board-Renderer, many per render.

A *sheet* is one rendered window: :func:`plan_sheet` picks up to ``per_sheet`` eligible parts in it that lie at least
``spacing_mm`` apart (so one part's defect never shows in another part's crop) and gives each a class with random
parameters (:func:`sample_op`). One Blender render of the sheet then yields one crop per chosen part
(:func:`crop_part`): square, ``FIELD_BODY_LENGTHS`` body lengths wide like the DeepVision app's crop field, centred on
the **nominal** position (the camera / AOI knows where the part should be, not where it went).

Labels (``labels.jsonl``, one line per crop) carry the class in Simple-Sim's names (:class:`~simple_sim.classes.DefectClass`),
the op with its parameters, refdes / package / family, the sheet and the crop geometry in board mm. A real crop of the
same part from the source image (``real/<id>.png``) is written next to it for sim-to-real pairs.
"""

from __future__ import annotations

import math
import random
from typing import Any

import cv2
import numpy as np

from simple_sim.classes import DefectClass
from simple_sim.board3d.substrate import BoardRaster

#: crop side / longest body side, as the DeepVision app cuts its part crops (``fieldMm`` = 3.04 body lengths)
FIELD_BODY_LENGTHS = 3.04
#: families whose builders support every op below (chips and gull-wing / QFN ICs)
ELIGIBLE_FAMILIES = ("resistor", "capacitor", "inductor", "led", "sot", "sot6", "soic", "qfn", "array_r", "array_c")

#: two-terminal chips: the only parts that can tombstone (a leaded IC lifts a lead, it does not stand up)
TOMBSTONE_FAMILIES = ("resistor", "capacitor", "inductor", "led")
#: a bridge to a neighbour's land is only drawn when the lands' edges are at most this far apart (mm)
BRIDGE_MAX_GAP_MM = 0.8

#: class -> default share of the crops
DEFAULT_MIX = {DefectClass.OK: 0.25, DefectClass.MISALIGNED: 0.2, DefectClass.MISSING: 0.15,
               DefectClass.TOMBSTONE: 0.13, DefectClass.EXCESS_SOLDER: 0.12, DefectClass.SOLDER_BRIDGE: 0.15}


def eligible(c: dict[str, Any], stages: set[str]) -> bool:
    return (c["stage"] in stages and not c["dnp"] and c.get("pose") is not None and c.get("body") is not None
            and c["family"] in ELIGIBLE_FAMILIES and (c["body"].get("h") or 0) > 0)


class NotApplicable(ValueError):
    """The class cannot happen on this part (tombstone on an IC, no land close enough for a bridge)."""


def _bridge(c: dict[str, Any], board3d: dict[str, Any] | None, rng: random.Random) -> dict[str, Any]:
    """Bridge op: between two neighbouring lands of the same IC, or from a chip land to the nearest land of a
    neighbouring part whose edge is within :data:`BRIDGE_MAX_GAP_MM`."""
    from simple_sim.board3d.scene import pad_world
    from simple_sim.board3d.footprints import footprint_for
    pads = footprint_for(c)["pads"]
    world = [pad_world(c, i) for i in range(len(pads))]
    if len(pads) > 2:                                   # IC / array: adjacent pins of one row
        pairs = [(i, j) for i in range(len(world)) for j in range(i + 1, len(world))]
        d = {p: math.dist(world[p[0]][:2], world[p[1]][:2]) for p in pairs}
        nearest = min(d.values())
        i, j = rng.choice([p for p in pairs if d[p] <= nearest * 1.05])
        return {"op": "bridge", "refdes": c["refdes"], "pad": i, "toPad": j}
    best = None
    for o in (board3d or {}).get("components", []):
        if o is c or o["dnp"] or not o.get("pose") or not o.get("body") or o["stage"] != c["stage"]:
            continue
        if math.dist((o["pose"]["x"], o["pose"]["y"]), (c["pose"]["x"], c["pose"]["y"])) > 6:
            continue
        for j in range(len(footprint_for(o)["pads"])):
            ow = pad_world(o, j)
            for i, cw in enumerate(world):
                gap = math.dist(cw[:2], ow[:2]) - (min(cw[2], cw[3]) + min(ow[2], ow[3])) / 2
                if gap <= BRIDGE_MAX_GAP_MM and (best is None or gap < best[0]):
                    best = (gap, i, o["refdes"], j)
    if best is None:
        raise NotApplicable("no neighbouring land close enough for a bridge")
    return {"op": "bridge", "refdes": c["refdes"], "pad": best[1], "to": best[2], "toPad": best[3]}


def sample_op(c: dict[str, Any], cls: DefectClass, rng: random.Random, board3d: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """A defect op of class *cls* for component *c* with random parameters (None for OK). Raises
    :class:`NotApplicable` when the class cannot happen on this part."""
    ref, l, w = c["refdes"], c["body"]["l"], c["body"]["w"]
    if cls == DefectClass.OK:
        return None
    if cls == DefectClass.SOLDER_BRIDGE:
        return _bridge(c, board3d, rng)
    if cls == DefectClass.MISSING:
        return {"op": "remove", "refdes": ref}
    if cls == DefectClass.TOMBSTONE:
        if c["family"] not in TOMBSTONE_FAMILIES:
            raise NotApplicable(f"{c['family']} parts do not tombstone")
        return {"op": "tombstone", "refdes": ref, "tiltDeg": round(rng.uniform(45, 85), 1), "raisedSign": rng.choice([-1, 1])}
    if cls == DefectClass.EXCESS_SOLDER:
        return {"op": "excess", "refdes": ref}
    if cls == DefectClass.MISALIGNED:
        # clearly off: shift of 25-50 % of the body width across, 15-30 % of the length along, or a rotation of 8-25 deg
        kind = rng.choice(["shift", "rotate", "both"])
        a = math.radians(c["pose"]["rotDeg"])
        along, across = (rng.choice([-1, 1]) * rng.uniform(0.15, 0.3) * l, rng.choice([-1, 1]) * rng.uniform(0.25, 0.5) * w) \
            if kind != "rotate" else (0.0, 0.0)
        drot = rng.choice([-1, 1]) * rng.uniform(8, 25) if kind != "shift" else 0.0
        dx, dy = along * math.cos(a) - across * math.sin(a), along * math.sin(a) + across * math.cos(a)
        return {"op": "move", "refdes": ref, "dx": round(dx, 4), "dy": round(dy, 4), "dRot": round(drot, 2)}
    raise ValueError(f"class {cls} is not supported by the board3d sampler")


def plan_sheet(board3d: dict[str, Any], window: BoardRaster, per_sheet: int, spacing_mm: float, rng: random.Random,
               mix: dict[DefectClass, float] | None = None, stages: set[str] | None = None) -> list[dict[str, Any]]:
    """Chosen parts of one sheet: ``[{"component", "class", "op"}]``."""
    stages = stages or {"smd"}
    mix = mix or DEFAULT_MIX
    margin = 1.0
    cands = [c for c in board3d["components"] if eligible(c, stages)
             and window.x0 + margin + crop_half(c) <= c["pose"]["x"] <= window.x0 + window.w - margin - crop_half(c)
             and window.y0 + margin + crop_half(c) <= c["pose"]["y"] <= window.y0 + window.h - margin - crop_half(c)]
    rng.shuffle(cands)
    chosen: list[dict[str, Any]] = []
    for c in cands:
        if len(chosen) >= per_sheet:
            break
        if all(math.dist((c["pose"]["x"], c["pose"]["y"]), (o["component"]["pose"]["x"], o["component"]["pose"]["y"]))
               >= max(spacing_mm, crop_half(c) + crop_half(o["component"])) for o in chosen):
            classes, weights = list(mix), list(mix.values())
            while classes:
                cls = rng.choices(classes, weights=weights)[0]
                try:
                    chosen.append({"component": c, "class": cls, "op": sample_op(c, cls, rng, board3d)})
                    break
                except NotApplicable:              # e.g. tombstone on an IC: draw another class for this part
                    k = classes.index(cls)
                    del classes[k], weights[k]
    return chosen


def crop_half(c: dict[str, Any]) -> float:
    return FIELD_BODY_LENGTHS * max(c["body"]["l"], c["body"]["w"]) / 2


def crop_photo(image: np.ndarray, board_to_image: np.ndarray, c: dict[str, Any], size_px: int,
               min_inside: float = 0.9) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Square crop of a photo-view image around the part's nominal centre, as the app cuts its crops from a phone
    still: the centre through the board homography *board_to_image* (board mm -> image px), the field of
    :func:`crop_half` scaled by the homography's local size there, axis-aligned in the image (the perspective stays in).
    None when less than *min_inside* of the field lies inside the image."""
    half = crop_half(c)
    x, y = c["pose"]["x"], c["pose"]["y"]
    h = np.asarray(board_to_image, float).reshape(3, 3)
    pts = cv2.perspectiveTransform(np.array([[[x, y], [x + half, y], [x, y + half]]], np.float64), h)[0]
    (u, v), du, dv = pts[0], pts[1] - pts[0], pts[2] - pts[0]
    r = float(np.sqrt(abs(du[0] * dv[1] - du[1] * dv[0])))             # local scale: sqrt of the area ratio
    if not np.isfinite(r) or r < 2:
        return None
    hh, ww = image.shape[:2]
    inside = (min(u + r, ww) - max(u - r, 0)) * (min(v + r, hh) - max(v - r, 0)) / (4 * r * r)
    if inside < min_inside:
        return None
    src = np.float32([[u - r, v - r], [u + r, v - r], [u - r, v + r]])
    dst = np.float32([[0, 0], [size_px, 0], [0, size_px]])
    crop = cv2.warpAffine(image, cv2.getAffineTransform(src, dst), (size_px, size_px), flags=cv2.INTER_AREA if 2 * r > size_px else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return crop, {"centerMm": [x, y], "fieldMm": round(2 * half, 4), "centerPx": [round(float(u), 1), round(float(v), 1)], "fieldPx": round(2 * r, 1)}


def crop_part(image: np.ndarray, window: BoardRaster, c: dict[str, Any], size_px: int) -> tuple[np.ndarray, dict[str, float]]:
    """Square crop of *image* (rendered / real on *window*'s grid) around the part's nominal centre, resized."""
    half = crop_half(c)
    x, y = c["pose"]["x"], c["pose"]["y"]
    (u0, v0), (u1, v1) = window.to_px(np.array([[x - half, y + half], [x + half, y - half]]))
    src = np.float32([[u0, v0], [u1, v0], [u0, v1]])
    dst = np.float32([[0, 0], [size_px, 0], [0, size_px]])
    crop = cv2.warpAffine(image, cv2.getAffineTransform(src, dst), (size_px, size_px), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, {"centerMm": [x, y], "fieldMm": round(2 * half, 4)}

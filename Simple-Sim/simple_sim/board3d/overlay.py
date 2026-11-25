"""T0 check images: the scene's body boxes drawn onto the real images of the board.

* flatbed scan (stage ``smd``): board mm -> pixels through ``boardToImage`` of ``compare/<side>.json``;
* median render of the mounted board (stage ``mounted``): pixel ``(u, v)`` <-> board mm
  ``(x0 + (u + 0.5) / px, y0 + (v + 0.5) / px)``, with ``mirrored`` ``u' = W - 1 - u``
  (``modules/simple-aoi/py/render.py``).

Colours: green = family with a Blender builder, orange = family planned (T2), red = generic box,
grey = DNP site; thin outline = SMD, thick = THT / mount. Parts without a body get a magenta cross at the
PnP point.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

COLOURS = {"builder": (60, 200, 60), "planned": (0, 150, 255), "generic": (40, 40, 230)}  # BGR
DNP = (160, 160, 160)
NO_BODY = (230, 0, 230)


def corners(pose: dict[str, float], body: dict[str, float]) -> np.ndarray:
    """4 x 2 body corners in board mm (same order as the app's rectangle)."""
    a = math.radians(pose["rotDeg"])
    c, s = math.cos(a), math.sin(a)
    hl, hw = body["l"] / 2, body["w"] / 2
    pts = [(-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)]
    return np.array([(pose["x"] + x * c - y * s, pose["y"] + x * s + y * c) for x, y in pts])


def homography_mapper(h: list[float]) -> Callable[[np.ndarray], np.ndarray]:
    m = np.asarray(h, float).reshape(3, 3)

    def apply(pts: np.ndarray) -> np.ndarray:
        q = np.c_[pts, np.ones(len(pts))] @ m.T
        return q[:, :2] / q[:, 2:3]
    return apply


def frame_mapper(frame: dict[str, Any], width: int) -> Callable[[np.ndarray], np.ndarray]:
    px, x0, y0 = float(frame["pxPerMm"]), float(frame["x0"]), float(frame["y0"])

    def apply(pts: np.ndarray) -> np.ndarray:
        u = (pts[:, 0] - x0) * px - 0.5
        v = (pts[:, 1] - y0) * px - 0.5
        if frame.get("mirrored"):
            u = width - 1 - u
        return np.c_[u, v]
    return apply


def draw(image: np.ndarray, board3d: dict[str, Any], to_px: Callable[[np.ndarray], np.ndarray],
         stages: set[str], scale: float = 1.0) -> dict[str, int]:
    """Draw the components of *stages* onto *image* (in place). Returns counts drawn / without body."""
    lw = max(1, round(2 * scale))
    drawn = no_body = 0
    for c in board3d["components"]:
        if c["stage"] not in stages:
            continue
        if c["body"] is None or c["pose"] is None:
            x, y = to_px(np.array([[c["pnp"]["x"], c["pnp"]["y"]]]))[0]
            r = round(10 * scale)
            cv2.line(image, (round(x - r), round(y - r)), (round(x + r), round(y + r)), NO_BODY, lw)
            cv2.line(image, (round(x - r), round(y + r)), (round(x + r), round(y - r)), NO_BODY, lw)
            no_body += 1
            continue
        poly = to_px(corners(c["pose"], c["body"])).round().astype(np.int32)
        colour = DNP if c["dnp"] else COLOURS[c["familyStatus"]]
        cv2.polylines(image, [poly], True, colour, lw if c["stage"] == "smd" else 2 * lw, cv2.LINE_AA)
        drawn += 1
    return {"drawn": drawn, "withoutBody": no_body}


def legend(image: np.ndarray, scale: float = 1.0) -> None:
    rows = [("Blender builder", COLOURS["builder"]), ("planned family (T2)", COLOURS["planned"]),
            ("generic box", COLOURS["generic"]), ("DNP site", DNP), ("no body (PnP point)", NO_BODY)]
    fs, step = 0.6 * scale, round(26 * scale)
    cv2.rectangle(image, (0, 0), (round(300 * scale), step * len(rows) + round(10 * scale)), (255, 255, 255), -1)
    for i, (text, colour) in enumerate(rows):
        y = round(10 * scale) + step * i
        cv2.rectangle(image, (round(8 * scale), y), (round(28 * scale), y + round(16 * scale)), colour, -1)
        cv2.putText(image, text, (round(36 * scale), y + round(14 * scale)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                    max(1, round(scale)), cv2.LINE_AA)


def write_overlays(board3d: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    """``overlay_scan.png`` (SMD stage on the flatbed scan) and ``overlay_mounted.jpg`` (all stages on the
    mounted render), when the sources exist. Returns per image the path and the counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    scan = board3d["sources"].get("scan")
    if scan and scan.get("boardToImage"):
        img = cv2.imread(scan["path"], cv2.IMREAD_COLOR)
        if img is not None:
            counts = draw(img, board3d, homography_mapper(scan["boardToImage"]), {"smd"}, scale=img.shape[1] / 1700)
            legend(img, scale=img.shape[1] / 1700)
            path = out_dir / "overlay_scan.png"
            cv2.imwrite(str(path), img)
            result["scan"] = {"path": str(path), **counts}
    mounted = board3d["sources"].get("mountedRender")
    if mounted and mounted.get("frame"):
        img = cv2.imread(mounted["path"], cv2.IMREAD_COLOR)
        if img is not None:
            counts = draw(img, board3d, frame_mapper(mounted["frame"], img.shape[1]), {"smd", "tht", "mount"},
                          scale=img.shape[1] / 1700)
            legend(img, scale=img.shape[1] / 1700)
            path = out_dir / "overlay_mounted.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            result["mounted"] = {"path": str(path), **counts}
    return result

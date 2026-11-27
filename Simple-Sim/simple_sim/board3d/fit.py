"""T2b: measure every part's real pose and body size on the rectified scan.

The scene's poses come from the PnP and the package library (where the part *should* be); the real part sits a
few hundredths of a millimetre and degrees off, and library body sizes are nominal. Rendering at those poses
puts the body's edges next to the real ones, which shows as bright / dark seams in the photo-textured render.

:func:`fit_part` searches, around the scene pose, the body rectangle whose outline lies on the strongest colour
edges of the scan: edge strength = gradient magnitude of the image in CIE L*a*b* (the tan / black body and the
bright end caps against the green mask), sampled with bilinear interpolation along the rectangle sides, averaged.
The search is coarse-to-fine: position and rotation at the nominal size, then length / width scale, then a fine
position / rotation step. A fit is accepted when it improves the edge score by :data:`MIN_GAIN` and stays inside
the search box (a fit at the border means the real part is further away or something else wins); otherwise the
part keeps its pose (``fit.accepted = false``).

Results go into the scene as ``component["fit"] = {x, y, rotDeg, l, w, score0, score, gain, accepted}``;
:func:`simple_sim.board3d.scene.apply_fit` makes an accepted fit the part's pose and body size (nominal and real).

CLI: ``python -m simple_sim.board3d.fit --board3d outputs/board3d/<product>/<side> [--families resistor,capacitor,...]`` (writes the
fits into ``board3d.json``, keeps a ``board3d.before-fit.json`` once, and ``fit_report.json``).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.board3d.substrate import BoardRaster

#: search box around the scene pose
SHIFT_MM, ROT_DEG, SCALE_RANGE = 0.2, 4.0, (0.85, 1.2)
#: a fit must raise the mean edge strength by this factor
MIN_GAIN = 1.05
#: families whose fit improves the render. Measured on the first product (TOP) (25.11.2026, full board, Delta E per part vs the scan,
#: fitted - unfitted): soic -2.4, array_c -2.0, array_r -1.2, sot -1.1, led -0.8, inductor -0.4, bga -0.4, crystal -0.3,
#: filter -0.3, qfn -0.2, sot6 0.0 - but resistor +0.6 and capacitor +0.6 (0402/0603 chips: the edge between solder
#: fillet and end cap pulls the outline off the body), so those two are left out.
FIT_FAMILIES = ("inductor", "led", "diode", "filter", "power_inductor", "sot", "sot6", "soic", "qfn", "array_r", "array_c",
                "crystal", "bga", "electrolytic")


def edge_image(image_bgr: np.ndarray) -> np.ndarray:
    """Gradient magnitude in CIE L*a*b* (sum over the channels), float32, lightly smoothed."""
    lab = cv2.cvtColor(image_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (0, 0), 0.8)
    mag = np.zeros(lab.shape[:2], np.float32)
    for ch, w in zip(range(3), (1.0, 1.5, 1.5), strict=True):
        gx = cv2.Sobel(lab[..., ch], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lab[..., ch], cv2.CV_32F, 0, 1, ksize=3)
        mag += w * np.sqrt(gx * gx + gy * gy)
    return mag


def _outline_unit(n_per_side: int = 24) -> np.ndarray:
    """Points on the unit square outline [-0.5, 0.5]^2 (corners left out: they belong to two sides and round off)."""
    t = (np.arange(n_per_side) + 0.5) / n_per_side - 0.5
    t = t[np.abs(t) < 0.42]
    h = np.full_like(t, 0.5)
    return np.concatenate([np.c_[t, -h], np.c_[t, h], np.c_[-h, t], np.c_[h, t]])


_UNIT = _outline_unit()


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    fu, fv = u - u0, v - v0
    return (img[v0, u0] * (1 - fu) * (1 - fv) + img[v0, u0 + 1] * fu * (1 - fv)
            + img[v0 + 1, u0] * (1 - fu) * fv + img[v0 + 1, u0 + 1] * fu * fv)


def score_poses(edges: np.ndarray, raster: BoardRaster, x: np.ndarray, y: np.ndarray, rot: np.ndarray, l: np.ndarray,
                w: np.ndarray) -> np.ndarray:
    """Mean edge strength along the outline of every candidate rectangle (arrays of equal length, board mm / deg)."""
    a = np.radians(rot)[:, None]
    px = _UNIT[None, :, 0] * l[:, None]
    py = _UNIT[None, :, 1] * w[:, None]
    bx = x[:, None] + px * np.cos(a) - py * np.sin(a)
    by = y[:, None] + px * np.sin(a) + py * np.cos(a)
    u = (bx - raster.x0) * raster.s - 0.5
    v = (raster.y0 + raster.h - by) * raster.s - 0.5
    return _bilinear(edges, u.ravel(), v.ravel()).reshape(u.shape).mean(axis=1)


def _grid(*axes: np.ndarray) -> list[np.ndarray]:
    return [g.ravel() for g in np.meshgrid(*axes, indexing="ij")]


def fit_part(edges: np.ndarray, raster: BoardRaster, c: dict[str, Any]) -> dict[str, Any]:
    """Coarse-to-fine search of pose and body size for one component (see module doc)."""
    p, b = c["pose"], c["body"]
    x0, y0, r0, l0, w0 = p["x"], p["y"], p["rotDeg"], b["l"], b["w"]
    one = lambda v: np.array([v], float)  # noqa: E731
    s0 = float(score_poses(edges, raster, one(x0), one(y0), one(r0), one(l0), one(w0))[0])
    step = 1.0 / raster.s                                    # one pixel
    # Position and size depend on each other (a rectangle that is too small hugs two edges, off-centre by half the
    # size difference, and growing it would move those edges away again): search them jointly on a coarse grid ...
    d = np.linspace(-SHIFT_MM, SHIFT_MM, 9)
    dr = np.arange(-ROT_DEG, ROT_DEG + 1e-9, 1.0)
    sc_ = np.linspace(SCALE_RANGE[0], SCALE_RANGE[1], 8)
    gx, gy, gr, gl, gw = _grid(x0 + d, y0 + d, r0 + dr, l0 * sc_, w0 * sc_)
    sc = score_poses(edges, raster, gx, gy, gr, gl, gw)
    k = int(np.argmax(sc))
    x, y, r, l, w = gx[k], gy[k], gr[k], gl[k], gw[k]
    # ... then refine all five locally, twice, with shrinking steps
    for ds, rs, ss in ((d[1] - d[0], 0.5, sc_[1] - sc_[0]), (step / 2, 0.25, 0.0125)):
        fd = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * ds
        gx, gy, gr, gl, gw = _grid(x + fd, y + fd, r + np.array([-rs, 0.0, rs]), l * (1 + np.array([-ss, 0.0, ss])),
                                   w * (1 + np.array([-ss, 0.0, ss])))
        sc = score_poses(edges, raster, gx, gy, gr, gl, gw)
        k = int(np.argmax(sc))
        x, y, r, l, w = gx[k], gy[k], gr[k], gl[k], gw[k]
    x, y, r, s = float(x), float(y), float(r), float(sc[k])
    l, w = float(l), float(w)
    inside = abs(x - x0) < SHIFT_MM - 1e-6 and abs(y - y0) < SHIFT_MM - 1e-6 and abs(r - r0) < ROT_DEG - 1e-6 \
        and SCALE_RANGE[0] < l / l0 < SCALE_RANGE[1] and SCALE_RANGE[0] < w / w0 < SCALE_RANGE[1]
    gain = s / s0 if s0 > 0 else float("inf")
    return {"x": round(x, 4), "y": round(y, 4), "rotDeg": round(r % 360, 3), "l": round(l, 4), "w": round(w, 4),
            "score0": round(s0, 4), "score": round(s, 4), "gain": round(gain, 3), "accepted": bool(inside and gain >= MIN_GAIN)}


def fit_board3d(board3d: dict[str, Any], scan_rgb: np.ndarray, raster: BoardRaster, families: tuple[str, ...] = FIT_FAMILIES,
             stages: tuple[str, ...] = ("smd",)) -> dict[str, Any]:
    """Fit every eligible component in place (``component["fit"]``); returns a summary."""
    edges = edge_image(scan_rgb)
    n = acc = 0
    shifts, rots = [], []
    for c in board3d["components"]:
        if c["stage"] not in stages or c["dnp"] or c["family"] not in families or not c.get("pose") or not c.get("body"):
            continue
        if not c["body"].get("h"):
            continue
        f = fit_part(edges, raster, c)
        c["fit"] = f
        n += 1
        if f["accepted"]:
            acc += 1
            shifts.append(math.dist((f["x"], f["y"]), (c["pose"]["x"], c["pose"]["y"])))
            rots.append(abs((f["rotDeg"] - c["pose"]["rotDeg"] + 180) % 360 - 180))
    q = lambda v, p: round(float(np.percentile(v, p)), 3) if v else None  # noqa: E731
    return {"fitted": n, "accepted": acc, "shiftMm": {"median": q(shifts, 50), "p90": q(shifts, 90)},
            "rotDeg": {"median": q(rots, 50), "p90": q(rots, 90)}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="3D-Board-Renderer: fit every part's pose / body size to the flatbed scan")
    ap.add_argument("--board3d", type=Path, required=True)
    ap.add_argument("--families", default=",".join(FIT_FAMILIES))
    a = ap.parse_args(argv)
    path = a.board3d / "board3d.json"
    board3d = json.loads(path.read_text(encoding="utf-8"))
    if not (a.board3d / "substrate" / "substrate.json").is_file():   # first run: build the rectified scan
        from simple_sim.board3d.render import prepare_substrate
        prepare_substrate(board3d, a.board3d)
    sub = json.loads((a.board3d / "substrate" / "substrate.json").read_text())
    r = sub["raster"]
    raster = BoardRaster.from_meta(r)
    scan = cv2.imread(str(a.board3d / "substrate" / "scan_rgb.png"), cv2.IMREAD_COLOR)
    t0 = time.monotonic()
    summary = fit_board3d(board3d, scan, raster, tuple(a.families.split(",")))
    summary["seconds"] = round(time.monotonic() - t0, 1)
    backup = a.board3d / "board3d.before-fit.json"
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(board3d, indent=1, ensure_ascii=False), encoding="utf-8")
    (a.board3d / "fit_report.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

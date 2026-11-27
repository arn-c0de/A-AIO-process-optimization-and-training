"""T6: reproduce a reviewed real defect as a defect op (parameter search against the real crop).

A registered phone still (``geometry.json`` + ``still.jpg``, :mod:`simple_sim.board3d.photoreg`) carries the board's
reviewed faults (``faults``: centre in board mm, the app's fault name). For one fault :func:`fit_fault` finds the part,
draws candidate ops of the fault's class (:func:`simple_sim.board3d.dataset.sample_op`, the dataset sampler's own
ranges), renders each through the still's camera **only in the part's crop** (``cropPx``: a border render of a small
window's scene, the still as texture) and scores it against the real crop (:func:`crop_score`). The nominal part (no
op) is scored too: a fit that does not beat it did not find the defect.

The best ops give the sampler realistic shape parameters (``board3d_dataset.py --fitted``) instead of guessed ranges.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from simple_sim.classes import DefectClass
from simple_sim.board3d.dataset import NotApplicable, crop_half, eligible, sample_op
from simple_sim.board3d.substrate import BoardRaster

#: the app's fault names -> Simple-Sim class (the faults a defect op can reproduce)
FAULT_CLASS = {"fehlt": DefectClass.MISSING, "loetbruecke": DefectClass.SOLDER_BRIDGE, "grabstein": DefectClass.TOMBSTONE,
               "versetzt": DefectClass.MISALIGNED, "verdreht": DefectClass.MISALIGNED}
#: the reviewed centre lies at most this far from the part's placement (mm)
MATCH_MM = 0.6
#: the crop is rendered so its field is about this many px wide (the app cuts ~49 px)
CROP_PX = 64
#: shape parameters an op of each kind hands on to the sampler (``--fitted``); a move's offsets depend on the part
SHAPE_KEYS = {"tombstone": ("tiltDeg", "domeMm"), "bridge": ("widthScale", "heightMm"), "remove": ("residue", "domeMm"),
              "excess": ("grow", "heightScale")}


def find_part(board3d: dict[str, Any], centre: tuple[float, float], stages: set[str]) -> dict[str, Any] | None:
    """The eligible part whose placement is nearest to *centre* (within :data:`MATCH_MM`)."""
    best, dist = None, MATCH_MM
    for c in board3d["components"]:
        if not eligible(c, stages):
            continue
        d = math.dist((c["pose"]["x"], c["pose"]["y"]), centre)
        if d <= dist:
            best, dist = c, d
    return best


def crop_box(board_to_image: np.ndarray, c: dict[str, Any], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Integer px box ``(u0, v0, u1, v1)`` of the part's crop field in an image of *shape* (rows, cols); None when
    it is not fully inside."""
    half = crop_half(c)
    x, y = c["pose"]["x"], c["pose"]["y"]
    pts = cv2.perspectiveTransform(np.array([[[x, y], [x + half, y], [x, y + half]]], np.float64), board_to_image)[0]
    (u, v), du, dv = pts[0], pts[1] - pts[0], pts[2] - pts[0]
    r = float(np.sqrt(abs(du[0] * dv[1] - du[1] * dv[0])))
    u0, v0, u1, v1 = (int(round(t)) for t in (u - r, v - r, u + r, v + r))
    if u0 < 0 or v0 < 0 or u1 > shape[1] or v1 > shape[0] or u1 - u0 < 8:
        return None
    return u0, v0, u1, v1


def crop_score(render: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """How well a rendered crop matches the real one: SSIM on grey (both slightly blurred: sample noise, JPEG),
    gradient-magnitude correlation (shape, robust to a colour offset) and mean Delta E; ``score`` = mean of SSIM and
    the gradient correlation minus 0.005 x Delta E (higher is better)."""
    from simple_sim.board3d.render import _delta_e, _ssim
    a, b = (cv2.GaussianBlur(x, (0, 0), 0.7) for x in (render, real))
    ssim = _ssim(a, b)

    def grad(x: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).ravel()
        return (m - m.mean()) / (m.std() + 1e-6)

    gc = float(np.mean(grad(a) * grad(b)))
    de = float(_delta_e(a, b).mean())
    return {"ssim": round(ssim, 4), "gradCorr": round(gc, 4), "deltaE": round(de, 2), "score": round((ssim + gc) / 2 - 0.005 * de, 4)}


def render_crop(scene: dict[str, Any], folder: Path, geo: dict[str, Any], source: str, c: dict[str, Any], scale: float,
                box: tuple[int, int, int, int], name: str, samples: int = 16, device: str = "GPU") -> np.ndarray:
    """The crop *box* (px at *scale*) of the still's camera view of *scene*, rendered alone: the scene of a window
    around the part (its crop field + 3 mm: neighbours that show in the field), the still as texture, room light."""
    from simple_sim.board3d.camera import blender_camera
    from simple_sim.board3d.render import GLOSSY_WORLD, prepare_substrate, render_window
    sub = prepare_substrate(scene, folder, source=source)
    w, h = (int(v) for v in geo["sourceSize"])
    size = (round(w * scale), round(h * scale))
    cam = blender_camera(list(geo["boardToSource"]), w, h)
    r = crop_half(c) + 3.0
    window = BoardRaster(c["pose"]["x"] - r, c["pose"]["y"] - r, 2 * r, 2 * r, sub["raster"]["pxPerMm"])
    res = render_window(scene, folder, window, name, samples=samples, device=device, camera=cam, size_px=size,
                        light={"preset": "room", "glossyWorld": GLOSSY_WORLD},
                        job_extra={"board": {"clip": True}, "cropPx": list(box)})
    img = cv2.imread(res["image"], cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(res["image"])
    want = (box[2] - box[0], box[3] - box[1])
    return img if img.shape[1::-1] == want else cv2.resize(img, want, interpolation=cv2.INTER_AREA)


def fit_fault(board3d: dict[str, Any], folder: Path, geometry_path: Path, fault: dict[str, Any], rng: random.Random, *,
              candidates: int = 10, samples: int = 16, device: str = "GPU", keep: Path | None = None) -> dict[str, Any] | None:
    """Best op for one reviewed *fault* of the still: ``{"fault", "refdes", "class", "nominal": score, "best": {"op",
    ...score}, "tried": n}``; None when the fault has no op class or no eligible part at its place. With *keep* the
    real crop, the nominal and the best render are written there side by side."""
    from simple_sim.board3d.render import still_source
    from simple_sim.board3d.scene import apply_defects
    from simple_sim.io_utils import load_json
    cls = FAULT_CLASS.get(fault.get("fault"))
    if cls is None:
        return None
    geo = load_json(geometry_path)
    source = still_source(geometry_path, clean_faults=True)     # the real defect must not be in the texture
    stages = {"smd", "tht", "mount"}
    c = find_part(board3d, tuple(fault["centerMm"]), stages)
    if c is None:
        return None
    still = cv2.imread(str(Path(geometry_path).parent / "still.jpg"), cv2.IMREAD_COLOR)
    h_full = np.asarray(geo["boardToSource"], float).reshape(3, 3)
    box_full = crop_box(h_full, c, still.shape[:2])
    if box_full is None:
        return None
    scale = min(1.0, CROP_PX / max(1, box_full[2] - box_full[0]))
    h_img = np.diag([scale, scale, 1.0]) @ h_full
    small = cv2.resize(still, (round(still.shape[1] * scale), round(still.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    box = crop_box(h_img, c, small.shape[:2])
    if box is None:
        return None
    real = small[box[1]:box[3], box[0]:box[2]]
    name = f"fit_{Path(geometry_path).parent.name}_{c['refdes']}"
    nominal = render_crop(board3d, folder, geo, source, c, scale, box, name, samples, device)
    out: dict[str, Any] = {"fault": fault.get("fault"), "faultId": fault.get("id"), "refdes": c["refdes"], "package": c.get("package"),
                           "family": c["family"], "class": str(cls), "nominal": crop_score(nominal, real), "photo": str(geometry_path)}
    best, best_img, tried = None, None, 0
    for _ in range(candidates * 3):
        if tried >= candidates:
            break
        try:
            op = sample_op(c, cls, rng, board3d)
        except NotApplicable:
            break
        tried += 1
        img = render_crop(apply_defects(board3d, [op]), folder, geo, source, c, scale, box, name, samples, device)
        sc = crop_score(img, real)
        if best is None or sc["score"] > best["score"]:
            best, best_img = {"op": op, **sc}, img
    out["best"], out["tried"] = best, tried
    out["beatsNominal"] = bool(best and best["score"] > out["nominal"]["score"])
    for p in (folder / "renders").glob(f"{name}.*"):
        p.unlink(missing_ok=True)
    if keep is not None and best_img is not None:
        keep.mkdir(parents=True, exist_ok=True)
        z = lambda x: cv2.resize(x, (128, 128), interpolation=cv2.INTER_CUBIC)  # noqa: E731
        cv2.imwrite(str(keep / f"{name}_{fault.get('fault')}.png"), np.hstack([z(real), z(nominal), z(best_img)]))
    return out


def shape_params(fitted: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per op kind the shape parameters (:data:`SHAPE_KEYS`) of the fits that beat the nominal part."""
    out: dict[str, list[dict[str, Any]]] = {}
    for f in fitted:
        best = f.get("best") or {}
        op = best.get("op") or {}
        if not f.get("beatsNominal") or op.get("op") not in SHAPE_KEYS:
            continue
        shape = {k: op[k] for k in SHAPE_KEYS[op["op"]] if k in op}
        if shape:
            out.setdefault(op["op"], []).append(shape)
    return out

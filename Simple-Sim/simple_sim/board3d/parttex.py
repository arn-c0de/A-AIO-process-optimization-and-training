"""Per-part textures from many photos: every face of every part (top and four walls), fused from the DeepVision stills.

The board-wide mosaic (:mod:`simple_sim.board3d.photomosaic`) is a top view only: walls got a flat material or a
pixel column of the top edge. Here every part gets five **face tiles** in its own frame (mm, ``TEX_PX_PER_MM``):

* ``top``: the footprint (body plus its lands) seen from above; texels inside the body outline sit at the body
  height, the rest at the lead / land height;
* ``+x`` / ``-x`` / ``+y`` / ``-y``: the four walls of the body box, ``z`` from the board to the top.

Every texel is a 3D board point. Per photo (camera from the registration homography, :func:`photomosaic.photo_camera`,
plus the residual field) it is projected into the still and sampled where the face looks at the camera and no taller
part is in the way (a ray march over the board's height map). The stills come from many units and carry glare, so a
face is not averaged over all photos (that blurs at the 0.02-0.06 mm registration residual and mixes units): per
face and photo a score - edge energy, disagreement with the per-texel median, the cosine of the view and the
sampling density - picks the photos (softmax), blended per texel; exposure is matched per part and photo on its top
face first.

:func:`evaluate` measures how well a face texture predicts a photo it was not built from (k-fold over the photos):
mean CIE76 Delta E per part and face, against the current textures (the mosaic top; walls from the mosaic just inside
the outline). :func:`refine_pose` moves each part (x, y, rotation) to where the photos agree best on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable

import cv2
import numpy as np

from simple_sim.board3d.cleanlook import CLEAN_SCALE, body_box, clean_atlas, clean_tile, print_mask
from simple_sim.board3d.footprints import footprint_for
from simple_sim.board3d.photomosaic import field_grid, height_map, photo_camera
from simple_sim.board3d.substrate import BoardRaster
from simple_sim.io_utils import load_json

#: texel density of the face tiles (the stills show the board at ~20 px/mm)
TEX_PX_PER_MM = 20.0
FACES = ("top", "+x", "-x", "+y", "-y")
#: a face counts as seen by a photo above this cosine between its normal and the view ray
MIN_COS = 0.2
#: softmax temperature of the per-face photo scores (as the mosaic's tile fusion)
TEMPERATURE = 0.15
#: height map for the occlusion test (px/mm) and the ray march
HEIGHT_PX_PER_MM, RAY_STEPS = 4.0, 24
#: lands / leads outside the body: texel height (mm) and the margin of the top tile around the body and its lands
LOW_Z, TOP_MARGIN_MM = 0.08, 0.25


@dataclass
class Face:
    name: str
    local: np.ndarray            # (rows, cols, 3) texel points in the part frame (mm)
    normal: np.ndarray           # (3,) in the part frame
    bounds: tuple[float, float, float, float]   # (u0, u1, v0, v1) mm: cols along u, rows along v (row 0 at v1)


@dataclass
class PartTex:
    refdes: str
    pose: tuple[float, float, float]            # x, y (mm), rotation (deg)
    size: tuple[float, float, float]            # l, w, h
    faces: list[Face]
    kind: str = "box"                           # body kind (board3d.footprints.board_part ``board``)
    samples: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = field(default_factory=dict)
    # face -> [(photo index, texel indices, BGR uint8 samples, weights (cos x density))]


def _grid(u0: float, u1: float, v0: float, v1: float) -> tuple[np.ndarray, np.ndarray]:
    cols = max(2, int(round((u1 - u0) * TEX_PX_PER_MM)))
    rows = max(2, int(round((v1 - v0) * TEX_PX_PER_MM)))
    u = u0 + (np.arange(cols) + 0.5) * (u1 - u0) / cols
    v = v1 - (np.arange(rows) + 0.5) * (v1 - v0) / rows
    return np.meshgrid(u, v)


def part_faces(l: float, w: float, h: float, pads: list) -> list[Face]:
    """The five face tiles of a body ``l`` x ``w`` x ``h`` (part frame) with its lands (``pads`` (cx, cy, w, h))."""
    ex = max([l / 2] + [abs(px) + pw / 2 for px, _py, pw, _ph in pads]) + TOP_MARGIN_MM
    ey = max([w / 2] + [abs(py) + ph / 2 for _px, py, _pw, ph in pads]) + TOP_MARGIN_MM
    gx, gy = _grid(-ex, ex, -ey, ey)
    inside = (np.abs(gx) <= l / 2) & (np.abs(gy) <= w / 2)
    faces = [Face("top", np.stack([gx, gy, np.where(inside, h, min(LOW_Z, h))], -1), np.array([0.0, 0.0, 1.0]), (-ex, ex, -ey, ey))]
    if h > 0.05:
        for sx in (1, -1):
            gy2, gz = _grid(-w / 2, w / 2, 0.0, h)
            faces.append(Face("+x" if sx > 0 else "-x", np.stack([np.full_like(gy2, sx * l / 2), gy2, gz], -1),
                              np.array([float(sx), 0.0, 0.0]), (-w / 2, w / 2, 0.0, h)))
        for sy in (1, -1):
            gx2, gz = _grid(-l / 2, l / 2, 0.0, h)
            faces.append(Face("+y" if sy > 0 else "-y", np.stack([gx2, np.full_like(gx2, sy * w / 2), gz], -1),
                              np.array([0.0, float(sy), 0.0]), (-l / 2, l / 2, 0.0, h)))
    return faces


def _to_board(local: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, rot = pose
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    p = local.reshape(-1, 3)
    return np.c_[x + p[:, 0] * c - p[:, 1] * s, y + p[:, 0] * s + p[:, 1] * c, p[:, 2]]


def _rot_normal(n: np.ndarray, rot: float) -> np.ndarray:
    a = math.radians(rot)
    return np.array([n[0] * math.cos(a) - n[1] * math.sin(a), n[0] * math.sin(a) + n[1] * math.cos(a), n[2]])


def parts_of(board3d: dict[str, Any], stages: set[str]) -> list[PartTex]:
    """A :class:`PartTex` per placed, populated part of *stages* with a body height (fits applied by the caller)."""
    out = []
    for c in board3d["components"]:
        p, b = c.get("pose"), c.get("body")
        if not p or not b or c.get("dnp") or c.get("stage") not in stages:
            continue
        fp = footprint_for(c)
        h = float(fp["height"])
        if h <= 0:
            continue
        l, w = float(b["l"]), float(b["w"])
        pose = (float(p["x"]), float(p["y"]), float(p["rotDeg"]))
        out.append(PartTex(c["refdes"], pose, (l, w, h), part_faces(l, w, h, fp["pads"]), str(fp["part"].get("board", "box"))))
    return out


def remap_points(img: np.ndarray, x: np.ndarray, y: np.ndarray, interp: int = cv2.INTER_LINEAR, *,
                 border: int = cv2.BORDER_REPLICATE, value: float = 0.0) -> np.ndarray:
    """*img* sampled at the points (x, y) px; returns (N,) or (N, C). The point lists are laid out as a 2D map
    (cv2.remap accepts at most 32767 rows)."""
    n = len(x)
    cols = 1024
    rows = max(1, -(-n // cols))
    mx = np.zeros(rows * cols, np.float32)
    my = np.zeros(rows * cols, np.float32)
    mx[:n], my[:n] = x, y
    out = cv2.remap(img, mx.reshape(rows, cols), my.reshape(rows, cols), interp, borderMode=border, borderValue=value)
    return out.reshape(rows * cols, -1)[:n] if out.ndim == 3 else out.reshape(-1)[:n]


class PhotoView:
    """One still with its camera, residual field and the board's height map (for the occlusion test)."""

    def __init__(self, geometry: Path, heights: np.ndarray, hraster: BoardRaster, field_raster: BoardRaster):
        self.path = Path(geometry)
        self.geo = load_json(self.path)
        self.image = cv2.imread(str(self.path.parent / "still.jpg"), cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(self.path.parent / "still.jpg")
        self.k, self.r, self.t = photo_camera(self.geo)
        self.centre = -self.r.T @ self.t
        self.heights, self.hr = heights, hraster
        self.fr = field_raster
        self.dx, self.dy = field_grid(self.geo.get("boardField"), field_raster)

    def _field(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        px = self.fr.to_px(pts[:, :2])
        return remap_points(self.dx, px[:, 0], px[:, 1]), remap_points(self.dy, px[:, 0], px[:, 1])

    def project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixel (x, y) of board points (mm, z up) and the camera depth."""
        dx, dy = self._field(pts)
        q = np.c_[pts[:, 0] - dx, pts[:, 1] - dy, pts[:, 2]]
        cam = q @ self.r.T + self.t
        pix = cam @ self.k.T
        return pix[:, :2] / pix[:, 2:3], cam[:, 2]

    def occluded(self, pts: np.ndarray, start_mm: float = 0.12) -> np.ndarray:
        """Is a taller part between the point and the camera (ray march over the height map)?"""
        d = self.centre[None, :] - pts
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        zmax = float(self.heights.max()) + 0.1
        rise = np.maximum(d[:, 2], 1e-3)
        reach = np.clip((zmax - pts[:, 2]) / rise, start_mm, None)
        hit = np.zeros(len(pts), bool)
        for i in range(1, RAY_STEPS + 1):
            s = start_mm + (reach - start_mm) * i / RAY_STEPS
            q = pts + d * s[:, None]
            px = self.hr.to_px(q[:, :2])
            hm = remap_points(self.heights, px[:, 0], px[:, 1], cv2.INTER_NEAREST, border=cv2.BORDER_CONSTANT)
            hit |= hm > q[:, 2] + 0.05
        return hit

    def sample(self, pts: np.ndarray, normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(indices of the texels this photo sees, their BGR samples, weights cos x sampling density)."""
        view = self.centre[None, :] - pts
        dist = np.linalg.norm(view, axis=1)
        cos = (view @ normal) / dist
        ok = cos > MIN_COS
        if not ok.any():
            return np.zeros(0, np.int64), np.zeros((0, 3), np.uint8), np.zeros(0, np.float32)
        idx = np.nonzero(ok)[0]
        uv, depth = self.project(pts[idx])
        hgt, wid = self.image.shape[:2]
        inb = (depth > 0) & (uv[:, 0] >= 1) & (uv[:, 1] >= 1) & (uv[:, 0] < wid - 2) & (uv[:, 1] < hgt - 2)
        idx, uv, depth = idx[inb], uv[inb], depth[inb]
        if len(idx):
            vis = ~self.occluded(pts[idx])
            idx, uv, depth = idx[vis], uv[vis], depth[vis]
        if not len(idx):
            return np.zeros(0, np.int64), np.zeros((0, 3), np.uint8), np.zeros(0, np.float32)
        bgr = remap_points(self.image, uv[:, 0], uv[:, 1])
        density = self.k[0, 0] / depth            # photo px per mm at the texel
        return idx.astype(np.int32), bgr, (cos[idx] * density / 20.0).astype(np.float16)


def collect(parts: list[PartTex], geometries: list[Path], board3d: dict[str, Any], stages: set[str],
            log: Callable[[str], None] = print) -> list[str]:
    """Sample every face texel of every part from every photo (one still in memory at a time)."""
    hr = BoardRaster(0.0, 0.0, float(board3d["widthMm"]), float(board3d["heightMm"]), HEIGHT_PX_PER_MM)
    heights = height_map(board3d, hr, stages).astype(np.float32)
    fr = BoardRaster(0.0, 0.0, float(board3d["widthMm"]), float(board3d["heightMm"]), 4.0)
    used = []
    world = {(p.refdes, f.name): _to_board(f.local, p.pose) for p in parts for f in p.faces}
    normals = {(p.refdes, f.name): _rot_normal(f.normal, p.pose[2]) for p in parts for f in p.faces}
    for g in geometries:
        try:
            view = PhotoView(g, heights, hr, fr)
        except (OSError, ValueError, KeyError) as e:
            log(f"  skip {g}: {e}")
            continue
        used.append(str(g))
        pi = len(used) - 1
        for p in parts:
            for f in p.faces:
                idx, bgr, wgt = view.sample(world[(p.refdes, f.name)], normals[(p.refdes, f.name)])
                if len(idx):
                    p.samples.setdefault(f.name, []).append((pi, idx, bgr, wgt))
        log(f"  sampled photo {pi + 1}/{len(geometries)}")
        del view
    return used


# --------------------------------------------------------------------------- fusion

def _dense(face: Face, entries: list, photos: set[int] | None = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(K, N, 3) float32 samples with NaN where unseen, (K, N) weights, photo ids, for the photos in *photos*."""
    n = face.local.shape[0] * face.local.shape[1]
    keep = [e for e in entries if photos is None or e[0] in photos]
    vals = np.full((len(keep), n, 3), np.nan, np.float32)
    wts = np.zeros((len(keep), n), np.float32)
    for i, (_pi, idx, bgr, w) in enumerate(keep):
        vals[i, idx] = bgr
        wts[i, idx] = w
    return vals, wts, [e[0] for e in keep]


def exposure_gains(top_vals: np.ndarray) -> np.ndarray:
    """(K, 3) per-photo gains matching each photo to the per-texel median on the part's top (robust median of ratios)."""
    with np.errstate(all="ignore"):
        med = np.nanmedian(top_vals, 0)
        ratio = med[None] / np.maximum(top_vals, 4.0)
        g = np.nanmedian(ratio, 1)
    g = np.where(np.isfinite(g), g, 1.0)
    return np.clip(g, 0.5, 2.0).astype(np.float32)


#: per photo and face shift search (mm): the stills show many units, every one placed a little differently
ALIGN_MAX_MM = 0.3


def align(img: np.ndarray, seen: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """*img* (rows, cols, 3 float, NaN-free where *seen*) shifted onto *ref* by phase correlation (sub-texel, at most
    ALIGN_MAX_MM); returns (shifted image, shifted mask, (dx, dy) texels). Unseen texels take *ref* for the correlation."""
    rows, cols = seen.shape
    if rows < 8 or cols < 8 or seen.mean() < 0.5:
        return img, seen, (0.0, 0.0)
    a = cv2.cvtColor(np.where(seen[..., None], img, ref).astype(np.float32), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(ref.astype(np.float32), cv2.COLOR_BGR2GRAY)
    win = cv2.createHanningWindow((cols, rows), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(b, a, win)
    lim = ALIGN_MAX_MM * TEX_PX_PER_MM
    if resp < 0.05 or abs(dx) > lim or abs(dy) > lim:
        return img, seen, (0.0, 0.0)
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    out = cv2.warpAffine(np.where(seen[..., None], img, ref).astype(np.float32), m, (cols, rows), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    msk = cv2.warpAffine(seen.astype(np.float32), m, (cols, rows), flags=cv2.INTER_NEAREST, borderValue=0) > 0.5
    return out, msk, (float(dx), float(dy))


def align_stack(vals: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """(K, N, 3) samples (NaN unseen) with every photo aligned onto the per-texel median (in place copy)."""
    rows, cols = shape
    with np.errstate(all="ignore"):
        ref = np.nan_to_num(np.nanmedian(vals, 0)).reshape(rows, cols, 3)
    out = vals.copy()
    for i in range(len(vals)):
        seen = np.isfinite(vals[i, :, 0]).reshape(rows, cols)
        img = np.nan_to_num(vals[i]).reshape(rows, cols, 3)
        a, m, _d = align(img, seen, ref)
        out[i] = np.where(m.reshape(-1, 1), a.reshape(-1, 3), np.nan)
    return out


def fuse_face(vals: np.ndarray, wts: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fused BGR tile (rows, cols, 3) uint8 and the coverage mask. Per photo a face score (as the mosaic's tiles):
    edge energy, disagreement with the per-texel median, mean view weight; softmax across photos; per-texel blend
    with the photo's view weight, the median where nothing sees the texel."""
    rows, cols = shape
    seen = np.isfinite(vals[..., 0])
    with np.errstate(all="ignore"):
        med = np.nanmedian(vals, 0)
    if not seen.any():
        return np.zeros((rows, cols, 3), np.uint8), np.zeros((rows, cols), bool)
    scores = np.full(len(vals), -np.inf, np.float32)
    for i in range(len(vals)):
        m = seen[i]
        if m.mean() < 0.3:
            continue
        img = np.where(m[:, None], vals[i], np.nan_to_num(med)).reshape(rows, cols, 3)
        g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 0.7), cv2.CV_32F) ** 2
        energy = float(lap.reshape(-1)[m].mean())
        disagree = float(np.nanmean(np.abs(vals[i][m] - med[m])))
        scores[i] = 0.5 * math.log1p(energy) - 4.0 * math.log1p(disagree / 3.0) + 1.5 * math.log(max(float(wts[i][m].mean()), 1e-3))
    if not np.isfinite(scores).any():
        scores = np.where(seen.any(1), 0.0, -np.inf).astype(np.float32)
    best = scores[np.isfinite(scores)].max()
    sw = np.where(np.isfinite(scores), np.exp((scores - best) / TEMPERATURE), 0.0).astype(np.float32)
    w = sw[:, None] * wts * seen
    acc = w.sum(0)
    out = np.nansum(np.nan_to_num(vals) * w[..., None], 0) / np.maximum(acc, 1e-9)[:, None]
    cover = acc > 1e-9
    any_seen = seen.any(0)
    out = np.where(cover[:, None], out, np.nan_to_num(med))
    tile = np.clip(out + 0.5, 0, 255).astype(np.uint8).reshape(rows, cols, 3)
    mask = (cover | any_seen).reshape(rows, cols)
    if not mask.all() and mask.any():
        tile = cv2.inpaint(tile, (~mask).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return tile, mask


def build(part: PartTex, photos: set[int] | None = None) -> dict[str, np.ndarray]:
    """Fused face tiles of *part* from the photos in *photos* (all when None), exposure matched on the top face."""
    top = next(f for f in part.faces if f.name == "top")
    tv, _tw, tids = _dense(top, part.samples.get("top", []), photos)
    gains = dict(zip(tids, exposure_gains(tv))) if len(tids) else {}
    tiles = {}
    for f in part.faces:
        vals, wts, ids = _dense(f, part.samples.get(f.name, []), photos)
        if not len(ids):
            continue
        for i, pi in enumerate(ids):
            vals[i] *= gains.get(pi, np.ones(3, np.float32))[None, :]
        vals = align_stack(vals, f.local.shape[:2])
        tiles[f.name] = fuse_face(vals, wts, f.local.shape[:2])[0]
    return tiles


# --------------------------------------------------------------------------- evaluation

def _lab(bgr: np.ndarray) -> np.ndarray:
    x = np.clip(bgr, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3) * np.array([100 / 255, 1, 1], np.float32) \
        - np.array([0, 128, 128], np.float32)


#: candidate textures per face group, chosen per part by their k-fold error (:func:`select`). Walls are seen at
#: grazing angles only (the stills are near-frontal): their detail is mostly registration noise, a vertical colour
#: profile or one colour often predicts an unseen photo better.
TOP_VARIANTS = ("sharp", "blur0.7", "blur1.5", "mosaic")
WALL_VARIANTS = ("sharp", "blur1.5", "blur3", "rows", "flat", "current")


def variant(tile: np.ndarray, name: str, current: np.ndarray | None = None) -> np.ndarray | None:
    """*tile* (fused face) turned into candidate *name*."""
    if name == "sharp":
        return tile
    if name.startswith("blur"):
        return cv2.GaussianBlur(tile, (0, 0), float(name[4:]))          # sigma in texels
    if name == "rows":            # one colour per height (wall): the vertical profile
        prof = np.median(tile.astype(np.float32), 1, keepdims=True)
        return np.clip(np.broadcast_to(prof, tile.shape) + 0.5, 0, 255).astype(np.uint8)
    if name == "flat":
        return np.broadcast_to(np.median(tile.reshape(-1, 3), 0).astype(np.uint8), tile.shape).copy()
    if name in ("mosaic", "current"):
        return current
    raise ValueError(name)


def evaluate(part: PartTex, n_folds: int = 4, current: dict[str, np.ndarray] | None = None) -> dict[str, dict[str, float]]:
    """k-fold over the photos: build from the other folds, compare every candidate (:data:`TOP_VARIANTS` /
    :data:`WALL_VARIANTS`) with each held-out photo (exposure matched) on the texels it sees. Returns face group ->
    candidate -> mean CIE76 Delta E (``current`` / ``mosaic``: the current textures, :func:`current_tiles`; they
    are built from all photos, the held-out ones included, so their error is optimistic)."""
    ids = sorted({e[0] for es in part.samples.values() for e in es})
    if len(ids) < 2:
        return {}
    folds = [set(ids[i::n_folds]) for i in range(min(n_folds, len(ids)))]
    acc: dict[str, dict[str, list[float]]] = {"top": {v: [] for v in TOP_VARIANTS}, "walls": {v: [] for v in WALL_VARIANTS}}
    top = next(f for f in part.faces if f.name == "top")
    tv, _tw, tids = _dense(top, part.samples.get("top", []))
    gains = dict(zip(tids, exposure_gains(tv)))
    for held in folds:
        tiles = build(part, set(ids) - held)
        for f in part.faces:
            grp = "top" if f.name == "top" else "walls"
            if f.name not in tiles:
                continue
            cands = {v: variant(tiles[f.name], v, (current or {}).get(f.name)) for v in acc[grp]}
            rows, cols = f.local.shape[:2]
            ref = tiles[f.name].astype(np.float32)
            for pi, idx, bgr, _w in part.samples.get(f.name, []):
                if pi not in held or not len(idx):
                    continue
                dense = np.zeros((rows * cols, 3), np.float32)
                dense[idx] = bgr * gains.get(pi, np.ones(3))[None, :]
                seen = np.zeros(rows * cols, bool)
                seen[idx] = True
                # the held-out unit sits a little elsewhere: compare appearance, not placement (same shift for all)
                a, m, _d = align(dense.reshape(rows, cols, 3), seen.reshape(rows, cols), ref)
                m = m.reshape(-1)
                if m.sum() < 4:
                    continue
                truth = _lab(a.reshape(-1, 3)[m])
                for v, t in cands.items():
                    if t is not None:
                        acc[grp][v].append(float(np.linalg.norm(_lab(t.reshape(-1, 3)[m]) - truth, axis=1).mean()))
    return {g: {v: round(float(np.mean(e)), 2) for v, e in d.items() if e} for g, d in acc.items() if any(d.values())}


#: a sharper candidate wins when its error is within this factor of the best: smoothing lowers the error against a
#: noisy photo a little, but erases prints and edges (realism over the last per cent of Delta E)
SHARP_TOLERANCE = 1.04


def select(scores: dict[str, dict[str, float]]) -> dict[str, str]:
    """Per face group the sharpest candidate (in :data:`TOP_VARIANTS` / :data:`WALL_VARIANTS` order) whose k-fold error
    is within :data:`SHARP_TOLERANCE` of the lowest."""
    out = {}
    for g, d in scores.items():
        if not d:
            continue
        best = min(d.values())
        order = TOP_VARIANTS if g == "top" else WALL_VARIANTS
        out[g] = next(v for v in order if v in d and d[v] <= best * SHARP_TOLERANCE)
    return out


#: wall texels closer than this (Delta E) to the board around the part are board, not wall: the app's body outline is
#: a little larger than the real part (connectors, inductors), the photos then see the board where the box's wall is
BOARD_DE = 14.0
#: a wall that is board over more than this share is dropped (the renderer falls back to another wall)
BOARD_SHARE_MAX = 0.5


def board_colour(top_tile: np.ndarray, rim_px: int = 3) -> np.ndarray:
    """Median BGR of the top tile's outer rim (the board around the part and its lands)."""
    rim = np.concatenate([top_tile[:rim_px].reshape(-1, 3), top_tile[-rim_px:].reshape(-1, 3),
                          top_tile[:, :rim_px].reshape(-1, 3), top_tile[:, -rim_px:].reshape(-1, 3)])
    return np.median(rim, 0)


def clean_wall(tile: np.ndarray, board_bgr: np.ndarray) -> np.ndarray | None:
    """*tile* with its board-coloured texels filled from the rest of the wall; None when it is mostly board."""
    de = np.linalg.norm(_lab(tile.reshape(-1, 3)) - _lab(board_bgr.reshape(1, 3)), axis=1).reshape(tile.shape[:2])
    board = de < BOARD_DE
    if board.mean() > BOARD_SHARE_MAX:
        return None
    if board.any():
        tile = cv2.inpaint(tile, board.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    return tile


def current_tiles(part: PartTex, mosaic: np.ndarray, raster: BoardRaster, inset_mm: float = 0.3) -> dict[str, np.ndarray]:
    """The current textures in the face-tile layout: the mosaic on the top; on the walls the mosaic just inside the
    outline (one colour per wall, as the renderer's ``photo_sides``)."""
    out = {}
    for f in part.faces:
        pts = _to_board(f.local, part.pose)
        if f.name != "top":
            l, w, _h = part.size
            loc = f.local.reshape(-1, 3).copy()
            loc[:, :2] = np.sign(loc[:, :2]) * np.maximum(np.abs(loc[:, :2]) - inset_mm, 0.0)
            pts = _to_board(loc, part.pose)
        px = raster.to_px(pts[:, :2])
        col = remap_points(mosaic, px[:, 0], px[:, 1])
        if f.name != "top":
            col = np.broadcast_to(np.median(col, 0, keepdims=True), col.shape)
        out[f.name] = np.ascontiguousarray(col).reshape(*f.local.shape[:2], 3)
    return out


#: parts taller than this (mm) get no wall textures and keep a fused top only below TALL_TOP_MAX_DE: the camera comes
#: from the board homography without lens distortion, 13-16 mm above the board its projection misses (USB / RJ45
#: walls showed other parts, the hand holding the board)
TALL_MM, TALL_TOP_MAX_DE = 8.0, 22.0


def gate(part: PartTex, choice: dict[str, str], scores: dict[str, dict[str, float]]) -> tuple[dict[str, str], list[str]]:
    """Per-part limits on the chosen candidates: gull-wing walls take the vertical profile (the photos show the leads
    in front of the wall, the 3D leads stand there already); tall parts lose their walls and, when the top predicts
    badly, the top. Returns (choice, faces to leave out)."""
    choice, drop = dict(choice), []
    # walls no held-out photo could test were seen by one or two photos only: their vertical profile, not their detail
    choice.setdefault("walls", "rows")
    choice.setdefault("top", "sharp")
    if part.kind == "gullwing":
        choice["walls"] = "rows"
    if part.size[2] > TALL_MM:
        drop = ["+x", "-x", "+y", "-y"]
        top = scores.get("top", {}).get(choice.get("top", ""), 1e9)
        if top > TALL_TOP_MAX_DE:
            drop.append("top")
    return choice, drop


# --------------------------------------------------------------------------- body outline from the photos

#: an edge is trimmed when at least this much (mm) of the body next to it is board colour in the fused top, by at most
#: PHOTO_FIT_MAX of the body size; a part that is board everywhere is reported as absent, not changed
PHOTO_FIT_MIN_MM, PHOTO_FIT_MAX, ABSENT_SHARE = 0.15, 0.45, 0.85
#: the photo fit trims only parts at least PHOTO_FIT_MIN_SIZE_MM on both sides, by at most PHOTO_FIT_MAX_TOTAL of a
#: dimension, and never chips: their size comes from the IPC name and is right, at 0.5 mm the soft edge of the photo
#: looks like board (0402s lost 45 % of their width); a D-sub lost 90 % of its length
PHOTO_FIT_MIN_SIZE_MM, PHOTO_FIT_MAX_TOTAL = 3.0, 0.25
PHOTO_FIT_SKIP_KINDS = ("chip",)


def photo_fit(part: PartTex, top: np.ndarray, bounds: tuple[float, float, float, float]) -> dict[str, Any] | None:
    """Trim the body outline to the photos: per edge, the body rows / columns next to it that are mostly board colour
    (the fused top tile, :func:`board_colour`) are cut off. Returns ``{"x", "y", "l", "w", "trimMm": {edge: mm}}``,
    ``{"absent": True}`` when the photos show board over the whole body, or None (nothing to trim)."""
    l, w, _h = part.size
    if part.kind in PHOTO_FIT_SKIP_KINDS or min(l, w) < PHOTO_FIT_MIN_SIZE_MM:
        return None
    u0, u1, v0, v1 = bounds
    rows, cols = top.shape[:2]
    board = np.linalg.norm(_lab(top.reshape(-1, 3)) - _lab(board_colour(top).reshape(1, 3)), axis=1).reshape(rows, cols) < BOARD_DE
    c0, c1 = (int(round((x - u0) / (u1 - u0) * cols)) for x in (-l / 2, l / 2))
    r0, r1 = (int(round((v1 - y) / (v1 - v0) * rows)) for y in (w / 2, -w / 2))
    body = board[r0:r1, c0:c1]
    if body.shape[0] < 4 or body.shape[1] < 4:
        return None
    if body.mean() > ABSENT_SHARE:
        return {"absent": True}

    def run(lines: np.ndarray) -> float:
        n = 0
        for line in lines:
            if line.mean() <= 0.6:
                break
            n += 1
        return n / TEX_PX_PER_MM

    trim = {"-x": run(body.T), "+x": run(body.T[::-1]), "+y": run(body), "-y": run(body[::-1])}
    trim = {k: min(v, PHOTO_FIT_MAX * (l if "x" in k else w)) for k, v in trim.items() if v >= PHOTO_FIT_MIN_MM}
    if not trim:
        return None
    for axis, size in (("x", l), ("y", w)):           # at most PHOTO_FIT_MAX_TOTAL of the dimension, shared by both edges
        tot = trim.get("-" + axis, 0.0) + trim.get("+" + axis, 0.0)
        if tot > PHOTO_FIT_MAX_TOTAL * size:
            for k in ("-" + axis, "+" + axis):
                if k in trim:
                    trim[k] *= PHOTO_FIT_MAX_TOTAL * size / tot
    dl = trim.get("-x", 0.0) + trim.get("+x", 0.0)
    dw = trim.get("+y", 0.0) + trim.get("-y", 0.0)
    sx = (trim.get("-x", 0.0) - trim.get("+x", 0.0)) / 2          # centre shift in the part frame
    sy = (trim.get("-y", 0.0) - trim.get("+y", 0.0)) / 2
    x, y, rot = part.pose
    a = math.radians(rot)
    return {"x": round(x + sx * math.cos(a) - sy * math.sin(a), 4), "y": round(y + sx * math.sin(a) + sy * math.cos(a), 4),
            "l": round(l - dl, 4), "w": round(w - dw, 4), "trimMm": {k: round(v, 3) for k, v in trim.items()}}


# --------------------------------------------------------------------------- clean look (print agreement)

def print_agreement(part: PartTex) -> float | None:
    """IoU of the print masks of the top fused from the even and from the odd photos; None when a half finds no
    print (the stills show several units: some carry no visible marking, glare covers it on others)."""
    ids = sorted({e[0] for e in part.samples.get("top", [])})
    if len(ids) < 4:
        return None
    a, b = build(part, set(ids[0::2])).get("top"), build(part, set(ids[1::2])).get("top")
    if a is None or b is None:
        return None
    top = next(f for f in part.faces if f.name == "top")
    c0, r0, c1, r1 = body_box(part.size, top.bounds, a.shape[:2])
    ma, _, _ = print_mask(a[max(r0, 0):r1, max(c0, 0):c1])
    mb, _, _ = print_mask(b[max(r0, 0):r1, max(c0, 0):c1])
    if ma.sum() == 0 or mb.sum() == 0:
        return None          # one half shows no print (other units / glare on it): no evidence against it
    k = np.ones((3, 3), np.uint8)
    da, db = cv2.dilate(ma, k) > 0, cv2.dilate(mb, k) > 0
    inter = ((ma > 0) & db).sum() + ((mb > 0) & da).sum()
    return round(float(inter / max((ma > 0).sum() + (mb > 0).sum(), 1)), 3)


# --------------------------------------------------------------------------- whole board

def _chunks(parts: list[PartTex], cell_mm: float = 30.0) -> list[list[PartTex]]:
    cells: dict[tuple[int, int], list[PartTex]] = {}
    for p in parts:
        cells.setdefault((int(p.pose[0] // cell_mm), int(p.pose[1] // cell_mm)), []).append(p)
    return [cells[k] for k in sorted(cells)]


def pack(tiles: dict[str, dict[str, np.ndarray]], width: int = 4096, pad: int = 2) -> tuple[np.ndarray, dict[str, dict[str, list[int]]]]:
    """Shelf-pack every face tile into one atlas; returns (BGR atlas, refdes -> face -> [x, y, w, h] px)."""
    items = sorted(((r, f, t) for r, fs in tiles.items() for f, t in fs.items()), key=lambda it: -it[2].shape[0])
    x = y = shelf = 0
    rects: dict[str, dict[str, list[int]]] = {}
    for r, f, t in items:
        h, w = t.shape[:2]
        if x + w + pad > width:
            x, y, shelf = 0, y + shelf + pad, 0
        rects.setdefault(r, {})[f] = [x, y, w, h]
        x += w + pad
        shelf = max(shelf, h)
    atlas = np.zeros((y + shelf + pad, width, 3), np.uint8)
    for r, f, t in items:
        ax, ay, w, h = rects[r][f]
        atlas[ay:ay + h, ax:ax + w] = t
    return atlas, rects


def score_summary(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Mean error per group and variant over all parts of a :func:`reconstruct` report, plus ``chosen`` (the error of
    the variant each part picked)."""
    agg: dict[str, dict[str, list[float]]] = {}
    for r in report["parts"].values():
        for g, d in r["scores"].items():
            for v, e in d.items():
                agg.setdefault(g, {}).setdefault(v, []).append(e)
            best = d.get(r["choice"].get(g))
            if best is not None:
                agg.setdefault(g, {}).setdefault("chosen", []).append(best)
    return {g: {v: round(float(np.mean(e)), 2) for v, e in d.items()} for g, d in agg.items()}


def reconstruct(board3d: dict[str, Any], geometries: list[Path], stages: set[str], *, mosaic: np.ndarray | None = None,
                mosaic_raster: BoardRaster | None = None, n_folds: int = 4,
                log: Callable[[str], None] = print) -> tuple[np.ndarray, dict[str, Any]]:
    """Face tiles for every part from every photo, the candidate per part and face group chosen by its k-fold error,
    packed into one atlas. Returns (atlas BGR, report: per part pose / size / faces (atlas rect, bounds in the part
    frame) / chosen candidates / scores)."""
    parts = parts_of(board3d, stages)
    tiles: dict[str, dict[str, np.ndarray]] = {}
    report: dict[str, Any] = {"parts": {}, "photos": [str(g) for g in geometries], "texPxPerMm": TEX_PX_PER_MM,
                              "folds": n_folds, "variants": {"top": TOP_VARIANTS, "walls": WALL_VARIANTS}}
    chunks = _chunks(parts)
    for ci, chunk in enumerate(chunks):
        collect(chunk, geometries, board3d, stages, log=lambda m: None)
        for p in chunk:
            cur = current_tiles(p, mosaic, mosaic_raster) if mosaic is not None else None
            scores = evaluate(p, n_folds, current=cur)
            choice, gated = gate(p, select(scores), scores)
            full = build(p)
            for f in gated:
                full.pop(f, None)
            out = {}
            board_bgr = board_colour(full["top"]) if "top" in full else None
            dropped = []
            for f in p.faces:
                if f.name not in full:
                    continue
                v = choice.get("top" if f.name == "top" else "walls", "sharp")
                t = variant(full[f.name], v, (cur or {}).get(f.name))
                t = np.ascontiguousarray(t if t is not None else full[f.name])
                if f.name != "top" and board_bgr is not None:
                    t = clean_wall(t, board_bgr)
                    if t is None:
                        dropped.append(f.name)
                        continue
                out[f.name] = t
            tiles[p.refdes] = out
            fit = photo_fit(p, full["top"], next(f.bounds for f in p.faces if f.name == "top")) if "top" in full else None
            iou = print_agreement(p) if "top" in full else None
            report["parts"][p.refdes] = {"pose": list(p.pose), "size": list(p.size), "choice": choice, "scores": scores,
                                         "boardWalls": dropped, "photoFit": fit, "gated": gated, "kind": p.kind,
                                         "printIoU": iou,
                                         "faces": {f.name: {"bounds": list(f.bounds)} for f in p.faces if f.name in out}}
            p.samples.clear()
        log(f"  chunk {ci + 1}/{len(chunks)}: {len(chunk)} parts")
    atlas, rects = pack(tiles)
    for r, fs in rects.items():
        for f, rect in fs.items():
            report["parts"][r]["faces"][f]["rect"] = rect
    report["atlasSize"] = [int(atlas.shape[1]), int(atlas.shape[0])]
    return atlas, report


__all__ = [
    "CLEAN_SCALE",
    "FACES",
    "TEX_PX_PER_MM",
    "Face",
    "PartTex",
    "PhotoView",
    "build",
    "clean_atlas",
    "clean_tile",
    "collect",
    "current_tiles",
    "evaluate",
    "exposure_gains",
    "fuse_face",
    "gate",
    "pack",
    "part_faces",
    "parts_of",
    "photo_fit",
    "print_mask",
    "reconstruct",
    "score_summary",
    "select",
    "variant",
]

"""Register a phone photo of a populated board side against the board's photo mosaic (``substrate_photos``).

The DeepVision runs carry their own camera (``geometry.json``: ``boardToSource``, board mm -> photo px). Photos judged
in the app without such a run (the prepared defect boards ``prep1`` / ``prep2``) have none, so the 3D-Board-Renderer
could not look at them through their camera. :func:`register_photo` finds it from the image alone:

1. **coarse:** SIFT on the mosaic (10 px/mm) against the photo at a few scales, ratio test, RANSAC homography; the
   scale with the most inliers wins;
2. **fine:** ECC on gradient magnitudes (robust against the photo's colour and the defect boards' residue), photo
   warped onto the mosaic raster, coarse to fine at 5 / 10 / 20 px/mm.

The result has the layout of a DeepVision run's ``auto/<view>/`` folder (``geometry.json`` + ``still.jpg``), so
``board3d_dataset.py --photos`` takes it as it is.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from simple_sim.board3d.substrate import BoardRaster

#: photo scales tried for the coarse match (photo px -> working px); a board filling the frame needs the small ones
COARSE_SCALES = (0.2, 0.3, 0.4, 0.6)
#: the mosaic's resolution for the coarse match (px/mm)
COARSE_PX_PER_MM = 10.0
#: ECC pyramid (px/mm of the mosaic raster)
FINE_PX_PER_MM = (5.0, 10.0, 20.0)
#: an ECC step may move the raster centre at most this far (mm): a larger jump is a wrong local optimum
MAX_FINE_STEP_MM = 1.0


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(3.0, (8, 8)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


def _grad(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / (float(m.mean()) + 1e-6)


def plausible(board_to_photo: np.ndarray, raster: BoardRaster, shape: tuple[int, ...]) -> bool:
    """The raster's outline lands in the photo as a convex quadrilateral in front of the camera, between 3 % and
    150 % of the frame and with opposite sides within a factor of 2.5 (a phone photo, not a degenerate fit)."""
    corners = np.array([[raster.x0, raster.y0], [raster.x0 + raster.w, raster.y0], [raster.x0 + raster.w, raster.y0 + raster.h],
                        [raster.x0, raster.y0 + raster.h]], float)
    hom = np.c_[corners, np.ones(4)] @ np.asarray(board_to_photo, float).T
    if np.any(hom[:, 2] <= 0):
        return False
    q = hom[:, :2] / hom[:, 2:]
    e = np.roll(q, -1, 0) - q
    cross = e[:, 0] * np.roll(e, -1, 0)[:, 1] - e[:, 1] * np.roll(e, -1, 0)[:, 0]
    if not (np.all(cross > 0) or np.all(cross < 0)):
        return False
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
    frame = shape[0] * shape[1]
    lens = np.linalg.norm(e, axis=1)
    return 0.03 * frame <= area <= 1.5 * frame and max(lens[0] / lens[2], lens[2] / lens[0], lens[1] / lens[3], lens[3] / lens[1]) <= 2.5


def coarse(photo: np.ndarray, mosaic: np.ndarray, raster: BoardRaster, scales: tuple[float, ...] = COARSE_SCALES) -> tuple[np.ndarray, dict[str, Any]]:
    """SIFT + RANSAC: board mm -> photo px. Raises ``ValueError`` when no scale gives a homography."""
    s = COARSE_PX_PER_MM / raster.s
    mos = cv2.resize(mosaic, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=30000)
    km, dm = sift.detectAndCompute(_gray(mos), None)
    if dm is None:
        raise ValueError("mosaic: no features")
    kmp = np.float32([k.pt for k in km]) / s                     # full mosaic raster px
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    best: tuple[int, np.ndarray, dict[str, Any]] | None = None
    for ps in scales:
        small = cv2.resize(photo, None, fx=ps, fy=ps, interpolation=cv2.INTER_AREA)
        kp, dp = sift.detectAndCompute(_gray(small), None)
        if dp is None or len(kp) < 10:
            continue
        good = [m[0] for m in matcher.knnMatch(dm, dp, k=2) if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 8:
            continue
        src = kmp[[g.queryIdx for g in good]]
        dst = np.float32([kp[g.trainIdx].pt for g in good]) / ps
        h, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0 / ps, maxIters=20000, confidence=0.999)
        if h is None:
            continue
        b = h @ np.linalg.inv(raster.pixel_to_board())
        n = int(inl.sum())
        if plausible(b / b[2, 2], raster, photo.shape) and (best is None or n > best[0]):
            best = (n, b, {"scale": ps, "matches": len(good), "inliers": n})
    if best is None or best[0] < 12:
        raise ValueError(f"coarse registration failed ({'no homography' if best is None else f'{best[0]} inliers'})")
    b = best[1] / best[1][2, 2]
    return b, best[2]


def fine(photo: np.ndarray, mosaic: np.ndarray, raster: BoardRaster, board_to_photo: np.ndarray,
         levels: tuple[float, ...] = FINE_PX_PER_MM) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """ECC homography refinement on gradient images, coarse to fine; a level that fails or jumps more than
    :data:`MAX_FINE_STEP_MM` is skipped."""
    b = np.asarray(board_to_photo, float).copy()
    hist = []
    for s in levels:
        r = BoardRaster(raster.x0, raster.y0, raster.w, raster.h, s)
        ref = _grad(cv2.resize(mosaic, (r.cols, r.rows), interpolation=cv2.INTER_AREA))
        t = b @ r.pixel_to_board()
        warped = _grad(cv2.warpPerspective(photo, t, (r.cols, r.rows), flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP))
        w = np.eye(3, dtype=np.float32)
        try:
            cc, w = cv2.findTransformECC(ref, warped, w, cv2.MOTION_HOMOGRAPHY,
                                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
        except cv2.error:
            hist.append({"pxPerMm": s, "ok": False})
            continue
        c = np.array([r.cols / 2, r.rows / 2, 1.0])
        d = w.astype(float) @ c
        step_mm = float(np.hypot(*(d[:2] / d[2] - c[:2]))) / s
        if step_mm > MAX_FINE_STEP_MM:
            hist.append({"pxPerMm": s, "ok": False, "stepMm": round(step_mm, 3)})
            continue
        b = t @ w.astype(float) @ np.linalg.inv(r.pixel_to_board())
        b /= b[2, 2]
        hist.append({"pxPerMm": s, "ok": True, "cc": round(float(cc), 4), "stepMm": round(step_mm, 4)})
    return b, hist


def register_photo(photo: np.ndarray, mosaic: np.ndarray, raster: BoardRaster) -> tuple[np.ndarray, dict[str, Any]]:
    """board mm -> photo px of *photo* (BGR) against *mosaic* on *raster*, and the registration stats. Raises
    ``ValueError`` when the photo does not register (another board, the other side, too little of it in frame)."""
    b0, st = coarse(photo, mosaic, raster)
    b, hist = fine(photo, mosaic, raster, b0)
    ok = [h for h in hist if h.get("ok")]
    if not ok:
        raise ValueError("fine registration failed at every level")
    return b, {"coarse": st, "fine": hist, "cc": ok[-1]["cc"]}


def geometry(board_to_photo: np.ndarray, photo_shape: tuple[int, ...], *, product: str | None, side: str | None,
             stats: dict[str, Any], source: str) -> dict[str, Any]:
    """A DeepVision-style ``geometry.json`` for the registered photo."""
    return {"schemaVersion": 1, "kind": "photo", "source": "board3d-register", "file": "still.jpg",
            "sourceSize": [int(photo_shape[1]), int(photo_shape[0])], "boardToSource": np.asarray(board_to_photo, float).ravel().tolist(),
            "product": product, "side": side, "refined": True, "registration": stats, "original": source}

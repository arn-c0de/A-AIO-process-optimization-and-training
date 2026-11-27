"""T4 ``phone``: the camera of a real photo from its board homography (DeepVision ``geometry.json``).

``boardToSource`` H maps board mm (z = 0) to photo pixels: ``H ~ K [r1 r2 t]``. With square pixels and the principal
point in the image centre, K has one unknown, the focal length f in pixels: the board axes are orthogonal,
``r1 . r2 = 0``, which for ``B = K^-1 H`` gives one equation in ``1/f^2`` (:func:`focal_from_homography`; the
equal-length condition ``|r1| = |r2|`` gives a second estimate, both are averaged when they agree). The pose
follows from ``K^-1 H`` (:func:`pose_from_homography`: scale from the mean column norm, r3 = r1 x r2, nearest
rotation by SVD). Lens distortion is not modelled (assumption: the app's refined pose already absorbs most of it).

:func:`blender_camera` turns it into the job's ``camera``: location (mm), rotation matrix columns in Blender's camera
convention (looks along -Z, +Y up; OpenCV's camera looks along +Z with +Y down), lens in mm for a 36 mm sensor.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SENSOR_WIDTH_MM = 36.0
#: focal length / longer image side of the phones in the data (26 mm full-frame equivalent main camera). Median of
#: the homography estimates on 31 TOP photos of the first product with a fine pose was 0.709-0.754 per resolution (25.11.2026)
PHONE_F_RATIO = 0.72
#: the homography estimate is used instead of the prior only above this tilt and when both constraints agree
MIN_TILT_FOR_ESTIMATE_DEG, MAX_ESTIMATE_SPREAD = 15.0, 0.10


def as_mat3(h: Any) -> np.ndarray:
    """A 3x3 float matrix from a flat 9-list (as stored in the JSON files) or any 3x3 array-like."""
    return np.asarray(h, float).reshape(3, 3)


def board_to_source(geo: dict[str, Any]) -> np.ndarray:
    """``boardToSource`` of a DeepVision ``geometry.json`` (board mm -> photo px) as a 3x3 matrix."""
    return as_mat3(geo["boardToSource"])


def focal_from_homography(h: np.ndarray, cx: float, cy: float) -> dict[str, float]:
    """Focal length (px) from one plane homography, principal point (cx, cy), square pixels. Returns
    ``{"f", "fOrtho", "fNorm"}`` (the two constraints; ``f`` their mean or the one that is valid)."""
    h = as_mat3(h)
    t = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    g = t @ h                                  # principal point at the origin: K = diag(f, f, 1)
    a, b = g[:, 0], g[:, 1]
    # r1 . r2 = 0:  (a0 b0 + a1 b1) / f^2 + a2 b2 = 0
    f_ortho = math.sqrt(-(a[0] * b[0] + a[1] * b[1]) / (a[2] * b[2])) if a[2] * b[2] != 0 and -(a[0] * b[0] + a[1] * b[1]) / (a[2] * b[2]) > 0 else float("nan")
    # |r1| = |r2|:  (a0^2 + a1^2 - b0^2 - b1^2) / f^2 + a2^2 - b2^2 = 0
    den = b[2] ** 2 - a[2] ** 2
    num = a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2
    f_norm = math.sqrt(num / den) if den != 0 and num / den > 0 else float("nan")
    vals = [v for v in (f_ortho, f_norm) if math.isfinite(v)]
    if not vals:
        raise ValueError("homography too close to fronto-parallel: focal length not observable")
    f = vals[0] if len(vals) == 1 or abs(vals[0] - vals[1]) > 0.25 * vals[0] else sum(vals) / 2
    return {"f": f, "fOrtho": f_ortho, "fNorm": f_norm}


def pose_from_homography(h: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with X_cam = R X_board + t (OpenCV camera: x right, y down, z forward), board in front (t_z > 0)."""
    b = np.linalg.inv(k) @ as_mat3(h)
    lam = 2.0 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
    if b[2, 2] * lam < 0:
        lam = -lam                             # the board must lie in front of the camera
    r1, r2, t = lam * b[:, 0], lam * b[:, 1], lam * b[:, 2]
    r = np.c_[r1, r2, np.cross(r1, r2)]
    u, _, vt = np.linalg.svd(r)
    r = u @ vt
    if np.linalg.det(r) < 0:
        r = u @ np.diag([1, 1, -1]) @ vt
    return r, t


def choose_focal(h: np.ndarray, width: int, height: int) -> tuple[float, str, dict[str, Any]]:
    """(f in px, source, estimates): the homography estimate when it is well conditioned (tilt, agreement),
    else the phone prior :data:`PHONE_F_RATIO` x longer side. Near-frontal photos constrain f badly (the 22
    frontal phone photos gave 0.57-1.0 x the longer side)."""
    prior = PHONE_F_RATIO * max(width, height)
    try:
        est = focal_from_homography(h, width / 2, height / 2)
    except ValueError:
        return prior, "prior", {}
    fo, fn = est["fOrtho"], est["fNorm"]
    k = np.array([[prior, 0, width / 2], [0, prior, height / 2], [0, 0, 1]])
    r, _ = pose_from_homography(h, k)
    tilt = math.degrees(math.acos(min(1.0, abs(r[2, 2]))))
    if math.isfinite(fo) and math.isfinite(fn) and abs(fo - fn) <= MAX_ESTIMATE_SPREAD * est["f"] and tilt >= MIN_TILT_FOR_ESTIMATE_DEG:
        return est["f"], "homography", est
    return prior, "prior", est


def blender_camera(board_to_source: list[float], width: int, height: int, f_px: float | None = None) -> dict[str, Any]:
    """The job ``camera`` of a photo: perspective, at the photo's pose and focal length."""
    h = as_mat3(board_to_source)
    cx, cy = width / 2, height / 2
    if f_px is None:
        f, source, focal = choose_focal(h, width, height)
    else:
        f, source, focal = f_px, "given", {}
    k = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    r, t = pose_from_homography(h, k)
    centre = -r.T @ t
    # Blender camera axes in world: x = CV x, y = -CV y, z = -CV z (rows of R are the CV axes in world coordinates)
    rot = np.c_[r[0], -r[1], -r[2]]
    reproj = k @ np.c_[r[:, 0], r[:, 1], t]
    reproj /= reproj[2, 2]
    err = float(np.abs(reproj - h / h[2, 2]).max())
    return {"type": "PERSP", "locationMm": centre.tolist(), "rotation": rot.tolist(), "focalPx": f,
            "lensMm": f * SENSOR_WIDTH_MM / max(width, height), "sensorWidthMm": SENSOR_WIDTH_MM, "widthPx": width, "heightPx": height,
            "focalSource": source, "focalEstimates": focal, "homographyResidual": err,
            "distanceMm": float(np.linalg.norm(centre)), "tiltDeg": math.degrees(math.acos(max(-1.0, min(1.0, abs(r[2, 2])))))}

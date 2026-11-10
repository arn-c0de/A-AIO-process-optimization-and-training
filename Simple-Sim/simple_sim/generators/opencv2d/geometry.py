"""Part geometry shared by the flat and photoreal 2D styles (and the 3D bake): nominal sampling, pad
rectangles, body pose of a defect, lifted corner, solder bridge and the rng-free defect selector."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np

from simple_sim.board.footprints import footprint_pads
from simple_sim.classes import NEVER_TOMBSTONE


def sample_nominal_geometry(
    roi_config: dict[str, Any],
    rng: np.random.Generator,
    geometry_ranges: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Sample nominal component geometry."""
    _ = roi_config  # reserved for future use
    defaults = {
        "pad_width": [25.0, 35.0],
        "pad_height": [30.0, 40.0],
        "pad_spacing": [50.0, 65.0],
        "component_length": [55.0, 65.0],
        "component_width": [25.0, 35.0],
    }
    gr = geometry_ranges or {}

    def _range(name: str) -> tuple[float, float]:
        r = gr.get(name, defaults[name])
        return float(r[0]), float(r[1])

    result = {
        "pad_width": float(rng.uniform(*_range("pad_width"))),
        "pad_height": float(rng.uniform(*_range("pad_height"))),
        "pad_spacing": float(rng.uniform(*_range("pad_spacing"))),
        "component_length": float(rng.uniform(*_range("component_length"))),
        "component_width": float(rng.uniform(*_range("component_width"))),
    }

    if "pad_spacing_y" in gr:
        lo, hi = float(gr["pad_spacing_y"][0]), float(gr["pad_spacing_y"][1])
        result["pad_spacing_y"] = float(rng.uniform(lo, hi))

    return result


def pad_positions(footprint: str, nominal: dict[str, float], img_shape: tuple[int, int]) -> list[dict[str, int]]:
    """Pad rectangles of *footprint* as ``{"x", "y", "w", "h"}`` dicts (geometry: :func:`simple_sim.board.footprints.footprint_pads`)."""
    return [p.as_dict() for p in footprint_pads(footprint, nominal, img_shape)]


def param_selector(defect_params: dict[str, Any]) -> int:
    """Deterministic integer derived from defect params (no RNG consumption)."""
    key = "|".join(f"{float(defect_params.get(k, 0.0)):.6f}" for k in ("shift_x", "shift_y", "rotation_deg", "tilt_deg"))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def draw_solder_bridge(
    img: np.ndarray,
    nominal: dict[str, float],
    footprint: str,
    body_center: tuple[int, int],
    body_size: tuple[int, int],
    selector: int,
    tilt_deg: float,
) -> None:
    """Draw a solder bridge (bright solder blob) joining two adjacent pads."""
    pads = pad_positions(footprint, nominal, img.shape[:2])
    if len(pads) < 2:
        return
    centers = np.array([(p["x"] + p["w"] / 2.0, p["y"] + p["h"] / 2.0) for p in pads], dtype=np.float32)

    # Candidate pairs: nearest-neighbour pads (within 5% of the smallest pitch).
    dists = {}
    for a in range(len(pads)):
        for b in range(a + 1, len(pads)):
            dists[(a, b)] = float(np.linalg.norm(centers[a] - centers[b]))
    min_d = min(dists.values())
    pairs = sorted(pair for pair, d in dists.items() if d <= min_d * 1.05)
    a, b = pairs[selector % len(pairs)]
    p0, p1 = centers[a].copy(), centers[b].copy()

    axis = p1 - p0
    horizontal = abs(axis[0]) >= abs(axis[1])
    pad_perp = min(pads[a]["h"], pads[b]["h"]) if horizontal else min(pads[a]["w"], pads[b]["w"])
    pad_along = min(pads[a]["w"], pads[b]["w"]) if horizontal else min(pads[a]["h"], pads[b]["h"])

    # If the straight path runs under the body, route the bridge along the body edge
    # (half on the body, half on the substrate) so it stays visible.
    body_len, body_wid = body_size
    mid = (p0 + p1) / 2.0
    rel = mid - np.array(body_center, dtype=np.float32)
    if abs(rel[0]) < body_len / 2.0 and abs(rel[1]) < body_wid / 2.0:
        perp_idx = 1 if horizontal else 0
        body_half_perp = (body_wid if horizontal else body_len) / 2.0
        offset = min(body_half_perp, 0.85 * pad_perp / 2.0)
        sign = np.sign(rel[perp_idx]) if abs(rel[perp_idx]) > 0.5 else (1.0 if (selector >> 3) & 1 else -1.0)
        p0[perp_idx] += sign * offset
        p1[perp_idx] += sign * offset

    # Bridge thickness varies slightly with the sampled tilt (0..2 deg).
    thickness = max(3, int(round(pad_perp * (0.35 + 0.05 * min(2.0, abs(tilt_deg))))))
    solder = (195, 195, 200)
    highlight = (240, 240, 240)
    q0 = tuple(int(round(v)) for v in p0)
    q1 = tuple(int(round(v)) for v in p1)
    cv2.line(img, q0, q1, solder, thickness, lineType=cv2.LINE_AA)
    blob_r = max(2, int(round(min(pad_along, pad_perp) * 0.35)))
    cv2.circle(img, q0, blob_r, solder, -1, lineType=cv2.LINE_AA)
    cv2.circle(img, q1, blob_r, solder, -1, lineType=cv2.LINE_AA)
    # Specular streak along the bridge (shiny solder).
    cv2.line(img, q0, q1, highlight, max(1, thickness // 4), lineType=cv2.LINE_AA)


def lifted_corner_geometry(
    box: np.ndarray,
    center: tuple[int, int],
    comp_width: int,
    tilt_deg: float,
    selector: int,
) -> dict[str, Any]:
    """Geometry of a body with one corner raised (shared by flat and photoreal styles)."""
    pts = box.astype(np.float32)
    k = selector % 4
    corner = pts[k]
    diag = corner - np.array(center, dtype=np.float32)
    diag /= max(1e-6, float(np.linalg.norm(diag)))

    # Raised corner appears displaced outward (closer to camera) proportional to tilt.
    # Capped so large bodies (QFN/SOIC) do not get implausibly long lifts.
    lift = min(comp_width * 0.35, 18.0) * float(np.sin(np.radians(min(90.0, tilt_deg))))
    lifted = pts.copy()
    lifted[k] = corner + diag * lift * 0.6

    prev_pt, next_pt = pts[(k - 1) % 4], pts[(k + 1) % 4]
    mid_prev = (corner + prev_pt) / 2.0
    mid_next = (corner + next_pt) / 2.0

    # Contact shadow cast onto the substrate under the lifted corner (light from top-left).
    shadow_shift = diag * lift * 0.9 + np.array([lift * 0.4, lift * 0.4], dtype=np.float32)
    shadow = np.array([corner, mid_prev, mid_next], dtype=np.float32) + shadow_shift
    return {
        "lifted": lifted,
        "facet": np.array([lifted[k], mid_prev, mid_next], dtype=np.float32),
        "shadow": np.vstack([shadow, [mid_next, mid_prev]]),  # simple polygon: c', mp', mn', mn, mp
    }


def body_geometry(
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    tolerances: dict[str, Any] | None,
    img_shape: tuple[int, int],
) -> dict[str, Any]:
    """Resolve body center, size (length, width), rotation and tombstone state from defect params."""
    h, w = img_shape
    # round() (not int()) so shifts just past tolerance do not truncate back inside it.
    cx = w // 2 + int(round(defect_params["shift_x"]))
    cy = h // 2 + int(round(defect_params["shift_y"]))
    tilt_deg = abs(defect_params["tilt_deg"])
    thresh = float(tolerances.get("tombstone_tilt_deg", 75.0)) if tolerances is not None else 75.0
    # SOLDER_BRIDGE / CORNER_LIFT carry tilt but are never tombstones.
    is_tombstone = tilt_deg >= thresh and defect_params["type"] not in NEVER_TOMBSTONE
    return {
        "center": (cx, cy),
        "length": int(nominal["component_length"]),
        "width": int(nominal["component_width"]),
        "rotation_deg": float(defect_params["rotation_deg"]),
        "tilt_deg": tilt_deg,
        "is_tombstone": is_tombstone,
    }

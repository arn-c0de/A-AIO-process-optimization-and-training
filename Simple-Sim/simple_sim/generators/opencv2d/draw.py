"""Drawing primitives for OpenCV 2D renderer."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np

from simple_sim.board.footprints import footprint_pads
from simple_sim.classes import NEVER_TOMBSTONE


def draw_substrate(img: np.ndarray, config: dict[str, Any]) -> None:
    """Draw green PCB substrate background."""
    img[:] = config["substrate_color"]


def _compute_pad_positions(footprint: str, nominal: dict[str, float], img_shape: tuple[int, int]) -> list[dict[str, int]]:
    """Pad rectangles of *footprint* as ``{"x", "y", "w", "h"}`` dicts (geometry: :func:`simple_sim.board.footprints.footprint_pads`)."""
    return [p.as_dict() for p in footprint_pads(footprint, nominal, img_shape)]


def draw_pads(img: np.ndarray, nominal: dict[str, float], color: tuple[int, int, int], footprint: str = "chip_2pad") -> None:
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
    for pad in pads:
        cv2.rectangle(img, (pad["x"], pad["y"]), (pad["x"] + pad["w"], pad["y"] + pad["h"]), color, -1)


def draw_solder(img: np.ndarray, nominal: dict[str, float], footprint: str = "chip_2pad") -> None:
    solder_color = (200, 200, 200)
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
    h, w = img.shape[:2]
    for pad in pads:
        x, y, pw, ph = pad["x"], pad["y"], pad["w"], pad["h"]
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, int(x + pw))
        y1 = min(h, int(y + ph))
        if x1 <= x0 or y1 <= y0:
            continue

        solder_region = img[y0:y1, x0:x1].copy()
        overlay = np.full_like(solder_region, solder_color)
        blended = cv2.addWeighted(solder_region, 0.7, overlay, 0.3, 0)
        if blended is None:
            continue
        img[y0:y1, x0:x1] = blended


def _param_selector(defect_params: dict[str, Any]) -> int:
    """Deterministic integer derived from defect params (no RNG consumption)."""
    key = "|".join(f"{float(defect_params.get(k, 0.0)):.6f}" for k in ("shift_x", "shift_y", "rotation_deg", "tilt_deg"))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def _darken_poly(img: np.ndarray, pts: np.ndarray, factor: float) -> None:
    """Multiply pixels inside polygon *pts* by *factor* (soft contact shadow)."""
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    alpha = (mask.astype(np.float32) / 255.0)[..., None] * (1.0 - factor)
    img[:] = np.clip(img.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


def _draw_solder_bridge(
    img: np.ndarray,
    nominal: dict[str, float],
    footprint: str,
    body_center: tuple[int, int],
    body_size: tuple[int, int],
    selector: int,
    tilt_deg: float,
) -> None:
    """Draw a solder bridge (bright solder blob) joining two adjacent pads."""
    pads = _compute_pad_positions(footprint, nominal, img.shape[:2])
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
        "shadow": np.vstack([shadow, [mid_prev, mid_next]]),
    }


def _draw_corner_lift(
    img: np.ndarray,
    box: np.ndarray,
    center: tuple[int, int],
    comp_width: int,
    color: tuple[int, int, int],
    tilt_deg: float,
    selector: int,
) -> None:
    """Draw the body with one corner raised: shadow under it, skewed quad, lit corner."""
    geo = lifted_corner_geometry(box, center, comp_width, tilt_deg, selector)
    _darken_poly(img, geo["shadow"], 0.5)

    body = np.round(geo["lifted"]).astype(np.int32)
    cv2.fillPoly(img, [body], color)

    # Highlight on the raised (tilted towards the light) corner facet.
    facet = np.round(geo["facet"]).astype(np.int32)
    bright = tuple(int(min(255, c * 1.45 + 20)) for c in color)
    overlay = img.copy()
    cv2.fillPoly(overlay, [facet], bright)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, dst=img)
    cv2.line(img, tuple(facet[1]), tuple(facet[2]), tuple(int(c * 0.6) for c in color), 1, lineType=cv2.LINE_AA)

    edge = tuple(int(min(255, max(0, c * 1.2))) for c in color)
    cv2.polylines(img, [body], True, edge, 1)


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
    thresh = float(tolerances["tombstone_tilt_deg"]) if tolerances is not None else 75.0
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


def draw_component(
    img: np.ndarray,
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    color: tuple[int, int, int],
    tolerances: dict[str, Any] | None = None,
    footprint: str = "chip_2pad",
) -> None:
    defect_type = defect_params["type"]
    if defect_type == "MISSING":
        return

    geo = body_geometry(nominal, defect_params, tolerances, img.shape[:2])
    center = geo["center"]
    if geo["is_tombstone"]:
        comp_width = int(geo["width"] * 0.2)
        comp_height = int(geo["length"] * 1.5)
    else:
        comp_width, comp_height = geo["width"], geo["length"]

    rect_size = (comp_height, comp_width)
    box = cv2.boxPoints((center, rect_size, geo["rotation_deg"]))

    if defect_type == "CORNER_LIFT":
        _draw_corner_lift(img, box, center, comp_width, color, geo["tilt_deg"], _param_selector(defect_params))
        return

    box = box.astype(np.int32)

    cv2.fillPoly(img, [box], color)
    edge = tuple(int(min(255, max(0, c * 1.2))) for c in color)
    cv2.polylines(img, [box], True, edge, 1)

    if defect_type == "SOLDER_BRIDGE":
        # Drawn on top of the body so a bridge routed along the body edge stays visible.
        _draw_solder_bridge(
            img, nominal, footprint, center, rect_size, _param_selector(defect_params), geo["tilt_deg"],
        )

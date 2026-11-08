"""High-level ROI rendering pipeline for OpenCV 2D backend."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from simple_sim.classes import SOLDER_ONLY_CLASSES

from .draw import draw_component, draw_pads, draw_solder, draw_substrate
from .filters import apply_post_filters
from .geometry import body_geometry, pad_positions
from .photoreal import DEFAULT_SUPERSAMPLING, render_photoreal_base, solder_defect_layout

# ``render.post_filters`` (photoreal only):
#   legacy  domain blur/noise and the augmentation rotation are applied to the finished camera image
#           (so the sensor noise of the camera model is blurred/rotated and noise is added twice).
#   camera  the rotation is a camera roll before the camera model, the domain blur is combined with the
#           lens blur and the domain noise with the read noise: one blur, one noise, no interpolated
#           noise. Brightness/contrast/colour and the extended image filters are applied as before.
from simple_sim.generators.options import DEFAULT_POST_FILTERS, POST_FILTER_MODES  # noqa: E402 (I7: cv2-free)
_CAMERA_KEYS = ("blur_sigma", "noise_stddev", "rotation_deg")


def post_filter_mode(config: dict[str, Any]) -> str:
    """``render.post_filters`` of a render config (default ``legacy``)."""
    mode = str(config.get("post_filters", DEFAULT_POST_FILTERS))
    if mode not in POST_FILTER_MODES:
        raise ValueError(f"render.post_filters must be one of {POST_FILTER_MODES}, got {mode!r}")
    return mode


def _draw_flat_solder_defects(img, nominal, defect_params, tolerances, footprint, rng) -> None:
    """SOLDER_BALL / EXCESS_SOLDER for the legacy flat style (same layout as photoreal).

    The layout gets ``ss=DEFAULT_SUPERSAMPLING`` although the flat image is at camera resolution: the
    flat style always did so (minimum ball size), and changing it would change every flat image.
    """
    if defect_params["type"] not in SOLDER_ONLY_CLASSES:
        return
    pads = pad_positions(footprint, nominal, img.shape[:2])
    geo = body_geometry(nominal, defect_params, tolerances, img.shape[:2])
    rect = cv2.boxPoints(((float(geo["center"][0]), float(geo["center"][1])), (geo["length"], geo["width"]), geo["rotation_deg"]))
    layout = solder_defect_layout(defect_params["type"], pads, rect, rng, ss=DEFAULT_SUPERSAMPLING)
    for i in layout["excess"]:
        p = pads[i]
        cv2.ellipse(img, (p["x"] + p["w"] // 2, p["y"] + p["h"] // 2), (int(p["w"] * 0.65), int(p["h"] * 0.65)),
                    0, 0, 360, (235, 235, 235), -1, lineType=cv2.LINE_AA)
    for x, y, r in layout["balls"]:
        cv2.circle(img, (int(x), int(y)), max(2, int(r)), (220, 220, 220), -1, lineType=cv2.LINE_AA)


def render_roi(
    nominal: dict[str, float],
    defect_params: dict[str, Any],
    augment: dict[str, float],
    roi_size: tuple[int, int],
    config: dict[str, Any],
    tolerances: dict[str, Any] | None,
    rng: np.random.Generator,
    footprint: str = "chip_2pad",
    *,
    camera_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Render complete ROI with defects and augmentation (see :data:`POST_FILTER_MODES`).

    *camera_rng* (photoreal only): a separate generator for the sensor model (lens blur draw, illumination
    gradient, noise, sharpening, JPEG quality). ``None`` (default) keeps the legacy behaviour of drawing them
    from *rng*, bit-identical. ``tools/export_deepvision.py`` renders pairs with it: the same *rng* and
    *camera_rng* for a reference and its defect probes gives one "photo" of the board (same board, pose,
    joint light and sensor draws, only the inspected part differs, since the part's draws come after the
    board's), another *camera_rng* for an OK probe gives the same board under another exposure.
    """
    width, height = roi_size
    photoreal = str(config.get("style", "photoreal")) == "photoreal"
    in_camera = photoreal and post_filter_mode(config) == "camera"
    if in_camera:
        post = {k: float(augment.get(k, 0.0)) for k in _CAMERA_KEYS}
        augment = {**augment, "blur_sigma": 0.0, "noise_stddev": 0.0, "rotation_deg": 0.0}
        img = render_photoreal_base(nominal, defect_params, config, tolerances, rng, footprint, roi_size, post,
                                    camera_rng=camera_rng)
    elif photoreal:
        img = render_photoreal_base(nominal, defect_params, config, tolerances, rng, footprint, roi_size,
                                    light_roll_deg=float(augment.get("rotation_deg", 0.0)), camera_rng=camera_rng)
    else:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        draw_substrate(img, config)
        draw_pads(img, nominal, tuple(config["copper_color"]), footprint=footprint)
        draw_solder(img, nominal, footprint=footprint)
        draw_component(
            img, nominal, defect_params, tuple(config["component_color"]), tolerances=tolerances, footprint=footprint
        )
        _draw_flat_solder_defects(img, nominal, defect_params, tolerances, footprint, rng)

    img = apply_post_filters(img, augment, rng)

    if abs(augment.get("rotation_deg", 0.0)) > 0.1:
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, augment["rotation_deg"], 1.0)
        img = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    return img

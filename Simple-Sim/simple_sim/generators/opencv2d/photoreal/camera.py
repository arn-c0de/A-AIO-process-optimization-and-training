"""Camera model shared by the 2D and 3D backends: viewing pose (tilt, zoom, roll), the view warp and
the sensor model (blur, illumination, highlight roll-off, noise, ISP sharpening, JPEG).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .imageops import clip8, grid, smooth_noise


# Camera model parameters (``render.camera``); tools/calibrate_camera.py fits the AOI ones to real ROIs.
# Ranges are sampled per image.
CAMERA_DEFAULTS: dict[str, Any] = {
    "blur_sigma": [0.55, 0.95],    # lens/optics blur (Gaussian sigma in camera pixels)
    "read_noise": [1.0, 4.0],      # read noise variance (8-bit levels^2)
    "shot_noise": 0.04,            # shot noise variance per intensity level
    "tilt_deg": [0.0, 0.0],        # oblique view: angle between optical axis and board normal
    "zoom": [1.0, 1.0],            # >1: camera farther away, the part appears this much smaller
    "defocus": [0.0, 0.0],         # extra blur (sigma) at the far/near edge of a tilted view (depth of field)
    "sharpen": [0.0, 0.0],         # phone ISP edge sharpening (unsharp mask amount)
    "jpeg_quality": [100, 100],    # < 95 compresses
    "capture_scale": [1.0, 1.0],   # sensor pixels per ROI pixel (<1: low-res capture, upscaled to the ROI like a small crop)
    "chroma_shift": [0.0, 0.0],    # lateral chromatic aberration: R/B shift at the ROI border (sensor pixels)
    "sensor_px_per_mm": [0.0, 0.0],  # > 0: sensor resolution; replaces capture_scale by sensor px/mm x roi.mm_per_px
    "channel_shift": [0.0, 0.0],   # line-scanner RGB misregistration: B and R shifted apart along one axis (sensor px)
    "channel_axis": "random",      # that axis: "random" (per image), "x" or "y" (a scanner's fixed scan direction)
    "crop_part_lengths": [0.0, 0.0],  # > 0 (2D): field of view = this many body lengths; replaces zoom (< 1 magnifies)
    "max_oversampling": 0.0,       # > 0 (2D): cap on supersampling x zoom (rendered px per output px and axis)
}


# ``render.camera.preset``: a fixed AOI camera (default) or handheld phone photos.
CAMERA_PRESETS: dict[str, dict[str, Any]] = {
    "aoi": {},
    # Phone photos of a whole board, cut into small per-part crops (real 0603 crops are ~50 px wide,
    # i.e. ~10 px/mm) and upscaled to the ROI: soft, blocky, sharpening halos, colour fringes.
    "phone": {"tilt_deg": [5.0, 40.0], "zoom": [1.0, 1.5], "blur_sigma": [0.5, 1.2], "defocus": [0.5, 2.5],
              "read_noise": [4.0, 16.0], "shot_noise": 0.08, "sharpen": [0.3, 0.9], "jpeg_quality": [70, 92],
              "capture_scale": [0.2, 0.7], "chroma_shift": [0.0, 0.8]},
    # QUINS AOI flatbed scans (DeepVision xpa, 23.7 px/mm, see .look): low resolution, very little
    # noise (flat-area high-pass std ~0.9 levels on the crops), slight RGB misregistration at edges.
    "quins": {"sensor_px_per_mm": [22.0, 25.5], "blur_sigma": [0.6, 0.95], "read_noise": [0.3, 1.5],
              "shot_noise": 0.015, "channel_shift": [1.6, 3.2], "channel_axis": "x",
              # DeepVision cuts every QUINS crop at 3x the body length (measured: 3.02-3.04 for all chip forms).
              "crop_part_lengths": [3.0, 3.0]},
}


CHANNEL_AXES = ("random", "x", "y")
_RANGE_LIMITS = {"tilt_deg": 60.0, "jpeg_quality": 100.0, "capture_scale": 1.0}


@dataclass(frozen=True, slots=True, eq=False)
class CameraSettings(Mapping):
    """Effective camera parameters of one render (:data:`CAMERA_DEFAULTS` keys), immutable.

    A read-only mapping, so ``c["blur_sigma"]``, ``c.get(..)``, ``dict(c)`` and ``{**c, ...}`` work as on
    the dict :func:`camera_settings` used to return (and ``==`` compares with plain dicts). Values are
    kept exactly as configured (ranges stay ``[lo, hi]`` lists, so every random draw is unchanged).
    Unknown keys fail on construction; :func:`validate_camera_config` checks the value ranges.
    """

    blur_sigma: Any
    read_noise: Any
    shot_noise: Any
    tilt_deg: Any
    zoom: Any
    defocus: Any
    sharpen: Any
    jpeg_quality: Any
    capture_scale: Any
    chroma_shift: Any
    sensor_px_per_mm: Any
    channel_shift: Any
    channel_axis: Any
    crop_part_lengths: Any
    max_oversampling: Any

    def __post_init__(self) -> None:
        if self.channel_axis not in CHANNEL_AXES:
            raise ValueError(f"render.camera.channel_axis must be one of {list(CHANNEL_AXES)}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any] | None) -> CameraSettings:
        """Defaults < preset < explicit ``render.camera`` keys."""
        cfg = dict(cfg or {})
        preset = str(cfg.pop("preset", "aoi"))
        if preset not in CAMERA_PRESETS:
            raise ValueError(f"render.camera.preset must be one of {sorted(CAMERA_PRESETS)}")
        unknown = set(cfg) - set(CAMERA_DEFAULTS)
        if unknown:
            raise ValueError(f"render.camera: unknown keys {sorted(unknown)} (known: {sorted(set(CAMERA_DEFAULTS) | {'preset'})})")
        return cls(**{**CAMERA_DEFAULTS, **CAMERA_PRESETS[preset], **cfg})

    def __getitem__(self, key: str) -> Any:
        if key not in CAMERA_DEFAULTS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(CAMERA_DEFAULTS)

    def __len__(self) -> int:
        return len(CAMERA_DEFAULTS)

    def __repr__(self) -> str:
        return f"CameraSettings({dict(self)!r})"


def camera_settings(cfg: Mapping[str, Any] | None) -> CameraSettings:
    """Effective camera parameters: defaults < preset < explicit ``render.camera`` keys (:class:`CameraSettings`)."""
    if isinstance(cfg, CameraSettings):
        return cfg
    return CameraSettings.from_config(cfg)


def validate_camera_config(camera: Any) -> None:
    """Raise ``ValueError`` unless *camera* is a valid ``render.camera`` mapping (preset + overrides).

    ``config._validate_camera`` runs this at config time, so typos and bad ranges fail before rendering.
    """
    if camera is None:
        return
    if not isinstance(camera, dict) or set(camera) - set(CAMERA_DEFAULTS) - {'preset'}:
        raise ValueError(f"render.camera must be a mapping with keys from {sorted(set(CAMERA_DEFAULTS) | {'preset'})}")
    if camera.get('preset', 'aoi') not in CAMERA_PRESETS:
        raise ValueError(f"render.camera.preset must be one of {sorted(CAMERA_PRESETS)}")
    for key, default in CAMERA_DEFAULTS.items():
        if key not in camera or not isinstance(default, list):
            continue
        rng = camera[key]
        limit = _RANGE_LIMITS.get(key, float('inf'))
        if not (isinstance(rng, list) and len(rng) == 2 and all(isinstance(v, (int, float)) for v in rng)
                and 0 <= rng[0] <= rng[1] <= limit):
            raise ValueError(f"render.camera.{key} must be [min, max] with 0 <= min <= max <= {_RANGE_LIMITS.get(key, 'inf')}")
    if 'capture_scale' in camera and camera['capture_scale'][0] < 0.05:
        raise ValueError("render.camera.capture_scale must be in [0.05, 1] (sensor pixels per ROI pixel)")
    if 'zoom' in camera and camera['zoom'][0] < 1:
        raise ValueError("render.camera.zoom must be >= 1 (1 = profile scale, larger = farther away)")
    if 'shot_noise' in camera and not (isinstance(camera['shot_noise'], (int, float)) and camera['shot_noise'] >= 0):
        raise ValueError("render.camera.shot_noise must be a number >= 0")
    if 'channel_axis' in camera and camera['channel_axis'] not in CHANNEL_AXES:
        raise ValueError(f"render.camera.channel_axis must be one of {list(CHANNEL_AXES)}")
    if 'max_oversampling' in camera and not (isinstance(camera['max_oversampling'], (int, float))
                                             and camera['max_oversampling'] >= 0):
        raise ValueError("render.camera.max_oversampling must be a number >= 0 (0 = off)")
    CameraSettings.from_config(camera)  # the merged settings must construct as well


def sample_camera_pose(cam: dict[str, Any], rng: np.random.Generator, roll_deg: float = 0.0) -> dict[str, float]:
    """Viewing geometry for one image: tilt (deg), its azimuth in the image (deg), zoom, and the roll
    of the camera about its optical axis (deg, counter-clockwise in the image; ``render.post_filters:
    camera`` feeds the augmentation rotation in here so it happens before blur, noise and JPEG)."""
    return {"tilt": float(rng.uniform(*cam["tilt_deg"])), "azimuth": float(rng.uniform(0.0, 360.0)),
            "zoom": float(rng.uniform(*cam["zoom"])), "roll": float(roll_deg)}


def fit_crop(pose: dict[str, float], cam: dict[str, Any], rng: np.random.Generator, part_length: float,
             size: tuple[int, int]) -> dict[str, float]:
    """*pose* with the zoom that shows ``crop_part_lengths`` x *part_length* (pixels at profile scale) across
    the smaller side of *size*, as the inspection pipeline crops its ROIs. Unchanged (and no random draw)
    when ``crop_part_lengths`` is off."""
    k = draw_range(rng, cam["crop_part_lengths"])
    if k <= 0.0:
        return pose
    return {**pose, "zoom": max(0.05, k * float(part_length) / float(min(size)))}


def effective_supersampling(ss: int, pose: dict[str, float], cam: dict[str, Any]) -> int:
    """``render.camera.max_oversampling``: the supersampling to render a zoomed-out view with.

    A view zoomed out by ``pose['zoom']`` renders ``ss x zoom`` board pixels per output pixel and axis;
    with ``crop_part_lengths`` a large package (QFN, TSSOP) needs zoom 2-6, i.e. a 1-3 Mpx canvas at
    4-12x oversampling that is averaged away again. With a cap > 0 the supersampling is lowered so
    ``ss x zoom <= cap`` where possible (never below 1, never raised); 0 (default) keeps *ss*.
    """
    cap = float(cam.get("max_oversampling", 0.0))
    zoom = float(pose.get("zoom", 1.0))
    if cap <= 0.0 or zoom <= 0.0 or ss * zoom <= cap:
        return ss
    return max(1, min(ss, int(np.floor(cap / zoom + 1e-9))))


def roll_parts(pose: dict[str, float]) -> tuple[int, float]:
    """``(k, jitter)``: the roll as k quarter turns (exact, via ``np.rot90``) plus a residual angle."""
    roll = float(pose.get("roll", 0.0))
    k = int(round(roll / 90.0))
    return k % 4, roll - 90.0 * k


def view_canvas(size: tuple[int, int], pose: dict[str, float]) -> tuple[int, int]:
    """Board area to render so that *size* is still covered after zooming out, foreshortening and rolling."""
    grow = pose["zoom"] / max(np.cos(np.radians(pose["tilt"])), 0.5) * (1.1 if pose["tilt"] > 0 else 1.0)
    k, jitter = roll_parts(pose)
    if jitter != 0.0:  # a rotated rectangle must still cover the output: grow by |cos| + |sin|
        j = np.radians(jitter)
        grow *= abs(np.cos(j)) + abs(np.sin(j))
    if k % 2 == 1:  # a quarter turn swaps the axes of the rendered board
        size = (size[1], size[0])
    return int(np.ceil(size[0] * grow)), int(np.ceil(size[1] * grow))


def apply_view(img: np.ndarray, pose: dict[str, float], size: tuple[int, int]) -> np.ndarray:
    """Look at a top-down render of the board (from :func:`view_canvas`) from farther away and at an angle.

    The board plane is tilted by ``pose['tilt']`` about an in-plane axis perpendicular to
    ``pose['azimuth']`` and projected with a pinhole camera; the output is centred on the part.
    ``pose['roll']`` turns the camera (quarter turns exactly with ``np.rot90``, the rest in the
    projection), positive = counter-clockwise in the image like ``cv2.getRotationMatrix2D``.
    """
    width, height = size
    k, jitter = roll_parts(pose)
    if k:
        img = np.ascontiguousarray(np.rot90(img, k))
    z = pose["zoom"]
    small = img
    if z != 1.0:  # > 1: farther away (area average); < 1: a magnified crop (crop_part_lengths)
        small = cv2.resize(img, (max(1, int(round(img.shape[1] / z))), max(1, int(round(img.shape[0] / z)))),
                           interpolation=cv2.INTER_AREA if z > 1.0 else cv2.INTER_CUBIC)
    sh, sw = small.shape[:2]
    if pose["tilt"] <= 0.0 and jitter == 0.0 and (sw, sh) == (width, height):
        return small
    t, a = np.radians(pose["tilt"]), np.radians(pose["azimuth"])
    d = np.array([np.cos(a), np.sin(a)])                              # tilt direction in the image
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float64) - [sw / 2.0, sh / 2.0]
    along, across = corners @ d, corners @ np.array([-d[1], d[0]])
    dist = 4.0 * max(width, height)                                   # pinhole distance: mild keystone
    depth = along * np.sin(t)                                         # towards the camera on one side
    scale = dist / (dist - depth)
    proj = (np.outer(along * np.cos(t), d) + np.outer(across, [-d[1], d[0]])) * scale[:, None]
    if jitter != 0.0:
        j = np.radians(jitter)
        proj = proj @ np.array([[np.cos(j), -np.sin(j)], [np.sin(j), np.cos(j)]])  # image-CCW for positive jitter
    dst = (proj + [width / 2.0, height / 2.0]).astype(np.float32)
    m = cv2.getPerspectiveTransform((corners + [sw / 2.0, sh / 2.0]).astype(np.float32), dst)
    return cv2.warpPerspective(small, m, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def draw_range(rng: np.random.Generator, bounds: Any) -> float:
    """``uniform(lo, hi)``, but without a random draw for a fixed value (``lo == hi``): optional camera
    effects that are off (aoi: capture_scale [1, 1], chroma_shift [0, 0]) leave the later draws alone."""
    lo, hi = float(bounds[0]), float(bounds[1])
    return lo if lo == hi else float(rng.uniform(lo, hi))


def camera_model(img: np.ndarray, rng: np.random.Generator, cfg: dict[str, Any] | None = None,
                 pose: dict[str, float] | None = None, *, extra_blur: float = 0.0,
                 extra_noise_var: float = 0.0, mm_per_px: float | None = None,
                 axis_roll_deg: float = 0.0) -> np.ndarray:
    """Turn an ideal render (2D supersampled or 3D) into a camera image. BGR in, uint8 out.

    Lens blur (plus depth-of-field falloff along the tilt of an oblique *pose*), uneven
    illumination, soft highlight roll-off instead of hard clipping, signal-dependent sensor noise
    that is slightly correlated by demosaicing, and for phones lateral chromatic aberration, ISP
    sharpening and JPEG. With ``capture_scale`` < 1 all of this happens at the lower sensor
    resolution and the result is upscaled to the input size (a small crop fed to the model).
    *cfg* is ``render.camera`` (see :func:`camera_settings`). *extra_blur* (Gaussian sigma, sensor
    pixels) is combined with the lens blur and *extra_noise_var* (8-bit levels^2) added to the read
    noise: ``render.post_filters: camera`` passes the domain's ``blur_sigma`` / ``noise_stddev`` here
    instead of applying them a second time on the finished camera image. *mm_per_px* (ROI scale) is
    needed for ``sensor_px_per_mm``: the capture scale is then sensor px/mm x mm/px (at most 1).
    *axis_roll_deg*: rotation the caller applies to the finished image (counter-clockwise, like
    ``cv2.getRotationMatrix2D``); a fixed ``channel_axis`` is turned against it so the scan direction
    stays put in the final image, as the camera roll of a *pose* already happened before this model.
    """
    c = camera_settings(cfg)
    out_h, out_w = img.shape[:2]
    img = img.astype(np.float32)
    sensor = draw_range(rng, c["sensor_px_per_mm"])
    if sensor > 0.0:
        if mm_per_px is None:
            raise ValueError("render.camera.sensor_px_per_mm needs the ROI scale (mm_per_px)")
        scale = min(1.0, sensor * float(mm_per_px))
    else:
        scale = draw_range(rng, c["capture_scale"])
    if scale < 0.999:  # the sensor sampled the part with fewer pixels than the ROI has
        img = cv2.resize(img, (max(8, int(round(out_w * scale))), max(8, int(round(out_h * scale)))),
                         interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    sigma0 = max(1e-3, float(np.hypot(float(rng.uniform(*c["blur_sigma"])), max(0.0, float(extra_blur)))))
    blurred = cv2.GaussianBlur(img, (0, 0), sigma0)
    defocus = float(rng.uniform(*c["defocus"]))
    yy, xx = grid(h, w)
    if pose is not None and pose["tilt"] > 0.0 and defocus > 0.0:
        a = np.radians(pose["azimuth"])  # focus on the part; sharpness falls off towards near/far edge
        off = np.abs((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / (0.5 * max(h, w))
        weight = np.clip(off * np.sin(np.radians(pose["tilt"])) / np.sin(np.radians(40.0)), 0.0, 1.0)[..., None]
        far = cv2.GaussianBlur(img, (0, 0), float(np.hypot(sigma0, defocus)))
        blurred = blurred * (1.0 - weight) + far * weight
    img = blurred
    shift = draw_range(rng, c["chroma_shift"])
    if shift > 0.0:  # lateral CA: R magnified, B shrunk around the optical centre
        for ch, k in ((2, 1.0), (0, -1.0)):
            m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, 1.0 + k * shift / (0.5 * max(h, w)))
            img[..., ch] = cv2.warpAffine(img[..., ch], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    split = draw_range(rng, c["channel_shift"])
    if split > 0.0:  # line scanner: the colour lines see the board at slightly different times
        if c["channel_axis"] == "random":
            dx, dy = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[int(rng.integers(4))]
        else:
            dx, dy = (1.0, 0.0) if c["channel_axis"] == "x" else (0.0, 1.0)
            if axis_roll_deg != 0.0:  # the image is turned afterwards: pre-rotate the axis the other way
                r = np.radians(axis_roll_deg)
                dx, dy = float(dx * np.cos(r) - dy * np.sin(r)), float(dx * np.sin(r) + dy * np.cos(r))
        for ch, k in ((0, 0.5), (2, -0.5)):
            m = np.float32([[1, 0, k * split * dx], [0, 1, k * split * dy]])
            img[..., ch] = cv2.warpAffine(img[..., ch], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a = float(rng.uniform(0.0, 2.0 * np.pi))
    ramp = ((xx - w / 2.0) * np.cos(a) + (yy - h / 2.0) * np.sin(a)) / max(h, w)
    img *= (1.0 + float(rng.uniform(0.04, 0.12)) * ramp + 0.03 * smooth_noise((h, w), max(h, w) / 3.0, rng))[..., None]
    knee = 205.0
    img = np.where(img > knee, knee + (255.0 - knee) * (1.0 - np.exp(-(img - knee) / (255.0 - knee))), img)
    sigma = np.sqrt(float(rng.uniform(*c["read_noise"])) + max(0.0, float(extra_noise_var))
                    + float(c["shot_noise"]) * np.clip(img, 0.0, None))
    noise = rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.5) * 1.6  # demosaic correlation, variance roughly kept
    noise += rng.normal(0.0, 0.35, (h, w, 1)).astype(np.float32)  # luminance-dominated
    img = img + sigma * noise
    amount = float(rng.uniform(*c["sharpen"]))
    if amount > 0.0:
        img = img + amount * (img - cv2.GaussianBlur(img, (0, 0), 1.2))
    out = clip8(img)
    quality = int(round(float(rng.uniform(*c["jpeg_quality"]))))
    if quality < 95:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    if (h, w) != (out_h, out_w):  # small crop resized to the model input, as the inspection pipeline does
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    return out

"""T4 ``phone``: make a Cycles render of the board3d read like the real photo it stands in for.

The phone textures are the *median* of many photos (noise-free, one exposure for all) and the board3d has no desk, so a
raw render looks too clean, has the median's colour balance and floats on an empty background. Three measured steps,
each fitted against the real photo inside the board outline:

* :func:`fit_colour_matrix` / :func:`apply_colour_matrix` - a 3 x 3 colour matrix + offset (linear light) that maps
  the texture to this photo (exposure, white balance, the phone's saturation and black level), fitted on the board and
  the low parts; the photo textures go to Blender corrected, under a neutral sky.
* :func:`colour_gain` - the older per-channel gain, fitted on the bare board alone (its red channel sits in the
  phone's toe there and ran into the 0.4 clamp; with :func:`fit_tone_lut` on top black parts turned teal and metal
  pink). Kept for ``photo_view(colour="gain")``.
* :func:`composite` - the render (transparent film + shadow catcher) over the real photo (``backplate``): desk,
  bracket and the board's shadow come from the photo, the board from the board3d.
* :func:`fit_camera_fx` / :func:`camera_fx` - lens softness (Gaussian blur, sigma chosen by SSIM) and sensor noise
  (the missing part of the high-pass energy on the bare solder mask), then JPEG like the phone.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Linear light (0..1) -> 8-bit sRGB."""
    x = np.clip(x, 0.0, 1.0)
    return (np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


#: greys (linear light) that :func:`fit_colour_matrix` keeps grey
NEUTRAL_GREYS = (0.05, 0.2, 0.5, 0.9)


def fit_colour_matrix(texture: np.ndarray, real: np.ndarray, mask: np.ndarray, *, step: int = 7,
                      iterations: int = 5, neutral: float = 0.0) -> np.ndarray | None:
    """4 x 3 matrix *M* (linear light, BGR) with ``real ~ [texture, 1] @ M`` on *mask* (both BGR on the same grid,
    blurred by the caller), least squares reweighted against outliers (parallax, stickers, parts that differ between
    boards). Unlike :func:`colour_gain` it carries the phone's saturation and black level: the bare board alone is one
    colour, so the mask should hold the low parts too (black ICs, white resistors, the silk screen). Per phone view
    on 811d1c / f48a23 / da70eb held-out blurred dE 13.2 / 15.5 / 16.1 (gain) -> 10.3 / 13.2 / 14.3. None when the mask
    is too small.

    *neutral* > 0 asks the greys of :data:`NEUTRAL_GREYS` to stay grey (equal B, G, R out), weighted as that share of
    the samples: the board and the parts hold few bright neutrals, and unconstrained the fit's answer to white was
    magenta (0.82 / 0.64 / 0.93), which turned the bare aluminium of the electrolytic lids lilac."""
    sel = mask > 0
    if sel.sum() < 5000:
        return None
    src = srgb_to_linear(texture)[sel][::step].astype(np.float64)
    dst = srgb_to_linear(real)[sel][::step].astype(np.float64)
    x = np.hstack([src, np.ones((len(src), 1))])
    # the grey constraint couples the three output channels: rows [g, -g, 0] and [0, g, -g] on the stacked columns
    cons = []
    for g in NEUTRAL_GREYS:
        xg = np.array([g, g, g, 1.0])
        cons.append(np.concatenate([xg, -xg, np.zeros(4)]))
        cons.append(np.concatenate([np.zeros(4), xg, -xg]))
    c_rows = np.array(cons)
    wt = np.ones(len(src))
    m = np.zeros((4, 3))
    for _ in range(iterations):
        xw = x * wt[:, None]
        gram = x.T @ xw
        a = np.kron(np.eye(3), gram)
        b = np.concatenate([xw.T @ dst[:, k] for k in range(3)])
        if neutral > 0:
            lam = neutral * float(wt.sum()) / len(NEUTRAL_GREYS)
            a = a + lam * c_rows.T @ c_rows
        vec = np.linalg.solve(a + 1e-9 * np.eye(12), b)
        m = vec.reshape(3, 4).T
        r = np.linalg.norm(x @ m - dst, axis=1)
        s = 1.5 * float(np.median(r)) + 1e-6
        wt = np.where(r < s, 1.0, s / np.maximum(r, 1e-12))
    return m.astype(np.float32)


def apply_colour_matrix(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """:func:`fit_colour_matrix` on a BGR or BGRA image (alpha kept), row by row (whole-board textures are large)."""
    out = img.copy()
    mat, off = np.asarray(m, np.float32)[:3], np.asarray(m, np.float32)[3]
    for y0 in range(0, img.shape[0], 512):
        lin = srgb_to_linear(img[y0:y0 + 512, :, :3])
        out[y0:y0 + 512, :, :3] = linear_to_srgb(lin @ mat + off)
    return out


def colour_gain(texture: np.ndarray, real: np.ndarray, mask: np.ndarray, clamp: tuple[float, float] = (0.4, 2.5)) -> list[float]:
    """RGB gain (linear light) with ``real ~ gain * texture`` on *mask* (both BGR images on the same grid): ratio of
    the linearised trimmed means (5-95 %) per channel. Means, not least squares: the bare mask is nearly one colour,
    so any slope fit there regresses to zero. The phone's tone curve / black level is left to :func:`fit_tone_lut`."""
    sel = mask > 0
    if sel.sum() < 500:
        return [1.0, 1.0, 1.0]
    gains = []
    for ch in range(3):
        means = []
        for img in (texture, real):
            v = img[..., ch][sel].astype(np.float32)
            lo, hi = np.percentile(v, (5, 95))
            means.append(float(v[(v >= lo) & (v <= hi)].mean()))
        gains.append(float(srgb_to_linear(np.float32(means[1])) / max(float(srgb_to_linear(np.float32(means[0]))), 1e-4)))
    return [round(float(np.clip(g, *clamp)), 4) for g in gains[::-1]]       # BGR -> RGB (Blender colour)


def residual_homography(real: np.ndarray, render: np.ndarray, mask: np.ndarray, *, iterations: int = 300) -> tuple[np.ndarray, float] | None:
    """(W, ecc) with ``render(W x) ~ real(x)`` (3x3, photo px -> render px) inside *mask*: ECC on the gradient magnitudes
    (robust to the colour / tone differences), coarse to fine. None when ECC does not converge. The render already
    holds the parts' parallax, so what is left is the camera's own error (a DeepVision homography off by ~90 px)."""
    def grad(img):
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.5)
        m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
        return m / max(float(m[mask > 0].mean()), 1e-6)
    a, b = grad(real), grad(render)
    warp, cc = np.eye(3, dtype=np.float32), 0.0
    try:
        for sigma in (3.0, 1.0):
            cc, warp = cv2.findTransformECC(cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma), warp,
                                            cv2.MOTION_HOMOGRAPHY, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-7),
                                            mask, 5)
    except cv2.error:
        return None
    return warp.astype(np.float64), float(cc)


def fit_tone_lut(render: np.ndarray, real: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel 256-entry LUT (BGR) that maps the render's value distribution on *mask* to the photo's (quantile
    matching, 1-99 %, monotone). Insensitive to a residual misregistration, which a pixel-wise colour regression is
    not (the bare board alone has no colour range: an affine fit there collapsed to a constant)."""
    sel = mask > 0
    lut = np.tile(np.arange(256, dtype=np.float32), (3, 1))
    if sel.sum() < 500:
        return lut.astype(np.uint8)
    qs = np.linspace(1, 99, 50)
    for ch in range(3):
        src = np.percentile(render[..., ch][sel], qs)
        dst = np.percentile(real[..., ch][sel], qs)
        src = np.maximum.accumulate(src + np.arange(len(src)) * 1e-3)    # strictly increasing for interp
        lut[ch] = np.interp(np.arange(256), src, dst)                    # flat beyond the 1st / 99th percentile
        # keep the extremes reachable: blend linearly from the end points to 0 / 255
        lo, hi = src[0], src[-1]
        below, above = np.arange(256) < lo, np.arange(256) > hi
        lut[ch][below] = dst[0] * np.arange(256)[below] / max(lo, 1e-3)
        lut[ch][above] = dst[-1] + (255 - dst[-1]) * (np.arange(256)[above] - hi) / max(255 - hi, 1e-3)
    return np.clip(lut + 0.5, 0, 255).astype(np.uint8)


def apply_tone_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """:func:`fit_tone_lut` on BGR or BGRA (alpha kept)."""
    out = img.copy()
    for ch in range(3):
        out[..., ch] = lut[ch][img[..., ch]]
    return out


def composite(rgba: np.ndarray, backplate: np.ndarray) -> np.ndarray:
    """Straight-alpha BGRA render over the BGR *backplate* (same size)."""
    if rgba.ndim == 2 or rgba.shape[2] == 3:
        return rgba
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgba[..., :3].astype(np.float32) * a + backplate.astype(np.float32) * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def noise_sigma(img: np.ndarray, mask: np.ndarray) -> float:
    """Sensor noise (grey levels) by Immerkaer's Laplacian-difference operator, on the smoothest third of *mask*
    (structure there - traces, silkscreen, edges - is what the plain high-pass std mistook for noise)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    resp = np.abs(cv2.filter2D(g, -1, k))
    grad = cv2.GaussianBlur(np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1)), (0, 0), 2.0)
    sel = mask > 0
    if sel.sum() < 300:
        return 0.0
    calm = sel & (grad <= np.percentile(grad[sel], 33))
    return float(np.sqrt(np.pi / 2) * resp[calm].mean() / 6.0)


def sharpness(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean |Laplacian of Gaussian (sigma 1)| on *mask*: edge contrast at the 2-3 px scale, little noise-driven."""
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    sel = mask > 0
    return float(lap[sel].mean()) if sel.sum() else 0.0


def camera_fx(img: np.ndarray, *, blur_sigma: float = 0.0, noise_std: float = 0.0, jpeg_quality: int | None = None,
              seed: int = 0) -> np.ndarray:
    """Lens blur, luminance-weighted sensor noise (stronger in the darks, like a phone's), JPEG round trip. No
    sharpening: matching the photo's edge contrast that way cost SSIM 0.478 -> 0.415 (it amplifies the residual
    misregistration), measured 25.11.2026."""
    out = img.astype(np.float32)
    if blur_sigma > 0.05:
        out = cv2.GaussianBlur(out, (0, 0), blur_sigma)
    if noise_std > 0.05:
        rng = np.random.default_rng(seed)
        # shot noise ~ sqrt(signal): scale the noise by the local brightness relative to mid grey; mostly luminance
        # noise with a little chroma. White (uncorrelated), so that :func:`noise_sigma` measures what was added.
        level = np.sqrt(np.clip(out.mean(2, keepdims=True), 8, 255) / 128.0)
        n = rng.normal(0.0, 1.0, out.shape[:2] + (1,)).astype(np.float32) * 0.95
        n = n + rng.normal(0.0, 0.3, out.shape).astype(np.float32)
        n *= noise_std / max(float(n.std()), 1e-6)
        out = out + n * level
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    if jpeg_quality:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def fit_camera_fx(real: np.ndarray, render: np.ndarray, board_mask: np.ndarray, flat_mask: np.ndarray) -> dict[str, Any]:
    """Lens blur sigma (0 ... 1.4 px) that brings the render's edge contrast (:func:`sharpness`) on *board_mask* to the
    photo's, and the noise std the photo has on *flat_mask* beyond the render's (:func:`noise_sigma`). Matched on
    image statistics, not by maximising SSIM: a residual misregistration would buy blur that no lens has."""
    target = sharpness(real, board_mask)
    best = (abs(np.log(max(sharpness(render, board_mask), 1e-6) / max(target, 1e-6))), 0.0)
    for s in (0.3, 0.5, 0.7, 0.9, 1.1, 1.4):
        v = abs(np.log(max(sharpness(cv2.GaussianBlur(render, (0, 0), s), board_mask), 1e-6) / max(target, 1e-6)))
        if v < best[0]:
            best = (v, s)
    sigma = best[1]
    blurred = cv2.GaussianBlur(render, (0, 0), sigma) if sigma else render
    nr, nb = noise_sigma(real, flat_mask), noise_sigma(blurred, flat_mask)
    noise = float(np.sqrt(max(0.0, nr * nr - nb * nb)))
    return {"blurSigma": sigma, "noiseStd": round(min(noise, 8.0), 2), "noiseReal": round(nr, 2), "noiseRender": round(nb, 2),
            "sharpnessReal": round(target, 2), "sharpnessRender": round(sharpness(render, board_mask), 2)}

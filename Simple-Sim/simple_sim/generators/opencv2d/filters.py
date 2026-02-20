"""Image filter primitives for OpenCV 2D renderer."""

from __future__ import annotations

import cv2
import numpy as np


def apply_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    ksize = int(sigma * 6)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (max(3, ksize), max(3, ksize)), sigma)


def apply_noise(img: np.ndarray, stddev: float, rng: np.random.Generator) -> np.ndarray:
    if stddev <= 0:
        return img
    noise = rng.normal(0, stddev, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    return np.clip((img.astype(np.float32) - 128.0) * factor + 128.0, 0, 255).astype(np.uint8)


def apply_perspective_transform(img: np.ndarray, strength: float, angle_x: float, angle_y: float) -> np.ndarray:
    if strength <= 0 or (abs(angle_x) < 0.1 and abs(angle_y) < 0.1):
        return img
    h, w = img.shape[:2]
    angle_x_rad = np.deg2rad(angle_x * strength)
    angle_y_rad = np.deg2rad(angle_y * strength)
    shift_x = w * 0.15 * np.tan(angle_y_rad)
    shift_y = h * 0.15 * np.tan(angle_x_rad)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[shift_x, shift_y], [w - shift_x, shift_y], [w + shift_x, h - shift_y], [-shift_x, h - shift_y]])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_motion_blur(img: np.ndarray, strength: float, angle: float) -> np.ndarray:
    if strength <= 0:
        return img
    ksize = int(strength * 10) + 1
    if ksize < 3:
        return img
    kernel = np.zeros((ksize, ksize))
    kernel[int((ksize - 1) / 2), :] = np.ones(ksize)
    kernel = kernel / ksize
    rotation_matrix = cv2.getRotationMatrix2D((ksize // 2, ksize // 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (ksize, ksize))
    return cv2.filter2D(img, -1, kernel)


def apply_chromatic_aberration(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    shift = int(strength * 2)
    if shift == 0:
        return img
    b, g, r = cv2.split(img)
    r_shifted = cv2.warpAffine(r, np.float32([[1, 0, shift], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    b_shifted = cv2.warpAffine(b, np.float32([[1, 0, -shift], [0, 1, 0]]), (w, h), borderMode=cv2.BORDER_REPLICATE)
    return cv2.merge([b_shifted, g, r_shifted])


def apply_vignetting(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    center_x, center_y = w // 2, h // 2
    ygrid, xgrid = np.ogrid[:h, :w]
    dist = np.sqrt((xgrid - center_x) ** 2 + (ygrid - center_y) ** 2)
    dist = dist / np.sqrt(center_x**2 + center_y**2)
    vignette = np.clip(1.0 - (dist * strength * 0.7), 0.2, 1.0)
    return (img.astype(np.float32) * np.stack([vignette] * 3, axis=-1)).astype(np.uint8)


def apply_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_hue_shift(img: np.ndarray, shift_deg: float) -> np.ndarray:
    if abs(shift_deg) < 0.1:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + shift_deg / 2.0) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_sharpen(img: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return img
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    kernel = (kernel - 1) * strength + 1
    kernel[1, 1] = 5 * strength
    kernel = kernel / kernel.sum() * 5
    return cv2.filter2D(img, -1, kernel)


def apply_lens_distortion(img: np.ndarray, k1: float, k2: float) -> np.ndarray:
    if abs(k1) < 1e-6 and abs(k2) < 1e-6:
        return img
    h, w = img.shape[:2]
    camera_matrix = np.array([[w, 0, w // 2], [0, w, h // 2], [0, 0, 1]], dtype=np.float32)
    return cv2.undistort(img, camera_matrix, np.array([k1, k2, 0, 0], dtype=np.float32))


def apply_jpeg_compression(img: np.ndarray, quality: int) -> np.ndarray:
    if quality >= 95:
        return img
    _, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def apply_shadow(img: np.ndarray, strength: float, size: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0 or size <= 0:
        return img
    h, w = img.shape[:2]
    shadow_w, shadow_h = int(w * size), int(h * size)
    if shadow_w < 1 or shadow_h < 1:
        return img
    x_pos = int(rng.integers(0, max(1, w - shadow_w)))
    y_pos = int(rng.integers(0, max(1, h - shadow_h)))
    iy, ix = np.ogrid[:shadow_h, :shadow_w]
    dist = np.sqrt(((iy - shadow_h / 2) / (shadow_h / 2)) ** 2 + ((ix - shadow_w / 2) / (shadow_w / 2)) ** 2)
    patch_mask = np.where(dist < 1.0, 1.0 - (1.0 - dist) * strength, 1.0).astype(np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    y1, x1 = min(y_pos + shadow_h, h), min(x_pos + shadow_w, w)
    mask[y_pos:y1, x_pos:x1] = np.minimum(mask[y_pos:y1, x_pos:x1], patch_mask[: y1 - y_pos, : x1 - x_pos])
    return (img.astype(np.float32) * np.stack([mask] * 3, axis=-1)).astype(np.uint8)


def apply_reflection(img: np.ndarray, strength: float, size: float, rng: np.random.Generator) -> np.ndarray:
    if strength <= 0 or size <= 0:
        return img
    h, w = img.shape[:2]
    refl_w, refl_h = int(w * size), int(h * size)
    if refl_w < 1 or refl_h < 1:
        return img
    x_pos = int(rng.integers(0, max(1, w - refl_w)))
    y_pos = int(rng.integers(0, max(1, h - refl_h)))
    iy, ix = np.ogrid[:refl_h, :refl_w]
    dist = np.sqrt(((iy - refl_h / 2) / (refl_h / 2)) ** 2 + ((ix - refl_w / 2) / (refl_w / 2)) ** 2)
    patch = np.where(dist < 1.0, (1.0 - dist) * strength * 255, 0.0).astype(np.float32)
    reflection = np.zeros((h, w), dtype=np.float32)
    y1, x1 = min(y_pos + refl_h, h), min(x_pos + refl_w, w)
    reflection[y_pos:y1, x_pos:x1] = patch[: y1 - y_pos, : x1 - x_pos]
    return np.clip(img.astype(np.float32) + np.stack([reflection] * 3, axis=-1), 0, 255).astype(np.uint8)


def apply_dust_particles(img: np.ndarray, density: float, size: float, rng: np.random.Generator) -> np.ndarray:
    if density <= 0 or size <= 0:
        return img
    h, w = img.shape[:2]
    result = img.copy()
    num_particles = int(w * h * density * 0.001)
    for _ in range(num_particles):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        particle_size = int(rng.uniform(size * 0.5, size * 1.5))
        darkness = rng.uniform(0.3, 0.8)
        if particle_size <= 1:
            if y < h and x < w:
                result[y, x] = (result[y, x].astype(np.float32) * darkness).astype(np.uint8)
        else:
            cv2.circle(
                result,
                (x, y),
                particle_size,
                (result[min(y, h - 1), min(x, w - 1)].astype(np.float32) * darkness).astype(np.uint8).tolist(),
                -1,
                lineType=cv2.LINE_AA,
            )
    return result


def apply_color_temperature(img: np.ndarray, kelvin: int) -> np.ndarray:
    if kelvin == 5500:
        return img
    if kelvin < 5500:
        factor = (5500 - kelvin) / 3000.0
        b_scale, g_scale, r_scale = 1.0 - factor * 0.3, 1.0 - factor * 0.1, 1.0
    else:
        factor = (kelvin - 5500) / 2000.0
        b_scale, g_scale, r_scale = 1.0 + factor * 0.3, 1.0, 1.0 - factor * 0.2
    b, g, r = cv2.split(img.astype(np.float32))
    b = np.clip(b * b_scale, 0, 255)
    g = np.clip(g * g_scale, 0, 255)
    r = np.clip(r * r_scale, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)

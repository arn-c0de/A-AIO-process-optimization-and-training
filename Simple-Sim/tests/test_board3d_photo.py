"""3D-Board-Renderer photo look (no Blender): bare-board fill, texture filter choice, phone colour / camera matching."""

import cv2
import numpy as np

from simple_sim.board3d import photofx
from simple_sim.board3d.render import texture_interpolation
from simple_sim.board3d.substrate import BoardRaster, bare_board, push_pull_fill


def _board(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((120, 160, 3), (40, 120, 40), np.float32)
    img += np.linspace(0, 20, 160)[None, :, None]                       # slow lighting gradient
    img += rng.normal(0, 3, img.shape)                                   # solder-mask grain
    return np.clip(img, 0, 255).astype(np.uint8)


def test_push_pull_fill_is_smooth_and_follows_the_surroundings():
    img = _board()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:80, 50:110] = 255
    img[40:80, 50:110] = (20, 20, 20)                                    # a dark body
    fill = push_pull_fill(img, mask)
    inside = fill[45:75, 55:105]
    assert abs(inside[..., 1].mean() - 120 - 10) < 6                     # green of the board, not the dark body
    assert inside[..., 1].std() < 6                                      # no streaks
    assert inside[:, -1, 1].mean() > inside[:, 0, 1].mean()              # keeps the gradient


def test_bare_board_keeps_known_pixels_and_adds_grain():
    img = _board()
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[40:80, 50:110] = 255
    img[40:80, 50:110] = (20, 20, 20)
    out = bare_board(img, mask, np.zeros_like(mask))
    keep = mask == 0
    assert np.array_equal(out[keep], img[keep])
    hp = lambda x: (x.astype(np.float32) - cv2.GaussianBlur(x.astype(np.float32), (0, 0), 2.0))[45:75, 55:105].std()  # noqa: E731
    assert hp(out[..., 1]) > 1.0                                          # grain, not a flat patch


def test_texture_interpolation():
    raster = {"x0": 0.0, "y0": 0.0, "widthMm": 135.0, "heightMm": 99.0, "pxPerMm": 24.0}
    assert texture_interpolation(BoardRaster(64, 41, 20, 20, 24.0), raster, None) == "Closest"
    assert texture_interpolation(BoardRaster(64.01, 41, 20, 20, 24.0), raster, None) == "Linear"   # off the texel grid
    assert texture_interpolation(BoardRaster(64, 41, 20, 20, 12.0), raster, None) == "Linear"
    assert texture_interpolation(BoardRaster(64, 41, 20, 20, 24.0), raster, {"type": "PERSP"}) == "Linear"


def test_colour_gain_and_tone_lut():
    tex = _board()
    lin = photofx.srgb_to_linear(tex) * np.array([0.6, 0.8, 1.5], np.float32)     # BGR gains in linear light
    real = np.clip(255 * np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055) + 0.5, 0, 255).astype(np.uint8)
    gain = photofx.colour_gain(tex, real, np.full(tex.shape[:2], 255, np.uint8))
    assert np.allclose(gain, [1.5, 0.8, 0.6], atol=0.06)                 # RGB order for Blender
    lifted = np.clip(tex.astype(np.float32) * 0.8 + 25, 0, 255).astype(np.uint8)  # a phone's lifted blacks
    lut = photofx.fit_tone_lut(tex, lifted, np.full(tex.shape[:2], 255, np.uint8))
    assert np.abs(photofx.apply_tone_lut(tex, lut).astype(int) - lifted).mean() < 2.0
    assert np.all(np.diff(lut.astype(int), axis=1) >= 0)                 # monotone


def test_composite_and_camera_fx():
    plate = np.full((40, 40, 3), 200, np.uint8)
    rgba = np.zeros((40, 40, 4), np.uint8)
    rgba[10:30, 10:30] = (0, 100, 0, 255)                                # the board
    rgba[30:35, 10:30, 3] = 128                                          # shadow catcher: half shadow
    out = photofx.composite(rgba, plate)
    assert out[20, 20].tolist() == [0, 100, 0] and out[0, 0].tolist() == [200, 200, 200]
    assert 95 <= out[32, 20, 0] <= 105
    img = _board()
    noisy = photofx.camera_fx(img, noise_std=4.0, seed=1)
    assert abs(photofx.noise_sigma(noisy, np.full(img.shape[:2], 255, np.uint8)) - photofx.noise_sigma(img, np.full(img.shape[:2], 255, np.uint8))) > 1.0
    fx = photofx.fit_camera_fx(cv2.GaussianBlur(img, (0, 0), 1.0), img, np.full(img.shape[:2], 255, np.uint8),
                               np.full(img.shape[:2], 255, np.uint8))
    assert 0.5 <= fx["blurSigma"] <= 1.4                                 # finds the lens blur


def test_mosaic_field_warp_and_fusion():
    from simple_sim.board3d.photomosaic import field_grid, fuse, warp_photo
    raster = BoardRaster(0, 0, 20, 10, 10.0)
    dx, dy = field_grid([5, 5, 0.2, -0.1] * 1 + [6, 5, 0.2, -0.1, 5, 6, 0.2, -0.1, 6, 6, 0.2, -0.1], raster)
    assert abs(float(dx[50, 55]) - 0.2) < 1e-4 and abs(float(dy[50, 55]) + 0.1) < 1e-4
    assert field_grid(None, raster)[0].max() == 0.0
    # a photo that is the board raster itself, 10 px/mm, y up: H maps board mm -> photo px
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, 100:] = (0, 0, 255)                                           # right half red (x >= 10 mm)
    geo = {"boardToSource": [10, 0, 0, 0, -10, 100, 0, 0, 1], "sourceSize": [200, 100]}
    out = warp_photo(img, geo, raster, None)
    assert out[50, 120].tolist() == [0, 0, 255] and out[50, 80].tolist() == [0, 0, 0]
    # fusion: the sharp photo wins where it agrees with the others, not the one with a bright blob
    rng = np.random.default_rng(0)
    base = np.clip(_board() + rng.normal(0, 6, (120, 160, 3)), 1, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(base, (0, 0), 1.5)
    blob = base.copy()
    cv2.circle(blob, (80, 60), 25, (250, 250, 250), -1)
    fused, rep = fuse([soft, base, soft.copy(), blob], 10.0)
    assert rep["tilesWon"][3] < rep["tilesWon"][1]
    assert fused[60, 80].mean() < 150                                    # the glare blob is not taken over

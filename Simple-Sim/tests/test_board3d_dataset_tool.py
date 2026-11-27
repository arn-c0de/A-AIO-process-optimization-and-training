"""board3d_dataset helpers: the phone-camera look of the crops."""

import random

import numpy as np

from tools.board3d_dataset import apply_phone_fx, phone_fx


def test_phone_fx_parameters_and_effect():
    rng = random.Random(0)
    fx = [phone_fx(rng, 128) for _ in range(50)]
    assert all(32 <= f["lowPx"] <= 64 and 55 <= f["jpegQuality"] <= 92 for f in fx)
    img = np.zeros((128, 128, 3), np.uint8)
    img[:, 64:] = 255                                          # a sharp edge
    out = apply_phone_fx(img, {"lowPx": 40, "blurSigma": 0.8, "noiseStd": 2.0, "jpegQuality": 70}, seed=1)
    assert out.shape == img.shape and out.dtype == np.uint8
    row = out[64, :, 1].astype(int)
    assert 20 < row[64] < 235 or 20 < row[63] < 235            # the edge is soft now
    assert abs(int(row[:40].mean()) - 0) < 15 and abs(int(row[-40:].mean()) - 255) < 15   # levels kept

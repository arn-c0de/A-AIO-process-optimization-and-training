"""board3d_dataset helpers: the phone-camera look of the crops, crops of phone views, photo lists."""

import random

import numpy as np
import pytest

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


def test_crop_photo_follows_the_homography_and_rejects_off_frame():
    import cv2
    from simple_sim.board3d.dataset import crop_photo
    img = np.zeros((400, 600, 3), np.uint8)
    h = np.array([[10.0, 0, 50], [0, -10.0, 350], [0, 0, 1]])           # board mm -> px, 10 px/mm, y up
    cv2.circle(img, (int(10 * 20 + 50), int(350 - 10 * 15)), 4, (255, 255, 255), -1)   # a mark at board (20, 15)
    c = {"pose": {"x": 20.0, "y": 15.0, "rotDeg": 0}, "body": {"l": 2.0, "w": 1.0}}
    crop, meta = crop_photo(img, h, c, 64)
    assert crop.shape == (64, 64, 3) and meta["fieldPx"] == pytest.approx(2 * 3.04 * 2.0 / 2 * 10, rel=0.02)
    ys, xs = np.nonzero(crop[..., 0] > 128)
    assert abs(xs.mean() - 32) < 2 and abs(ys.mean() - 32) < 2       # the mark sits in the middle
    far = {"pose": {"x": 200.0, "y": 15.0, "rotDeg": 0}, "body": {"l": 2.0, "w": 1.0}}
    assert crop_photo(img, h, far, 64) is None


def test_photo_geometries_filters_product_and_side(tmp_path):
    import json
    from tools.board3d_dataset import photo_geometries
    for name, product, side in (("a", "p1", "top"), ("b", "p1", "bottom"), ("c", "p2", "top"), ("d", "p1", "top")):
        d = tmp_path / name
        d.mkdir()
        (d / "geometry.json").write_text(json.dumps({"product": product, "side": side, "boardToSource": [1, 0, 0, 0, 1, 0, 0, 0, 1]}))
        if name != "d":
            (d / "still.jpg").write_bytes(b"x")                    # d has no still: skipped
    got = photo_geometries(tmp_path, {"sources": {"product": "p1"}, "side": "top"})
    assert [p.parent.name for p in got] == ["a"]

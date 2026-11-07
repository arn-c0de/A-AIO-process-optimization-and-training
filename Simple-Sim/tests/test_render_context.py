"""Typed render state: resolved ``RenderContext`` fields, immutable looks, ``CameraSettings``, texture cache."""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from simple_sim.board import BoardOptions
from simple_sim.generators.opencv2d.photoreal import CAMERA_DEFAULTS, LOOKS, RenderContext, camera_settings
from simple_sim.generators.opencv2d.photoreal.camera import CameraSettings, validate_camera_config
from simple_sim.generators.opencv2d.photoreal.look import _LOOKS, LEGACY_LOOK, QUINS_LOOK
from simple_sim.generators.opencv2d.photoreal.substrate import _decoded_texture, load_texture


def test_for_render_resolves_the_config_once():
    cfg = {"look": "quins", "texture_dir": "/tmp/x", "component_type": "capacitor", "component_color": [1, 2, 3],
           "marking_probability": 0.25, "mm_per_px": 0.02, "board": {"routing": "none"}}
    ctx = RenderContext.for_render(cfg, np.random.default_rng(0), 90.0, post_roll_deg=90.0)
    assert ctx.look is QUINS_LOOK and ctx.post_roll_deg == 90.0
    assert (ctx.texture_dir, ctx.component_type, ctx.component_color) == ("/tmp/x", "capacitor", (1, 2, 3))
    assert ctx.marking_probability == 0.25 and ctx.mm_per_px == 0.02
    assert ctx.camera == {"preset": "quins"} and ctx.mask_gloss == 0.0 and ctx.rework == QUINS_LOOK.rework
    assert ctx.board == BoardOptions(routing="none")
    empty = RenderContext.for_render(None, np.random.default_rng(0))
    defaults = RenderContext()
    for name in ("texture_dir", "component_type", "component_color", "marking_probability", "mm_per_px", "camera",
                 "mask_gloss", "rework", "board", "post_roll_deg", "look"):
        assert getattr(empty, name) == getattr(defaults, name), name
    with pytest.raises(ValueError, match="render.board"):
        RenderContext.for_render({"board": {"routing": "nope"}}, np.random.default_rng(0))


def test_look_registry_and_immutability():
    assert LOOKS == tuple(_LOOKS)
    for look in (LEGACY_LOOK, QUINS_LOOK):
        for arr in (look.copper_under_mask, look.cap_body, look.end_cap, look.lead, *look.bodies.values()):
            assert not arr.flags.writeable
        with pytest.raises(TypeError):
            look.bodies["x"] = np.zeros(3)  # type: ignore[index]
        with pytest.raises(TypeError):
            look.cap_len["x"] = 0.1  # type: ignore[index]
    with pytest.raises(ValueError):
        QUINS_LOOK.cap_body[0] = 0.0


def test_camera_settings_is_a_frozen_mapping():
    c = camera_settings({"preset": "quins", "read_noise": [1.0, 2.0]})
    assert isinstance(c, CameraSettings) and c["channel_axis"] == "x" and c["read_noise"] == [1.0, 2.0]
    assert c.get("max_oversampling", 1.0) == 0.0 and set(c) == set(CAMERA_DEFAULTS)
    assert {**c, "zoom": [2.0, 2.0]}["zoom"] == [2.0, 2.0] and dict(c) == c
    assert camera_settings(c) is c
    with pytest.raises(AttributeError):
        c.zoom = [1.0, 1.0]  # type: ignore[misc]
    with pytest.raises(KeyError):
        c["preset"]
    with pytest.raises(ValueError, match="unknown keys"):
        camera_settings({"blur_sigmaa": [1, 2]})
    with pytest.raises(ValueError, match="channel_axis"):
        camera_settings({"channel_axis": "z"})
    with pytest.raises(ValueError, match="preset"):
        camera_settings({"preset": "nope"})


def test_camera_config_validation_is_shared_with_config():
    from simple_sim.config import _validate_camera

    for bad in ({"blur_sigmaa": [1, 2]}, {"tilt_deg": [0, 90]}, {"channel_axis": "z"}, {"preset": "nope"}):
        with pytest.raises(ValueError, match="render.camera"):
            validate_camera_config(bad)
        with pytest.raises(ValueError, match="render.camera"):
            _validate_camera(bad)
    validate_camera_config({"preset": "phone", "jpeg_quality": [80, 90]})


def test_texture_photos_are_decoded_once_per_process(tmp_path):
    path = tmp_path / "board.png"
    cv2.imwrite(str(path), np.full((12, 16, 3), 77, np.uint8))
    _decoded_texture.cache_clear()
    a, b = load_texture(path), load_texture(path)
    assert a is b and not a.flags.writeable and _decoded_texture.cache_info().hits == 1
    cv2.imwrite(str(path), np.full((12, 16, 3), 9, np.uint8))
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))  # replaced on disk: decoded again
    c = load_texture(path)
    assert c is not a and int(c[0, 0, 0]) == 9
    with pytest.raises(OSError):
        load_texture(tmp_path / "missing.png")

"""Backend registry (``simple_sim.generators.backends``)."""

from __future__ import annotations

import pytest

from simple_sim.classes import Backend
from simple_sim.generators.backends import BACKENDS, DEFAULT_BACKEND, backend_of, parse_backend, supporting


def test_every_backend_enum_member_is_registered():
    assert set(BACKENDS) == set(Backend)
    assert backend_of(None).backend is DEFAULT_BACKEND is Backend.OPENCV_2D
    assert backend_of({"backend": "blender_3d"}).needs_blender


def test_unknown_backend_message():
    with pytest.raises(ValueError, match="Unsupported render backend: vulkan"):
        parse_backend("vulkan")


def test_capabilities_drive_config_validation():
    from simple_sim.config import _validate_domains

    assert supporting("domain_camera") == ["opencv_2d"]
    dom = {"d": {"lighting_brightness": [1, 1], "blur_sigma": [0, 0], "noise_stddev": [0, 0], "camera": {"preset": "phone"}}}
    _validate_domains(dom, {"backend": "opencv_2d"})
    with pytest.raises(ValueError, match="only supported by render.backend: opencv_2d"):
        _validate_domains(dom, {"backend": "blender_3d"})

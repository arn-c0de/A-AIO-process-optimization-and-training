"""GUI Profile Preview tab logic (tk-free): renders per-state previews with chosen settings."""

from pathlib import Path

import pytest

from gui.tabs.profile_preview.logic import PreviewCancelled, PreviewRequest, ProfilePreviewLogic

SIM_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def logic(tmp_path):
    lg = ProfilePreviewLogic(SIM_ROOT)
    lg.out_root = tmp_path
    return lg


def test_renders_variants_per_state_with_ultra(logic):
    req = PreviewRequest("chip_0603_resistor@2", "opencv_2d", states=["OK", "TOMBSTONE"], variants=2,
                         supersampling=4, camera="phone")
    seen = []
    images = logic.render(req, on_image=seen.append, on_log=lambda _s: None, should_stop=lambda: False)
    assert [(im.state, im.variant) for im in images] == [("OK", 1), ("OK", 2), ("TOMBSTONE", 1), ("TOMBSTONE", 2)]
    assert seen == images and all(im.path.is_file() for im in images)
    assert "tilt" in images[2].info


def test_stop_cancels(logic):
    req = PreviewRequest("chip_0603_resistor@2", "opencv_2d", states=["OK"], variants=3)
    with pytest.raises(PreviewCancelled):
        logic.render(req, on_image=lambda _i: None, on_log=lambda _s: None, should_stop=lambda: True)


def test_configs_for_lists_matching_backend_first(logic):
    names = logic.configs_for("sot23_transistor@2", "opencv_2d")
    assert names[0] == "run_sot23.yaml" and "run_0001_3d.yaml" not in names


def test_sidecar_keeps_defect_for_overlay(logic):
    from gui.components.overlay_renderer import draw_defect_overlay
    from gui.tabs.profile_preview.logic import read_sidecar
    import cv2

    req = PreviewRequest("chip_0603_resistor@2", "opencv_2d", states=["TOMBSTONE"], variants=1)
    (im,) = logic.render(req, on_image=lambda _i: None, on_log=lambda _s: None, should_stop=lambda: False)
    side = read_sidecar(im.path)
    assert side["defect"]["type"] == "TOMBSTONE" and side["nominal"]
    out = draw_defect_overlay(cv2.imread(str(im.path)), side["defect"], side["nominal"])
    assert out.shape == (256, 256, 3)

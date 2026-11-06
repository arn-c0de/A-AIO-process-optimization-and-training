"""Render-section validation shared by both backends (the 3D bake uses the photoreal board settings)."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest

from simple_sim.config import load_config, validate_config

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("run", ["run_sot23_3d.yaml", "run_0001.yaml"])
@pytest.mark.parametrize("key,value", [("mask_gloss", 5), ("mask_gloss", True), ("rework_artifacts", "lots"),
                                       ("texture_dir", "/nonexistent/textures"), ("solder", "shiny")])
def test_board_look_settings_validated_for_both_backends(run, key, value):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / run))
    cfg["render"][key] = value
    with pytest.raises(ValueError, match=f"render.{key}"):
        validate_config(cfg)


def test_flat_style_warns_about_ignored_photoreal_settings(caplog):
    cfg = copy.deepcopy(load_config(ROOT / "configs" / "run_0001.yaml"))
    cfg["render"].update(style="flat", supersampling=4, board={"routing": "escape"})
    with caplog.at_level(logging.WARNING, logger="simple_sim.config"):
        validate_config(cfg)
    assert "render.board" in caplog.text and "render.supersampling" in caplog.text

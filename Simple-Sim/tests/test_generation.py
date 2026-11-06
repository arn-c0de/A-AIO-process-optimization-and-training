"""Tests for simple_sim.generation (shared generator helpers, Blender post-filter, CLI --json)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from simple_sim.cli_utils import parse_result
from simple_sim.generation.blender import blender_settings, postprocess_blender_images
from simple_sim.generation.dataset import read_existing_max_index

SIM_ROOT = Path(__file__).resolve().parent.parent


def test_postprocess_skips_neutral_and_filters_others(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"images/{i}.png"), img)
    before = (tmp_path / "images/0.png").read_bytes()
    postprocess_blender_images(
        [
            {"image_path": "images/0.png", "seed": 1, "augment": {"rotation_deg": 90.0, "brightness_factor": 1.0}},
            {"image_path": "images/1.png", "seed": 1, "augment": {"brightness_factor": 1.5}},
        ],
        tmp_path,
    )
    assert (tmp_path / "images/0.png").read_bytes() == before
    assert int(cv2.imread(str(tmp_path / "images/1.png")).mean()) > 100


def test_blender_settings_defaults() -> None:
    assert blender_settings({}) == {"executable": "blender", "samples": 64, "device": "CPU", "engine": "BLENDER_EEVEE"}


def _meta(i: str, path: str) -> SimpleNamespace:
    return SimpleNamespace(id=i, image_path=path)  # only the fields read_existing_max_index uses


def test_read_existing_max_index() -> None:
    assert read_existing_max_index([]) == -1
    assert read_existing_max_index([_meta("r/d/train/000007", "images/000007.png"), _meta("weird", "images/000012.png")]) == 12


def test_generate_cli_json_and_extend(tmp_path: Path) -> None:
    cfg = yaml.safe_load((SIM_ROOT / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {k: 2 for k in cfg["classes"]}
    cfg["roi"].update(width_px=64, height_px=64)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / "ds"
    cmd = [sys.executable, str(SIM_ROOT / "scripts" / "generate.py"), "--config", str(cfg_path), "--out", str(out), "--json"]
    n = 2 * len(cfg["classes"])
    res = parse_result(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    assert res["ok"] and res["total_samples"] == res["new_samples"] == n
    assert sum(res["splits"].values()) == n and sum(res["classes"].values()) == n
    res2 = parse_result(subprocess.run(cmd + ["--extend"], capture_output=True, text=True, check=True).stdout)
    assert res2["extend"] and res2["total_samples"] == 2 * n and res2["new_samples"] == n
    assert len(list((out / "images").glob("*.png"))) == 2 * n

    bad = subprocess.run(cmd[:3] + [str(tmp_path / "missing.yaml")] + cmd[4:], capture_output=True, text=True)
    assert bad.returncode == 1 and parse_result(bad.stdout)["ok"] is False


def test_generate_profile_dataset_cli_json(tmp_path: Path) -> None:
    cfg = yaml.safe_load((SIM_ROOT / "configs" / "run_profile_cls.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {k: 2 for k in cfg["classes"]}
    cfg["roi"].update(width_px=64, height_px=64)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SIM_ROOT / "scripts" / "generate_profile_dataset.py"), "--config", str(cfg_path),
         "--out", str(tmp_path / "ds"), "--json"],
        capture_output=True, text=True, check=True,
    )
    res = parse_result(proc.stdout)
    n_prof = len(cfg["run"]["component_profiles"])
    assert res["ok"] and res["total_samples"] == 8 * n_prof
    assert set(res["profiles"]) == set(cfg["run"]["component_profiles"])

"""Renderer version in the dataset manifest: ``generate --extend`` must not mix two looks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from simple_sim.generators.version import RENDERER_VERSION
from simple_sim.manifest import check_renderer_version

ROOT = Path(__file__).parent.parent


def _tiny_config(tmp_path: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / "run_0001.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = dict.fromkeys(list(cfg["classes"])[:2], 1)
    cfg["roi"].update(width_px=64, height_px=64)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def test_check_renderer_version():
    check_renderer_version({"generator": {"renderer_version": RENDERER_VERSION}}, RENDERER_VERSION)
    with pytest.raises(ValueError, match="Renderer version mismatch"):
        check_renderer_version({"generator": {"renderer_version": "1"}}, RENDERER_VERSION)
    with pytest.raises(ValueError, match="unknown"):
        check_renderer_version({"generator": {}}, RENDERER_VERSION)  # v1.0.3 datasets have none
    check_renderer_version({"generator": {}}, RENDERER_VERSION, allow_mismatch=True)


def test_generate_records_version_and_extend_refuses_mismatch(tmp_path):
    from simple_sim.generation.dataset import generate_dataset

    cfg = _tiny_config(tmp_path)
    out = tmp_path / "ds"
    generate_dataset(cfg, out, project_root=ROOT)
    manifest_path = out / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["generator"]["renderer_version"] == RENDERER_VERSION
    assert manifest["extend_history"][0]["renderer_version"] == RENDERER_VERSION

    generate_dataset(cfg, out, extend=True, project_root=ROOT)  # same version: fine
    assert len(json.loads(manifest_path.read_text())["extend_history"]) == 2

    manifest = json.loads(manifest_path.read_text())
    del manifest["generator"]["renderer_version"]  # as written by v1.0.3
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Renderer version mismatch"):
        generate_dataset(cfg, out, extend=True, project_root=ROOT)
    generate_dataset(cfg, out, extend=True, project_root=ROOT, allow_renderer_mismatch=True)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["generator"]["renderer_version"] == RENDERER_VERSION
    assert len(manifest["extend_history"]) == 3

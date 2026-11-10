"""``generate_dataset`` as a library call: config mapping, explicit image filters / profiles dir, the
pure sampling step ``plan_records``."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from simple_sim.config import load_config
from simple_sim.filter_settings import normalize_image_filters
from simple_sim.generation.dataset import generate_dataset, plan_records, resolve_run

ROOT = Path(__file__).resolve().parent.parent


def _small_config() -> dict:
    cfg = load_config(ROOT / "configs" / "run_0001.yaml")
    cfg["classes"] = {"OK": 2, "MISSING": 1, "TOMBSTONE": 1}
    cfg["roi"].update(width_px=96, height_px=96)
    return cfg


def _images(ds: Path) -> dict[str, bytes]:
    return {p.name: cv2.imread(str(p), cv2.IMREAD_UNCHANGED).tobytes() for p in sorted((ds / "images").glob("*.png"))}


def test_config_mapping_equals_config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    cfg = _small_config()
    before = json.dumps(cfg, sort_keys=True, default=str)
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")  # keep the class order
    a = generate_dataset(path, tmp_path / "from_file", project_root=ROOT)
    b = generate_dataset(cfg, tmp_path / "from_dict", project_root=ROOT, profiles_dir=ROOT / "configs" / "profiles")
    assert json.dumps(cfg, sort_keys=True, default=str) == before  # the caller's mapping is not modified
    assert a["total_samples"] == b["total_samples"] == 4
    assert _images(tmp_path / "from_file") == _images(tmp_path / "from_dict")
    assert (tmp_path / "from_file" / "labels.jsonl").read_text() == (tmp_path / "from_dict" / "labels.jsonl").read_text()


def test_image_filters_parameter_wins_over_the_environment(tmp_path, monkeypatch):
    filters = {"enable_blur": False, "enable_grain": False, "cardinal_rotation_90": False}
    cfg = _small_config()
    monkeypatch.setenv("IMAGE_FILTERS", json.dumps(filters))
    generate_dataset(cfg, tmp_path / "env", project_root=ROOT)
    monkeypatch.setenv("IMAGE_FILTERS", "{}")
    generate_dataset(cfg, tmp_path / "param", project_root=ROOT, image_filters=filters)
    assert _images(tmp_path / "env") == _images(tmp_path / "param")


def test_plan_records_is_pure_and_deterministic():
    run = resolve_run(_small_config(), profiles_dir=ROOT / "configs" / "profiles", root=ROOT)
    seen = []
    kw = {"image_filters": normalize_image_filters(None), "start_index": 5}
    a = plan_records(run, progress=lambda cls, i, n, rec: seen.append((cls, i, n, rec["index"])), **kw)
    b = plan_records(run, **kw)
    assert [r["index"] for r in a] == [5, 6, 7, 8]
    assert seen[0] == ("OK", 1, 2, 5) and seen[-1] == ("TOMBSTONE", 1, 1, 8)
    assert json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)
    assert all(isinstance(r["seed"], (int, np.integer)) for r in a)
    assert run.config["render"]["component_type"]  # profile defaults merged into the render section


def test_extend_refuses_other_look_options(tmp_path, monkeypatch):
    """``--extend`` compares the resolved look options with the dataset's config.yaml."""
    import pytest

    from simple_sim.manifest import RendererMismatchError

    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    cfg = _small_config()
    ds = tmp_path / "ds"
    generate_dataset(cfg, ds, project_root=ROOT)
    same = {**cfg, "render": {**cfg["render"], "solder": "legacy", "supersampling": 2}}  # explicit defaults: same look
    assert generate_dataset(same, ds, project_root=ROOT, extend=True)["new_samples"] == 4
    for change in ({"solder": "meniscus"}, {"supersampling": 3}, {"style": "flat"}, {"camera": {"preset": "phone"}},
                   {"board": {"routing": "escape"}}, {"post_filters": "camera"}):
        other = {**cfg, "render": {**cfg["render"], **change}}
        with pytest.raises(RendererMismatchError, match="Render options differ"):
            generate_dataset(other, ds, project_root=ROOT, extend=True)
    other = {**cfg, "defects": {"placement_sampling": "continuous"}}
    with pytest.raises(RendererMismatchError, match="placement_sampling"):
        generate_dataset(other, ds, project_root=ROOT, extend=True)
    res = generate_dataset({**cfg, "render": {**cfg["render"], "solder": "meniscus"}}, ds, project_root=ROOT,
                           extend=True, allow_renderer_mismatch=True)
    assert res["total_samples"] == 12


def test_look_options_resolve_defaults():
    from simple_sim.generation.render_options import look_differences, look_options

    cfg = _small_config()
    opts = look_options(cfg)
    assert opts["render.style"] == "photoreal" and opts["render.supersampling"] == 2 and opts["render.solder"] == "legacy"
    assert look_differences(cfg, {**cfg, "render": {**cfg["render"], "look": "legacy", "mask_gloss": 0.5}}) == []
    assert look_differences(cfg, {**cfg, "render": {**cfg["render"], "mask_gloss": 0.9}}) == ["render.mask_gloss: 0.5 -> 0.9"]

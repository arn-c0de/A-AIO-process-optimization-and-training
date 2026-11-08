"""Golden hashes of every 2D render variant (all part families, all classes, every ``render`` option).

``test_render_golden.py`` pins the default look through the whole dataset pipeline; this file pins
the renderer itself, case by case, so a refactor or speed-up of ``simple_sim.generators.opencv2d``
can be checked bit for bit:

* ``base/<config>/<class>``: every 2D run config with every defect class, default options,
* ``<variant>/<config>/<class>``: one ``render`` option changed (style, supersampling, solder model,
  post filters, board router, camera preset, target look, gloss/rework/marking, texture substrate, all extended
  image filters),
* ``bake/...`` and ``camera/...``: the Blender board bake (``render_board`` + its scene) and the
  camera model on a fixed image.

Update after an *intended* look change (and bump ``RENDERER_VERSION``):
``.venv/bin/python -m tests.test_render_golden_variants --update``. Skipped when OpenCV/NumPy differ
from the versions that wrote the file (rasterisation depends on the build).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from simple_sim.classes import VALID_CLASSES
from simple_sim.config import load_config, merge_profile_render
from simple_sim.filter_settings import normalize_image_filters
from simple_sim.generation.common import sample_record
from simple_sim.generators.opencv2d import render_roi
from simple_sim.generators.version import RENDERER_VERSION
from simple_sim.profile_hash import load_profile

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = Path(__file__).resolve().parent / "golden" / "render_variants.json"
PROFILES = ROOT / "configs" / "profiles"

CONFIGS_2D = ("run_0001", "run_0001_phone", "run_0402_c", "run_0402_r", "run_0603_ferrite", "run_0603_led",
              "run_1210_c", "run_2520_inductor", "run_array_c", "run_array_r", "run_qfn32", "run_sot23",
              "run_sot23_6", "run_sot363", "run_tssop14", "run_tssop20")
CLASSES = tuple(sorted(VALID_CLASSES))
VARIANT_CONFIGS = ("run_0001", "run_qfn32", "run_sot23", "run_array_r", "run_0603_led", "run_2520_inductor")
VARIANT_CLASSES = ("OK", "TOMBSTONE", "SOLDER_BRIDGE", "EXCESS_SOLDER", "MISALIGNED")

_ALL_FILTERS = {k: True for k in normalize_image_filters(None) if k.startswith("enable_")}
_ALL_FILTERS.update(perspective_strength=1.0, perspective_angle_x=4.0, motion_blur_strength=1.0, shadow_strength=0.4,
                    reflection_strength=0.6, vignetting_strength=1.0, chromatic_strength=1.0, distortion_k1=0.1,
                    dust_density=0.5, sharpen_strength=1.0, jpeg_quality=70, color_temperature_kelvin=4000)
_REALISM_FILTERS = {"filter_mode": "realism", "realism_enabled": True}

# name -> (render overrides, image filters)
VARIANTS: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {
    "flat": ({"style": "flat"}, None),
    "ss1": ({"supersampling": 1}, None),
    "ss3": ({"supersampling": 3}, None),
    "ss4": ({"supersampling": 4}, None),
    "meniscus": ({"solder": "meniscus"}, None),
    "post_camera": ({"post_filters": "camera"}, None),
    "route_none": ({"board": {"routing": "none"}}, None),
    "route_escape": ({"board": {"routing": "escape", "pour_probability": 1.0}}, None),
    "route_escape_drc": ({"board": {"routing": "escape", "design_rules": "hdi", "drc": "warn"}}, None),
    "phone": ({"camera": {"preset": "phone"}}, None),
    "look_quins": ({"look": "quins"}, None),
    "look_dv_top": ({"look": "dv_top"}, None),
    "look_dv_bottom": ({"look": "dv_bottom"}, None),
    "gloss_rework": ({"mask_gloss": 1.0, "rework_artifacts": 1.0, "marking_probability": 1.0}, None),
    "texture": ({"texture_dir": "<textures>"}, None),
    "filters_all": ({}, _ALL_FILTERS),
    "filters_realism": ({}, _REALISM_FILTERS),
    "flat_filters": ({"style": "flat"}, _ALL_FILTERS),
}


def _versions() -> dict[str, str]:
    return {"opencv": cv2.__version__, "numpy": np.__version__}


def _md5(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest() + f":{arr.shape}:{arr.dtype}"


def make_textures(directory: Path) -> Path:
    """Two deterministic board 'photos' for ``render.texture_dir`` (PNG, so lossless)."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    for i in range(2):
        img = rng.integers(20, 110, size=(96, 128, 3), dtype=np.uint8)
        img = cv2.GaussianBlur(img, (0, 0), 2.0)
        cv2.imwrite(str(directory / f"tex_{i}.png"), img)
    return directory


class _Case:
    def __init__(self, run: str):
        cfg = load_config(ROOT / "configs" / f"{run}.yaml")
        self.config = cfg
        self.profile = load_profile(cfg["run"]["component_profile"], PROFILES)
        self.footprint = self.profile["component"]["footprint"]
        self.geometry_ranges = self.profile.get("geometry_ranges")
        self.tolerances = self.profile.get("tolerances")
        self.render = merge_profile_render(cfg["render"], self.profile, cfg.get("roi"))
        self.roi = (int(cfg["roi"]["width_px"]), int(cfg["roi"]["height_px"]))

    def render_one(self, class_name: str, index: int, overrides: dict[str, Any], filters: dict[str, Any] | None,
                   textures: Path | None) -> np.ndarray:
        image_filters = normalize_image_filters(filters)
        rec, _ = sample_record(index, class_name, config=self.config, run_seed=self.config["run"]["seed"],
                               geometry_ranges=self.geometry_ranges, tolerances=self.tolerances,
                               image_filters=image_filters, enable_cardinal_rotation_90=True,
                               footprint=self.footprint)
        render = {**self.render, **overrides}
        if render.get("texture_dir") == "<textures>":
            render["texture_dir"] = str(textures)
        return render_roi(nominal=rec["nominal"], defect_params=rec["defect"], augment=rec["augment"],
                          roi_size=self.roi, config=render, tolerances=self.tolerances,
                          rng=np.random.default_rng(int(rec["seed"])), footprint=self.footprint)


def iter_cases(textures: Path) -> Iterator[tuple[str, Callable[[], str]]]:
    """(name, thunk returning the hash) for every golden case, in a fixed order."""
    cases: dict[str, _Case] = {}

    def case(run: str) -> _Case:
        if run not in cases:
            cases[run] = _Case(run)
        return cases[run]

    for run in CONFIGS_2D:
        for i, cls in enumerate(CLASSES):
            yield f"base/{run}/{cls}", lambda run=run, cls=cls, i=i: _md5(case(run).render_one(cls, 11 + i, {}, None, textures))
    for name, (overrides, filters) in VARIANTS.items():
        for run in VARIANT_CONFIGS:
            for i, cls in enumerate(VARIANT_CLASSES):
                yield (f"{name}/{run}/{cls}",
                       lambda run=run, cls=cls, i=i, o=overrides, f=filters: _md5(case(run).render_one(cls, 101 + i, o, f, textures)))
    for run in ("run_0001", "run_qfn32", "run_sot23"):
        for ss in (1, 2, 3):
            yield f"bake/{run}/ss{ss}", lambda run=run, ss=ss: _bake_hash(case(run), ss)
    for preset in ("aoi", "phone", "quins"):
        yield f"camera/{preset}", lambda preset=preset: _camera_hash(preset)


def _bake_hash(c: _Case, ss: int) -> str:
    from simple_sim.generators.opencv2d.photoreal import render_board

    rec, _ = sample_record(5, "OK", config=c.config, run_seed=c.config["run"]["seed"], geometry_ranges=c.geometry_ranges,
                           tolerances=c.tolerances, image_filters=normalize_image_filters(None),
                           enable_cardinal_rotation_90=True, footprint=c.footprint)
    config = {"component_type": c.render.get("component_type", ""), "supersampling": ss,
              "mm_per_px": c.render.get("mm_per_px"), "board": {"routing": "escape"} if ss == 3 else None}
    w, h = c.roi
    texture, scene = render_board(rec["nominal"], rec["defect"], config, c.tolerances, np.random.default_rng(5),
                                  c.footprint, (2 * w, 2 * h), crowd=4.0)
    return _md5(texture) + "|" + hashlib.md5(json.dumps(scene, sort_keys=True, default=float).encode()).hexdigest()


def _camera_hash(preset: str) -> str:
    from simple_sim.generators.opencv2d.photoreal import camera_model, camera_settings, sample_camera_pose

    rng = np.random.default_rng(3)
    img = cv2.GaussianBlur(rng.integers(0, 255, size=(200, 240, 3), dtype=np.uint8), (0, 0), 1.5)
    cam = {"preset": preset}
    pose = sample_camera_pose(camera_settings(cam), np.random.default_rng(4))
    return _md5(camera_model(img, np.random.default_rng(9), cam, pose, mm_per_px=0.01))


def render_hashes(tmp_dir: Path) -> dict[str, str]:
    textures = make_textures(tmp_dir / "textures")
    return {name: thunk() for name, thunk in iter_cases(textures)}


def test_render_variants_match_golden(tmp_path, monkeypatch):
    if not GOLDEN.exists():
        pytest.fail(f"missing {GOLDEN}; create it with: python -m tests.test_render_golden_variants --update")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if golden.get("versions") != _versions():
        pytest.skip(f"golden hashes were written with {golden.get('versions')}, running {_versions()}")
    assert golden["renderer_version"] == RENDERER_VERSION, (
        "RENDERER_VERSION changed: regenerate (python -m tests.test_render_golden_variants --update)")
    monkeypatch.delenv("IMAGE_FILTERS", raising=False)
    got = render_hashes(tmp_path)
    assert set(got) == set(golden["hashes"]), "the set of golden cases changed"
    changed = sorted(k for k in golden["hashes"] if got[k] != golden["hashes"][k])
    assert not changed, f"{len(changed)} render variants changed, e.g. {changed[:12]}"


def main(argv: list[str]) -> int:
    if "--update" not in argv:
        print(__doc__)
        return 2
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        hashes = render_hashes(Path(tmp))
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps({"renderer_version": RENDERER_VERSION, "versions": _versions(), "hashes": hashes},
                                 indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(hashes)} hashes to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

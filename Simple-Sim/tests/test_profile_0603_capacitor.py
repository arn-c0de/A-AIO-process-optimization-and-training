"""Profile ``chip_0603_capacitor@1`` (CAPC1608 for the DeepVision BOTTOM exports) and the round-2 configs."""

from pathlib import Path

import numpy as np

from simple_sim.board.footprints import family_of, profile_footprint
from simple_sim.config import load_config, merge_profile_render, validate_config
from simple_sim.defects import sample_defect_params
from simple_sim.generators.opencv2d import render_roi, sample_nominal_geometry
from simple_sim.profile_hash import load_profile

ROOT = Path(__file__).resolve().parents[1]
R2_CONFIGS = ("run_0402_r_dv_bottom_missing_r2_res10", "run_0402_r_dv_bottom_missing_r2_res15", "run_0402_r_dv_bottom_missing_r2_res20",
              "run_0402_r_dv_bottom_bridges_r2", "run_0402_c_dv_bottom_bridges_r2", "run_0603_c_dv_bottom_r2", "run_rescaxs_dv_bottom_r2")


def test_profile_is_the_0603_geometry_with_a_capacitor_body():
    cap = load_profile("chip_0603_capacitor@1", ROOT / "configs" / "profiles")
    res = load_profile("chip_0603_resistor@2", ROOT / "configs" / "profiles")
    assert cap["geometry_ranges"] == res["geometry_ranges"] and cap["tolerances"] == res["tolerances"]
    assert family_of(profile_footprint(cap), cap["component"]["type"]) == "capacitor"
    assert cap["component"]["package"] == "0603 (CAPC1608)" and cap["component"]["nominal_dims_mm"]["length"] == 1.6
    assert "TOMBSTONE" in cap["defect_set"] and "SOLDER_BRIDGE" in cap["defect_set"]
    # The dark default colour keeps the look's ceramic colour: the body renders tan, not black.
    cfg = load_config(ROOT / "configs" / "run_0603_c_dv_bottom_r2.yaml")
    render = merge_profile_render(dict(cfg["render"]), dict(cap), cfg["roi"])
    rng = np.random.default_rng(1)
    nominal = sample_nominal_geometry({}, rng, cap["geometry_ranges"])
    defect = sample_defect_params("OK", rng, cap["tolerances"])
    img = render_roi(nominal, defect, {"rotation_deg": 0.0}, (256, 256), {**render, "camera": {"preset": "aoi"}}, cap["tolerances"],
                     np.random.default_rng(2), footprint="chip_2pad")
    centre = img[120:136, 120:136].reshape(-1, 3).mean(axis=0)
    assert centre[2] > centre[0] + 10 and centre.mean() > 60   # tan ceramic: red above blue, not a black body


def test_round2_configs_target_the_bottom_app_crops():
    for name in R2_CONFIGS:
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        validate_config(cfg)
        assert cfg["render"]["look"] == "dv_bottom" and cfg["render"]["post_filters"] == "camera"
        assert cfg["render"]["camera"]["preset"] == "dv_bottom"
        lo, hi = cfg["render"]["camera"]["crop_part_lengths"]
        prof = load_profile(cfg["run"]["component_profile"], ROOT / "configs" / "profiles")
        body_px = float(np.mean(prof["geometry_ranges"]["component_length"]))
        field_mm = 0.5 * (lo + hi) * body_px * float(cfg["roi"]["mm_per_px"])
        if "0603" in name:
            assert abs(field_mm - 4.55) < 0.2 and abs(cfg["roi"]["mm_per_px"] - 1.6 / body_px) < 1e-3   # real 76 px crops
        elif "rescaxs" in name:
            assert abs(field_mm - 4.24) < 0.15                                                         # real 68 px crops
        else:
            assert abs(field_mm - 3.12) < 0.1                                                          # real 50 px crops
    residues = {load_config(ROOT / "configs" / f"{n}.yaml")["render"]["missing_residue"] for n in R2_CONFIGS if "missing_r2" in n}
    assert residues == {1.0, 1.5, 2.0}
    missing = load_config(ROOT / "configs" / "run_0402_r_dv_bottom_missing_r2_res15.yaml")["classes"]
    assert missing["MISSING"] > 3 * (missing["SOLDER_BRIDGE"] + missing["EXCESS_SOLDER"])

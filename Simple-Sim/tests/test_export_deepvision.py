"""tools/export_deepvision.py: reference/probe pairs for the DeepVision judge (schema, pairing, mapping, crops)."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from simple_sim.classes import VALID_CLASSES
from tools.export_deepvision import (FAULT_MAPPING, FAULTS, Exporter, choose_classes, export, fault_classes, fault_of, field_mm,
                                     main, package_code, side_of)

ROOT = Path(__file__).resolve().parents[1]
DEEPVISION_FAULTS = {"fehlt", "versetzt", "verdreht", "grabstein", "loetbruecke", "loetfehler", "fremdkoerper"}  # app spelling


def _small_config(tmp_path: Path, src: str = "run_0402_c_dv_top.yaml", **classes) -> Path:
    cfg = yaml.safe_load((ROOT / "configs" / src).read_text(encoding="utf-8"))
    cfg["roi"] = {**cfg["roi"], "width_px": 96, "height_px": 96}   # 96 px ROI: fast renders, same mm_per_px
    cfg["classes"] = classes or {"OK": 10, "MISSING": 10, "MISALIGNED": 5, "TOMBSTONE": 5}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def test_fault_mapping_covers_every_class_with_the_app_spelling():
    assert set(FAULTS) == VALID_CLASSES
    assert FAULTS["OK"] is None and set(v for v in FAULTS.values() if v) | {"verdreht"} == DEEPVISION_FAULTS
    assert FAULTS["SOLDER_BRIDGE"] == FAULTS["EXCESS_SOLDER"] == "loetbruecke"
    assert FAULTS["TOMBSTONE"] == "grabstein" and FAULTS["CORNER_LIFT"] == "loetfehler" and FAULTS["SOLDER_BALL"] == "fremdkoerper"
    assert FAULTS["MISSING"] == "fehlt" and set(FAULT_MAPPING) == VALID_CLASSES and "verdreht" in FAULT_MAPPING["MISALIGNED"]
    # MISALIGNED: rotation alone over tolerance -> verdreht, otherwise versetzt
    tol = {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0}
    assert fault_of("MISALIGNED", {"shift_x": 0.5, "shift_y": 0.5, "rotation_deg": 12.0}, tol) == "verdreht"
    assert fault_of("MISALIGNED", {"shift_x": 6.0, "shift_y": 0.0, "rotation_deg": 1.0}, tol) == "versetzt"
    assert fault_of("MISALIGNED", {"shift_x": 6.0, "shift_y": 0.0, "rotation_deg": 12.0}, tol) == "versetzt"
    assert fault_of("MISALIGNED", {"shift_x": 0.0, "shift_y": 0.0, "rotation_deg": -4.0}, None) == "verdreht"  # default tol 3 deg
    assert fault_of("TOMBSTONE", {"rotation_deg": 80.0}, tol) == "grabstein" and fault_of("OK", {}, tol) is None


def test_helpers():
    assert package_code({"component": {"package": "0402 (CAPC1005)"}}) == "CAPC1005"
    assert package_code({"component": {"package": "0804 4x (RESCAXS50P100X200-8)"}}) == "RESCAXS50P100X200-8"
    assert package_code({"component": {"package": "SOT-23"}}) == "SOT-23"
    assert package_code({"profile": {"profile_id": "x@1"}}) == "x@1"
    assert side_of({"render": {"look": "dv_top"}}, None) == "top" and side_of({"render": {"look": "quins"}}, "bottom") == "bottom"
    with pytest.raises(ValueError, match="--side"):
        side_of({"render": {"look": "quins"}}, None)
    assert fault_classes({"classes": {"OK": 5, "MISSING": 3, "TOMBSTONE": 0, "SOLDER_BRIDGE": 7}}) == {"MISSING": 3, "SOLDER_BRIDGE": 7}
    with pytest.raises(ValueError, match="positive count"):
        fault_classes({"classes": {"OK": 5}})
    weights = {"MISSING": 100, "TOMBSTONE": 1, "SOLDER_BRIDGE": 100}
    assert choose_classes(weights, None, np.random.default_rng(0)) == list(weights)
    picks = [tuple(choose_classes(weights, 2, np.random.default_rng(s))) for s in range(40)]
    assert all(len(p) == 2 and len(set(p)) == 2 for p in picks)
    assert sum("TOMBSTONE" in p for p in picks) < 20  # weighted: the rare class is drawn rarely
    # Field of view: crop_part_lengths x body length (dv looks) or the ROI at the profile scale.
    assert field_mm({"mm_per_px": 0.01, "look": "dv_top"}, {"component_length": 100.0}, 256) == pytest.approx(3.05)
    assert field_mm({"mm_per_px": 0.01}, {"component_length": 100.0}, 256) == pytest.approx(2.56)


def test_export_writes_pairs_records_and_crops(tmp_path):
    cfg = _small_config(tmp_path)
    out = tmp_path / "export"
    manifest = export(cfg, out, pairs=3, side=None, px_per_mm=16.1, max_size=0, faults_per_pair=2, seed=None, workers=2)
    records = [json.loads(l) for l in (out / "records.ndjson").read_text(encoding="utf-8").splitlines()]
    assert manifest["pairs"] == 3 and manifest["renders"] == len(records) == 3 * (2 + 2)
    assert manifest["side"] == "top" and manifest["look"] == "dv_top" and manifest["profile"] == "chip_0402_capacitor@1"
    assert manifest["form"] == "CAPC1005" and manifest["board"] == "sim-chip_0402_capacitor@1-dv_top-run_0402_c_dv_top"
    assert manifest["renderer_version"] and manifest["created"] and manifest["config"]["roi"]["width_px"] == 96
    assert manifest["classes"]["OK"] == 6 and sum(manifest["classes"].values()) == 12
    assert manifest["faultMapping"] == FAULT_MAPPING and sum(manifest["faults"].values()) == 12 and manifest["faults"]["ok"] == 6
    profile_tol = manifest["config"] and yaml.safe_load((ROOT / "configs" / "profiles" / "chip_0402_capacitor@1.yaml").read_text())["tolerances"]
    assert set(manifest["classes"]) <= {"OK", "MISSING", "MISALIGNED", "TOMBSTONE"}
    by_pair: dict[str, list[dict]] = {}
    for r in records:
        assert set(r) == {"id", "pairId", "role", "board", "side", "label", "fault", "form", "size", "fieldMm", "crop",
                          "cropColor", "sim"}
        assert r["role"] in ("ref", "probe") and r["label"] in ("ok", "fault") and r["side"] == "top"
        assert (r["fault"] is None) == (r["label"] == "ok") and (r["fault"] in DEEPVISION_FAULTS or r["fault"] is None)
        assert r["fault"] == fault_of(r["sim"]["derived_class"], r["sim"]["defect_params"], profile_tol) and r["sim"]["look"] == "dv_top"
        if r["sim"]["derived_class"] == "MISALIGNED":
            dp = r["sim"]["defect_params"]
            rotated = abs(dp["rotation_deg"]) > profile_tol["ok_rotation_deg"] and np.hypot(dp["shift_x"], dp["shift_y"]) <= profile_tol["ok_shift_px"]
            assert r["fault"] == ("verdreht" if rotated else "versetzt")
        assert r["crop"] == f"crops/{r['id']}.png" and r["cropColor"] == f"crops/rgb/{r['id']}.png"
        assert r["id"].startswith(r["pairId"]) and str(r["sim"]["seed"]).zfill(19) == r["pairId"]
        color, grey = cv2.imread(str(out / r["cropColor"]), cv2.IMREAD_UNCHANGED), cv2.imread(str(out / r["crop"]), cv2.IMREAD_UNCHANGED)
        assert color.shape == (r["size"], r["size"], 3) and grey.shape == (r["size"], r["size"])
        # 3.0-3.1 body lengths of a 1.0 mm 0402 at 16.1 px/mm: the app's 48-50 px crops
        assert 2.9 < r["fieldMm"] < 3.3 and 46 <= r["size"] <= 53
        by_pair.setdefault(r["pairId"], []).append(r)
    for _pid, rows in by_pair.items():
        roles = [(r["role"], r["sim"]["class"]) for r in rows]
        assert roles[0] == ("ref", "OK") and roles[1] == ("probe", "OK") and len(roles) == 4
        assert all(role == "probe" and cls != "OK" for role, cls in roles[2:])
        assert len({r["id"] for r in rows}) == 4 and len({r["board"] for r in rows}) == 1


def test_pairs_share_the_board_and_differ_only_at_the_part(tmp_path):
    """Reference and defect probe: same board, pose, light and sensor draws -> identical far from the part;
    the OK probe is another exposure of the same board (differs everywhere a little, but not in content)."""
    cfg = yaml.safe_load(_small_config(tmp_path, OK=1, MISSING=1).read_text())
    from simple_sim.profile_hash import load_profile

    profile = load_profile(cfg["run"]["component_profile"], ROOT / "configs" / "profiles")
    cfg["render"]["camera"] = {"preset": "dv_top", "tilt_deg": [0.0, 0.0], "defocus": [0.0, 0.0]}  # no view-dependent blur
    ex = Exporter(cfg, profile, side="top", px_per_mm=16.1, max_size=0, faults_per_pair=None, seed=3)
    plans = ex.plan_pair(0)
    imgs = {}
    for plan in plans:
        from simple_sim.generators.opencv2d import render_roi

        imgs[(plan["role"], plan["class"])] = render_roi(nominal=plan["nominal"], defect_params=plan["defect"], augment=plan["augment"],
                                                        roi_size=ex.roi, config=dict(ex.render_cfg), tolerances=dict(ex.tolerances),
                                                        rng=np.random.default_rng(plan["seed"]), footprint=ex.footprint,
                                                        camera_rng=np.random.default_rng([plan["seed"], plan["camera_salt"]])).astype(int)
    ref, missing, ok_probe = imgs[("ref", "OK")], imgs[("probe", "MISSING")], imgs[("probe", "OK")]
    border = np.ones(ref.shape[:2], bool)
    border[24:72, 24:72] = False                   # the part sits in the centre of the 96 px ROI
    assert np.abs(ref - missing)[border].mean() < 1.0        # same photo of the same board
    assert np.abs(ref - missing)[~border].mean() > 5.0       # the part is gone in the centre
    assert np.abs(ref - ok_probe)[border].mean() > 0.5       # other exposure / noise ...
    assert np.abs(ref.mean(axis=2) - ok_probe.mean(axis=2)).mean() < 25.0  # ... but the same board


def test_cli_and_blender_configs_are_rejected(tmp_path, capsys):
    cfg = _small_config(tmp_path, OK=1, MISSING=1)
    out = tmp_path / "cli"
    assert main(["--config", str(cfg), "--out", str(out), "--pairs", "1", "--faults-per-pair", "all", "--max-size", "40", "--workers", "1"]) == 0
    text = capsys.readouterr().out
    assert "3 renders in 1 pairs" in text and "40 px x3" in text
    cfg3d = yaml.safe_load(cfg.read_text())
    cfg3d["render"] = {"backend": "blender_3d", "blender": {"executable": "blender", "samples": 8}}
    cfg3d["run"]["component_profile"] = "chip_0603_resistor_3d@2"
    p = tmp_path / "cfg3d.yaml"
    p.write_text(yaml.safe_dump(cfg3d, sort_keys=False))
    with pytest.raises(ValueError, match="2D backend"):
        export(p, tmp_path / "x", pairs=1, side="top", px_per_mm=16.1, max_size=0, faults_per_pair=None, seed=None, workers=1)

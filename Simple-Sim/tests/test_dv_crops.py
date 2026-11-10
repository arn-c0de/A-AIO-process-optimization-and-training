"""tools/dv_crops.py: DeepVision app reviews (<side>.ndjson) -> class folders of eval_real / mix_real."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from simple_sim.classes import VALID_CLASSES
from tools.dv_crops import FAULT_FOLDERS, export, folder_of, latest_reviews, main, nearest_package, read_pnp, select


def _labels(root: Path, side: str = "top") -> Path:
    """A labels folder: reviews of 4 parts on one board, one part reviewed twice, one retracted, one without crop."""
    (root / side / "b1" / "rgb").mkdir(parents=True)
    rows = []

    def review(pid, label, fault=None, at="2026-10-12T15:00:00Z", center=(10.0, 10.0), crop=True, **extra):
        if crop:
            cv2.imwrite(str(root / side / "b1" / "rgb" / f"{pid}.png"), np.full((49, 49, 3), len(rows) * 20, np.uint8))
        rows.append({"board": "b1", "id": pid, "label": label, "fault": fault, "at": at, "size": 49, "fieldMm": 3.037,
                     "centerMm": list(center), "crop": f"{side}/b1/{pid}.png",
                     "cropColor": f"{side}/b1/rgb/{pid}.png" if crop else None, **extra})

    review("p1", "fault", "fehlt", at="2026-10-12T15:00:00Z")          # superseded by the later ok review
    review("p1", "ok", at="2026-10-13T15:00:00Z")
    review("p2", "fault", "loetbruecke", center=(20.0, 10.0))
    review("p3", "fault", "verdreht", center=(30.0, 10.0))
    review("p4", "fault", "beschaedigt", center=(40.0, 10.0))
    review("p5", "ok", center=(50.0, 10.0), retracted=True)
    review("p6", "ok", center=(60.0, 10.0), crop=False)
    review("p7", "dnp", center=(70.0, 10.0))
    review("p8", "fault", None, center=(80.0, 10.0))
    (root / f"{side}.ndjson").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return root


def _pnp(path: Path) -> Path:
    lines = [["C1", "CAP", "100n", "", "CAPC1005", "10.000", "10.000", "0", "NO"],
             ["R1", "RES", "10k", "", "RESC1005", "20.000", "10.000", "90", "NO"],
             ["R2", "RES", "10k", "", "RESC1005", "30.100", "10.100", "90", "NO"],
             ["C9", "CAP", "1u", "", "CAPC1608", "40.000", "10.000", "0", "YES"]]
    path.write_text("".join(",".join(l) + "\n" for l in lines), encoding="utf-8")
    return path


def test_folders_name_simple_sim_classes_or_defect_groups():
    for folder in FAULT_FOLDERS.values():
        assert folder.split("@", 1)[0] in VALID_CLASSES or folder.startswith("DEFECT@"), folder
    assert folder_of({"label": "ok"}) == "OK"
    assert folder_of({"label": "fault", "fault": "fehlt"}) == "MISSING"
    assert folder_of({"label": "fault", "fault": "loetbruecke"}) == "SOLDER_BRIDGE"
    assert folder_of({"label": "fault", "fault": "verdreht"}) == "MISALIGNED@verdreht"
    assert folder_of({"label": "fault", "fault": "grabstein"}) == "TOMBSTONE"
    assert folder_of({"label": "fault", "fault": "kratzer"}) == "DEFECT@kratzer"
    assert folder_of({"label": "fault"}) == "DEFECT@unspecified"
    assert folder_of({"label": "dnp"}) is None and folder_of({"label": "unclear"}) is None


def test_latest_review_per_part_wins_and_retracted_are_dropped(tmp_path):
    rows = [json.loads(l) for l in (_labels(tmp_path) / "top.ndjson").read_text().splitlines()]
    latest = {r["id"]: r for r in latest_reviews(rows)}
    assert latest["p1"]["label"] == "ok" and "p5" not in latest
    assert len(latest) == 7


def test_select_assigns_packages_and_skips_missing_crops(tmp_path):
    root = _labels(tmp_path / "labels")
    rows = [json.loads(l) for l in (root / "top.ndjson").read_text().splitlines()]
    placements = read_pnp(_pnp(tmp_path / "pnp.csv"))
    assert [p[2] for p in placements["top"]] == ["CAPC1005", "RESC1005", "RESC1005"] and placements["bottom"][0][2] == "CAPC1608"
    assert nearest_package([30.0, 10.0], placements["top"], 0.8) == ("RESC1005", pytest.approx(0.1414, abs=1e-3))
    assert nearest_package([80.0, 10.0], placements["top"], 0.8)[0] == "unknown"
    selected, skipped = select(rows, root, packages=None, placements=placements["top"], match_mm=0.8)
    assert {k: len(v) for k, v in selected.items()} == {"OK": 1, "SOLDER_BRIDGE": 1, "MISALIGNED@verdreht": 1,
                                                        "DEFECT@beschaedigt": 1, "DEFECT@unspecified": 1}
    assert skipped == {"no colour crop": 1, "label:dnp": 1}
    assert selected["OK"][0]["package"] == "CAPC1005" and selected["DEFECT@unspecified"][0]["package"] == "unknown"
    only_r, _ = select(rows, root, packages={"RESC1005"}, placements=placements["top"], match_mm=0.8)
    assert set(only_r) == {"SOLDER_BRIDGE", "MISALIGNED@verdreht"}


def test_export_copies_unchanged_and_writes_summary(tmp_path):
    root = _labels(tmp_path / "labels")
    out = tmp_path / "real"
    summary = export(root, "top", out, pnp=_pnp(tmp_path / "pnp.csv"), packages=None, limit_ok=500, limit=0, seed=0,
                     match_mm=0.8, dry_run=False)
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["DEFECT@beschaedigt", "DEFECT@unspecified",
                                                                   "MISALIGNED@verdreht", "OK", "SOLDER_BRIDGE"]
    files = list((out / "OK").iterdir())
    assert [f.name for f in files] == ["b1_p1.png"] and cv2.imread(str(files[0])).shape == (49, 49, 3)
    written = json.loads((out / "summary.json").read_text())
    assert written["folders"] == summary["folders"] == {"DEFECT@beschaedigt": 1, "DEFECT@unspecified": 1, "OK": 1,
                                                         "SOLDER_BRIDGE": 1, "MISALIGNED@verdreht": 1}
    assert written["px_per_mm"][1] == pytest.approx(49 / 3.037, abs=0.01) and written["packages"]["RESC1005"] == 2
    dry = export(root, "top", tmp_path / "dry", pnp=None, packages=None, limit_ok=500, limit=0, seed=0, match_mm=0.8,
                 dry_run=True)
    assert dry["dry_run"] and not (tmp_path / "dry").exists() and dry["packages"] == {"unknown": 5}


def test_cli_prints_counts_only(tmp_path, capsys):
    root = _labels(tmp_path / "labels")
    assert main(["--labels", str(root), "--side", "top", "--out", str(tmp_path / "o"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "total" in out and "b1_p1" not in out
    with pytest.raises(SystemExit, match="--packages needs --pnp"):
        main(["--labels", str(root), "--side", "top", "--out", str(tmp_path / "o"), "--packages", "CAPC1005"])
    with pytest.raises(SystemExit, match="no bottom.ndjson"):
        main(["--labels", str(root), "--side", "bottom", "--out", str(tmp_path / "o")])
    (root / "bottom.ndjson").write_text('{"board": "b", "id": "x", "label": "dnp"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="no crops matched"):
        main(["--labels", str(root), "--side", "bottom", "--out", str(tmp_path / "o")])

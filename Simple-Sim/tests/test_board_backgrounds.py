"""tools/board_backgrounds.py: empty board areas of autopilot stills -> render.texture_dir folders."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.board_backgrounds import (cut_window, find_runs, free_windows, inside_still, main, mask_check, part_boxes,
                                     window_transform)

MM_PER_PX = 0.02   # 50 px/mm textures
STILL_PX_PER_MM = 20.0


def _run_dir(root: Path, side: str = "top", photo: str = "IMG1.jpg", parts=None, shapes=False) -> Path:
    """A run folder: a synthetic 'still' of a green board (30 x 30 mm at 20 px/mm, with a bright part at every
    placement and a dark hole) plus geometry.json / components.json."""
    root.mkdir(parents=True, exist_ok=True)
    still = np.zeros((600, 600, 3), np.uint8)
    still[:] = (70, 95, 50)
    noise = np.random.default_rng(0).normal(0, 2, still.shape)
    still = np.clip(still.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    h = np.array([[STILL_PX_PER_MM, 0.0, 0.0], [0.0, STILL_PX_PER_MM, 0.0], [0.0, 0.0, 1.0]])
    parts = parts if parts is not None else [(5.0 + 5.0 * i, 5.0) for i in range(5)] + [(5.0 + 5.0 * i, 25.0) for i in range(5)]
    items = []
    for i, (x, y) in enumerate(parts):
        px, py = int(x * STILL_PX_PER_MM), int(y * STILL_PX_PER_MM)
        cv2.rectangle(still, (px - 10, py - 5), (px + 10, py + 5), (230, 230, 230), -1)   # bright part
        item = {"id": f"p{i}", "form": "form-02", "centerMm": [x, y], "fieldMm": 3.04, "size": 49}
        if shapes:
            item["shapes"] = [{"kind": "body", "points": [[x - 0.5, y - 0.25], [x + 0.5, y - 0.25], [x + 0.5, y + 0.25], [x - 0.5, y + 0.25]]}]
        items.append(item)
    cv2.circle(still, (int(15 * STILL_PX_PER_MM), int(12 * STILL_PX_PER_MM)), 8, (5, 5, 5), -1)  # a dark hole
    cv2.imwrite(str(root / "still.jpg"), still, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (root / "geometry.json").write_text(json.dumps({"file": photo, "side": side, "sourceSize": [600, 600],
                                                    "boardToSource": h.reshape(-1).tolist(), "quality": {"mode": "flow"}}))
    (root / "components.json").write_text(json.dumps({"side": side, "items": items}))
    return root


def test_part_boxes_from_field_or_body_polygon():
    boxes = part_boxes([{"centerMm": [10.0, 10.0], "fieldMm": 3.04}], keepout_mm=0.0)
    assert boxes.shape == (1, 4) and boxes[0].tolist() == pytest.approx([9.2, 9.2, 10.8, 10.8])  # 0.5 * 1.6 mm half side
    boxes = part_boxes([{"centerMm": [10.0, 10.0], "fieldMm": 0.0}], keepout_mm=0.4)
    assert boxes[0].tolist() == pytest.approx([9.0, 9.0, 11.0, 11.0])  # minimum half side 0.6 + keep-out
    boxes = part_boxes([{"centerMm": [10.0, 10.0], "shapes": [{"kind": "body", "points": [[9, 9.5], [11, 9.5], [11, 10.5], [9, 10.5]]}]}], 0.0)
    assert boxes[0].tolist() == pytest.approx([8.3, 8.8, 11.7, 11.2])  # polygon grown by 35 % of its longer side


def test_free_windows_avoid_parts_and_the_margin():
    parts = [{"centerMm": [x, y], "fieldMm": 3.04} for x in (5.0, 10.0, 15.0, 20.0, 25.0) for y in (5.0, 25.0)]
    wins = free_windows(parts, window_mm=4.0, margin_mm=2.0, keepout_mm=0.4, step_mm=1.0)
    assert wins, "the band between the two part rows must be free"
    boxes = part_boxes(parts, 0.4)
    for x, y in wins:
        assert 7.0 - 1e-9 <= x and x + 4.0 <= 23.0 + 1e-9 and 7.0 - 1e-9 <= y and y + 4.0 <= 23.0 + 1e-9
        hit = (boxes[:, 0] < x + 4.0) & (boxes[:, 2] > x) & (boxes[:, 1] < y + 4.0) & (boxes[:, 3] > y)
        assert not hit.any()
    assert free_windows(parts, window_mm=30.0, margin_mm=2.0, keepout_mm=0.4) == []


def test_window_transform_maps_board_mm_to_texture_px(tmp_path):
    run = find_runs(_run_dir(tmp_path / "r1"))[0]
    m = window_transform(run.homography, 10.0, 12.0, MM_PER_PX)
    still_pt = np.array([(10.0 + 1.0) * STILL_PX_PER_MM, (12.0 + 2.0) * STILL_PX_PER_MM, 1.0])  # board (11, 14) mm
    tex = m @ still_pt
    assert (tex[:2] / tex[2]).tolist() == pytest.approx([1.0 / MM_PER_PX, 2.0 / MM_PER_PX])
    assert inside_still(run.homography, 10.0, 12.0, 4.0, run.source_size)
    assert not inside_still(run.homography, 28.0, 12.0, 4.0, run.source_size)
    still = cv2.imread(str(run.folder / "still.jpg"))
    patch = cut_window(still, run.homography, 10.0, 12.0, 4.0, MM_PER_PX)
    assert patch.shape == (200, 200, 3)  # 4 mm at 50 px/mm


def test_mask_check_rejects_parts_holes_and_foreign_colours():
    green = np.full((64, 64, 3), (70, 95, 50), np.uint8)
    assert mask_check(green, 0.9) is None
    with_part = green.copy()
    with_part[20:30, 10:40] = (235, 235, 235)
    assert mask_check(with_part, 0.9).startswith("bright")
    with_hole = green.copy()
    with_hole[20:30, 20:30] = (3, 3, 3)
    assert mask_check(with_hole, 0.9).startswith("dark")
    with_print = green.copy()
    with_print[30:34, 5:60] = (170, 172, 168)   # grey silkscreen text, below the bright threshold
    assert mask_check(with_print, 0.9).startswith("print")
    assert mask_check(np.full((64, 64, 3), (40, 40, 150), np.uint8), 0.9) == "not mask-coloured"


def test_find_runs_keeps_one_run_per_photo_and_side(tmp_path):
    _run_dir(tmp_path / "a" / "run1", photo="P.jpg")
    _run_dir(tmp_path / "a" / "run2", photo="P.jpg", parts=[(5.0 + 2.0 * i, 5.0) for i in range(12)])  # more parts
    _run_dir(tmp_path / "b" / "run3", photo="Q.jpg", side="bottom")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "still.jpg").write_bytes(b"")  # no geometry: skipped
    runs = find_runs(tmp_path)
    assert [(r.side, r.photo, len(r.parts)) for r in runs] == [("bottom", "Q.jpg", 10), ("top", "P.jpg", 12)]


def test_cli_cuts_textures_and_writes_summary(tmp_path, capsys):
    _run_dir(tmp_path / "runs" / "r1")
    _run_dir(tmp_path / "runs" / "r2", side="bottom", photo="B.jpg")
    out = tmp_path / "tex"
    assert main(["--runs", str(tmp_path / "runs"), "--out", str(out), "--mm-per-px", str(MM_PER_PX), "--window-mm", "4",
                 "--margin-mm", "2", "--max-per-photo", "5", "--dry-run"]) == 0
    assert not out.exists() and "accepted" in capsys.readouterr().out
    assert main(["--runs", str(tmp_path / "runs"), "--out", str(out), "--mm-per-px", str(MM_PER_PX), "--window-mm", "4",
                 "--margin-mm", "2", "--max-per-photo", "5"]) == 0
    for side in ("top", "bottom"):
        files = sorted((out / side).glob("*.png"))
        assert 1 <= len(files) <= 5
        img = cv2.imread(str(files[0]))
        assert img.shape == (200, 200, 3) and mask_check(img, 0.9) is None
        summary = json.loads((out / side / "summary.json").read_text())
        assert summary["textures"] == len(files) and summary["still_px_per_mm"][1] == pytest.approx(STILL_PX_PER_MM, abs=0.1)
        assert "rejected: bright pixels (metal, print, label)" in summary["counts"] or summary["counts"]["accepted"] >= 1
    with pytest.raises(SystemExit, match="no usable runs"):
        main(["--runs", str(tmp_path / "empty"), "--out", str(out)])

"""board3d_simreal helpers: sheet split, real OK rows, real folder layout, scoring."""

import cv2
import numpy as np

from tools.board3d_simreal import real_folders, score, split_by_sheet, with_real_ok


def test_split_keeps_sheets_together():
    rows = [{"sheet": f"s{i // 5}"} for i in range(60)]
    va = split_by_sheet(rows)
    assert 0 < va.sum() < len(rows)
    for s in {r["sheet"] for r in rows}:
        assert len({bool(v) for r, v in zip(rows, va) if r["sheet"] == s}) == 1


def test_with_real_ok_adds_each_real_crop_once():
    rows = [{"image": "a.png", "real": "r1.png", "label": "MISSING", "sheet": "s"},
            {"image": "b.png", "real": "r2.png", "label": "OK", "sheet": "s"},
            {"image": "c.png", "label": "OK", "sheet": "s"},
            {"image": "r2.png", "label": "OK", "sheet": "s"}]           # already a row (--real-ok dataset)
    got = with_real_ok(rows)
    assert [r["image"] for r in got[4:]] == ["r1.png"] and got[4]["label"] == "OK"


def test_real_folders_and_score(tmp_path):
    for d in ("OK", "MISSING@empty_pad", "MISSING", "SHIFT"):
        (tmp_path / d).mkdir()
    for d, n in (("OK", 2), ("MISSING@empty_pad", 1), ("MISSING", 1)):
        for i in range(n):
            cv2.imwrite(str(tmp_path / d / f"{i}.png"), np.zeros((4, 4, 3), np.uint8))
    got = real_folders(tmp_path)
    assert [(c, len(ims)) for c, ims in got] == [("MISSING", 1), ("MISSING", 1), ("OK", 2)]    # empty SHIFT skipped
    res = score([("OK", "OK"), ("OK", "SHIFT"), ("MISSING", "MISSING"), ("MISSING", "OK"), ("SHIFT", "MISSING")])
    assert res["realOkSpecificity"] == 0.5 and res["realDefectRecall"] == round(2 / 3, 3)
    assert res["perClass"]["MISSING"] == {"n": 2, "flagged": 0.5, "exact": 0.5}
    assert res["balancedAcc"] == round((0.5 + 2 / 3) / 2, 3)

"""tools/real_crops.py: DeepVision QUINS manifest -> class folders of eval_real / mix_real."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from simple_sim.classes import VALID_CLASSES
from tools.real_crops import CLASS_FOLDERS, export, main, select


def _manifest(root: Path) -> Path:
    rows = []
    specs = [("form-02", "gut", "train", 4), ("form-02", "gut", "test", 3), ("form-02", "fehler:fehlt", "val", 2),
             ("form-01", "leer", "test", 2), ("form-02", "ungeprüft", "test", 2), ("form-02", "fehlalarm", "test", 1),
             ("form-03", "gut", "test", 5)]
    n = 0
    for form, cls, split, count in specs:
        for _ in range(count):
            n += 1
            crop = f"crops/x/s{n % 2}/{n:04d}.png"
            (root / crop).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / crop), np.full((72, 72, 3), n, np.uint8))
            rows.append({"crop": crop, "id": f"id{n:04d}", "form": form, "scan": f"s{n % 2}", "split": split, "class": cls})
    path = root / "manifest.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_folders_name_simple_sim_classes_or_defect_groups():
    for folder in CLASS_FOLDERS.values():
        cls = folder.split("@", 1)[0]
        assert cls in VALID_CLASSES or cls == "DEFECT", folder


def test_select_filters_form_split_and_class(tmp_path):
    rows = [json.loads(line) for line in _manifest(tmp_path).read_text(encoding="utf-8").splitlines()]
    by_folder = select(rows, {"form-02"}, {"test"}, None)
    assert {k: len(v) for k, v in by_folder.items()} == {"OK": 3, "OK@fehlalarm": 1}   # ungeprüft skipped
    by_folder = select(rows, {"form-02", "form-01"}, {"train", "val", "test"}, {"fehler:fehlt", "leer"})
    assert {k: len(v) for k, v in by_folder.items()} == {"MISSING": 2, "MISSING@empty_pad": 2}


def test_export_copies_crops_with_scan_prefix_and_limit(tmp_path):
    manifest = _manifest(tmp_path / "dv")
    out = tmp_path / "real"
    counts = export(manifest, out, {"form-02"}, {"train", "test"}, {"gut"}, limit=5, seed=0)
    assert counts == {"OK": 5}
    files = sorted(p.name for p in (out / "OK").iterdir())
    assert len(files) == 5 and all(name.startswith("s") and "_id" in name and name.endswith(".png") for name in files)
    img = cv2.imread(str(out / "OK" / files[0]))
    assert img.shape == (72, 72, 3)   # copied unchanged
    # Appending another class to the same folder keeps what is there.
    export(manifest, out, {"form-01"}, {"test"}, {"leer"}, limit=0, seed=0)
    assert sorted(p.name for p in out.iterdir()) == ["MISSING@empty_pad", "OK"]
    assert len(list((out / "MISSING@empty_pad").iterdir())) == 2


def test_cli_prints_counts_only_and_rejects_unknown_classes(tmp_path, capsys):
    manifest = _manifest(tmp_path / "dv")
    assert main(["--manifest", str(manifest), "--out", str(tmp_path / "o"), "--forms", "form-02", "--splits", "test"]) == 0
    out = capsys.readouterr().out
    assert "OK" in out and "total" in out and "id0" not in out
    with pytest.raises(SystemExit, match="unknown classes"):
        main(["--manifest", str(manifest), "--out", str(tmp_path / "o"), "--forms", "form-02", "--classes", "kaputt"])
    with pytest.raises(SystemExit, match="no crops matched"):
        main(["--manifest", str(manifest), "--out", str(tmp_path / "o2"), "--forms", "form-99"])

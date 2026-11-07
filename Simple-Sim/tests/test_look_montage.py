"""tools/look_montage.py: real-vs-synthetic montage and gap statistics per class."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tools.look_montage import build, format_stats, image_stats, main

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    from simple_sim.generation.dataset import generate_dataset

    tmp = tmp_path_factory.mktemp("montage")
    cfg = yaml.safe_load((ROOT / "configs" / "run_0402_c.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {"OK": 3, "MISSING": 3}
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    generate_dataset(tmp / "cfg.yaml", tmp / "ds", project_root=ROOT)
    return tmp / "ds"


def _crops(root: Path, folder: str, n: int, value: int, size: int = 72) -> None:
    (root / folder).mkdir(parents=True)
    for i in range(n):
        img = np.full((size, size, 3), value, np.uint8)
        img[20:52, 10:30] = 230  # a "terminal": bright pixels for the metal statistic
        cv2.imwrite(str(root / folder / f"{i}.png"), img)


def test_image_stats_measure_contrast_and_highlights():
    flat = [np.full((32, 32, 3), 100, np.uint8)]
    st = image_stats(flat)
    assert st["L_std"] == 0.0 and st[">200"] == 0.0 and st["metal"] is None
    bright = flat[0].copy()
    bright[:, 16:] = 250
    st = image_stats([bright])
    assert st[">200"] == 0.5 and st["L_std"] > 20 and st["metal"] == [250, 250, 250]
    assert "metal" in format_stats({"x": st}) and "50.0%" in format_stats({"x": st})


def test_montage_pairs_real_folders_with_synthetic_classes(dataset, tmp_path):
    real = tmp_path / "real"
    _crops(real, "OK", 5, 60)
    _crops(real, "MISSING@empty_pad", 2, 70)
    _crops(real, "DEFECT@loetfehler", 1, 80)
    montage, stats = build(real, dataset, size=64, per_row=4, stat_limit=10, with_stats=True)
    # rows: real DEFECT@loetfehler, real MISSING@empty_pad (sub-group: no synthetic row), real OK, synthetic OK
    assert list(stats) == ["real DEFECT@loetfehler", "real MISSING@empty_pad", "real OK", "synthetic OK"]
    assert montage.shape == (4 * 64, 4 * 64, 3)
    assert stats["real OK"][">200"] > 0 and stats["synthetic OK"]["L_std"] > 0
    # Fewer crops than tiles: the row is padded with black.
    assert montage[64 * 1 + 40, 3 * 64 + 32].tolist() == [0, 0, 0]


def test_cli_writes_the_png_and_prints_stats(dataset, tmp_path, capsys):
    real = tmp_path / "real"
    _crops(real, "MISSING", 3, 50)
    out = tmp_path / "m" / "montage.png"
    assert main(["--real", str(real), "--data", str(dataset), "--out", str(out), "--stats", "--per-row", "3", "--size", "48"]) == 0
    text = capsys.readouterr().out
    assert out.is_file() and "montage" in text and "synthetic MISSING" in text and "L_std" in text
    with pytest.raises(SystemExit, match="labels.jsonl"):
        main(["--real", str(real), "--data", str(tmp_path), "--out", str(out)])

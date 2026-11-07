"""tools/mix_real.py: real crops appended to a generated dataset."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from simple_sim.data_loader import ROIDataset
from tools.mix_real import mix_real

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    from simple_sim.generation.dataset import generate_dataset

    tmp = tmp_path_factory.mktemp("mix")
    cfg = yaml.safe_load((ROOT / "configs" / "run_0402_c.yaml").read_text(encoding="utf-8"))
    cfg["classes"] = {"OK": 3, "MISSING": 3}
    (tmp / "cfg.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    generate_dataset(tmp / "cfg.yaml", tmp / "ds", project_root=ROOT)
    return tmp / "ds"


def _crops(root: Path, folder: str, n: int, value: int) -> None:
    (root / folder).mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(root / folder / f"{i}.png"), np.full((48, 48, 3), value, np.uint8))


def test_mix_real_appends_rows_images_and_manifest(dataset, tmp_path):
    _crops(tmp_path, "OK", 3, 120)
    _crops(tmp_path, "MISSING@empty_pad", 4, 60)
    before = ROIDataset(dataset, "train", ["MISSING", "OK"])
    added = mix_real(dataset, tmp_path, limit=3)
    assert added == {"MISSING": 3, "OK": 3}
    after = ROIDataset(dataset, "train", ["MISSING", "OK"])
    assert len(after) == len(before) + 6
    x, y = after[len(after) - 1]
    assert tuple(x.shape[1:]) == (256, 256)
    manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["real_crops"][-1]["classes"] == {"MISSING": 3, "OK": 3}
    again = mix_real(dataset, tmp_path, limit=1, split="val")  # ids keep counting, no collisions
    assert again == {"MISSING": 1, "OK": 1}
    mix_real(dataset, tmp_path, limit=1, split="val", domain="other")  # another domain: same image series
    rows = [json.loads(line) for line in (dataset / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
    ids, paths = [r["id"] for r in rows], [r["image_path"] for r in rows]
    assert len(ids) == len(set(ids)) and len(paths) == len(set(paths))
    assert all((dataset / p).is_file() for p in paths)


def test_mix_real_rejects_untrained_or_unknown_classes(dataset, tmp_path):
    _crops(tmp_path, "TOMBSTONE", 1, 90)
    with pytest.raises(SystemExit, match="not trained"):
        mix_real(dataset, tmp_path)
    (tmp_path / "TOMBSTONE" / "0.png").unlink()
    (tmp_path / "TOMBSTONE").rmdir()
    _crops(tmp_path, "GOOD", 1, 90)
    with pytest.raises(SystemExit, match="unknown class"):
        mix_real(dataset, tmp_path)

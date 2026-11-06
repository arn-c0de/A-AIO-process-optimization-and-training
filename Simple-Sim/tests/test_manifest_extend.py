import json
from pathlib import Path

import pytest

from simple_sim.config_schema import parse_config_typed
from simple_sim.manifest import write_multi_profile_manifest
from simple_sim.schema import LabelRow


def _rows(n: int, profile_id: str) -> tuple[list[LabelRow], list[LabelRow]]:
    # The manifest writer only needs ids/class_name/profile_id, so LabelRow doubles as meta row.
    labels = [
        LabelRow(schema_version=2, id=f"run/domain_A/train/{i:06d}", class_name="OK", profile_id=profile_id)
        for i in range(n)
    ]
    return labels, labels


def test_multi_profile_manifest_extend_preserves_history(tmp_path: Path) -> None:
    profiles = [
        {"profile_id": "a", "profile_hash": "sha256:a", "profile_path": "a.yaml"},
        {"profile_id": "b", "profile_hash": "sha256:b", "profile_path": "b.yaml"},
    ]
    meta, labels = _rows(4, "a")
    write_multi_profile_manifest(tmp_path, "run", profiles, meta, labels, {"train": [r.id for r in meta]})
    first = json.loads((tmp_path / "dataset_manifest.json").read_text())

    meta2, labels2 = _rows(6, "a")
    write_multi_profile_manifest(
        tmp_path, "run", profiles, meta2, labels2, {"train": [r.id for r in meta2]},
        script_name="scripts/generate.py", existing=first,
    )
    second = json.loads((tmp_path / "dataset_manifest.json").read_text())

    assert second["created_at"] == first["created_at"]
    assert len(second["extend_history"]) == 2
    assert second["extend_history"][0] == first["extend_history"][0]
    assert second["extend_history"][1]["samples_added"] == 2
    assert second["dataset_stats"]["total_samples"] == 6
    assert second["generator"]["script"] == "scripts/generate.py"


def test_parse_config_typed_names_bad_section() -> None:
    cfg = {
        "run": {"run_id": "r1", "seed": 42, "schema_version": 2, "mode": ""},
        "roi": {"width_px": 128, "height_px": 128, "mm_per_px": 0.05},
        "train": {"model": "resnet18", "epochs": "many", "batch_size": 4, "lr": 0.001,
                  "optimizer": "adam", "weight_decay": 0.0},
        "eval": {"batch_size": 4},
    }
    with pytest.raises(ValueError, match="section 'train'"):
        parse_config_typed(cfg)

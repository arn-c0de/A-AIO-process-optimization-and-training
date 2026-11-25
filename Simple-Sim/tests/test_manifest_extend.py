import json
from pathlib import Path

import warnings

import pytest

from simple_sim.manifest import RendererMismatchError, write_multi_profile_manifest
from simple_sim.generation.dataset import generate_dataset
from simple_sim.generation.render_options import look_differences
from simple_sim.rows import LabelRow, read_jsonl, validate_jsonl_pair
import yaml
from tests.helpers import empty_dataset as _empty_dataset, ROOT, tiny_run_config as _config

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)  # plan A4: deprecated shim, still tested
    from simple_sim.config_schema import parse_config_typed


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
    write_multi_profile_manifest(tmp_path, "run", profiles=profiles, meta_rows=meta, label_rows=labels,
                                 splits={"train": [r.id for r in meta]})
    first = json.loads((tmp_path / "dataset_manifest.json").read_text())

    meta2, labels2 = _rows(6, "a")
    write_multi_profile_manifest(
        tmp_path, "run", profiles=profiles, meta_rows=meta2, label_rows=labels2,
        splits={"train": [r.id for r in meta2]}, script_name="scripts/generate.py", existing=first,
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


def test_extend_refuses_another_roi_size_or_scale(tmp_path):
    cfg = _config()
    assert look_differences(cfg, cfg) == []
    other = {**cfg, "roi": {**cfg["roi"], "width_px": 32, "height_px": 32}}
    assert any(d.startswith("roi.width_px") for d in look_differences(cfg, other))
    scaled = {**cfg, "roi": {**cfg["roi"], "mm_per_px": 0.08}}
    assert any(d.startswith("roi.mm_per_px") for d in look_differences(cfg, scaled))

    data = tmp_path / "ds"
    generate_dataset(cfg, data, project_root=ROOT)
    with pytest.raises(RendererMismatchError, match="roi.width_px"):
        generate_dataset(other, data, extend=True, project_root=ROOT)
    assert len(read_jsonl(data / "labels.jsonl", LabelRow)) == 8


def test_extend_with_a_new_class_adds_it_to_the_dataset_config(tmp_path):
    data = tmp_path / "ds"
    generate_dataset(_config(OK=4, MISSING=4), data, project_root=ROOT)
    res = generate_dataset(_config(TOMBSTONE=6), data, extend=True, project_root=ROOT)
    assert res["extend"] is True
    stored = yaml.safe_load((data / "config.yaml").read_text(encoding="utf-8"))
    # Training reads its class list from config.yaml: without the entry the new samples were dropped.
    assert stored["classes"] == {"OK": 4, "MISSING": 4, "TOMBSTONE": 6}
    validate_jsonl_pair(data / "meta.jsonl", data / "labels.jsonl")
    assert len(read_jsonl(data / "labels.jsonl", LabelRow)) == 14
    assert not list(data.glob("*.tmp"))


def test_extend_into_an_empty_directory_is_a_fresh_dataset(tmp_path):
    data = tmp_path / "ds"
    data.mkdir()
    res = generate_dataset(_config(), data, extend=True, project_root=ROOT)
    assert res["extend"] is False and (data / "dataset_manifest.json").exists()


def test_extending_a_multi_profile_manifest_keeps_foreign_keys(tmp_path):
    existing = {
        "manifest_version": 2, "created_at": "2026-01-01T00:00:00", "run_id": "r",
        "generator": {"version": "1", "git_commit": None, "script": "tools/create_multi_dataset.py", "renderer_version": "2"},
        "dataset_stats": {"total_samples": 1}, "extend_history": [{"samples_added": 1}],
        "real_crops": [{"added": 3}],
    }
    rows = [LabelRow(schema_version=2, id=f"s{i}", class_name="OK", profile_id="p@1") for i in range(3)]
    write_multi_profile_manifest(
        output_dir=tmp_path, run_id="r", profiles=[{"profile_id": "p@1", "profile_hash": "sha256:x", "profile_path": "p"}],
        meta_rows=rows, label_rows=rows, splits={"train": ["s0", "s1"], "val": ["s2"], "test": []},
        script_name="scripts/generate.py", existing=existing, renderer_version="3", git_commit="abc",
    )
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["real_crops"] == [{"added": 3}]
    assert manifest["generator"]["script"] == "scripts/generate.py"  # the script that wrote it last
    assert manifest["generator"]["renderer_version"] == "3"
    assert manifest["created_at"] == "2026-01-01T00:00:00" and len(manifest["extend_history"]) == 2
    assert manifest["dataset_stats"]["total_samples"] == 3


def test_backfill_manifest_survives_an_empty_run_section(tmp_path, capsys):
    from tools.backfill_manifest import STATUS_FAILED, backfill_dataset

    ds = _empty_dataset(tmp_path / "ds")
    (ds / "config.yaml").write_text("run:\n", encoding="utf-8")
    # No AttributeError on `run: null`; the dataset is reported as failed (unknown profile) instead.
    assert backfill_dataset(ds, "no_such_profile@1", tmp_path) == STATUS_FAILED
    capsys.readouterr()


def test_generate_extend_on_a_new_directory_writes_a_fresh_manifest(tmp_path):
    from simple_sim.config import load_config
    from simple_sim.dataset_files import DATASET_MANIFEST
    from simple_sim.generation.dataset import generate_dataset

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "run_0001.yaml")
    cfg["classes"] = dict.fromkeys(list(cfg["classes"])[:2], 4)
    out = tmp_path / "fresh"
    result = generate_dataset(cfg, out, extend=True)
    assert (out / DATASET_MANIFEST).is_file() and result["extend"] is False

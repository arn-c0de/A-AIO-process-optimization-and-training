from __future__ import annotations

import json
from pathlib import Path

import pytest

from gui.utils.dataset_ops import delete_samples, list_dataset_samples, move_samples
from simple_sim.schema import LabelRow, MetaRow, read_jsonl, write_jsonl


def _meta(sample_id: str, image_path: str, split: str = "train") -> MetaRow:
    return MetaRow(
        schema_version=2,
        id=sample_id,
        run_id="run_a",
        domain="sim",
        split=split,
        seed=1,
        image_path=image_path,
        render_backend="opencv_2d",
        footprint="chip_2pad",
        nominal={
            "pad_width": 10.0,
            "pad_height": 10.0,
            "pad_spacing": 20.0,
            "component_length": 30.0,
            "component_width": 15.0,
        },
        defect={
            "type": "OK",
            "shift_x": 0.0,
            "shift_y": 0.0,
            "rotation_deg": 0.0,
            "tilt_deg": 0.0,
        },
        augment={
            "blur_sigma": 0.0,
            "noise_stddev": 0.0,
            "brightness_factor": 1.0,
            "contrast_factor": 1.0,
            "rotation_deg": 0.0,
        },
        render_meta={},
    )


def _label(sample_id: str, class_name: str = "OK") -> LabelRow:
    return LabelRow(schema_version=2, id=sample_id, class_name=class_name, profile_id="chip_0603_resistor@1")


def _write_dataset(ds: Path, rows: list[tuple[str, str, str]], profile_id: str = "chip_0603_resistor@1") -> None:
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "images").mkdir(parents=True, exist_ok=True)

    meta_rows: list[MetaRow] = []
    label_rows: list[LabelRow] = []
    for sample_id, class_name, split in rows:
        image_name = f"{sample_id.split('/')[-1]}.png"
        image_rel = f"images/{image_name}"
        meta_rows.append(_meta(sample_id, image_rel, split=split))
        label_rows.append(_label(sample_id, class_name=class_name))
        (ds / image_rel).write_bytes(b"png")

    write_jsonl(ds / "meta.jsonl", meta_rows)
    write_jsonl(ds / "labels.jsonl", label_rows)
    manifest = {
        "manifest_version": 1,
        "created_at": "2026-02-20T00:00:00",
        "run_id": "run_a",
        "component_profile": {
            "profile_id": profile_id,
            "profile_hash": "sha256:abc",
            "profile_path": f"configs/profiles/{profile_id}.yaml",
        },
        "generator": {"version": "1.0.0", "git_commit": "abc", "script": "scripts/generate.py"},
        "dataset_stats": {"total_samples": len(rows), "splits": {}, "classes": {}},
        "extend_history": [],
    }
    with open(ds / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)


def test_delete_samples_updates_jsonl_and_manifest(tmp_path: Path) -> None:
    ds = tmp_path / "runs" / "ds_a"
    _write_dataset(
        ds,
        [
            ("run_a/sim/train/0001", "OK", "train"),
            ("run_a/sim/test/0002", "MISSING", "test"),
        ],
    )

    summary = delete_samples(ds, ["run_a/sim/train/0001"])

    assert summary.requested == 1
    assert summary.affected == 1
    meta_rows = read_jsonl(ds / "meta.jsonl", MetaRow)
    label_rows = read_jsonl(ds / "labels.jsonl", LabelRow)
    assert [r.id for r in meta_rows] == ["run_a/sim/test/0002"]
    assert [r.id for r in label_rows] == ["run_a/sim/test/0002"]
    assert not (ds / "images/0001.png").exists()

    with open(ds / "dataset_manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["dataset_stats"]["total_samples"] == 1
    assert manifest["dataset_stats"]["splits"]["test"] == 1
    assert manifest["dataset_stats"]["classes"]["MISSING"] == 1


def test_move_samples_transfers_rows_files_and_manifest(tmp_path: Path) -> None:
    src = tmp_path / "runs" / "src"
    dst = tmp_path / "runs" / "dst"
    _write_dataset(src, [("run_a/sim/train/0001", "OK", "train"), ("run_a/sim/val/0002", "TOMBSTONE", "val")])
    _write_dataset(dst, [("run_b/sim/train/1001", "OK", "train")])

    summary = move_samples(src, dst, ["run_a/sim/val/0002"])

    assert summary.affected == 1
    assert not (src / "images/0002.png").exists()
    assert (dst / "images/0002.png").exists()

    src_ids = {r.id for r in read_jsonl(src / "meta.jsonl", MetaRow)}
    dst_ids = {r.id for r in read_jsonl(dst / "meta.jsonl", MetaRow)}
    assert src_ids == {"run_a/sim/train/0001"}
    assert "run_a/sim/val/0002" in dst_ids

    with open(src / "dataset_manifest.json", "r", encoding="utf-8") as f:
        src_manifest = json.load(f)
    with open(dst / "dataset_manifest.json", "r", encoding="utf-8") as f:
        dst_manifest = json.load(f)
    assert src_manifest["dataset_stats"]["total_samples"] == 1
    assert dst_manifest["dataset_stats"]["total_samples"] == 2


def test_move_samples_dedupes_conflicting_ids_and_images(tmp_path: Path) -> None:
    src = tmp_path / "runs" / "src"
    dst = tmp_path / "runs" / "dst"
    conflict_id = "run_a/sim/train/0001"
    _write_dataset(src, [(conflict_id, "OK", "train")])
    _write_dataset(dst, [(conflict_id, "MISSING", "train")])

    move_samples(src, dst, [conflict_id])

    dst_meta = read_jsonl(dst / "meta.jsonl", MetaRow)
    dst_ids = {r.id for r in dst_meta}
    assert len(dst_ids) == 2
    assert any(s.endswith("__moved2") for s in dst_ids)

    image_paths = {r.image_path for r in dst_meta}
    assert len(image_paths) == 2
    assert any(p.endswith("__moved2.png") for p in image_paths)


def test_move_samples_rejects_profile_mismatch(tmp_path: Path) -> None:
    src = tmp_path / "runs" / "src"
    dst = tmp_path / "runs" / "dst"
    _write_dataset(src, [("run_a/sim/train/0001", "OK", "train")], profile_id="chip_0603_resistor@1")
    _write_dataset(dst, [("run_b/sim/train/1001", "OK", "train")], profile_id="qfn32_ic@1")

    with pytest.raises(ValueError, match="Profile mismatch"):
        move_samples(src, dst, ["run_a/sim/train/0001"])


def test_list_dataset_samples_returns_rows(tmp_path: Path) -> None:
    ds = tmp_path / "runs" / "ds_a"
    _write_dataset(ds, [("run_a/sim/train/0001", "OK", "train")])

    rows = list_dataset_samples(ds)

    assert len(rows) == 1
    assert rows[0].sample_id == "run_a/sim/train/0001"
    assert rows[0].class_name == "OK"

"""Tests for non-GUI logic in tools/: profile editor state store, label backfill, dataset merge."""

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate import generate_dataset
from simple_sim.manifest import read_dataset_manifest
from simple_sim.schema import LabelRow, MetaRow, read_jsonl, write_jsonl
from tools.backfill_labels_v2 import backfill_labels
from tools.create_multi_dataset import create_multi_dataset
from tools.profile_editor.state_store import EditorStateStore, to_form_text


def _store_with(data: dict) -> EditorStateStore:
    store = EditorStateStore()
    store.profile.data = data
    store.profile.dirty = False
    return store


# --- state_store ------------------------------------------------------------


def test_commit_form_text_unchanged_is_noop():
    store = _store_with({"component": {"package": "0603", "flag": True, "n": 3, "none": None}})
    for path, value in [
        (("component", "package"), "0603"),
        (("component", "flag"), True),
        (("component", "n"), 3),
        (("component", "none"), None),
    ]:
        assert store.commit_form_text("profile", path, to_form_text(value), source="form") is False
    assert store.profile.dirty is False
    assert store.profile.data["component"]["package"] == "0603"


def test_commit_form_text_keeps_strings_as_strings():
    store = _store_with({"component": {"package": "0805", "mode": "auto"}})
    assert store.commit_form_text("profile", ("component", "package"), "0603", source="form")
    assert store.commit_form_text("profile", ("component", "mode"), "yes", source="form")
    assert store.profile.data["component"] == {"package": "0603", "mode": "yes"}
    assert store.profile.dirty is True


def test_commit_form_text_parses_non_strings():
    store = _store_with({"g": {"w": [1.0, 2.0], "n": 3, "b": False}})
    store.commit_form_text("profile", ("g", "w"), "[1.5, 2.5]", source="form")
    store.commit_form_text("profile", ("g", "n"), "4", source="form")
    store.commit_form_text("profile", ("g", "b"), "true", source="form")
    assert store.profile.data["g"] == {"w": [1.5, 2.5], "n": 4, "b": True}


def test_profile_edits_do_not_rewrite_run_profile():
    store = EditorStateStore()
    store.profile.data = {"profile": {"profile_id": "a@1"}}
    store.run.path = Path("run_x.yaml")
    store.run.data = {"run": {"run_id": "x", "component_profile": "b@1"}}
    store.update_leaf("profile", ("profile", "profile_id"), "c@1", source="form")
    assert store.run.data["run"]["component_profile"] == "b@1"
    assert store.run.dirty is False
    assert store.run_profile_mismatch() is True

    assert store.link_run_to_profile() is True
    assert store.run.data["run"]["component_profile"] == "c@1"
    assert store.run.dirty is True
    assert store.run_profile_mismatch() is False


# --- dataset helpers --------------------------------------------------------


def _write_config(path: Path, run_id: str, profile_id: str) -> Path:
    config = {
        "run": {"run_id": run_id, "seed": 7, "schema_version": 1, "component_profile": profile_id},
        "roi": {"width_px": 64, "height_px": 64, "mm_per_px": 0.02},
        "classes": {"OK": 2, "MISSING": 2, "MISALIGNED": 2, "TOMBSTONE": 2},
        "tolerances": {"0603": {"ok_shift_px": 3.0, "ok_rotation_deg": 5.0, "tombstone_tilt_deg": 75.0}},
        "domains": {"domain_A": {"lighting_brightness": [0.9, 1.1], "blur_sigma": [0.0, 1.0], "noise_stddev": [0.0, 5.0]}},
        "splits": {
            "train_domains": ["domain_A"],
            "val_domains": ["domain_A"],
            "test_domains": ["domain_A"],
            "train_frac": 0.5,
            "val_frac": 0.25,
            "test_frac": 0.25,
        },
        "render": {
            "backend": "opencv_2d",
            "substrate_color": [40, 90, 40],
            "copper_color": [60, 120, 180],
            "component_color": [20, 20, 20],
            "solder_mask_alpha": 0.3,
        },
        "augment": {
            "rotation_deg_range": [-5, 5],
            "brightness_factor_range": [0.85, 1.15],
            "contrast_factor_range": [0.9, 1.1],
        },
        "train": {"model": "resnet18", "pretrained": False, "epochs": 1, "batch_size": 4, "num_workers": 0, "lr": 0.001,
                  "optimizer": "adam", "weight_decay": 0.0001},
        "eval": {"batch_size": 4, "num_workers": 0, "metrics": ["accuracy"]},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _make_dataset(tmp_path: Path, name: str, profile_id: str) -> Path:
    cfg = _write_config(tmp_path / f"{name}.yaml", name, profile_id)
    out = tmp_path / name
    generate_dataset(cfg, out)
    return out


def _downgrade_labels_to_v1(ds: Path) -> None:
    rows = read_jsonl(ds / "labels.jsonl", LabelRow)
    write_jsonl(ds / "labels.jsonl", [LabelRow(schema_version=1, id=r.id, class_name=r.class_name) for r in rows])


# --- backfill_labels_v2 -----------------------------------------------------


def test_backfill_labels_v2_fills_only_missing(tmp_path):
    ds = _make_dataset(tmp_path, "ds_a", "chip_0603_resistor@1")
    rows = read_jsonl(ds / "labels.jsonl", LabelRow)
    mixed = [LabelRow(schema_version=1, id=r.id, class_name=r.class_name) for r in rows]
    mixed[0] = LabelRow(schema_version=2, id=rows[0].id, class_name=rows[0].class_name, profile_id="keep_me@1")
    write_jsonl(ds / "labels.jsonl", mixed)

    filled = backfill_labels(ds)
    out = read_jsonl(ds / "labels.jsonl", LabelRow)
    assert filled == len(rows) - 1
    assert all(r.schema_version == 2 for r in out)
    assert out[0].profile_id == "keep_me@1"
    assert {r.profile_id for r in out[1:]} == {"chip_0603_resistor@1"}


def test_backfill_labels_v2_refuses_multi_profile_manifest(tmp_path):
    ds = _make_dataset(tmp_path, "ds_a", "chip_0603_resistor@1")
    _downgrade_labels_to_v1(ds)
    manifest_path = ds / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    cp = manifest.pop("component_profile")
    manifest["manifest_version"] = 2
    manifest["component_profiles"] = [cp, dict(cp, profile_id="other@1")]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="multi-profile"):
        backfill_labels(ds)


# --- create_multi_dataset ---------------------------------------------------


def test_create_multi_dataset_two_profiles(tmp_path):
    ds_a = _make_dataset(tmp_path, "ds_a", "chip_0603_resistor@1")
    ds_b = _make_dataset(tmp_path, "ds_b", "sot23_transistor@1")
    _downgrade_labels_to_v1(ds_a)  # v1 labels must pick up the manifest's profile
    b_meta = read_jsonl(ds_b / "meta.jsonl", MetaRow)
    for i, m in enumerate(b_meta):
        m.schema_version, m.render_meta = 2, {"backend": "test", "idx": i}
    write_jsonl(ds_b / "meta.jsonl", b_meta)

    out = tmp_path / "merged"
    create_multi_dataset([ds_a, ds_b], out)

    labels = read_jsonl(out / "labels.jsonl", LabelRow)
    meta = read_jsonl(out / "meta.jsonl", MetaRow)
    n_a = len(read_jsonl(ds_a / "meta.jsonl", MetaRow))
    n_b = len(read_jsonl(ds_b / "meta.jsonl", MetaRow))
    assert len(labels) == len(meta) == n_a + n_b
    assert all(r.schema_version == 2 for r in labels)
    assert [r.profile_id for r in labels] == ["chip_0603_resistor@1"] * n_a + ["sot23_transistor@1"] * n_b

    src_meta = read_jsonl(ds_a / "meta.jsonl", MetaRow) + read_jsonl(ds_b / "meta.jsonl", MetaRow)
    assert [m.render_meta for m in meta] == [m.render_meta for m in src_meta]
    assert meta[-1].render_meta == {"backend": "test", "idx": n_b - 1}
    assert all((out / m.image_path).exists() for m in meta)

    manifest = read_dataset_manifest(out / "dataset_manifest.json")
    assert manifest["manifest_version"] == 2
    assert [p["profile_id"] for p in manifest["component_profiles"]] == [
        "chip_0603_resistor@1",
        "sot23_transistor@1",
    ]
    assert manifest["dataset_stats"]["profile_counts"] == {
        "chip_0603_resistor@1": n_a,
        "sot23_transistor@1": n_b,
    }
    assert manifest["run_id"] == "merged"


def test_create_multi_dataset_single_profile_writes_v1_manifest(tmp_path):
    ds_a = _make_dataset(tmp_path, "ds_a", "chip_0603_resistor@1")
    ds_b = _make_dataset(tmp_path, "ds_b", "chip_0603_resistor@1")
    out = tmp_path / "merged"
    create_multi_dataset([ds_a, ds_b], out)

    manifest = read_dataset_manifest(out / "dataset_manifest.json")
    assert manifest["manifest_version"] == 1
    assert manifest["component_profile"]["profile_id"] == "chip_0603_resistor@1"
    assert {r.profile_id for r in read_jsonl(out / "labels.jsonl", LabelRow)} == {"chip_0603_resistor@1"}

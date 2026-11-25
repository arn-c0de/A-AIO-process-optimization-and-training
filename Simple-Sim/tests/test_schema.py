"""Test schema validation."""

import pytest
import tempfile
from pathlib import Path
from simple_sim.schema import MetaRow, LabelRow, write_jsonl, read_jsonl, validate_jsonl_pair
from simple_sim.rows import extend_jsonl_pair
import json
import os
from tests.helpers import ROOT, tiny_run_config as _config
from simple_sim.generation.dataset import generate_dataset


def test_meta_row_valid():
    """Test valid MetaRow creation."""
    row = MetaRow(
        schema_version=1,
        id="run_0001/domain_A/train/000001",
        run_id="run_0001",
        domain="domain_A",
        split="train",
        seed=12345,
        image_path="images/000001.png",
        render_backend="opencv_2d",
        footprint="0603",
        nominal={
            'pad_width': 30.0,
            'pad_height': 35.0,
            'pad_spacing': 55.0,
            'component_length': 60.0,
            'component_width': 30.0
        },
        defect={
            'type': 'OK',
            'shift_x': 1.0,
            'shift_y': -0.5,
            'rotation_deg': 2.0,
            'tilt_deg': 0.0
        },
        augment={
            'blur_sigma': 0.5,
            'noise_stddev': 3.0,
            'brightness_factor': 1.05,
            'contrast_factor': 1.0,
            'rotation_deg': -1.5
        }
    )
    assert row.id == "run_0001/domain_A/train/000001"


def test_meta_row_invalid_split():
    """Test MetaRow with invalid split."""
    with pytest.raises(ValueError):
        MetaRow(
            schema_version=1,
            id="run_0001/domain_A/invalid_split/000001",
            run_id="run_0001",
            domain="domain_A",
            split="invalid_split",
            seed=12345,
            image_path="images/000001.png",
            render_backend="opencv_2d",
            footprint="0603",
            nominal={
                'pad_width': 30.0,
                'pad_height': 35.0,
                'pad_spacing': 55.0,
                'component_length': 60.0,
                'component_width': 30.0
            },
            defect={'type': 'OK', 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0, 'tilt_deg': 0},
            augment={'blur_sigma': 0, 'noise_stddev': 0, 'brightness_factor': 1.0, 'contrast_factor': 1.0, 'rotation_deg': 0}
        )


def test_meta_row_valid_with_pad_spacing_y():
    """Test valid MetaRow with extra pad_spacing_y in nominal (SOT-23)."""
    row = MetaRow(
        schema_version=1,
        id="run_0001/domain_A/train/000001",
        run_id="run_0001",
        domain="domain_A",
        split="train",
        seed=12345,
        image_path="images/000001.png",
        render_backend="opencv_2d",
        footprint="sot23",
        nominal={
            'pad_width': 70.0,
            'pad_height': 100.0,
            'pad_spacing': 145.0,
            'pad_spacing_y': 95.0,
            'component_length': 85.0,
            'component_width': 40.0
        },
        defect={
            'type': 'OK',
            'shift_x': 0.0,
            'shift_y': 0.0,
            'rotation_deg': 0.0,
            'tilt_deg': 0.0
        },
        augment={
            'blur_sigma': 0.5,
            'noise_stddev': 3.0,
            'brightness_factor': 1.05,
            'contrast_factor': 1.0,
            'rotation_deg': -1.5
        }
    )
    assert row.footprint == "sot23"
    assert 'pad_spacing_y' in row.nominal


def test_label_row_valid():
    """Test valid LabelRow creation."""
    row = LabelRow(
        schema_version=1,
        id="run_0001/domain_A/train/000001",
        class_name="MISALIGNED"
    )
    assert row.class_name == "MISALIGNED"


def test_label_row_invalid_class():
    """Test LabelRow with invalid class."""
    with pytest.raises(ValueError):
        LabelRow(
            schema_version=1,
            id="run_0001/domain_A/train/000001",
            class_name="INVALID_CLASS"
        )


def test_write_read_jsonl():
    """Test JSONL write and read roundtrip."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        labels = [
            LabelRow(schema_version=1, id="test/domain/train/000001", class_name="OK"),
            LabelRow(schema_version=1, id="test/domain/train/000002", class_name="MISSING"),
        ]

        # Write
        write_jsonl(tmpdir / "labels.jsonl", labels)

        # Read
        read_labels = read_jsonl(tmpdir / "labels.jsonl", LabelRow)

        # Verify
        assert len(read_labels) == 2
        assert read_labels[0].id == "test/domain/train/000001"
        assert read_labels[0].class_name == "OK"
        assert read_labels[1].class_name == "MISSING"


def test_validate_jsonl_pair_mismatch():
    """Test validation catches ID mismatches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mismatched data
        meta_rows = [
            MetaRow(
                schema_version=1,
                id="test/domain/train/000001",
                run_id="test",
                domain="domain",
                split="train",
                seed=123,
                image_path="img.png",
                render_backend="opencv_2d",
                footprint="0603",
                nominal={'pad_width': 30, 'pad_height': 35, 'pad_spacing': 55,
                        'component_length': 60, 'component_width': 30},
                defect={'type': 'OK', 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0, 'tilt_deg': 0},
                augment={'blur_sigma': 0, 'noise_stddev': 0, 'brightness_factor': 1.0, 'contrast_factor': 1.0, 'rotation_deg': 0}
            )
        ]

        label_rows = [
            LabelRow(schema_version=1, id="test/domain/train/000002", class_name="OK")  # Different ID
        ]

        write_jsonl(tmpdir / "meta.jsonl", meta_rows)
        write_jsonl(tmpdir / "labels.jsonl", label_rows)

        with pytest.raises(ValueError):
            validate_jsonl_pair(tmpdir / "meta.jsonl", tmpdir / "labels.jsonl")


def test_extend_jsonl_pair_replaces_both_files_or_none(tmp_path):
    meta, labels = tmp_path / "meta.jsonl", tmp_path / "labels.jsonl"
    meta.write_text('{"id": "a"}', encoding="utf-8")  # no trailing newline
    labels.write_text('{"id": "a"}\n', encoding="utf-8")
    snapshot = tmp_path / "snapshot_meta.jsonl"
    os.link(meta, snapshot)
    row = LabelRow(schema_version=1, id="b", class_name="OK")

    extend_jsonl_pair(meta, [row], labels, [row])
    assert [json.loads(line)["id"] for line in meta.read_text(encoding="utf-8").splitlines()] == ["a", "b"]
    assert len(labels.read_text(encoding="utf-8").splitlines()) == 2
    assert snapshot.read_text(encoding="utf-8") == '{"id": "a"}'  # a hardlinked copy keeps the old rows

    before = (meta.read_bytes(), labels.read_bytes())
    with pytest.raises(TypeError):
        extend_jsonl_pair(meta, [row], labels, [object()])  # labels cannot be written
    assert (meta.read_bytes(), labels.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.jsonl", "meta.jsonl", "snapshot_meta.jsonl"]


def test_validate_jsonl_pair_reports_duplicate_ids(tmp_path):
    data = tmp_path / "ds"
    generate_dataset(_config(), data, project_root=ROOT)
    for name in ("meta.jsonl", "labels.jsonl"):
        lines = (data / name).read_text(encoding="utf-8").splitlines()
        (data / name).write_text("\n".join([*lines, lines[-1]]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate sample ids"):
        validate_jsonl_pair(data / "meta.jsonl", data / "labels.jsonl")


def test_read_jsonl_rejects_non_object_lines(tmp_path):
    path = tmp_path / "meta.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path, MetaRow)


def test_read_jsonl_wraps_non_dict_row_fields(tmp_path):
    """A row whose nominal/defect/augment is not an object is a schema error with the line number
    (it used to escape read_jsonl as a bare AttributeError)."""
    row = {
        "schema_version": 1, "id": "r/d/train/000001", "run_id": "r", "domain": "d", "split": "train", "seed": 1,
        "image_path": "images/a.png", "render_backend": "opencv_2d", "footprint": "0603",
        "nominal": [1, 2], "defect": {}, "augment": {},
    }
    path = tmp_path / "meta.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1.*nominal must be a dict"):
        read_jsonl(path, MetaRow)


@pytest.mark.parametrize("payload, match", [
    ([1, 2], "expected a JSON object"),
    ({"manifest_version": 1, "created_at": "", "run_id": "r", "generator": {}, "dataset_stats": {},
      "extend_history": [], "component_profile": "profile_id profile_hash profile_path"},
     "component_profile must be an object"),
])
def test_read_dataset_manifest_rejects_malformed_json(tmp_path, payload, match):
    from simple_sim.manifest import read_dataset_manifest

    path = tmp_path / "dataset_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        read_dataset_manifest(path)

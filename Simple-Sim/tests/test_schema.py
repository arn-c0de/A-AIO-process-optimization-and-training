"""Test schema validation."""

import pytest
import tempfile
from pathlib import Path
from simple_sim.schema import MetaRow, LabelRow, write_jsonl, read_jsonl, validate_jsonl_pair


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

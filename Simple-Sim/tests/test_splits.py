"""Test stratified splitting."""

import pytest
from simple_sim.schema import MetaRow, LabelRow
from simple_sim.splits import generate_splits, assert_no_overlap


def create_test_data(n_per_class=10):
    """Create test metadata and labels."""
    meta_rows = []
    label_rows = []

    idx = 0
    for class_name in ['OK', 'MISSING', 'MISALIGNED', 'TOMBSTONE']:
        for i in range(n_per_class):
            sample_id = f"test/domain_A/train/{idx:06d}"

            meta = MetaRow(
                schema_version=1,
                id=sample_id,
                run_id="test",
                domain="domain_A",
                split="train",
                seed=idx + 1,
                image_path=f"images/{idx:06d}.png",
                render_backend="opencv_2d",
                footprint="0603",
                nominal={'pad_width': 30, 'pad_height': 35, 'pad_spacing': 55,
                        'component_length': 60, 'component_width': 30},
                defect={'type': class_name, 'shift_x': 0, 'shift_y': 0, 'rotation_deg': 0, 'tilt_deg': 0},
                augment={'blur_sigma': 0, 'noise_stddev': 0, 'brightness_factor': 1.0, 'contrast_factor': 1.0, 'rotation_deg': 0}
            )
            meta_rows.append(meta)

            label = LabelRow(
                schema_version=1,
                id=sample_id,
                class_name=class_name
            )
            label_rows.append(label)

            idx += 1

    return meta_rows, label_rows


def test_generate_splits_no_overlap():
    """Test that splits have no overlap."""
    meta_rows, label_rows = create_test_data(10)

    config = {
        'splits': {
            'train_frac': 0.7,
            'val_frac': 0.15,
            'test_frac': 0.15
        }
    }

    splits = generate_splits(meta_rows, label_rows, config, run_seed=42)

    # Should not raise
    assert_no_overlap(splits)


def test_generate_splits_all_samples_assigned():
    """Test that all samples are assigned to a split."""
    meta_rows, label_rows = create_test_data(10)

    config = {
        'splits': {
            'train_frac': 0.7,
            'val_frac': 0.15,
            'test_frac': 0.15
        }
    }

    splits = generate_splits(meta_rows, label_rows, config, run_seed=42)

    total_samples = len(meta_rows)
    assigned_samples = len(splits['train']) + len(splits['val']) + len(splits['test'])

    assert assigned_samples == total_samples


def test_generate_splits_stratification():
    """Test that splits are stratified by class."""
    meta_rows, label_rows = create_test_data(100)

    config = {
        'splits': {
            'train_frac': 0.7,
            'val_frac': 0.15,
            'test_frac': 0.15
        }
    }

    splits = generate_splits(meta_rows, label_rows, config, run_seed=42)

    # Build label map
    id_to_label = {row.id: row.class_name for row in label_rows}

    # Check each split has all classes
    for split_name, sample_ids in splits.items():
        split_classes = set(id_to_label[sid] for sid in sample_ids)
        assert len(split_classes) == 4, f"{split_name} missing some classes"


def test_assert_no_overlap_detects_overlap():
    """Test that overlap detection works."""
    splits = {
        'train': ['id1', 'id2', 'id3'],
        'val': ['id4', 'id5'],
        'test': ['id3', 'id6']  # id3 overlaps with train
    }

    with pytest.raises(ValueError):
        assert_no_overlap(splits)

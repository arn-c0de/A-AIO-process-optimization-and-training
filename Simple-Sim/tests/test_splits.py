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
        for _i in range(n_per_class):
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


def _splits_config():
    return {'splits': {'train_frac': 0.7, 'val_frac': 0.15, 'test_frac': 0.15}}


def test_generate_splits_small_groups_get_val_and_test():
    """Largest-remainder rounding: 5 per class -> 3/1/1 (not 3/0/2)."""
    meta_rows, label_rows = create_test_data(5)
    splits = generate_splits(meta_rows, label_rows, _splits_config(), run_seed=42)
    assert (len(splits['train']), len(splits['val']), len(splits['test'])) == (12, 4, 4)


def test_generate_splits_deterministic():
    meta_rows, label_rows = create_test_data(7)
    a = generate_splits(meta_rows, label_rows, _splits_config(), run_seed=1)
    b = generate_splits(meta_rows, label_rows, _splits_config(), run_seed=1)
    assert a == b


def test_split_counts_largest_remainder():
    from simple_sim.splits import _split_counts
    assert _split_counts(5, (0.7, 0.15, 0.15)) == [3, 1, 1]
    assert _split_counts(100, (0.7, 0.15, 0.15)) == [70, 15, 15]
    assert _split_counts(3, (0.7, 0.15, 0.15)) == [1, 1, 1]
    assert _split_counts(2, (0.7, 0.15, 0.15)) == [2, 0, 0]
    assert _split_counts(7, (0.8, 0.2, 0.0)) == [6, 1, 0]
    for n in range(0, 40):
        assert sum(_split_counts(n, (0.7, 0.15, 0.15))) == n


def test_generate_splits_stratifies_by_profile():
    """Each (profile, class) group is split on its own, so every profile reaches val/test."""
    meta_a, labels_a = create_test_data(5)
    meta_rows, label_rows = [], []
    for pid in ('prof_a@1', 'prof_b@1'):
        for m, l in zip(meta_a, labels_a):
            sid = f"{pid}/{m.id}"
            meta_rows.append(MetaRow(**{**m.__dict__, 'id': sid}))
            label_rows.append(LabelRow(schema_version=2, id=sid, class_name=l.class_name, profile_id=pid))

    splits = generate_splits(meta_rows, label_rows, _splits_config(), run_seed=42)
    id_to_profile = {r.id: r.profile_id for r in label_rows}
    for split_name in ('val', 'test'):
        per_profile = {}
        for sid in splits[split_name]:
            per_profile[id_to_profile[sid]] = per_profile.get(id_to_profile[sid], 0) + 1
        assert per_profile == {'prof_a@1': 4, 'prof_b@1': 4}

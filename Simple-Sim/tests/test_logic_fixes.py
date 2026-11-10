"""Regression tests for logic errors found in the 2026-11 audit (one small test per fix)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from gui.utils.pipeline_helpers import precise_counts, profile_run_targets
from simple_sim.metrics import evaluate_success_criteria

cv2 = pytest.importorskip("cv2")


def test_motion_blur_keeps_brightness_and_position_for_every_angle():
    from simple_sim.generators.opencv2d.filters import apply_motion_blur

    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    for strength in (0.5, 1.0, 1.5):
        for angle in (0.0, 45.0, 90.0, 135.0):
            out = apply_motion_blur(img, strength, angle)
            assert abs(float(out[16:48, 16:48].mean()) - 120.0) <= 1.0, (strength, angle)
    dot = np.zeros((65, 65), dtype=np.float32)
    dot[32, 32] = 1.0
    ys, xs = np.nonzero(apply_motion_blur(dot, 0.5, 90.0) > 0)  # even kernel size used to shift the image
    assert abs(float(ys.mean()) - 32.0) < 0.26 and abs(float(xs.mean()) - 32.0) < 0.26


def test_chromatic_aberration_acts_below_half_strength():
    from simple_sim.generators.opencv2d.filters import apply_chromatic_aberration

    img = np.zeros((16, 32, 3), dtype=np.uint8)
    img[:, 16:] = 200
    assert not np.array_equal(apply_chromatic_aberration(img, 0.2), img)


def test_filter_angles_from_the_settings_are_applied():
    from simple_sim.generators.opencv2d.augment import apply_image_filter_overrides

    sampled = {"perspective_angle_x": 3.0, "perspective_angle_y": -2.0, "motion_blur_angle": 200.0, "rotation_deg": -50.0}
    filt = {"enable_perspective": True, "perspective_angle_x": 10.0, "enable_motion_blur": True, "motion_blur_angle": 45.0,
            "rotation_strength": 0.5}
    out = apply_image_filter_overrides(dict(sampled), filt, np.random.default_rng(0))
    assert out["perspective_angle_x"] == 10.0 and out["motion_blur_angle"] == 45.0
    assert out["perspective_angle_y"] == -2.0  # not set: the sampled angle stays
    assert out["rotation_deg"] == pytest.approx(-70.0)  # jitter scaled around the cardinal base (-90)


def test_success_criteria_skip_the_val_check_without_a_stored_val_accuracy():
    metrics = {"accuracy": 0.95, "per_class": {"OK": {"recall": 1.0, "support": 3}}}
    assert evaluate_success_criteria(None, metrics).passed
    assert not evaluate_success_criteria(0.5, metrics).passed


def test_empty_model_field_falls_back_to_the_run_id_model():
    out, model = profile_run_targets(Path("/d/x"), Path(""), "run_7", multi=False, models_root=Path("/m"))
    assert out == Path("/d/x") and model == Path("/m/run_7.pt")


def test_precise_per_class_counts_only_cover_the_current_class_set():
    settings = {"per_class": True, "class_names": ["OK", "MISSING"],
                "classes": {"OK": 5, "MISSING": 6, "TOMBSTONE": 7}}
    assert precise_counts(settings) == ({"OK": 5, "MISSING": 6}, None)


def test_board_drc_error_survives_pickling():
    from simple_sim.board.drc import BoardDrcError, DrcReport

    err = pickle.loads(pickle.dumps(BoardDrcError(DrcReport(), 7)))
    assert isinstance(err, BoardDrcError) and err.seed == 7


def test_generate_extend_on_a_new_directory_writes_a_fresh_manifest(tmp_path):
    from simple_sim.config import load_config
    from simple_sim.dataset_files import DATASET_MANIFEST
    from simple_sim.generation.dataset import generate_dataset

    cfg = load_config(Path(__file__).resolve().parent.parent / "configs" / "run_0001.yaml")
    cfg["classes"] = dict.fromkeys(list(cfg["classes"])[:2], 4)
    out = tmp_path / "fresh"
    result = generate_dataset(cfg, out, extend=True)
    assert (out / DATASET_MANIFEST).is_file() and result["extend"] is False

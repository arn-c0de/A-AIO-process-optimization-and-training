"""Sim-to-real metrics: same distribution ~ chance, disjoint distributions fully separable."""

import importlib.util
from pathlib import Path

import numpy as np
from simple_sim.images import list_synthetic_images
import pytest

_spec = importlib.util.spec_from_file_location("sim_to_real", Path(__file__).parent.parent / "tools" / "sim_to_real.py")
s2r = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2r)


def _unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_same_distribution_is_indistinguishable():
    rng = np.random.default_rng(0)
    real, syn = _unit(rng.normal(size=(300, 16))), _unit(rng.normal(size=(300, 16)))
    assert abs(s2r.one_nn_accuracy(real, syn) - 0.5) < 0.1
    assert 0.8 < s2r.nn_distance_ratio(real, syn) < 1.2


def test_shifted_distribution_is_separable():
    rng = np.random.default_rng(0)
    real = _unit(rng.normal(size=(200, 16)) * 0.1 + 1.0)
    syn = _unit(rng.normal(size=(200, 16)) * 0.1 - 1.0)
    assert s2r.one_nn_accuracy(real, syn) == 1.0
    assert s2r.nn_distance_ratio(real, syn) > 5.0


def test_sim_to_real_tools_skip_mixed_in_real_crops_and_balance_evenly(tmp_path):
    from tools.sim_to_real import one_nn_accuracy

    data = tmp_path / "ds"
    (data / "images" / "real").mkdir(parents=True)
    (data / "meta.jsonl").write_text("", encoding="utf-8")
    for rel in ("images/000000.png", "images/000001.png", "images/real/000002.png"):
        (data / rel).write_bytes(b"x")
    assert [p.name for p in list_synthetic_images(data / "images")] == ["000000.png", "000001.png"]
    assert [p.name for p in list_synthetic_images(data)] == ["000000.png", "000001.png"]
    assert [p.name for p in list_synthetic_images(data / "images" / "real")] == ["000002.png"]  # asked for directly

    # Synthetic features sorted class by class: the first half looks like the real ones, the second
    # does not. A prefix of the larger set would see the first class only (accuracy ~ chance).
    rng = np.random.default_rng(0)

    def cluster(axis: int, n: int) -> np.ndarray:
        x = np.eye(8)[[axis] * n] + rng.normal(0.0, 0.05, (n, 8))
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    real, syn = cluster(0, 6), np.concatenate([cluster(0, 6), cluster(1, 6)])
    assert one_nn_accuracy(real, syn) > one_nn_accuracy(real, syn[:6]) + 0.1
    assert one_nn_accuracy(syn, real) == pytest.approx(one_nn_accuracy(real, syn))

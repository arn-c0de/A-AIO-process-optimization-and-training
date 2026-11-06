"""Sim-to-real metrics: same distribution ~ chance, disjoint distributions fully separable."""

import importlib.util
from pathlib import Path

import numpy as np

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

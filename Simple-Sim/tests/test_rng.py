"""Test deterministic RNG."""

import pytest
from simple_sim.rng import derive_sample_seed, make_sample_id, parse_sample_id
import numpy as np


def test_derive_sample_seed_deterministic():
    """Test that same inputs produce same seed."""
    seed1 = derive_sample_seed(42, "domain_A", 100)
    seed2 = derive_sample_seed(42, "domain_A", 100)
    assert seed1 == seed2


def test_derive_sample_seed_different_inputs():
    """Test that different inputs produce different seeds."""
    seed1 = derive_sample_seed(42, "domain_A", 100)
    seed2 = derive_sample_seed(42, "domain_A", 101)  # Different index
    seed3 = derive_sample_seed(42, "domain_B", 100)  # Different domain
    seed4 = derive_sample_seed(43, "domain_A", 100)  # Different run seed

    assert seed1 != seed2
    assert seed1 != seed3
    assert seed1 != seed4


def test_derive_sample_seed_positive():
    """Test that seeds are always positive."""
    for i in range(100):
        seed = derive_sample_seed(i, f"domain_{i % 3}", i)
        assert seed >= 0


def test_make_sample_id():
    """Test sample ID formatting."""
    sample_id = make_sample_id("run_0001", "domain_A", "train", 42)
    assert sample_id == "run_0001/domain_A/train/000042"


def test_parse_sample_id():
    """Test sample ID parsing."""
    run_id, domain, split, index = parse_sample_id("run_0001/domain_A/train/000042")
    assert run_id == "run_0001"
    assert domain == "domain_A"
    assert split == "train"
    assert index == 42


def test_parse_sample_id_roundtrip():
    """Test make and parse roundtrip."""
    original = ("run_0001", "domain_A", "val", 123)
    sample_id = make_sample_id(*original)
    parsed = parse_sample_id(sample_id)
    assert parsed == original


def test_parse_sample_id_invalid():
    """Test parsing invalid IDs."""
    with pytest.raises(ValueError):
        parse_sample_id("invalid")

    with pytest.raises(ValueError):
        parse_sample_id("run/domain")


def test_render_stream_is_independent_of_the_sampling_stream():
    from simple_sim.generation.common import sample_record  # noqa: F401 (the sampler uses default_rng(seed))
    from simple_sim.rng import STREAM_BAKE, STREAM_POST, render_seed

    seeds = np.arange(1, 400)
    sampling = np.array([np.random.default_rng(int(s)).random() for s in seeds])
    for stream in (None, STREAM_BAKE, STREAM_POST):
        key = [render_seed(int(s)) if stream is None else render_seed(int(s), stream) for s in seeds]
        render = np.array([np.random.default_rng(k).random() for k in key])
        # The first render draw used to equal the first sampling draw (correlation 1.0).
        assert abs(np.corrcoef(sampling, render)[0, 1]) < 0.2


def test_render_record_does_not_replay_the_sampling_draws(monkeypatch):
    from simple_sim.generators.opencv2d import spec as spec_mod

    seen: dict[str, float] = {}
    monkeypatch.setattr(spec_mod, "render_sample", lambda _s, _n, _d, _a, rng, **_kw: seen.setdefault("first", rng.random()))
    spec_mod.render_record(None, {"seed": 1234, "nominal": {}, "defect": {}, "augment": {}})
    assert seen["first"] != np.random.default_rng(1234).random()
    assert seen["first"] == np.random.default_rng([1234, 1]).random()

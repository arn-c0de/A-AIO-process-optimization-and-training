"""Test deterministic RNG."""

import pytest
from simple_sim.rng import derive_sample_seed, make_sample_id, parse_sample_id


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

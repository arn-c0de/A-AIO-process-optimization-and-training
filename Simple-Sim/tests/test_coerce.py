"""simple_sim.coerce: the one lenient bool/float/int parser for settings."""

import math

from simple_sim.coerce import as_bool, as_float, as_int


def test_as_bool():
    assert as_bool(True) is True and as_bool(0) is False and as_bool(2.5) is True
    assert [as_bool(s) for s in ("1", " TRUE ", "yes", "On", "0", "no", "")] == [True] * 4 + [False] * 3
    assert as_bool(None) is False and as_bool(None, True) is True and as_bool([1], True) is True


def test_as_float():
    assert as_float("2.5", 0.0) == 2.5
    assert as_float(None, 1) == 1.0 and isinstance(as_float("x", 1), float)
    assert as_float("x", None) is None
    assert as_float(9.0, 1.0, 0.0, 4.0) == 4.0 and as_float(-1, 1.0, 0.0, 4.0) == 0.0
    assert as_float("bad", 7.0, 0.0, 4.0) == 4.0  # the default is clamped too (historic behaviour)
    assert math.isinf(as_float("inf", 0.0))


def test_as_int():
    assert as_int("3", 0) == 3 and as_int(2.9, 0) == 2
    assert as_int("2.5", 7) == 7  # int("2.5") fails -> default
    assert as_int(None, None) is None and as_int(float("inf"), 5) == 5
    assert as_int(99, 0, 0, 10) == 10

"""Data loader: per-image transforms run single-threaded in the main process (same tensors)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from simple_sim.data_loader import (
    IMAGENET_TRANSFORM,
    apply_transform,
    build_train_transform,
    preprocess_bgr,
    single_threaded_torch,
)


@pytest.fixture
def four_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(4)
    try:
        yield 4
    finally:
        torch.set_num_threads(before)


def test_single_threaded_torch_sets_one_thread_and_restores(four_threads):
    with single_threaded_torch():
        assert torch.get_num_threads() == 1
        with single_threaded_torch():  # nested: still one thread, inner exit keeps it
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == four_threads


def test_single_threaded_torch_restores_on_error(four_threads):
    with pytest.raises(RuntimeError):
        with single_threaded_torch():
            raise RuntimeError("boom")
    assert torch.get_num_threads() == four_threads


def test_apply_transform_is_bit_identical(four_threads):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 48, 3), dtype=np.uint8)
    assert torch.equal(apply_transform(IMAGENET_TRANSFORM, img), IMAGENET_TRANSFORM(img))
    torch.manual_seed(1)
    a = apply_transform(build_train_transform({"hflip": True, "brightness": 0.3}), img)
    torch.manual_seed(1)
    b = build_train_transform({"hflip": True, "brightness": 0.3})(img)
    assert torch.equal(a, b)
    assert torch.get_num_threads() == four_threads


def test_preprocess_bgr_keeps_thread_count(four_threads):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue in BGR -> last channel after conversion
    x = preprocess_bgr(img)
    assert x.shape == (3, 32, 32)
    assert x[2].mean() > x[0].mean()
    assert torch.get_num_threads() == four_threads

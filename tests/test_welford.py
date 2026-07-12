"""Tests for the Welford online mean/variance update used in NMRD.

File: tests/test_welford.py
"""
import numpy as np
import pytest

from nmrdfrommd import NMRD


def test_welford_first_update_initializes_mean():
    """First call (mean=None) should initialize mean to new_value and M2 to zero."""
    new_value = np.array([1.0, 2.0, 3.0])
    mean, M2 = NMRD._welford_update(None, None, new_value, count=1)

    np.testing.assert_allclose(mean, new_value)
    np.testing.assert_allclose(M2, np.zeros_like(new_value))

def test_welford_matches_numpy_mean_and_variance():
    """Running Welford updates over N samples should match numpy's batch mean/var."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=5.0, scale=2.0, size=(50, 4))  # 50 samples, 4 frequencies

    mean, M2 = None, None
    for count, sample in enumerate(samples, start=1):
        mean, M2 = NMRD._welford_update(mean, M2, sample, count)

    expected_mean = np.mean(samples, axis=0)
    expected_var = np.var(samples, axis=0, ddof=1)  # unbiased, matches n>1 case downstream
    welford_var = M2 / (len(samples) - 1)

    np.testing.assert_allclose(mean, expected_mean, rtol=1e-10)
    np.testing.assert_allclose(welford_var, expected_var, rtol=1e-10)

def test_welford_single_sample_gives_zero_variance():
    """With a single sample, M2 must be exactly zero (no spread yet)."""
    new_value = np.array([7.0, -3.0])
    mean, M2 = NMRD._welford_update(None, None, new_value, count=1)

    assert np.all(M2 == 0.0)
    np.testing.assert_allclose(mean, new_value)

def test_welford_is_order_independent_in_final_mean():
    """Final mean should not depend on the order samples are fed in."""
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(20, 3))

    def run(order):
        mean, M2 = None, None
        for count, sample in enumerate(samples[order], start=1):
            mean, M2 = NMRD._welford_update(mean, M2, sample, count)
        return mean

    order_a = np.arange(len(samples))
    order_b = order_a[::-1]

    mean_a = run(order_a)
    mean_b = run(order_b)

    np.testing.assert_allclose(mean_a, mean_b, rtol=1e-10)

def test_welford_returns_float64_regardless_of_input_dtype():
    """Ensures accumulation happens in float64 even if fed float32 samples."""
    new_value = np.array([1.0, 2.0], dtype=np.float32)
    mean, M2 = NMRD._welford_update(None, None, new_value, count=1)

    assert mean.dtype == np.float64
    assert M2.dtype == np.float64

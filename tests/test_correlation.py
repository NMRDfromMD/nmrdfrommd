import numpy as np
from scipy import constants as cst

from nmrdfrommd.correlation import autocorrelation_function, calculate_tau


def reference_autocorrelation(data):
    """Direct O(N²) autocorrelation for validation."""
    n = len(data)
    ac = np.zeros(n)

    for lag in range(n):
        ac[lag] = np.mean(data[: n - lag] * data[lag:])

    return ac

def test_autocorrelation_matches_reference():
    """FFT autocorrelation matches the direct implementation."""
    rng = np.random.default_rng(12345)
    data = rng.random(100)

    ac_fft = autocorrelation_function(
        data,
        use_wiener_khinchin=True,
    )

    ac_ref = reference_autocorrelation(data)

    np.testing.assert_allclose(
        ac_fft,
        ac_ref,
        rtol=1e-12,
        atol=1e-12,
    )

def test_wiener_khinchin_matches_legacy():
    """Wiener-Khinchin and legacy implementations give identical results."""
    rng = np.random.default_rng(12345)
    data = rng.random(100)

    ac_wk = autocorrelation_function(
        data,
        use_wiener_khinchin=True,
    )

    ac_legacy = autocorrelation_function(
        data,
        use_wiener_khinchin=False,
    )

    np.testing.assert_allclose(
        ac_wk,
        ac_legacy,
        rtol=1e-12,
        atol=1e-12,
    )

def test_autocorrelation_of_constant_signal():
    """Constant signal has constant autocorrelation."""
    data = np.ones(100)

    ac = autocorrelation_function(data, use_wiener_khinchin=True)

    np.testing.assert_allclose(ac, 1.0)

def test_autocorrelation_at_zero_lag():
    """Zero-lag autocorrelation equals the mean square value."""
    rng = np.random.default_rng(12345)
    data = rng.random(100)

    ac = autocorrelation_function(data, use_wiener_khinchin=True)

    assert np.isclose(ac[0], np.mean(data**2))

def test_autocorrelation_of_sine_wave():
    """Autocorrelation of a sine wave is periodic."""
    t = np.linspace(0, 10 * np.pi, 1000)
    data = np.sin(t)

    ac = autocorrelation_function(data, use_wiener_khinchin=True)

    assert ac[0] > 0
    assert ac[100] < ac[0]

def test_autocorrelation_of_constant_signal():
    """Constant signal has constant autocorrelation."""
    data = np.ones(100)

    ac = autocorrelation_function(data, use_wiener_khinchin=False)

    np.testing.assert_allclose(ac, 1.0)

def test_autocorrelation_at_zero_lag():
    """Zero-lag autocorrelation equals the mean square value."""
    rng = np.random.default_rng(12345)
    data = rng.random(100)

    ac = autocorrelation_function(data, use_wiener_khinchin=False)

    assert np.isclose(ac[0], np.mean(data**2))

def test_autocorrelation_of_sine_wave():
    """Autocorrelation of a sine wave is periodic."""
    t = np.linspace(0, 10 * np.pi, 1000)
    data = np.sin(t)

    ac = autocorrelation_function(data, use_wiener_khinchin=False)

    assert ac[0] > 0
    assert ac[100] < ac[0]

def test_calculate_tau_oneD_analytic():
    """One-dimensional analytic τ = 0.5 * J(0) / g(0)."""
    J = np.array([2.0])
    gij = np.array([1.0])

    tau = calculate_tau(J, gij, dim=1, oneDarray=True)

    expected = 0.5 * 2.0 / 1.0 / cst.pico

    np.testing.assert_allclose(tau[0], expected)

def test_calculate_tau_oneD_integral():
    """Integral definition matches expected area for constant function."""
    t = np.linspace(0, 10, 100)
    gij = np.ones_like(t)
    J = np.array([1.0])

    tau = calculate_tau(J, gij, dim=1, integral=True, t=t, oneDarray=True)

    expected = np.trapezoid(gij, t) / gij[0] / cst.pico

    np.testing.assert_allclose(tau[0], expected)

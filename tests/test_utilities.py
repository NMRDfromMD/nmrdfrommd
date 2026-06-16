from scipy import constants as cst
import numpy as np
import pytest

from nmrdfrommd.utilities import find_nearest, calculate_tau, compute_rij, \
    cartesian_to_spherical


def test_find_nearest_basic():
    """Returns index of closest value in simple case."""
    data = np.array([0, 2, 5, 10])

    idx = find_nearest(data, 6)

    assert idx == 2  # 5 is closest to 6

def test_find_nearest_tie():
    """In a tie, returns the first occurrence."""
    data = np.array([1, 4, 6, 10])

    idx = find_nearest(data, 5)  # equally close to 4 and 6

    assert idx == 1  # 4 is chosen (first match)

def test_find_nearest_invalid_shape():
    """Raises error if input is not 1D."""
    data = np.array([[1, 2], [3, 4]])

    with pytest.raises(ValueError):
        find_nearest(data, 2)

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

def test_compute_rij_no_pbc():
    """Compute direct displacement vector without periodic boundary conditions."""
    pos_i = np.array([1.0, 2.0, 3.0])
    pos_j = np.array([0.5, 1.0, 2.0])
    box = np.array([10.0, 10.0, 10.0])

    rij = compute_rij(pos_i, pos_j, box, pbc=False)

    np.testing.assert_allclose(rij, [0.5, 1.0, 1.0])

def test_compute_rij_pbc_wrap():
    """Verify minimum-image convention correctly wraps distances under PBC."""
    pos_i = np.array([9.8, 0.2, 0.2])
    pos_j = np.array([0.2, 0.2, 0.2])
    box = np.array([10.0, 10.0, 10.0])

    rij = compute_rij(pos_i, pos_j, box, pbc=True)

    # should wrap to -0.4 in x direction
    np.testing.assert_allclose(rij[0], -0.4)

def test_compute_rij_antisymmetry():
    """Ensure displacement vector is antisymmetric: r_ij = -r_ji."""
    pos_i = np.random.rand(3)
    pos_j = np.random.rand(3)
    box = np.array([10.0, 10.0, 10.0])

    rij = compute_rij(pos_i, pos_j, box, pbc=True)
    rji = compute_rij(pos_j, pos_i, box, pbc=True)

    np.testing.assert_allclose(rij, -rji)

def test_cartesian_to_spherical_basic():
    """Convert a Cartesian unit vector along x-axis to spherical coordinates."""
    rij = np.array([1.0, 0.0, 0.0])

    r, theta, phi = cartesian_to_spherical(rij)

    assert np.isclose(r, 1.0)
    assert np.isclose(theta, np.pi / 2)
    assert np.isclose(phi, 0.0)


def test_cartesian_to_spherical_batch():
    """Ensure function supports batched (N, 3) Cartesian input."""
    rij = np.array([[1, 0, 0],
                    [0, 1, 0]])

    r, theta, phi = cartesian_to_spherical(rij)

    assert r.shape == (2,)


def test_cartesian_to_spherical_zero():
    """Handle zero vector without crashing and return zero radius."""
    rij = np.array([0.0, 0.0, 0.0])

    r, theta, phi = cartesian_to_spherical(rij)

    assert np.isclose(r, 0.0)
from scipy import constants as cst
import numpy as np
import pytest

from nmrdfrommd.geometry import compute_rij, cartesian_to_spherical, spherical_harmonic_kernel


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

# def test_spherical_harmonic_kernel_basic():
#     """Kernel returns finite, correctly shaped output for simple input."""
#     r = 2.0
#     theta = np.pi / 2
#     phi = 0.0

#     alpha_m = {0: 1.0}

#     out = spherical_harmonic_kernel(r, theta, phi, alpha_m, isotropic=True)

#     assert out.shape == (1,)
#     assert np.isfinite(out[0])

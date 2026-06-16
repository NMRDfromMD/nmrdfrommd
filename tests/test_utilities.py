from scipy import constants as cst
import numpy as np
import pytest

from nmrdfrommd.utilities import find_nearest, spherical_harmonic_kernel


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

def test_spherical_harmonic_kernel_basic():
    """Kernel returns finite, correctly shaped output for simple input."""
    r = 2.0
    theta = np.pi / 2
    phi = 0.0

    alpha_m = {0: 1.0}

    out = spherical_harmonic_kernel(r, theta, phi, alpha_m, isotropic=True)

    assert out.shape == (1,)
    assert np.isfinite(out[0])


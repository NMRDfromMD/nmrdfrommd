from scipy import constants as cst
import numpy as np
import pytest

from nmrdfrommd.utilities import find_nearest


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

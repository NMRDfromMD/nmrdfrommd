import pytest
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader


@pytest.fixture
def universe_small():
    coords = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 0], [1.1, 0, 0], [0, 1.1, 0]],
    ])

    u = mda.Universe.empty(3, trajectory=True)
    u.add_TopologyAttr('name', ['H', 'H', 'H'])

    u.load_new(coords, format=MemoryReader)

    ag = u.atoms

    return u, ag

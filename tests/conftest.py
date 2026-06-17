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

@pytest.fixture
def twomol_universe():
    data = "datafiles/twomolecules/twomolecules.data"
    trj = "datafiles/twomolecules/twomolecules.xtc"
    u = mda.Universe(data, trj)
    group_H2O = u.select_atoms("type 2") # pick one H per molecule
    return u, group_H2O
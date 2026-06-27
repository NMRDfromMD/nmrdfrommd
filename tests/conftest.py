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

@pytest.fixture
def two_molecule_system():
    """
    3 atoms in residue 1 (atoms 0-2), 3 atoms in residue 2 (atoms 3-5)
    Molecule 1: H H O
    Molecule 2: H H O
    """
    resids = np.array([1, 1, 1, 2, 2, 2])
    indices = np.array([0, 1, 2, 3, 4, 5])
    atom_types = np.array(["H", "H", "O", "H", "H", "O"])
    return resids, indices, atom_types

from pathlib import Path

import pytest
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

TWOMOL_DIR = Path(__file__).parent / "datafiles" / "twomolecules"


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
def twomol_files():
    """Paths to the two-molecule LAMMPS data file and XTC trajectory."""
    return TWOMOL_DIR / "twomolecules.data", TWOMOL_DIR / "twomolecules.xtc"

@pytest.fixture
def twomol_universe(twomol_files):
    u = mda.Universe(*twomol_files)
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

RIGID_DIPOLE_DISTANCE = 2.0  # Angstrom

@pytest.fixture
def rigid_dipole_universe():
    """Two spins at a fixed distance along z, frozen over the whole trajectory.

    The dipole never moves, so the m=0 correlation function is a constant
    equal to (3 cos^2(theta) - 1)^2 / r^6 = 4 / r^6 (theta = 0).
    """
    n_frames = 64
    coords = np.zeros((n_frames, 2, 3), dtype=np.float32)
    coords[:, 0] = [25.0, 25.0, 25.0]
    coords[:, 1] = [25.0, 25.0, 25.0 + RIGID_DIPOLE_DISTANCE]

    u = mda.Universe.empty(2, n_residues=1, atom_resindex=[0, 0],
                           trajectory=True)
    u.add_TopologyAttr("name", ["H", "H"])
    u.add_TopologyAttr("resid", [1])
    u.load_new(coords, format=MemoryReader, dt=1.0,
               dimensions=[50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
    return u, u.atoms, RIGID_DIPOLE_DISTANCE

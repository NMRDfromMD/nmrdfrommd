import pytest
import MDAnalysis as mda
import numpy as np

from nmrdfrommd import NMRD


def test_verify_entry_valid(universe_small):
    u, ag = universe_small

    nmr = NMRD(u=u, atom_group=ag)
    nmr._verify_entry()  # should NOT raise

def test_invalid_type_analysis(universe_small):
    u, ag = universe_small

    nmr = NMRD(
        u=u,
        atom_group=ag,
        type_analysis="not_valid"
    )

    with pytest.raises(ValueError, match="Invalid type_analysis"):
        nmr._verify_entry()

def test_empty_atom_group(universe_small):
    u, ag = universe_small

    empty_ag = ag[:0]

    nmr = NMRD(
        u=u,
        atom_group=empty_ag
    )

    with pytest.raises(ValueError, match="atom_group"):
        nmr._verify_entry()

def test_empty_neighbor_group(universe_small):
    u, ag = universe_small

    empty_ag = ag[:0]

    nmr = NMRD(
        u=u,
        atom_group=ag,
        neighbor_group=empty_ag
    )

    with pytest.raises(ValueError, match="neighbor_group"):
        nmr._verify_entry()

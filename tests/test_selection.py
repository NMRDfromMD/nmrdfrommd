import numpy as np
import pytest

from nmrdfrommd.selection import select_neighbor_indices


def test_full_excludes_only_self(two_molecule_system):
    """'full' should return every atom except i_idx itself."""
    resids, indices = two_molecule_system
    index_j = select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="full")
    np.testing.assert_array_equal(index_j, [1, 2, 3, 4, 5])

def test_intra_molecular_keeps_same_residue_excluding_self(two_molecule_system):
    """'intra_molecular' should keep atoms with the same resid, excluding i_idx."""
    resids, indices = two_molecule_system
    index_j = select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="intra_molecular")
    np.testing.assert_array_equal(index_j, [1, 2])

def test_inter_molecular_keeps_other_residues(two_molecule_system):
    """'inter_molecular' should keep atoms from a different residue (i is
    excluded automatically since its own resid matches res_id_i)."""
    resids, indices = two_molecule_system
    index_j = select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="inter_molecular")
    np.testing.assert_array_equal(index_j, [3, 4, 5])

def test_empty_selection_raises_value_error():
    """A molecule with no other atoms should raise, not silently return empty."""
    resids = np.array([1, 2, 2, 2])
    indices = np.array([0, 1, 2, 3])

    with pytest.raises(ValueError):
        select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="intra_molecular")

def test_unknown_type_analysis_raises_key_error(two_molecule_system):
    """An unsupported type_analysis should fail loudly; validity is normally
    enforced upstream in _verify_entry, this just documents the contract."""
    resids, indices = two_molecule_system
    with pytest.raises(KeyError):
        select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="bogus")

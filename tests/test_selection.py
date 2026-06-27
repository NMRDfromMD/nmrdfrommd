import numpy as np
import pytest

from nmrdfrommd.selection import select_neighbor_indices


def test_full_excludes_only_self(two_molecule_system):
    """'full' should return every atom except i_idx itself."""
    resids, indices, _ = two_molecule_system
    index_j = select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="full")
    np.testing.assert_array_equal(index_j, [1, 2, 3, 4, 5])

def test_intra_molecular_keeps_same_residue_excluding_self(two_molecule_system):
    """'intra_molecular' should keep atoms with the same resid, excluding i_idx."""
    resids, indices, _ = two_molecule_system
    index_j = select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="intra_molecular")
    np.testing.assert_array_equal(index_j, [1, 2])

def test_inter_molecular_keeps_other_residues(two_molecule_system):
    """'inter_molecular' should keep atoms from a different residue (i is
    excluded automatically since its own resid matches res_id_i)."""
    resids, indices, _ = two_molecule_system
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
    resids, indices, _ = two_molecule_system
    with pytest.raises(KeyError):
        select_neighbor_indices(resids, indices, res_id_i=1, i_idx=0, type_analysis="bogus")

def test_intra_molecular_two_hydrogens_returns_one(two_molecule_system):
    """
    In a system where a residue contains two hydrogen atoms,
    selecting intra_molecular for one hydrogen should return exactly
    one other hydrogen, and only hydrogens.
    """
    resids, indices, atom_types = two_molecule_system

    # focus on hydrogen atoms only
    h_mask = (atom_types == "H")
    h_indices = indices[h_mask]
    h_resids = resids[h_mask]
    h_types = atom_types[h_mask]

    # pick first hydrogen in residue 1
    i_idx = h_indices[0]

    index_j = select_neighbor_indices(
        h_resids,
        h_indices,
        res_id_i=1,
        i_idx=i_idx,
        type_analysis="intra_molecular",
    )

    # ---- CRITICAL CHECKS ----

    # 1. exactly one neighbor
    assert len(index_j) == 1

    # 2. it must be a hydrogen (this is the key part you asked for)
    selected_types = h_types[np.isin(h_indices, index_j)]
    assert np.all(selected_types == "H")

    # 3. and it must be the correct hydrogen
    np.testing.assert_array_equal(index_j, [h_indices[1]])

def test_intra_molecular_hydrogen_other_molecule(two_molecule_system):
    """
    In a system with two identical molecules (H H O),
    selecting intra_molecular neighbors for a hydrogen in residue 1
    should return ONLY hydrogen atoms from the same residue,
    and NOT include oxygen or atoms from the other residue.
    """
    resids, indices, atom_types = two_molecule_system

    # focus on hydrogen atoms only
    h_mask = (atom_types == "H")
    h_indices = indices[h_mask]
    h_resids = resids[h_mask]
    h_types = atom_types[h_mask]

    # pick first hydrogen in residue 1
    i_global = h_indices[0]

    index_j = select_neighbor_indices(
        h_resids,
        h_indices,
        res_id_i=1,
        i_idx=i_global,
        type_analysis="intra_molecular",
    )

    # ---- ASSERTIONS ----

    # 1. must return exactly one neighbor
    assert len(index_j) == 1

    # 2. must be hydrogen ONLY
    selected_types = h_types[np.isin(h_indices, index_j)]
    assert np.all(selected_types == "H")

    # 3. must be from same residue (not other molecule)
    selected_resids = h_resids[np.isin(h_indices, index_j)]
    assert np.all(selected_resids == 1)

    # 4. must be the correct hydrogen partner
    np.testing.assert_array_equal(index_j, [h_indices[1]])

def test_inter_molecular_hydrogen_selects_other_molecule(two_molecule_system):
    """
    In a system with two identical molecules (H H O),
    selecting inter_molecular neighbors for a hydrogen in residue 1
    should return ONLY hydrogen atoms from residue 2.
    """
    resids, indices, atom_types = two_molecule_system

    # restrict to hydrogen only
    h_mask = (atom_types == "H")
    h_indices = indices[h_mask]
    h_resids = resids[h_mask]
    h_types = atom_types[h_mask]

    # pick first hydrogen in residue 1
    i_global = h_indices[0]

    index_j = select_neighbor_indices(
        h_resids,
        h_indices,
        res_id_i=1,
        i_idx=i_global,
        type_analysis="inter_molecular",
    )

    # ---- ASSERTIONS ----

    # 1. should return exactly 2 hydrogens (from other molecule)
    assert len(index_j) == 2

    # 2. must be hydrogens only
    selected_types = h_types[np.isin(h_indices, index_j)]
    assert np.all(selected_types == "H")

    # 3. must come from the OTHER residue (residue 2)
    selected_resids = h_resids[np.isin(h_indices, index_j)]
    assert np.all(selected_resids == 2)

    # 4. must NOT include any atom from same residue
    assert not np.any(selected_resids == 1)

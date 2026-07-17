#!/usr/bin/env python3
"""Test file for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2025 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import MDAnalysis as mda
import numpy as np

from nmrdfrommd import NMRD


@pytest.fixture
def twomol_universe():
    data = "datafiles/twomolecules/twomolecules.data"
    trj = "datafiles/twomolecules/twomolecules.xtc"
    return mda.Universe(data, trj)

def test_nmr_initialization(twomol_universe):
    # Use first two atoms for both target and neighbor groups
    atom_group = twomol_universe.atoms[:2]
    
    nmr = NMRD(
        u=twomol_universe,
        atom_group=atom_group,
        neighbor_group=atom_group,
        type_analysis='full',
        number_i=1,
        isotropic=True,
        frame_interval=0.02,
        hydrogen_per_atom=1.0,
        spin=0.5,
        pbc=True,
        num_log_points=50
    )
    
    # Assert attributes are correctly set
    assert nmr.u == twomol_universe
    assert nmr.atom_group == atom_group
    assert nmr.neighbor_group == atom_group
    assert nmr.type_analysis == 'full'
    assert nmr.number_i == 1
    assert nmr.isotropic is True
    assert np.isclose(nmr.frame_interval, 0.02)
    assert np.isclose(nmr.hydrogen_per_atom, 1.0)
    assert np.isclose(nmr.spin, 0.5)
    assert nmr.pbc is True
    assert nmr.num_log_points == 50
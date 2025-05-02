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
    u = mda.Universe(data, trj)
    group_H2O = u.select_atoms("type 2")
    return u, group_H2O

def test_shape_array(twomol_universe):
    """Assert that the data matrix dimension is consistent."""
    u, group_H2O = twomol_universe
    n_frames = u.trajectory.n_frames
    n_fft = len(np.arange(np.int32(n_frames / 2)+1))

    nmrd_iso = NMRD(
        u=u,
        atom_group=group_H2O,
        isotropic=True)
    nmrd_iso.run_analysis()

    assert nmrd_iso.data.shape == (1, n_frames, 1)
    assert nmrd_iso.gij.shape == (1, n_frames)
    assert nmrd_iso.t.shape == (n_frames,)
    assert nmrd_iso.J[0].shape == (n_fft,)
    assert nmrd_iso.f.shape == (n_fft,)
    assert nmrd_iso.R1.shape == (n_fft,)
    assert nmrd_iso.R2.shape == (n_fft,)
    
    nmrd_aniso = NMRD(
        u=u,
        atom_group=group_H2O,
        isotropic=False)
    nmrd_aniso.run_analysis()

    assert nmrd_aniso.data.shape == (3, n_frames, 1)
    assert nmrd_aniso.gij.shape == (3, n_frames)
    assert nmrd_aniso.J.shape == (3, n_fft)
    assert nmrd_aniso.t.shape == (n_frames,)
    assert nmrd_aniso.f.shape == (n_fft,)
    assert nmrd_aniso.R1.shape == (n_fft,)
    assert nmrd_aniso.R2.shape == (n_fft,)
    assert nmrd_aniso.gij.T[0].shape == (3,)

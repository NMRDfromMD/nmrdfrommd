#!/usr/bin/env python3
"""Test file for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2025 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from scipy import constants as cst
import MDAnalysis as mda
import numpy as np
import pytest

from nmrdfrommd import NMRD
from nmrdfrommd.geometry import compute_rij, cartesian_to_spherical


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
    assert nmrd_iso.results["gij"].shape == (1, n_frames)
    assert nmrd_iso.results["t"].shape == (n_frames,)
    assert nmrd_iso.results["J"][0].shape == (n_fft,)
    assert nmrd_iso.results["f"].shape == (n_fft,)
    assert nmrd_iso.results["R1"].shape == (n_fft,)
    assert nmrd_iso.results["R2"].shape == (n_fft,)
    
    nmrd_aniso = NMRD(
        u=u,
        atom_group=group_H2O,
        isotropic=False)
    nmrd_aniso.run_analysis()

    assert nmrd_aniso.data.shape == (3, n_frames, 1)
    assert nmrd_aniso.results["gij"].shape == (3, n_frames)
    assert nmrd_aniso.results["t"].shape == (n_frames,)
    assert nmrd_aniso.results["J"].shape == (3, n_fft,)
    assert nmrd_aniso.results["f"].shape == (n_fft,)
    assert nmrd_aniso.results["R1"].shape == (n_fft,)
    assert nmrd_aniso.results["R2"].shape == (n_fft,)
    assert nmrd_aniso.results["gij"].T[0].shape == (3, )

def test_distance(twomol_universe):
    """Assert that the calculated distance is correct."""
    u, group_H2O = twomol_universe

    nmrd_pbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=True)
    nmrd_pbc.run_analysis()

    rij = compute_rij(
        nmrd_pbc.position_i,
        nmrd_pbc.position_j,
        nmrd_pbc.box,
        nmrd_pbc.pbc)[0]
    
    assert np.isclose(rij[0], -8.0)
    assert np.isclose(rij[1], 0.0)
    assert np.isclose(rij[2], 0.0)

    nmrd_nopbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=False)
    nmrd_nopbc.run_analysis()

    rij = compute_rij(
        nmrd_nopbc.position_i,
        nmrd_nopbc.position_j,
        nmrd_nopbc.box,
        nmrd_nopbc.pbc)[0]

    assert np.isclose(rij[0], 10.0)
    assert np.isclose(rij[1], 0.0)
    assert np.isclose(rij[2], 0.0)

def test_distance(twomol_universe):
    """Assert that the calculated spherical functions are correct."""
    u, group_H2O = twomol_universe

    nmrd_pbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=True)
    nmrd_pbc.run_analysis()

    rij = compute_rij(
        nmrd_pbc.position_i,
        nmrd_pbc.position_j,
        nmrd_pbc.box,
        nmrd_pbc.pbc)[0]
    r, theta, phi = cartesian_to_spherical(rij)

    assert np.isclose(r, 8.0)
    assert np.isclose(theta, np.pi/2)
    assert np.isclose(phi, np.pi)

    nmrd_nopbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=False)
    nmrd_nopbc.run_analysis()

    rij = compute_rij(
        nmrd_nopbc.position_i,
        nmrd_nopbc.position_j,
        nmrd_nopbc.box,
        nmrd_nopbc.pbc)[0]
    r, theta, phi = cartesian_to_spherical(rij)

    assert np.isclose(r, 10.0)
    assert np.isclose(theta, np.pi/2)
    assert np.isclose(phi, 0)

def test_correlation(twomol_universe):
    """Assert that the calculated correlation is correct."""
    u, group_H2O = twomol_universe

    d_atoms = 8.0
    expected_gij = np.float32(np.round(1/d_atoms**6, 7))

    nmrd_pbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=True)
    nmrd_pbc.run_analysis()

    assert np.float32(np.round(nmrd_pbc.results["gij"][0][0], 7)) == expected_gij

    nmrd_noniso = NMRD(
        u=u,
        atom_group=group_H2O,
        isotropic=False,
        pbc=True)
    nmrd_noniso.run_analysis()

    assert np.float32(np.round(nmrd_noniso.results["gij"][0][0], 7)) == expected_gij

    d_atoms = 10.0
    expected_gij = np.float32(np.round(1/d_atoms**6, 7))

    nmrd_nopbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=False)
    nmrd_nopbc.run_analysis()

    assert np.float32(np.round(nmrd_nopbc.results["gij"][0][0], 7)) == expected_gij

def test_R1(twomol_universe):
    """Assert that the calculated value of R1 is correct."""
    u, group_H2O = twomol_universe

    nmrd_pbc = NMRD(
        u=u,
        atom_group=group_H2O,
        pbc=True)
    nmrd_pbc.run_analysis()

    # expected prefactor
    GAMMA = 267512897.63847807
    spin = 1/2
    K = ((3 / 2) * (cst.mu_0 / (4 * np.pi)) ** 2 *
            cst.hbar ** 2 * GAMMA ** 4 * spin * (1 + spin))
    prefactor = K / cst.angstrom ** 6

    dt = (nmrd_pbc.results["t"][1]-nmrd_pbc.results["t"][0]) * cst.pico

    # expected J
    d_atoms = 8.0
    expected_gij = np.float32(np.round(1/d_atoms**6, 7))
    J0_0 = expected_gij * dt * 2 * len(nmrd_pbc.results["t"]) # J in freq 0

    # expected R1
    R1_0 = prefactor * (J0_0 + 4 * J0_0) / 6

    assert np.float32(np.round(nmrd_pbc.results["R1"][0], 9)) == np.float32(np.round(R1_0, 9))

    # expected R2
    R2_0 = prefactor * (3/2 * J0_0 + (5/2) * J0_0 + J0_0) / 6

    assert np.float32(np.round(nmrd_pbc.results["R2"][0], 9)) == np.float32(np.round(R2_0, 9))

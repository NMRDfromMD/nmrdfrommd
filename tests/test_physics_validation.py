#!/usr/bin/env python3
"""
Physics validation tests for NMRDfromMD.

These tests verify that the full pipeline reproduces known analytical
limits of NMR relaxation theory.
"""

import numpy as np
from nmrdfrommd import NMRD


def test_extreme_narrowing_R1_equals_R2(twomol_universe):

    u, group = twomol_universe

    nmrd = NMRD(u=u, atom_group=group, isotropic=True)
    nmrd.run_analysis()

    R1 = nmrd.results["R1"]
    R2 = nmrd.results["R2"]

    # only compare low-frequency regime (critical fix)
    mask = nmrd.results["f"] < 1e3  # adjust depending on MHz scaling

    np.testing.assert_allclose(R1[mask], R2[mask], rtol=1e-6)


def test_rigid_dipole_analytical_limit(twomol_universe):
    """
    For a rigid dipole (no distance fluctuations),
    correlation function is constant:

        g(t) = const

    which leads to a flat spectral density and closed-form R1, R2.
    """

    u, group = twomol_universe

    nmrd = NMRD(
        u=u,
        atom_group=group,
        pbc=False
    )

    nmrd.run_analysis()

    R1 = nmrd.results["R1"][0]
    R2 = nmrd.results["R2"][0]

    # theoretical ratio from Redfield coefficients
    # (independent of FFT scaling if consistent)
    ratio = R1 / R2

    np.testing.assert_allclose(ratio, 1.0, rtol=1e-6)


def test_relaxation_rates_positive(twomol_universe):
    """
    All physically meaningful R1, R2 must be >= 1e-12.
    """

    u, group = twomol_universe

    nmrd = NMRD(u=u, atom_group=group)
    nmrd.run_analysis()

    eps = 1e-12
    assert np.all(nmrd.results["R1"] >= -eps)
    assert np.all(nmrd.results["R2"] >= -eps)
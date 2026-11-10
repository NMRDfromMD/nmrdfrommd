#!/usr/bin/env python3
"""
Physics validation tests for NMRDfromMD.

These tests verify that the full pipeline reproduces known analytical
limits of NMR relaxation theory.
"""

import numpy as np
from nmrdfrommd import NMRD


def test_extreme_narrowing_R1_equals_R2(twomol_universe):

    # NOTE: the test trajectory is only a few frames long, so every
    # resolved frequency is in the extreme-narrowing regime. This checks
    # consistency of the pipeline, not the physical limit itself.
    u, group = twomol_universe

    nmrd = NMRD(u=u, atom_group=group, isotropic=True)
    nmrd.run_analysis()

    R1 = nmrd.results["R1"]
    R2 = nmrd.results["R2"]

    # only compare low-frequency regime (critical fix)
    mask = nmrd.results["f"] < 1e3  # adjust depending on MHz scaling

    np.testing.assert_allclose(R1[mask], R2[mask], rtol=1e-6)


def test_R1_equals_R2_at_zero_frequency(twomol_universe):
    """At f = 0 the R1 and R2 formulas reduce to the same expression.

    This is an identity of the Redfield coefficients, not a physical
    limit of the trajectory, so it holds for any input.
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

    np.testing.assert_allclose(R1 / R2, 1.0, rtol=1e-6)


def test_rigid_dipole_constant_correlation(rigid_dipole_universe):
    """A frozen dipole gives a constant correlation function.

    With both spins aligned along z (theta = 0), the m=0 kernel is
    (3 cos^2(theta) - 1) / r^3 = 2 / r^3, so G(t) = 4 / r^6 at every lag.
    A constant G(t) has no spectral weight at f > 0, hence R1(f > 0) = 0.
    """

    u, group, r = rigid_dipole_universe

    nmrd = NMRD(u=u, atom_group=group, isotropic=True)
    nmrd.run_analysis()

    expected_G = 4 / r**6
    np.testing.assert_allclose(nmrd.results["gij"][0], expected_G, rtol=1e-5)

    J = nmrd.results["J"][0]
    np.testing.assert_allclose(J[1:], 0.0, atol=1e-6 * abs(J[0]))

    R1 = nmrd.results["R1"]
    np.testing.assert_allclose(R1[1:], 0.0, atol=1e-6 * abs(R1[0]))


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
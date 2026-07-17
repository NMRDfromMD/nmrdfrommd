#!/usr/bin/env python3
"""Relaxation functions for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import constants as cst
from scipy.interpolate import interp1d

from .utilities import find_nearest
from .units import ANGSTROM_TO_M


def compute_relaxation_rates(f, J, K, isotropic):
    """Compute relaxation rates R1 and R2 from spectral density J(f).

    Based on dipolar (BPP) relaxation expressions for spin-1/2 nuclei.

    Parameters
    ----------
    f : np.ndarray
        Frequency array, in MHz.
    J : np.ndarray
        Spectral density, shape (dim, n_freqs), in Angstrom^-6 * s;
        dim=1 if isotropic, else 3.
    K : float
        Dipolar relaxation prefactor, in m^6 * s^-2 (SI).
    isotropic : bool
        Whether to use the isotropic (dim=1) or anisotropic (dim=3) formula.

    Returns
    -------
    R1, R2 : np.ndarray
        Relaxation rates on the same frequency grid as f.
    """
    prefactor = K / ANGSTROM_TO_M ** 6  # Angstrom^6 * s^-2 (K rescaled m^6 -> Angstrom^6 to match J)
    idx0 = find_nearest(f, 0.0)         # index, dimensionless
    J0 = J[0]                           # Angstrom^-6 * s

    if isotropic:
        J02 = interp1d(f, J0, fill_value="extrapolate")(2 * f)            # Angstrom^-6 * s
        R1 = prefactor * (J0 + 4 * J02) / 6                               # s^-1
        R2 = prefactor * (3 / 2 * J0[idx0] + 5 / 2 * J0 + J02) / 6        # s^-1
    else:
        J1 = J[1]                                                         # Angstrom^-6 * s
        J2 = interp1d(f, J[2], fill_value="extrapolate")(2 * f)           # Angstrom^-6 * s
        R1 = prefactor * (J1 + J2)                                        # s^-1
        R2 = prefactor * (J0[idx0] + 10 * J1 + J2) / 4                    # s^-1

    return R1, R2  # s^-1, s^-1

def compute_relaxation_times(f, R1, R2, R1_err=None, R2_err=None):
    """Compute T1/T2 relaxation times and uncertainties.

    Parameters
    ----------
    f : np.ndarray
        Frequency array, in MHz.
    R1, R2 : np.ndarray
        Relaxation rates, in s^-1.
    R1_err, R2_err : np.ndarray, optional
        Uncertainty on relaxation rates, in s^-1.

    Returns
    -------
    T1, T2 : np.ndarray
        Relaxation times, in s.
    T1_err, T2_err : np.ndarray
        Uncertainties on relaxation times, in s.
    """

    T1 = np.full_like(R1, np.inf, dtype=float)
    T2 = np.full_like(R2, np.inf, dtype=float)

    mask1 = ~np.isclose(R1, 0.0)
    mask2 = ~np.isclose(R2, 0.0)

    T1[mask1] = 1.0 / R1[mask1]
    T2[mask2] = 1.0 / R2[mask2]

    T1_err = None
    T2_err = None

    if R1_err is not None:
        T1_err = np.zeros_like(R1)
        T1_err[mask1] = R1_err[mask1] / R1[mask1]**2

    if R2_err is not None:
        T2_err = np.zeros_like(R2)
        T2_err[mask2] = R2_err[mask2] / R2[mask2]**2

    return T1, T2, T1_err, T2_err
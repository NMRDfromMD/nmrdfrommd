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

def compute_relaxation_times(f, R1, R2, target_frequency=None):
    """Compute T1 and T2 relaxation times from relaxation rates R1, R2.

    Values are evaluated at the frequency closest to target_frequency
    (or to zero, if target_frequency is None). Protects against
    divide-by-zero when a rate is numerically zero.

    Parameters
    ----------
    f : np.ndarray
        Frequency array, in MHz.
    R1 : np.ndarray
        Longitudinal relaxation rate, in s^-1, same shape as f.
    R2 : np.ndarray
        Transverse relaxation rate, in s^-1, same shape as f.
    target_frequency : float, optional
        Frequency at which to evaluate T1/T2, in MHz. If None, f=0 is used.

    Returns
    -------
    T1, T2 : float
        Relaxation times, in s. np.inf if the corresponding rate is ~0.
    """
    target = 0.0 if target_frequency is None else target_frequency  # MHz
    idx = find_nearest(f, target)                                   # index, dimensionless

    R1_val = R1[idx]  # s^-1
    R2_val = R2[idx]  # s^-1

    eps = 1e-20  # s^-1 (safety floor, sized for R1/R2 ~ 0.1-1e4 s^-1)
    T1 = np.inf if np.isclose(R1_val, 0.0) else 1.0 / (R1_val + eps)  # s
    T2 = np.inf if np.isclose(R2_val, 0.0) else 1.0 / (R2_val + eps)  # s

    return T1, T2  # s, s

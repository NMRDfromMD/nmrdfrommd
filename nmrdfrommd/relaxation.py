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


def compute_relaxation_rates(f, J, K, isotropic):
    """Compute relaxation rates R1 and R2 from spectral density J(f).

    Based on dipolar (BPP) relaxation expressions for spin-1/2 nuclei.

    Parameters
    ----------
    f : np.ndarray
        Frequency array.
    J : np.ndarray
        Spectral density, shape (dim, n_freqs); dim=1 if isotropic, else 3.
    K : float
        Dipolar relaxation prefactor.
    isotropic : bool
        Whether to use the isotropic (dim=1) or anisotropic (dim=3) formula.

    Returns
    -------
    R1, R2 : np.ndarray
        Relaxation rates on the same frequency grid as f.
    """
    prefactor = K / cst.angstrom ** 6
    idx0 = find_nearest(f, 0.0)
    J0 = J[0]

    if isotropic:
        J02 = interp1d(f, J0, fill_value="extrapolate")(2 * f)
        R1 = prefactor * (J0 + 4 * J02) / 6
        R2 = prefactor * (3 / 2 * J0[idx0] + 5 / 2 * J0 + J02) / 6
    else:
        J1 = J[1]
        J2 = interp1d(f, J[2], fill_value="extrapolate")(2 * f)
        R1 = prefactor * (J1 + J2)
        R2 = prefactor * (1 / 4) * (J0[idx0] + 10 * J1 + J2)

    return R1, R2

def compute_relaxation_times(f, R1, R2, target_frequency=None):
    """Compute T1 and T2 relaxation times from relaxation rates R1, R2.

    Values are evaluated at the frequency closest to target_frequency
    (or to zero, if target_frequency is None). Protects against
    divide-by-zero when a rate is numerically zero.

    Parameters
    ----------
    f : np.ndarray
        Frequency array.
    R1 : np.ndarray
        Longitudinal relaxation rate, same shape as f.
    R2 : np.ndarray
        Transverse relaxation rate, same shape as f.
    target_frequency : float, optional
        Frequency at which to evaluate T1/T2. If None, f=0 is used.

    Returns
    -------
    T1, T2 : float
        Relaxation times. np.inf if the corresponding rate is ~0.
    """
    target = 0.0 if target_frequency is None else target_frequency
    idx = find_nearest(f, target)

    R1_val = R1[idx]
    R2_val = R2[idx]

    eps = 1e-20
    T1 = np.inf if np.isclose(R1_val, 0.0) else 1.0 / (R1_val + eps)
    T2 = np.inf if np.isclose(R2_val, 0.0) else 1.0 / (R2_val + eps)

    return T1, T2

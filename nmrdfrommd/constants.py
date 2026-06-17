#!/usr/bin/env python3
"""Constants for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import constants as cst


ALPHA_M = (
    np.sqrt(16 * np.pi / 5),
    np.sqrt(8 * np.pi / 15),
    np.sqrt(32 * np.pi / 15),
)

# Gyromagnetic ratios in rad/s/T
GYROMAGNETIC_RATIOS = {
    "H": 2 * np.pi * 42.576e6,
    "C": 2 * np.pi * 10.705e6,
    "N": 2 * np.pi * -4.316e6,
    "F": 2 * np.pi * 40.053e6,
    "P": 2 * np.pi * 17.235e6,
}

def get_gyromagnetic_ratio(atom: str) -> float:
    """Return the gyromagnetic ratio (rad/s/T) of a given atom symbol."""
    try:
        return GYROMAGNETIC_RATIOS[atom.upper()]
    except KeyError:
        raise ValueError(f"Unknown atom type '{atom}'. Add it to GYROMAGNETIC_RATIOS.")
    
def dipolar_prefactor(gamma, spin):
    """
    Return dipolar relaxation prefactor K.

    Parameters
    ----------
    gamma : float
        Gyromagnetic ratio (rad s⁻¹ T⁻¹).
    spin : float
        Nuclear spin quantum number.

    Returns
    -------
    float
        Dipolar prefactor K (m⁶ s⁻²).
    """
    return (
        (3 / 2)
        * (cst.mu_0 / (4 * np.pi)) ** 2 * cst.hbar**2
        * gamma**4 * spin * (spin + 1)
    )

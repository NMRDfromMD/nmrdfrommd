#!/usr/bin/env python3
"""Geometry for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.special import sph_harm_y, sph_harm


def compute_rij(pos_i, pos_j, box, pbc=True):
    """
    Compute displacement vector r_ij = r_i - r_j with optional periodic boundary conditions.

    Parameters
    ----------
    pos_i, pos_j : array-like
        Particle positions (shape: (3,) or (..., 3)).
    box : array-like
        Box lengths in x, y, z directions.
    pbc : bool
        If True, apply minimum-image convention.

    Returns
    -------
    np.ndarray
        Displacement vector(s) with same shape as input positions.
    """
    dr = np.asarray(pos_i) - np.asarray(pos_j)

    if pbc:
        box = np.asarray(box)[:3]
        dr = dr - box * np.round(dr / box)

    return dr

def cartesian_to_spherical(rij):
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    rij : array-like
        Cartesian coordinates (..., 3)

    Returns
    -------
    tuple of np.ndarray
        r, theta, phi
    """
    rij = np.asarray(rij)

    x = rij[..., 0]
    y = rij[..., 1]
    z = rij[..., 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    return r, theta, phi

def spherical_harmonic_kernel(r, theta, phi, alpha_m, isotropic=True):
    """Evaluate the spherical harmonics-based F function.
    
    If isotropic=True, takes only the real part of the spherical harmonic product.
    If isotropic=False, returns the full complex result.
    """
    # Define the m values based on whether the system is isotropic
    m_values = [0] if isotropic else [0, 1, 2]
    F_val = []
    for m in m_values:
        # Calculate the spherical harmonic for the current m value
        sph_harm_value = sph_harm(m, 2, phi, theta)
        # Apply the coefficient alpha_m, adjust for isotropy, and scale by r^3
        if isotropic:
            F_val.append((alpha_m[m] * sph_harm_value).real / r**3)
        else:
            F_val.append(alpha_m[m] * sph_harm_value / r**3)
    return F_val

# def spherical_harmonic_kernel(r, theta, phi, alpha_m, isotropic=True, l=2):
#     """
#     Compute spherical-harmonics-based F function.

#     Parameters
#     ----------
#     r, theta, phi : float or array-like
#         Spherical coordinates.
#     alpha_m : dict or array-like
#         Coefficients indexed by m.
#     isotropic : bool
#         If True, take only real part of m=0 term.
#     l : int
#         Angular momentum quantum number.

#     Returns
#     -------
#     np.ndarray
#         F values.
#     """
#     r = np.asarray(r)

#     if np.any(r == 0):
#         raise ValueError("r must be non-zero to avoid division by zero.")

#     if isotropic:
#         m_values = [0]
#     else:
#         m_values = [0, 1, 2]

#     F_vals = []

#     for m in m_values:
#         Ylm = sph_harm_y(m, l, phi, theta)

#         val = alpha_m[m] * Ylm / (r ** 3)

#         if isotropic:
#             val = val.real

#         F_vals.append(val)

#     return np.array(F_vals)

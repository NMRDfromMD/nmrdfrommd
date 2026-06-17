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
from scipy.special import sph_harm_y


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
    """Evaluate the rank-2 spherical harmonic kernel F_m(r, theta, phi).

    Computes F_m = alpha_m * Y_2^m(theta, phi) / r**3 for each required m.
    For isotropic (bulk) systems only m=0 is needed, and only its real
    part is returned since Y_2^0 is real-valued.

    Parameters
    ----------
    r : np.ndarray
        Inter-atomic distances, in Angstrom.
    theta : np.ndarray
        Polar (colatitudinal) angle, in [0, pi], same shape as r.
    phi : np.ndarray
        Azimuthal (longitudinal) angle, in [0, 2*pi], same shape as r.
    alpha_m : np.ndarray
        Rank-2 spherical harmonic normalization coefficients, indexed by m.
    isotropic : bool, default True
        If True, only m=0 is evaluated and its real part returned.
        If False, m=0, 1, 2 are evaluated and the full complex result
        is returned.

    Returns
    -------
    np.ndarray
        F values, shape (1, *r.shape) if isotropic else (3, *r.shape).
    """
    m_values = [0] if isotropic else [0, 1, 2]
    F_val = []
    for m in m_values:
        Y2m = sph_harm_y(2, m, theta, phi)
        F_m = alpha_m[m] * Y2m / r**3
        F_val.append(F_m.real if isotropic else F_m)
    return np.stack(F_val)

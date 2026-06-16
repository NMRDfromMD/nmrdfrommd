#!/usr/bin/env python3
"""Geometry for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2025 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

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

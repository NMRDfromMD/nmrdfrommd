#!/usr/bin/env python3
"""Utilities for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2025 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy.special import sph_harm_y
from scipy import constants as cst

def find_nearest(data, value):
    """
    Return the index of the element in `data` closest to `value`.

    Parameters
    ----------
    data : array-like
        1D array of numeric values.
    value : float
        Value to find the nearest match for.

    Returns
    -------
    int
        Index of the element in `data` closest to `value`.
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("Input `data` must be a 1D array.")
    return np.abs(data - value).argmin()

def calculate_tau(J, gij, dim, integral=False, t=None, oneDarray=False):
    """
    Compute the correlation time τ = 0.5 * J(0) / G(0) or from integral.

    Parameters
    ----------
    J : np.ndarray
        Spectral density array (shape: [dim, N] or [N]).
    gij : np.ndarray
        Time correlation function values (shape: [dim, N] or [N]).
    dim : int
        Dimensionality of the system (e.g., 3 for m = -1, 0, +1).
    integral : bool, optional
        If True, use integral of gij to compute τ.
    t : np.ndarray, optional
        Time vector (required if integral=True).
    oneDarray : bool, optional
        If True, assume 1D inputs for isotropic case.

    Returns
    -------
    np.ndarray
        Correlation times τ in picoseconds.
    """
    if oneDarray:
        if integral:
            if t is None:
                raise ValueError("Time vector `t` must be provided when integral=True.")
            tau = np.trapz(gij, t) / gij[0]
        else:
            tau = 0.5 * J[0] / gij[0]
        return np.array([tau / cst.pico])  # ensure array output
    else:
        tau = []
        for m in range(dim):
            if integral:
                if t is None:
                    raise ValueError("Time vector `t` must be provided when integral=True.")
                tau_m = np.trapz(gij[m], t) / gij[m, 0]
            else:
                tau_m = 0.5 * J[m][0] / gij[0][m]
            tau.append(tau_m / cst.pico)
        return np.array(tau)

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

def spherical_harmonic_kernel(r, theta, phi, alpha_m, isotropic=True, l=2):
    """
    Compute spherical-harmonics-based F function.

    Parameters
    ----------
    r, theta, phi : float or array-like
        Spherical coordinates.
    alpha_m : dict or array-like
        Coefficients indexed by m.
    isotropic : bool
        If True, take only real part of m=0 term.
    l : int
        Angular momentum quantum number.

    Returns
    -------
    np.ndarray
        F values.
    """
    r = np.asarray(r)

    if np.any(r == 0):
        raise ValueError("r must be non-zero to avoid division by zero.")

    if isotropic:
        m_values = [0]
    else:
        m_values = range(-l, l + 1)

    F_vals = []

    for m in m_values:
        Ylm = sph_harm_y(m, l, phi, theta)

        val = alpha_m[m] * Ylm / (r ** 3)

        if isotropic:
            val = val.real

        F_vals.append(val)

    return np.array(F_vals)

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

def log_bin(x, y, num_bins):
    """
    Bin y-values based on logarithmically spaced intervals in x.

    Parameters
    ----------
    x : array_like
        Independent variable (e.g. frequency). Must be > 0.
    y : array_like
        Dependent variable (e.g. spectral density) to be averaged in each bin.
    num_bins : int, optional
        Number of logarithmic bins to use (default is 20).
    remove_empty : bool, optional
        If True (default), bins without any points are removed.
        If False, NaN will be inserted in those bins.

    Returns
    -------
    bin_centers : ndarray
        Logarithmic bin centers (geometric mean of bin edges).
    binned_y : ndarray
        Mean y-values within each bin.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Filter out zero or negative x values
    mask = x > 0
    x = x[mask]
    y = y[mask]

    # Define log-spaced bins
    log_min = np.log10(x.min())
    log_max = np.log10(x.max())
    log_bins = np.logspace(log_min, log_max, num_bins + 1)
    bin_centers = np.sqrt(log_bins[:-1] * log_bins[1:])  # geometric mean

    # Digitize x into bins
    indices = np.digitize(x, log_bins)

    # Compute mean of y-values in each bin
    binned_y = np.array([
        y[indices == i].mean() if np.any(indices == i) else np.nan
        for i in range(1, len(log_bins))
    ])

    return bin_centers, binned_y

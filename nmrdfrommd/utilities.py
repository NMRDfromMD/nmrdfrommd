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
from scipy.special import sph_harm
from scipy import constants as cst

def fourier_transform(data, normalize=False):
    """
    Compute the Fourier transform of a time-domain signal.

    Converts time-domain data into frequency-domain data using FFT.

    Input:
        data[:, 0] -- Time (in picoseconds)
        data[:, 1] -- Signal amplitude (arbitrary units)

    Output:
        data[:, 0] -- Frequency (in MHz)
        data[:, 1] -- Fourier-transformed signal (scaled to s * signal)

    Parameters
    ----------
    data : np.ndarray
        2D array with shape (N, 2) where column 0 is time (ps) and column 1 is signal.
    normalize : bool, optional
        If True, apply unitary normalization to the FFT (i.e., divide by sqrt(N)).

    Returns
    -------
    np.ndarray
        Transformed data with shape (M, 2), where M = N//2 + 1:
        Column 0 is frequency (MHz), column 1 is complex signal (s * signal units).
    """
    if data.shape[1] != 2:
        raise ValueError("Input data must be a 2D array with two columns: time (ps), signal.")

    if len(data) < 2:
        raise ValueError("Need at least two samples to compute time step.")

    dt_ps = data[1, 0] - data[0, 0] # in picoseconds
    dt = dt_ps * cst.pico # convert to seconds

    n = len(data)
    freqs = np.fft.rfftfreq(n, dt) / cst.mega  # in MHz
    fft_norm = 'ortho' if normalize else None
    spectrum = np.fft.rfft(data[:, 1], norm=fft_norm) * dt * 2

    return np.column_stack((freqs, spectrum))

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
    """Compute distance vector between atoms, with optional PBC."""
    if pbc:
        rij = np.remainder(pos_i - pos_j + box[:3]/2., box[:3]) - box[:3]/2
    else:
        rij = pos_i - pos_j
    return rij.T

def cartesian_to_spherical(rij):
    """Convert Cartesian to spherical coordinates."""
    x, y, z = rij
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return r, theta, phi

def compute_F(r, theta, phi, alpha_m, isotropic=True):
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

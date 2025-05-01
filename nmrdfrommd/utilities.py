#!/usr/bin/env python3
"""Utilities for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import constants as cst


def autocorrelation_function(data: np.ndarray,
                             use_wiener_khinchin: bool = False,
                             use_gpu: bool = False) -> np.ndarray:
    """
    Calculate the autocorrelation of a 1D array using FFT.

    Parameters
    ----------
    data : np.ndarray
        Input signal (units: Å-3).
    use_wiener_khinchin : bool, optional
        If True, use Wiener-Khinchin theorem for more memory-efficient FFT.
        If False (default), use legacy double-zero-padded method.
    use_gpu : bool, optional
        If True, compute autocorrelation using CuPy on GPU (requires cupy).

    Returns
    -------
    np.ndarray
        Autocorrelation function (units: Å-6).
    """
    data = np.asarray(data)
    n = len(data)

    if use_gpu:
        try:
            import cupy as cp
        except ImportError:
            raise ImportError("CuPy is not installed.")

        data_cp = cp.asarray(data)
        n_fft = 2 ** int(cp.ceil(cp.log2(2 * n - 1)))
        fdata = cp.fft.fft(data_cp, n=n_fft)
        power = fdata * cp.conj(fdata)
        ac = cp.fft.ifft(power).real[:n]
        normalization = cp.arange(n, 0, -1)
        return cp.asnumpy(ac / normalization)

    elif use_wiener_khinchin:
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fdata = np.fft.fft(data, n=n_fft)
        power_spectrum = fdata * np.conj(fdata)
        autocorr = np.fft.ifft(power_spectrum).real[:n]

    else:
        n_pad = 2 ** int(np.ceil(np.log2(n)))
        data_padded = np.pad(data, (0, n_pad - n))
        data_padded = np.pad(data_padded, (0, data_padded.size))
        fdata = np.fft.fft(data_padded)
        power_spectrum = fdata * np.conj(fdata)
        autocorr = np.fft.ifft(power_spectrum).real[:n]

    normalization = np.arange(n, 0, -1)
    return autocorr / normalization

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

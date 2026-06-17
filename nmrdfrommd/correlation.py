#!/usr/bin/env python3
"""Correlation functions for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import constants as cst


def autocorrelation_function(
        data: np.ndarray,
        use_wiener_khinchin: bool = False,
        use_gpu: bool = False) -> np.ndarray:
    """
    Calculate the autocorrelation of a 1D signal using FFT.

    Parameters
    ----------
    data : np.ndarray
        Input signal (units: Å⁻³).
    use_wiener_khinchin : bool, optional
        If True, use the Wiener-Khinchin theorem with minimal zero-padding.
        If False, use the legacy double-zero-padded implementation.
    use_gpu : bool, optional
        If True, compute the autocorrelation with CuPy on a GPU.

    Returns
    -------
    np.ndarray
        Autocorrelation function (units: Å⁻⁶).
    """
    data = np.asarray(data)
    n = len(data)

    if use_gpu:
        try:
            import cupy as cp
        except ImportError:
            raise ImportError("CuPy is not installed.")

        n_fft = 2 ** int(cp.ceil(cp.log2(2 * n - 1)))

        signal = cp.asarray(data)
        fft_signal = cp.fft.fft(signal, n=n_fft)
        power_spectrum = fft_signal * cp.conj(fft_signal)

        autocorr = cp.fft.ifft(power_spectrum).real[:n]
        normalization = cp.arange(n, 0, -1)

        return cp.asnumpy(autocorr / normalization)

    if use_wiener_khinchin:
        # Minimal zero-padding required for linear autocorrelation.
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))

        fft_signal = np.fft.fft(data, n=n_fft)
        power_spectrum = fft_signal * np.conj(fft_signal)

    else:
        # Legacy implementation using double zero-padding.
        n_fft = 2 ** int(np.ceil(np.log2(n)))

        padded_data = np.pad(data, (0, n_fft - n))
        padded_data = np.pad(padded_data, (0, padded_data.shape[0]))

        fft_signal = np.fft.fft(padded_data)
        power_spectrum = fft_signal * np.conj(fft_signal)

    autocorr = np.fft.ifft(power_spectrum).real[:n]
    normalization = np.arange(n, 0, -1)

    return autocorr / normalization

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
            tau = np.trapezoid(gij, t) / gij[0]
        else:
            tau = 0.5 * J[0] / gij[0]
        return np.array([tau / cst.pico])

    else:
        tau = []
        for m in range(dim):
            if integral:
                if t is None:
                    raise ValueError("Time vector `t` must be provided when integral=True.")
                tau_m = np.trapezoid(gij[m], t) / gij[m, 0]
            else:
                tau_m = 0.5 * J[m][0] / gij[0][m]
            tau.append(tau_m / cst.pico)

        return np.array(tau)

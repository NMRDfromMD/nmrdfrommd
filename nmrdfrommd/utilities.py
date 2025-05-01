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
        Input signal (units: Å⁻³).
    use_wiener_khinchin : bool, optional
        If True, use Wiener-Khinchin theorem for more memory-efficient FFT.
        If False (default), use legacy double-zero-padded method.
    use_gpu : bool, optional
        If True, compute autocorrelation using CuPy on GPU (requires cupy).

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
            raise ImportError("CuPy is not installed. Install it with `pip install cupy`.")

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

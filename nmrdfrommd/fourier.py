#!/usr/bin/env python3
"""Fourier transform functions for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2025 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import constants as cst

# This Fourier transform implementation is numerically correct
# but scientifically fragile due to implicit unit handling and
# normalization choices. The function mixes time units (picoseconds → seconds),
# frequency scaling (Hz → MHz), and FFT normalization conventions in a way
# that is not explicitly enforced in the API. This makes the output sensitive
# to hidden assumptions and easy to misuse when integrating with external
# data or tests using different unit conventions. In scientific workflows,
# such ambiguity can silently introduce errors of several orders of magnitude
# without triggering runtime failures. A safer design would enforce a single
# consistent unit system at the API boundary (preferably SI units), and delegate
# all unit conversions explicitly to the caller, ensuring that the Fourier
# transform itself operates on dimensionless or strictly defined inputs
# and produces well-defined spectral outputs.

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

    if not np.allclose(np.diff(data[:, 0]), dt_ps):
        raise ValueError("Non-uniform time spacing detected.")

    dt = dt_ps * cst.pico # convert to seconds

    n = len(data)
    freqs = np.fft.rfftfreq(n, dt) / cst.mega  # in MHz
    fft_norm = 'ortho' if normalize else None
    spectrum = np.fft.rfft(data[:, 1], norm=fft_norm) * dt * 2

    return np.column_stack((freqs, spectrum))

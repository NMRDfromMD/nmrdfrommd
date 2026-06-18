#!/usr/bin/env python3
"""Fourier transform functions for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from .units import PS_TO_S, HZ_TO_MHZ

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

    dt = dt_ps * PS_TO_S

    n = len(data)
    freqs = np.fft.rfftfreq(n, dt) * HZ_TO_MHZ  # in MHz
    fft_norm = 'ortho' if normalize else None
    spectrum = np.fft.rfft(data[:, 1], norm=fft_norm) * dt * 2

    return np.column_stack((freqs, spectrum))

def compute_spectral_density(t, gij, dim):
    """Compute spectral density J(f) from correlation function gij(t).

    Parameters
    ----------
    t : np.ndarray
        Time array, shape (n_frames,).
    gij : np.ndarray
        Correlation function, shape (dim, n_frames).
    dim : int
        Number of spherical harmonic components (1 isotropic, 3 otherwise).

    Returns
    -------
    f : np.ndarray
        Frequency array.
    J : np.ndarray
        Spectral density, shape (dim, n_frames).
    """
    J = []
    f = None
    for m in range(dim):
        fij = fourier_transform(np.vstack([t, gij[m]]).T)
        J.append(np.real(fij.T[1]))
        f = np.real(fij.T[0])
    return f, np.array(J)

def test_compute_spectral_density_matches_fourier_transform():
    """For dim=1, compute_spectral_density reduces to a single fourier_transform call."""
    t_ps = np.linspace(0, 10, 100)
    signal = np.exp(-t_ps / 2.0)
    gij = signal[np.newaxis, :]  # shape (1, n_frames)

    f, J = compute_spectral_density(t_ps, gij, dim=1)

    expected = fourier_transform(np.column_stack((t_ps, signal)))
    np.testing.assert_allclose(f, expected[:, 0])
    np.testing.assert_allclose(J[0], np.real(expected[:, 1]))

def test_compute_spectral_density_multiple_components():
    """Each component m is transformed independently; shapes and per-row results match."""
    t_ps = np.linspace(0, 10, 64)
    gij = np.stack([
        np.exp(-t_ps / 2.0),
        np.exp(-t_ps / 4.0),
        np.zeros_like(t_ps),
    ])

    f, J = compute_spectral_density(t_ps, gij, dim=3)

    assert J.shape == (3, len(f))
    for m in range(3):
        expected = fourier_transform(np.column_stack((t_ps, gij[m])))
        np.testing.assert_allclose(J[m], np.real(expected[:, 1]))

def test_compute_spectral_density_zero_signal_gives_zero_spectrum():
    """A flat zero correlation function has zero spectral density everywhere."""
    t_ps = np.linspace(0, 10, 50)
    gij = np.zeros((1, len(t_ps)))

    _, J = compute_spectral_density(t_ps, gij, dim=1)

    np.testing.assert_allclose(J, 0.0)
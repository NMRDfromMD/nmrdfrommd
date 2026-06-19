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


def fourier_transform(
    data,
    *,
    normalize=False,
    spectrum_scaling="two_sided_time_domain",
    output_freq_unit="MHz"
):
    """
    Convention-explicit Fourier transform.

    Assumptions (fixed by MDA):
    - time is in picoseconds (ps)
    - sampling is uniform
    - output frequency default unit is MHz

    Parameters
    ----------
    data : np.ndarray
        Shape (N, 2): [time_ps, signal]

    normalize : bool
        If True, apply orthonormal FFT normalization (numpy 'ortho').

    spectrum_scaling : str
        Defines Fourier convention:

        - "raw_rfft"
            np.fft.rfft(x) (no scaling)

        - "continuous_time"
            Approximates continuous FT:
            FFT * dt

        - "two_sided_time_domain"
            Continuous FT + one-sided correction:
            FFT * dt * 2

    output_freq_unit : str
        "MHz" or "Hz"

    Returns
    -------
    np.ndarray
        Column 0: frequency
        Column 1: complex spectrum
    """

    if data.shape[1] != 2:
        raise ValueError("Input must be (N, 2): time_ps, signal")

    if len(data) < 2:
        raise ValueError("Need at least two samples")

    # --- time handling (fixed by MDA contract) ---
    dt_ps = data[1, 0] - data[0, 0]

    if not np.allclose(np.diff(data[:, 0]), dt_ps):
        raise ValueError("Non-uniform time spacing detected")

    dt = dt_ps * PS_TO_S
    n = len(data)

    # --- frequency axis (pure FFT definition) ---
    freqs_hz = np.fft.rfftfreq(n, dt)

    # --- scaling convention (explicit choice) ---
    spectrum = np.fft.rfft(data[:, 1], norm="ortho" if normalize else None)

    if spectrum_scaling == "raw_rfft":
        pass

    elif spectrum_scaling == "continuous_time":
        spectrum = spectrum * dt

    elif spectrum_scaling == "two_sided_time_domain":
        spectrum = spectrum * dt * 2

    else:
        raise ValueError(f"Unknown spectrum_scaling: {spectrum_scaling}")

    # --- unit conversion (explicit boundary step) ---
    if output_freq_unit == "Hz":
        freqs = freqs_hz
    elif output_freq_unit == "MHz":
        freqs = freqs_hz * HZ_TO_MHZ
    else:
        raise ValueError(f"Unknown output_freq_unit: {output_freq_unit}")

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
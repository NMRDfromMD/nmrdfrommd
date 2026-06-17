#!/usr/bin/env python3
"""Utilities for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np


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

#!/usr/bin/env python3
"""Constants for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2025 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

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
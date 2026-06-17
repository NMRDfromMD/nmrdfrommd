#!/usr/bin/env python3
"""election functions for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np

from .errors import EMPTY_J_GROUP


def select_neighbor_indices(neighbor_resids, neighbor_indices, res_id_i, i_idx, type_analysis):
    """Select neighbor atom indices j according to the analysis type.

    Parameters
    ----------
    neighbor_resids : np.ndarray
        Residue id of each atom in the neighbor group.
    neighbor_indices : np.ndarray
        Atom indices of the neighbor group.
    res_id_i : int
        Residue id of the target atom i.
    i_idx : int
        Atom index of the target atom i.
    type_analysis : str
        One of "full", "intra_molecular", "inter_molecular".

    Returns
    -------
    np.ndarray
        Indices of neighbor atoms j satisfying the selection criterion.

    Raises
    ------
    ValueError
        If no neighbor atoms satisfy the criterion.
    """
    conditions = {
        "intra_molecular": (neighbor_resids == res_id_i) & (neighbor_indices != i_idx),
        "inter_molecular": (neighbor_resids != res_id_i),
        "full": (neighbor_indices != i_idx),
    }
    index_j = neighbor_indices[conditions[type_analysis]]

    if len(index_j) == 0:
        raise ValueError(EMPTY_J_GROUP)

    return index_j
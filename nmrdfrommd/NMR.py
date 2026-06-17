#!/usr/bin/env python3
"""Main file for NMRDfromMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import constants as cst
from scipy.interpolate import interp1d
import MDAnalysis as mda
import logging

# Set up basic configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)
logging.getLogger("MDAnalysis").setLevel(logging.WARNING) # Suppress MDAnalysis logs

from .correlation import autocorrelation_function, normalize_correlation
from .fourier import compute_spectral_density
from .geometry import compute_rij, cartesian_to_spherical, spherical_harmonic_kernel
from .constants import get_gyromagnetic_ratio, ALPHA_M, dipolar_prefactor
from .utilities import find_nearest, log_bin_arrays
from .relaxation import compute_relaxation_rates
from .errors import NON_ORTHOGONAL_BOX, INVALID_TYPE_ANALYSIS


class NMRD:
    """Calculate NMR relaxation time from MDAnalysis universe.

    Parameters
    ----------
    u : MDAnalysis.Universe
        MDAnalysis universe containing all the information describing
        the molecular dynamics system.
    atom_group : MDAnalysis.AtomGroup
        Target atom groups for NMR calculation.
    neighbor_group : MDAnalysis.AtomGroup
        Neighbor atom groups. If not specified, atom_group is used.
    type_analysis : str, default ``full``
        Type of analysis, which can be ``full``, ``intra_molecular``,
        or ``inter_molecular``.
    number_i : int, default 0
        Number of atom of the target group to consider for the calculation.
        If ``number_i = 0``, all atoms are considered.
    isotropic : bool, default ``True``
        If isotropic is true, only the spherical harmonic of order 0 is considered, 
        which is usually valid for bulk systems. For non-isotropic systems,
        use ``False``.
    target_frequency : int, default ``None``
        Frequency at which ``T1`` and ``T2`` are calculated.
        If ``None``, ``f = 0`` is used.
    frame_interval : float, default ``None``
        Can be used to specify a different time interface between frames than the 
        one detected by MDAnalysis.
    hydrogen_per_atom : float, default 1.0
        Specify the number of hydrogen per atom, useful for 
        coarse-grained simulations.
    pdb : bool, default True
        To turn off/on the periodic boundary condition treatment.            
    """

    def __init__(self, 
                u: mda.Universe,
                atom_group: mda.AtomGroup,
                neighbor_group: mda.AtomGroup = None,
                type_analysis: str = "full",
                number_i: int = 0,
                isotropic: bool = True,
                target_frequency: float = None,
                frame_interval: float = None,
                hydrogen_per_atom: float = 1.0,
                spin: float = 1/2,
                pbc: bool = True,
                num_log_points: int = 100,
                seed: int = None):
        
        """Initialize class and store parameters."""
        self.u = u
        self.atom_group = atom_group
        self.neighbor_group = atom_group if neighbor_group is None else neighbor_group # default: self-interaction case
        self.type_analysis = type_analysis
        self.number_i = number_i
        self.isotropic = isotropic
        self.dim = 1 if isotropic else 3
        self.target_frequency = target_frequency
        self.frame_interval = frame_interval
        self.hydrogen_per_atom = hydrogen_per_atom
        self.spin = spin
        self.pbc = pbc
        self.num_log_points = num_log_points
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # placeholder attributes (set during analysis)
        self.index_i = None
        self.group_i = None
        self.group_j = None
        self.alpha_m = None
        self.K = None
        self.GAMMA = None

        # For storing results
        self.results = {}

    def run_analysis(self):
        """
        Run the full NMR relaxation analysis pipeline.

        This method executes the complete workflow:
            1. Initialization of physical constants and atom selection
            2. Correlation data collection from the MD trajectory
            3. Fourier transform and spectral density computation
            4. Relaxation time calculation (T1, T2)

        Returns
        -------
        dict
            Dictionary containing all computed results
        """
        self.initialize()
        self.collect_data()
        self.finalize()
        return self.results

    def initialize(self):
        """Prepare the calculation"""
        self._verify_entry()
        self._initialize_physical_constants()
        self._select_atom_group()

    def _verify_entry(self):
        """Verify that inputs are valid for NMR analysis."""

        allowed_analysis = {"inter_molecular", "intra_molecular", "full"}

        if self.type_analysis not in allowed_analysis:
            raise ValueError(INVALID_TYPE_ANALYSIS)

        if self.atom_group.n_atoms == 0:
            raise ValueError("atom_group is empty (n_atoms=0).")

        if self.neighbor_group.n_atoms == 0:
            raise ValueError("neighbor_group is empty (n_atoms=0).")

    def _initialize_physical_constants(self):
        """
        Initialize physical constants and dipolar interaction prefactors.

        Sets the gyromagnetic ratio of the spin bearers, computes the dipolar relaxation
        prefactor K, and loads the spherical-harmonic normalization
        coefficients used in the correlation function calculation.

        Notes
        -----
        The current implementation assumes proton (¹H) spins for the
        relaxation calculation.
        """
        self.GAMMA = get_gyromagnetic_ratio("H") # H is enforced, improve in the future

        self.K = dipolar_prefactor(self.GAMMA, self.spin)

        self.alpha_m = ALPHA_M

    def _select_atom_group(self):
        """
        Select target atoms for the NMR calculation.

        If ``number_i`` is 0 or larger than the size of ``atom_group``,
        all atoms are selected. Otherwise, ``number_i`` atoms are chosen
        randomly without replacement.
        """
        indices = self.atom_group.atoms.indices

        if self.number_i == 0 or self.number_i > len(indices):
            if self.number_i > len(indices):
                logger.warning(
                    "`number_i` is larger than the number of atoms in "
                    "`atom_group`. All atoms will be selected."
                )
            self.index_i = np.array(indices)
        else:
            self.index_i = self._rng.choice(
                indices,
                size=self.number_i,
                replace=False,
            )

    def collect_data(self):
        """Collect data by looping over atoms, time, and evaluate correlation"""
        # Loop on all the atom of group i
        for i_idx in self.index_i:

            self.select_atoms_group_i(i_idx)
            self.select_atoms_group_j(i_idx)

            if i_idx == self.index_i[0]:
                self.initialise_data()

            self.loop_over_trajectory()
            self.calculate_correlation_ij()

    def select_atoms_group_i(self, i_idx):
        """Select atoms of group i for calculation."""
        self.group_i = self.u.select_atoms(f'index {i_idx}')
        self.resids_i = self.group_i.resids

    def select_atoms_group_j(self, i_idx):
        res_id_i = self.resids_i[0]

        conditions = {
            "intra_molecular": (self.neighbor_group.resids == res_id_i) & (self.neighbor_group.indices != i_idx),
            "inter_molecular": (self.neighbor_group.resids != res_id_i),
            "full": (self.neighbor_group.indices != i_idx),
        }

        mask = conditions[self.type_analysis]
        index_j = self.neighbor_group.atoms.indices[mask]

        if len(index_j) == 0:
            raise ValueError("Empty atom groups j...")

        self.group_j = self.u.select_atoms(f'index {" ".join(map(str, index_j))}')

    def initialise_data(self):
        """Initialise arrays.

        Create an array of zeros for the data and the correlation function.
        If anisotropic, the spherical harmonic may be complex, so dtype=complex64
        is used.
        Create an array for of values separated by timestep for the time. 
        """
        n_frames = self.u.trajectory.n_frames
        if self.isotropic:
            self.data = np.zeros((self.dim, n_frames,
                                    self.group_j.atoms.n_atoms),
                                    dtype=np.float32)
        else:
            self.data = np.zeros((self.dim, n_frames,
                                    self.group_j.atoms.n_atoms),
                                    dtype=np.complex64)
        self.results["gij"] = np.zeros((self.dim,  n_frames),
                                dtype=np.float32)
        if self.frame_interval is None:
            self.timestep = np.round(self.u.trajectory.dt, 4)
        else:
            self.timestep = self.frame_interval
        self.results["t"] = np.arange(n_frames) * self.timestep

    def loop_over_trajectory(self):
        """Loop of the MDA trajectory and extract rij. 
        
        Run over the MDA trajectory. If start, stop, or step are
        specified, only a sub-part of the trajectory is analyzed.
        """
        for cpt, ts in enumerate(self.u.trajectory):
            self.position_i = self.group_i.atoms.positions
            self.position_j = self.group_j.atoms.positions
            self.box = ts.dimensions

            # Ensure that the box is orthonormal
            angles = self.box[3:]
            if not (np.allclose(angles, angles[0]) and np.isclose(angles[0], 90.0)):
                raise ValueError(NON_ORTHOGONAL_BOX)
        
            rij = compute_rij(self.position_i, self.position_j, self.box, self.pbc)
            r, theta, phi = cartesian_to_spherical(rij)
            F_val = spherical_harmonic_kernel(r, theta, phi, self.alpha_m, self.isotropic)
            self.data[:, cpt] = F_val

    def calculate_correlation_ij(self):
        """Calculate the correlation function."""
        for idx_j in range(self.group_j.atoms.n_atoms):
            for m in range(self.dim):
                self.results["gij"][m] += autocorrelation_function(self.data[m, :, idx_j])
        self.results["gij"] = np.real(self.results["gij"])

    def finalize(self):
        # calculate spectrums
        self.normalize_Gij()
        self.calculate_spectral_density()
        self.calculate_spectrum()
        self.calculate_relaxationtime()

    def normalize_Gij(self):
        """Divide Gij by the number of spin pairs.
        Optional, for coarse grained model, apply a coefficient "hydrogen_per_atom" != 1
        """
        self.results["gij"] = normalize_correlation(
            self.results["gij"], len(self.index_i), self.hydrogen_per_atom
        )

    def calculate_spectral_density(self):
        """Calculate spectral density J from the Fourier transform of the correlation function."""
        self.results["f"], self.results["J"] = compute_spectral_density(
            self.results["t"], self.results["gij"], self.dim)

    def calculate_spectrum(self):
        """Calculate relaxation rates R1 and R2 from spectral density J."""
        self.results["R1"], self.results["R2"] = compute_relaxation_rates(
            self.results["f"], self.results["J"], self.K, self.isotropic)
            
        self.results["log"] = log_bin_arrays(
            self.results["f"],
            {"R1": self.results["R1"], "R2": self.results["R2"]},
            self.num_log_points)

    def calculate_relaxationtime(self):
        """Calculate T1 and T2 relaxation times from spectral density.

        If target_frequency is None, values are taken at the frequency
        closest to zero. Otherwise, values are evaluated at the nearest
        frequency to target_frequency.

        Protects against divide-by-zero numerical issues.
        """

        f = self.results["f"]
        R1 = self.results["R1"]
        R2 = self.results["R2"]

        # choose frequency index
        if self.target_frequency is None:
            idx = find_nearest(f, 0.0)
        else:
            idx = find_nearest(f, self.target_frequency)

        R1_val = R1[idx]
        R2_val = R2[idx]

        # numerical safety (avoid divide-by-zero warnings)
        eps = 1e-20

        if np.isclose(R1_val, 0.0):
            self.results["T1"] = np.inf
        else:
            self.results["T1"] = 1.0 / (R1_val + eps)

        if np.isclose(R2_val, 0.0):
            self.results["T2"] = np.inf
        else:
            self.results["T2"] = 1.0 / (R2_val + eps)

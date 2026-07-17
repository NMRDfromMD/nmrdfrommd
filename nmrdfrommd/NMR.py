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
from .utilities import log_bin_arrays
from .selection import select_neighbor_indices
from .relaxation import compute_relaxation_rates, compute_relaxation_times
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
    frame_interval : float, default ``None``
        Can be used to specify a different time interface between frames than the 
        one detected by MDAnalysis.
    hydrogen_per_atom : float, default 1.0
        Specify the number of hydrogen per atom, useful for 
        coarse-grained simulations.
    pbc : bool, default True
        To turn off/on the periodic boundary condition treatment.            
    """

    def __init__(self, 
                u: mda.Universe,
                atom_group: mda.AtomGroup,
                neighbor_group: mda.AtomGroup = None,
                type_analysis: str = "full",
                number_i: int = 0,
                isotropic: bool = True,
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
        self.frame_interval = frame_interval
        self.hydrogen_per_atom = hydrogen_per_atom
        self.spin = spin
        self.pbc = pbc
        self.num_log_points = num_log_points
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # placeholder attributes set during analysis
        self.index_i = None
        self.group_i = None
        self.group_j = None
        self.resids_i = None
        self.alpha_m = None
        self.K = None
        self.GAMMA = None
        self.timestep = None
        self.data = None
        self.position_i = None
        self.position_j = None
        self.box = None
        self._n_samples = None

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

    # --- initialize() stage ---
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
                replace=False
                )

    # --- collect_data() stage ---
    def collect_data(self):
        """Collect data by looping over atoms, time, and evaluate correlation."""
        self._initialize_accumulators()

        # Loop on all the atom of group i
        for i_idx in self.index_i:

            self._select_atoms_group_i(i_idx)
            self._select_atoms_group_j(i_idx)
            self._allocate_data_buffer()
            self._loop_over_trajectory()

            gij_i = self._calculate_correlation_ij()
            self.results["gij"] += gij_i  # accumulate raw Gij over atoms

            self._update_error_statistics(gij_i)

    def _initialize_accumulators(self):
        """Initialize the time axis and the correlation accumulator."""

        n_frames = self.u.trajectory.n_frames
        self.results["gij"] = np.zeros((self.dim, n_frames), dtype=np.float32)

        # initialize the Welford statistics,
        self._n_samples = 0
        self.results["R1_mean"] = None
        self.results["R1_M2"] = None
        self.results["R2_mean"] = None
        self.results["R2_M2"] = None
        self.results["gij_mean"] = None
        self.results["gij_M2"] = None

        if self.frame_interval is None:
            self.timestep = np.round(self.u.trajectory.dt, 4)
        else:
            self.timestep = self.frame_interval
        self.results["t"] = np.arange(n_frames) * self.timestep

    def _select_atoms_group_i(self, i_idx):
        """Select atoms of group i for calculation."""
        self.group_i = self.u.atoms[[i_idx]]
        self.resids_i = self.group_i.resids

    def _select_atoms_group_j(self, i_idx):
        """Select atoms of group j for calculation."""
        res_id_i = self.resids_i[0]
        index_j = select_neighbor_indices(
            self.neighbor_group.resids,
            self.neighbor_group.atoms.indices,
            res_id_i,
            i_idx,
            self.type_analysis,
        )
        self.group_j = self.u.select_atoms(f'index {" ".join(map(str, index_j))}')

    def _allocate_data_buffer(self):
        """Allocate the per-atom F-value buffer, sized to the current group_j."""
        n_frames = self.u.trajectory.n_frames
        dtype = np.float32 if self.isotropic else np.complex64
        self.data = np.zeros((self.dim, n_frames, self.group_j.atoms.n_atoms), dtype=dtype)

    def _loop_over_trajectory(self):
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

    def _calculate_correlation_ij(self):
        """Calculate the correlation function for the current atom i."""
        gij = np.zeros((self.dim, self.u.trajectory.n_frames), dtype=np.float32)

        for idx_j in range(self.group_j.atoms.n_atoms):
            for m in range(self.dim):
                gij[m] += autocorrelation_function(self.data[m, :, idx_j])

        return np.real(gij)

    def _update_error_statistics(self, gij_i):
        """Compute per-atom R1/R2 and fold into running Welford statistics."""
        gij_i_norm = normalize_correlation(gij_i.copy(), 1, self.hydrogen_per_atom)
        f_i, J_i = compute_spectral_density(self.results["t"], gij_i_norm, self.dim)
        R1_i, R2_i = compute_relaxation_rates(f_i, J_i, self.K, self.isotropic)

        self._n_samples += 1
        self.results["gij_mean"], self.results["gij_M2"] = self._welford_update(
            self.results["gij_mean"], self.results["gij_M2"], gij_i_norm, self._n_samples)        
        self.results["R1_mean"], self.results["R1_M2"] = self._welford_update(
            self.results["R1_mean"], self.results["R1_M2"], R1_i, self._n_samples)
        self.results["R2_mean"], self.results["R2_M2"] = self._welford_update(
            self.results["R2_mean"], self.results["R2_M2"], R2_i, self._n_samples)

    @staticmethod
    def _welford_update(mean, M2, new_value, count):
        """One online-Welford step, vectorized over the frequency axis."""
        new_value = np.asarray(new_value, dtype=np.float64)
        if mean is None:
            mean = np.zeros_like(new_value)
            M2 = np.zeros_like(new_value)
        delta = new_value - mean  # deviation before mean update
        mean = mean + delta / count
        M2 = M2 + delta * (new_value - mean)  # deviation after mean update
        return mean, M2

    # --- finalize() stage ---
    def finalize(self):
        """Compute spectral density, relaxation rates/times, and error estimates."""
        self._normalize_gij()
        self._calculate_spectral_density()
        self._calculate_spectrum()
        self._calculate_error_estimates()
        self._calculate_relaxationtime()

    def _normalize_gij(self):
        """Divide Gij by the number of spin pairs.
        Optional, for coarse grained model, apply a coefficient "hydrogen_per_atom" != 1
        """
        self.results["gij"] = normalize_correlation(
            self.results["gij"], len(self.index_i), self.hydrogen_per_atom)

    def _calculate_spectral_density(self):
        """Calculate spectral density J from the Fourier transform of the correlation function."""
        self.results["f"], self.results["J"] = compute_spectral_density(
            self.results["t"], self.results["gij"], self.dim)

    def _calculate_spectrum(self):
        """Calculate relaxation rates R1 and R2 from spectral density J."""
        self.results["R1"], self.results["R2"] = compute_relaxation_rates(
            self.results["f"], self.results["J"], self.K, self.isotropic)
            
        self.results["log"] = log_bin_arrays(
            self.results["f"],
            {"R1": self.results["R1"], "R2": self.results["R2"]},
            self.num_log_points)

    def _calculate_error_estimates(self):
        """Std and SEM of R1/R2 across atoms, from the Welford accumulators."""
        n = self._n_samples
        for key in ("R1", "R2", "gij"):
            M2 = self.results[f"{key}_M2"]
            variance = M2 / (n - 1) if n > 1 else np.zeros_like(M2)  # unbiased
            self.results[f"{key}_std"] = np.sqrt(variance)
            self.results[f"{key}_err"] = self.results[f"{key}_std"] / np.sqrt(n)

    def _calculate_relaxationtime(self):
        """Calculate T1/T2 spectra from R1/R2 and uncertainties."""

        (
            self.results["T1"],
            self.results["T2"],
            self.results["T1_err"],
            self.results["T2_err"],
        ) = compute_relaxation_times(
            self.results["f"],
            self.results["R1"],
            self.results["R2"],
            self.results["R1_err"],
            self.results["R2_err"],
        )

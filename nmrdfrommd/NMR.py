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

from .correlation import autocorrelation_function
from .fourier import fourier_transform
from .geometry import compute_rij, cartesian_to_spherical, spherical_harmonic_kernel
from .constants import get_gyromagnetic_ratio, ALPHA_M, dipolar_prefactor
from .utilities import find_nearest, log_bin


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
        self.results = {
            "gij": None,
            "t": None,
            "J": None,
            "f": None,
            "R1": None,
            "R2": None,
            "T1": None,
            "T2": None,
        }

    def run_analysis(self):
        """Run full NMR analysis pipeline.
        
        # Example of use:
        nmr = NMR(u, atom_group, type_analysis='inter_molecular')
        nmr.run_analysis()
        """
        self.initialize()
        self.collect_data()
        self.finalize()

    def initialize(self):
        """Prepare the calculation"""
        self._verify_entry()
        self._initialize_physical_constants()
        self._select_atom_group()

    def _verify_entry(self):
        """Verify that inputs are valid for NMR analysis."""

        allowed_analysis = {"inter_molecular", "intra_molecular", "full"}

        if self.type_analysis not in allowed_analysis:
            raise ValueError(
                f"Invalid type_analysis='{self.type_analysis}'. "
                f"Must be one of {sorted(allowed_analysis)}."
            )

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
        for cpt_i, _ in enumerate(self.index_i):
            self.cpt_i = cpt_i
            self.select_atoms_group_i()
            self.select_atoms_group_j()
            if cpt_i == 0:
                self.initialise_data()
            self.loop_over_trajectory()
            self.calculate_correlation_ij()

    def select_atoms_group_i(self):
        """Select atoms of the group i for the calculation."""
        self.group_i = self.u.select_atoms('index ' + str(self.index_i[self.cpt_i]))
        self.resids_i = self.group_i.resids[self.group_i.atoms.indices == self.index_i[self.cpt_i]]

    def select_atoms_group_j(self):
        """Select atoms of the group j for the calculation.

        For intra molecular analysis, group j are made of atoms of the
        same residue as group i.
        For inter molecular analysis, group j are made of atoms of
        different residues as group i.
        For full analysis, group j are made of atoms that are not in group i.
        """
        res_id_i = self.resids_i[0]
        idx_i = self.index_i[self.cpt_i]
        
        conditions = {
            "intra_molecular": lambda: (self.neighbor_group.resids == res_id_i) & (self.neighbor_group.indices != idx_i),
            "inter_molecular": lambda: (self.neighbor_group.resids != res_id_i),
            "full": lambda: (self.neighbor_group.indices != idx_i),
        }
        
        if self.type_analysis not in conditions:
            raise ValueError(f"Unknown type_analysis: {self.type_analysis}")
        
        index_j = self.neighbor_group.atoms.indices[conditions[self.type_analysis]()]
        if len(index_j) == 0:
            raise ValueError("Empty atom groups j. Wrong combination of type_analysis and group selection?")
        
        str_j = ' '.join(map(str, index_j))
        self.group_j = self.u.select_atoms(f'index {str_j}')

    def initialise_data(self):
        """Initialise arrays.

        Create an array of zeros for the data and the correlation function.
        If anisotropic, the spherical harmonic may be complex, so dtype=complex64
        is used.
        Create an array for of values separated by timestep for the time. 
        """
        if self.isotropic:
            self.data = np.zeros((self.dim, self.u.trajectory.n_frames,
                                    self.group_j.atoms.n_atoms),
                                    dtype=np.float16)
        else:
            self.data = np.zeros((self.dim, self.u.trajectory.n_frames,
                                    self.group_j.atoms.n_atoms),
                                    dtype=np.complex64)
        self.results["gij"] = np.zeros((self.dim,  self.u.trajectory.n_frames),
                                dtype=np.float32)
        if self.frame_interval is None:
            self.timestep = np.round(self.u.trajectory.dt, 4)
        else:
            self.timestep = self.frame_interval
        self.results["t"] = np.arange(self.u.trajectory.n_frames) * self.timestep

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
                raise ValueError("NMRforMD does not accept non-orthogonal box"
                                 "You can use triclinic_to_orthorhombic from the package"
                                 "lipyphilic to convert the trajectory file.")
        
            rij = compute_rij(self.position_i, self.position_j, self.box, self.pbc)
            r, theta, phi = cartesian_to_spherical(rij)
            F_val = spherical_harmonic_kernel(r, theta, phi, self.alpha_m, self.isotropic)
            # F_val = np.array(F_val)  # shape: (dim, n_j_atoms)
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
        self.calculate_fourier_transform()
        self.calculate_spectrum()
        self.calculate_relaxationtime()

    def normalize_Gij(self):
        """Divide Gij by the number of spin pairs.
        
        Optional, for coarse grained model, apply a coefficient "hydrogen_per_atom" != 1
        """
        # normalise gij by the number of iteration (or number of pair spin)
        self.results["gij"] /= self.cpt_i+1
        if self.hydrogen_per_atom != 1:
            self.results["gij"] *= np.float32(self.hydrogen_per_atom)

    def calculate_fourier_transform(self):
        """Calculate spectral density J.
        
        Calculate the spectral density J from the 
        Fourier transform of the correlation function.
        """
        # for coarse grained models, possibly more than 1 hydrogen per atom
        self.results["J"]  = []
        for m in range(self.dim):
            fij = fourier_transform(np.vstack([self.results["t"], self.results["gij"][m]]).T)
            self.results["J"] .append(np.real(fij.T[1]))
        self.results["J"]  = np.array(self.results["J"] )
        self.results["f"] = np.real(fij.T[0])

    def calculate_spectrum(self):
        """Calculate relaxation rates R1 and R2 from spectral density J."""
        prefactor = self.K / cst.angstrom ** 6

        J0 = interp1d(self.results["f"], self.results["J"] [0], fill_value="extrapolate")(self.results["f"])
        if self.isotropic:
            J02 = interp1d(self.results["f"], self.results["J"] [0], fill_value="extrapolate")(2 * self.results["f"])
            self.results["R1"]  = prefactor * (J0 + 4 * J02) / 6
            self.results["R2"]  = prefactor * (3/2 * J0[0] + (5/2) * J0 + J02) / 6
        else:
            J1 = interp1d(self.results["f"], self.results["J"] [1], fill_value="extrapolate")(self.results["f"])
            J2 = interp1d(self.results["f"], self.results["J"] [2], fill_value="extrapolate")(2 * self.results["f"])
            self.results["R1"]  = prefactor * (J1 + J2)
            self.results["R2"]  = prefactor * (1/4) * (J0[0] + 10 * J1 + J2)

        _, R1_log = log_bin(self.results["f"], self.results["R1"] , num_bins=self.num_log_points)
        f_log, R2_log = log_bin(self.results["f"], self.results["R2"] , num_bins=self.num_log_points)
        self.f_log = f_log
        self.R1_log = R1_log
        self.R2_log = R2_log

    def calculate_relaxationtime(self):
        """Calculate the relaxation time at a given frequency target_frequency (default is 0)"""
        if self.target_frequency is None:
            self.results["T1"]  = 1/self.results["R1"] [0]
            self.results["T2"]  = 1/self.results["R2"] [0]
        else:
            idx = find_nearest(self.results["f"], self.target_frequency)
            self.results["T1"]  = 1 / self.results["R1"] [idx]
            self.results["T2"]  = 1 / self.results["R2"] [idx]

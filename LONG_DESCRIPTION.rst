NMRDfromMD is a Python toolkit for computing dipolar NMR relaxation times,
T1 and T2, directly from molecular dynamics trajectories. Built on top of
MDAnalysis, it works with trajectory files from any MDAnalysis-compatible
simulation engine, including LAMMPS and GROMACS.

Rather than relying on simplified analytical models, NMRDfromMD computes
relaxation times from the actual inter-atomic dipole-dipole correlation
functions sampled during the simulation. This makes it well suited for
systems where bulk relaxation theories break down, such as confined
fluids, interfaces, porous media, and biomolecules.

Features:

- Compute T1 and T2 from any MDAnalysis-readable trajectory.
- Separate intra-molecular and inter-molecular contributions to the
  relaxation rate, or compute the combined signal in one pass.
- Support for both isotropic (bulk) and anisotropic (interfacial or
  confined) systems.
- Adjustable hydrogen-per-atom scaling, for use with coarse-grained
  models.

The current implementation targets spin-1/2 nuclei (1H) and dipolar
relaxation; quadrupolar relaxation is not covered, and simulation boxes
are assumed orthorhombic.

Documentation, including the underlying theory and common pitfalls of
NMR relaxation calculations, is available at
https://nmrdfrommd.github.io

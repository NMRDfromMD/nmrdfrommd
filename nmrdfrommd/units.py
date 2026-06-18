"""Unit conventions and conversion factors for NMRDfromMD.

Internal convention (matches MDAnalysis and common NMR usage):
    length    : Angstrom (Å)    -- matches MDAnalysis .positions
    time      : picosecond (ps) -- matches MDAnalysis .trajectory.dt
    frequency : MHz             -- conventional for NMR relaxation
Anything else (K, gamma, ...) is SI, converted to/from the above only
at the points imported from here.
"""
from scipy import constants as cst

PS_TO_S = cst.pico
HZ_TO_MHZ = 1 / cst.mega
ANGSTROM_TO_M = cst.angstrom

"""Real-space TDC coupling via KD-tree nearest-neighbour search."""

from __future__ import annotations

from math import pi
from typing import Dict, Any, Tuple

import numpy as np
from scipy.spatial import cKDTree

from eecc.constants import eps0, e_charge, Angstrom, eV_to_cm1
from eecc.io.cube import read_cube


# ============================================================
# === Read Gaussian Cube Density =============================
# ============================================================

def read_cube_density(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin adapter: read cube via read_cube(units='angstrom') and return (origin, vectors, rho)."""
    cube = read_cube(filename, units='angstrom')
    origin = cube['origin']
    vectors = np.array([cube['vx'], cube['vy'], cube['vz']])
    rho = cube['rho']
    return origin, vectors, rho


# ============================================================
# === Build Coordinates and Charges ==========================
# ============================================================

def cube_to_points(
    origin: np.ndarray, vectors: np.ndarray, rho: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert cube density to coords (meters) and charges (Coulombs)."""
    nx, ny, nz = rho.shape

    voxel_volume_A3 = abs(np.linalg.det(vectors))
    voxel_volume_m3 = voxel_volume_A3 * (Angstrom ** 3)

    x = origin[0] + np.arange(nx) * vectors[0][0]
    y = origin[1] + np.arange(ny) * vectors[1][1]
    z = origin[2] + np.arange(nz) * vectors[2][2]

    XA, YA, ZA = np.meshgrid(x, y, z, indexing='ij')

    coords = np.stack([XA, YA, ZA], axis=-1).reshape(-1, 3) * Angstrom
    charges = rho.flatten() * voxel_volume_m3 * e_charge

    return coords, charges


# ============================================================
# === Real-space Coulomb Coupling via KD-tree ================
# ============================================================

def tdc_coupling_kdtree(
    coordsA: np.ndarray,
    qA: np.ndarray,
    coordsB: np.ndarray,
    qB: np.ndarray,
    eps_r: float = 1.0,
    cutoff_Ang: float = 20.0,
) -> Tuple[float, float]:
    """Compute Coulomb coupling via KD-tree. Returns (J_eV, J_cm1)."""
    cutoff_m = cutoff_Ang * Angstrom

    treeB = cKDTree(coordsB)

    J = 0.0
    for i in range(len(coordsA)):
        r1 = coordsA[i]
        q1 = qA[i]

        idx = treeB.query_ball_point(r1, cutoff_m)
        if not idx:
            continue

        r2 = coordsB[idx]
        q2 = qB[idx]

        d = np.linalg.norm(r2 - r1, axis=1)
        J += np.sum(q1 * q2 / d)

    J *= 1.0 / (4 * pi * eps0 * eps_r)

    J_eV = J / e_charge
    J_cm1 = J_eV * eV_to_cm1

    return J_eV, J_cm1

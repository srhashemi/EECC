"""Transition multipoles and TrESP-specific grid helpers (Bohr units)."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def is_axis_aligned(axes_bohr: np.ndarray, tol: float = 1e-10) -> bool:
    """True if axes are approximately (Lx,0,0), (0,Ly,0), (0,0,Lz)."""
    M = np.array(axes_bohr, dtype=float)
    off = M.copy()
    np.fill_diagonal(off, 0.0)
    return bool(np.all(np.abs(off) < tol))


def grid_spacing(
    axes_bohr: np.ndarray, shape: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """Grid spacing from TrESP axes (Bohr).

    Each row of *axes_bohr* is the step vector between consecutive grid
    points (as stored in a Gaussian cube file).  The spacing is simply
    the norm of each step vector.
    """
    dx = float(np.linalg.norm(axes_bohr[0]))
    dy = float(np.linalg.norm(axes_bohr[1]))
    dz = float(np.linalg.norm(axes_bohr[2]))
    return dx, dy, dz


def grid_lengths(
    axes_bohr: np.ndarray, shape: Tuple[int, int, int]
) -> Tuple[float, float, float]:
    """Total grid extent from axes and shape (Bohr).

    Returns ``(nx * dx, ny * dy, nz * dz)`` — the full span of the grid
    (note: last grid point is at ``origin + (n-1)*step``).
    """
    dx, dy, dz = grid_spacing(axes_bohr, shape)
    nx, ny, nz = shape
    return dx * nx, dy * ny, dz * nz


def grid_coordinates(
    origin: np.ndarray,
    axes_bohr: np.ndarray,
    shape: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 1D coordinate arrays xs, ys, zs (Bohr) for voxel centres."""
    nx, ny, nz = shape
    Lx, Ly, Lz = grid_lengths(axes_bohr, shape)
    x0, y0, z0 = origin
    xs = x0 + np.arange(nx) * (Lx / nx)
    ys = y0 + np.arange(ny) * (Ly / ny)
    zs = z0 + np.arange(nz) * (Lz / nz)
    return xs, ys, zs


def transition_multipoles(
    rho: np.ndarray,
    origin: np.ndarray,
    axes_bohr: np.ndarray,
) -> Tuple[float, np.ndarray, float]:
    """Compute net transition charge Q and dipole mu for diagnostics.

    Returns (Q, mu, dV) where Q in e, mu in Bohr·e, dV in Bohr³.
    """
    nx, ny, nz = rho.shape
    dx, dy, dz = grid_spacing(axes_bohr, (nx, ny, nz))
    dV = dx * dy * dz

    xs, ys, zs = grid_coordinates(origin, axes_bohr, (nx, ny, nz))
    Q = np.sum(rho) * dV

    rho_x = np.sum(np.sum(rho, axis=2), axis=1)
    rho_y = np.sum(np.sum(rho, axis=2), axis=0)
    rho_z = np.sum(np.sum(rho, axis=1), axis=0)
    mu = np.array([
        np.dot(xs, rho_x),
        np.dot(ys, rho_y),
        np.dot(zs, rho_z)
    ]) * dV

    return Q, mu, dV

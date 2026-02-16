"""Transition ESP computation via FFT Poisson solver and CHELPG-like sampling."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from eecc.constants import FOUR_PI, ANG_TO_BOHR
from eecc.tresp.multipoles import grid_lengths


# ---------- vdW radius table ----------

def vdW_radius_bohr(Z: int) -> float:
    """Return van der Waals radius in Bohr for atomic number *Z*."""
    table_ang = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80, 17: 1.75, 35: 1.85}
    return table_ang.get(int(Z), 1.70) * ANG_TO_BOHR


# ============================================================
# === FFT Coulomb potential (Poisson solver) =================
# ============================================================

def fft_coulomb_potential(
    rho: np.ndarray,
    dx: float, dy: float, dz: float,
    pad: int = 2,
) -> np.ndarray:
    """Compute Coulomb potential V on the grid from charge density rho using FFT Poisson.

    Parameters
    ----------
    rho : (nx, ny, nz) in e/Bohr³
    dx, dy, dz : grid spacings in Bohr
    pad : zero-padding factor per dimension (>=1)

    Returns
    -------
    V : (nx, ny, nz) in Hartree/e
    """
    nx, ny, nz = rho.shape

    # enforce neutrality (transition density integral ~ 0)
    rho = rho - np.mean(rho)

    if pad > 1:
        Nx, Ny, Nz = nx * pad, ny * pad, nz * pad
        px = (Nx - nx) // 2
        py = (Ny - ny) // 2
        pz = (Nz - nz) // 2
        rho_pad = np.pad(rho, ((px, Nx - nx - px),
                               (py, Ny - ny - py),
                               (pz, Nz - nz - pz)), mode="constant")
    else:
        Nx, Ny, Nz = nx, ny, nz
        px = py = pz = 0
        rho_pad = rho

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX ** 2 + KY ** 2 + KZ ** 2

    rho_k = np.fft.fftn(rho_pad)
    rho_k[0, 0, 0] = 0.0  # neutrality
    Gk = np.zeros_like(k2)
    mask = k2 > 0
    Gk[mask] = FOUR_PI / k2[mask]
    V_pad = np.fft.ifftn(Gk * rho_k).real

    if pad > 1:
        V = V_pad[px:px + nx, py:py + ny, pz:pz + nz]
    else:
        V = V_pad
    return V


# ============================================================
# === CHELPG sampling ========================================
# ============================================================

def chelpg_points(
    atoms: List[Tuple[int, float, float, float]],
    shells_ang: Sequence[float] = (1.4, 1.8, 2.2),
    pps: int = 4000,
    seed: int = 1,
) -> np.ndarray:
    """Generate CHELPG-like sampling points (Bohr) on shells beyond vdW spheres.

    Parameters
    ----------
    atoms : list of (Z, x_bohr, y_bohr, z_bohr)
    shells_ang : shell offsets in Å beyond vdW radius
    pps : approximate total points per shell

    Returns
    -------
    P : (M, 3) points in Bohr
    """
    rng = np.random.default_rng(seed)
    offs_bohr = [s * ANG_TO_BOHR for s in shells_ang]
    coords = np.array([[x, y, z] for (Z, x, y, z) in atoms], dtype=float)
    vdW = np.array([vdW_radius_bohr(Z) for (Z, _, _, _) in atoms], dtype=float)

    pts = []
    n_atoms = len(atoms)
    for s in offs_bohr:
        r_shell = vdW + s
        n_each = max(1, pps // n_atoms)
        u = rng.standard_normal((n_atoms, n_each, 3))
        u /= np.linalg.norm(u, axis=2)[:, :, None]
        layer = coords[:, None, :] + u * r_shell[:, None, None]
        pts.append(layer.reshape(-1, 3))
    return np.vstack(pts)


# ============================================================
# === Trilinear interpolation ================================
# ============================================================

def trilinear_on_grid(
    V: np.ndarray,
    origin: np.ndarray,
    axes_bohr: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Trilinear interpolation of grid V at points P (Bohr).

    Returns (values, mask) where mask selects points inside the grid box.
    """
    nx, ny, nz = V.shape
    Lx, Ly, Lz = grid_lengths(axes_bohr, (nx, ny, nz))
    x0, y0, z0 = origin
    dx = Lx / nx; dy = Ly / ny; dz = Lz / nz

    fx = (P[:, 0] - x0) / dx
    fy = (P[:, 1] - y0) / dy
    fz = (P[:, 2] - z0) / dz

    mask = (fx >= 0) & (fx <= nx - 1) & (fy >= 0) & (fy <= ny - 1) & (fz >= 0) & (fz <= nz - 1)
    vals = np.zeros(P.shape[0], dtype=float)
    if not np.any(mask):
        return vals, mask

    i0 = np.floor(fx[mask]).astype(int); tx = fx[mask] - i0
    j0 = np.floor(fy[mask]).astype(int); ty = fy[mask] - j0
    k0 = np.floor(fz[mask]).astype(int); tz = fz[mask] - k0

    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)
    k1 = np.clip(k0 + 1, 0, nz - 1)

    c000 = V[i0, j0, k0]
    c100 = V[i1, j0, k0]
    c010 = V[i0, j1, k0]
    c110 = V[i1, j1, k0]
    c001 = V[i0, j0, k1]
    c101 = V[i1, j0, k1]
    c011 = V[i0, j1, k1]
    c111 = V[i1, j1, k1]

    c00 = c000 * (1 - tx) + c100 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    vals[mask] = c0 * (1 - tz) + c1 * tz
    return vals, mask

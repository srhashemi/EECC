"""FFT-based Transition Density Cube (TDC) Coulomb coupling."""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq
from scipy.interpolate import RegularGridInterpolator

from eecc.constants import EV_TO_CM, KE_EV_ANG, DEBYE_PER_EANG
from eecc.io.cube import (
    grid_spacing, build_axes_coordinates, monomer_separation, voxel_volume,
)
from eecc.coupling.dipole import extended_dipole_coupling_formula, point_dipole_coupling


# ============================================================
# === Transition dipole from cube ============================
# ============================================================

def transition_dipole_from_cube(cube: Dict[str, Any]) -> Dict[str, Any]:
    """Compute transition dipole (e·Å, Debye) by integrating rho·r over the grid."""
    rho = cube['rho']
    xs, ys, zs = build_axes_coordinates(cube)

    dx, dy, dz = grid_spacing(cube)
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')

    mux = float(np.sum(rho * X) * dV)
    muy = float(np.sum(rho * Y) * dV)
    muz = float(np.sum(rho * Z) * dV)

    mu = np.array([mux, muy, muz], dtype=float)
    mu_D = np.linalg.norm(mu) * DEBYE_PER_EANG

    return {'mu': mu, 'mu_D': mu_D}


# ============================================================
# === Density charge helpers =================================
# ============================================================

def integrate_density_total_charge(cube: Dict[str, Any]) -> float:
    """Return the total integrated transition charge ∫ rho dV."""
    rho = cube['rho']
    dx, dy, dz = grid_spacing(cube)
    dV = dx * dy * dz
    return float(np.sum(rho) * dV)


def neutralize_full_density(rho: np.ndarray) -> Tuple[np.ndarray, float]:
    """Subtract global mean so that ∫ rho dV = 0 (for uniform dV)."""
    mean = float(np.mean(rho))
    return rho - mean, mean


# ============================================================
# === FFT Coulomb Potential ==================================
# ============================================================

def _fft_coulomb_potential(
    rho: np.ndarray,
    dx: float, dy: float, dz: float,
    origin: np.ndarray,
    pad_factor: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Coulomb potential V(r) = ∫ ρ(r')/|r-r'| dr' via FFT.

    Returns
    -------
    V_padded : ndarray
        Coulomb potential on the full padded grid.
    padded_origin : ndarray
        Physical origin (Å) of the padded grid.
    """
    nx, ny, nz = rho.shape
    Nx = pad_factor * nx
    Ny = pad_factor * ny
    Nz = pad_factor * nz

    sx = (Nx - nx) // 2
    sy = (Ny - ny) // 2
    sz = (Nz - nz) // 2

    rp = np.zeros((Nx, Ny, Nz), dtype=float)
    rp[sx:sx+nx, sy:sy+ny, sz:sz+nz] = rho

    kx = 2.0 * np.pi * fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * fftfreq(Ny, d=dy)
    kz = 2.0 * np.pi * fftfreq(Nz, d=dz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX ** 2 + KY ** 2 + KZ ** 2

    dV = dx * dy * dz

    kernel = np.zeros_like(K2, dtype=float)
    mask = (K2 > 0.0)
    kernel[mask] = 4.0 * np.pi / K2[mask]

    rho_k = fftn(rp * dV)
    V_padded = ifftn(rho_k * kernel).real

    padded_origin = np.asarray(origin, dtype=float) - np.array([sx * dx, sy * dy, sz * dz])
    return V_padded, padded_origin


def _interpolate_potential(
    V_padded: np.ndarray,
    padded_origin: np.ndarray,
    dx: float, dy: float, dz: float,
    target_cube: Dict[str, Any],
) -> np.ndarray:
    """Interpolate a potential grid at the target cube's grid positions."""
    Vnx, Vny, Vnz = V_padded.shape
    xs_V = padded_origin[0] + np.arange(Vnx) * dx
    ys_V = padded_origin[1] + np.arange(Vny) * dy
    zs_V = padded_origin[2] + np.arange(Vnz) * dz

    interp = RegularGridInterpolator(
        (xs_V, ys_V, zs_V), V_padded,
        method='linear', bounds_error=False, fill_value=0.0,
    )

    xs_t, ys_t, zs_t = build_axes_coordinates(target_cube)
    Xt, Yt, Zt = np.meshgrid(xs_t, ys_t, zs_t, indexing='ij')
    pts = np.column_stack([Xt.ravel(), Yt.ravel(), Zt.ravel()])

    return interp(pts).reshape(target_cube['rho'].shape)


# ============================================================
# === TDC Coupling (Two Cubes) ===============================
# ============================================================

# Largest padded free-space grid (points) before tdc_coupling_fft gives up.
FREE_FFT_MAX_POINTS = 2.5e8

# Mean of 1/r over a cube of unit edge centred at the origin (self-term of the kernel).
_CUBE_MEAN_INV_R = 2.3800772


def _free_space_potential(
    rhoB: np.ndarray, spacing: Tuple[float, float, float], originB: np.ndarray,
    target_cube: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Potential of rho_B (e/Å) on a grid covering cubeB and the target cube.

    Hockney's method: the grid is zero-padded to twice its size and convolved
    with the real-space 1/r kernel, which gives the exact discrete sum
    sum_j q_j / |r - r_j| without periodic images, however far apart the cubes are.
    """
    h = np.asarray(spacing, float)
    originB = np.asarray(originB, float)
    lo, hi = originB.copy(), originB + (np.array(rhoB.shape) - 1) * h
    xs_t, ys_t, zs_t = build_axes_coordinates(target_cube)
    lo = np.minimum(lo, [xs_t[0], ys_t[0], zs_t[0]])
    hi = np.maximum(hi, [xs_t[-1], ys_t[-1], zs_t[-1]])
    # box aligned with cubeB's lattice, one spare point on each side for interpolation
    shift = np.ceil((originB - lo) / h - 1e-9).astype(int) + 1
    origin = originB - shift * h
    n = np.ceil((hi - origin) / h - 1e-9).astype(int) + 2
    if 8 * np.prod(n.astype(float)) > FREE_FFT_MAX_POINTS:
        raise ValueError(f"free-space FFT grid {tuple(2 * n)} is too large; use the direct TDC sum")

    q = np.zeros(2 * n)
    q[shift[0]:shift[0] + rhoB.shape[0], shift[1]:shift[1] + rhoB.shape[1],
      shift[2]:shift[2] + rhoB.shape[2]] = rhoB * np.prod(h)
    axes = [np.minimum(np.arange(2 * m), 2 * m - np.arange(2 * m)) * hk for m, hk in zip(n, h)]
    R = np.sqrt(axes[0][:, None, None] ** 2 + axes[1][None, :, None] ** 2 + axes[2][None, None, :] ** 2)
    R[0, 0, 0] = 1.0
    G = 1.0 / R
    G[0, 0, 0] = _CUBE_MEAN_INV_R / np.cbrt(np.prod(h))
    V = ifftn(fftn(q) * fftn(G)).real[:n[0], :n[1], :n[2]]
    return V, origin


def tdc_coupling_fft(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    dielectric: float = 1.0,
    pad_factor: int = 3,
    boundary: str = "free",
) -> Dict[str, Any]:
    """Compute TDC Coulomb coupling between two monomer cubes via FFT.

    Computes V_B on a grid, then interpolates V_B at cubeA's grid points.

    boundary='free' (default) solves the isolated problem on a grid covering both
    cubes and agrees with the direct sum at any separation (*pad_factor* is not used).
    boundary='periodic' reproduces the published workflow: cubeB's own grid padded
    by *pad_factor* with a periodic Coulomb kernel. Periodic images of B then add to
    the potential, and parts of cubeA outside the padded grid see zero potential, so
    the coupling is off by about 10-30 % for neighbours and more for distant pairs.

    Returns a dict with J_eV, J_cm1, transition dipoles, and point-dipole
    and extended-dipole comparison values.
    """
    # Separation between the monomers (atom centroids, or density centroids)
    Rvec_phys = monomer_separation(cubeA, cubeB)

    # Transition dipoles (computed independently on each cube)
    muA = transition_dipole_from_cube(cubeA)
    muB = transition_dipole_from_cube(cubeB)

    # --- FFT Coulomb potential of rho_B on cubeB's padded grid ---
    rhoB = cubeB['rho']
    dxB, dyB, dzB = grid_spacing(cubeB)

    rhoA = cubeA['rho']
    if boundary == "free":
        V, V_origin = _free_space_potential(rhoB, (dxB, dyB, dzB), cubeB['origin'], cubeA)
        VB_at_A = _interpolate_potential(V, V_origin, dxB, dyB, dzB, cubeA)
        # --- TDC Coulomb coupling: J = k_e * Σ ρ_A · φ_B · dV_A ---
        J_eV = (KE_EV_ANG / dielectric) * float(np.sum(rhoA * VB_at_A)) * voxel_volume(cubeA)
    elif boundary == "periodic":
        V_padded, V_origin = _fft_coulomb_potential(
            rhoB, dxB, dyB, dzB, cubeB['origin'], pad_factor=pad_factor,
        )
        VB_at_A = _interpolate_potential(V_padded, V_origin, dxB, dyB, dzB, cubeA)
        # --- TDC Coulomb coupling: J = k_e * Σ ρ_A · φ_B · dV_A ---
        # V_padded carries a factor dV_B (it is built from ρ_B·dV_B), so rescale
        # by dV_A/dV_B to integrate over cubeA's voxels.
        dV_ratio = voxel_volume(cubeA) / voxel_volume(cubeB)
        J_eV = (KE_EV_ANG / dielectric) * float(np.sum(rhoA * VB_at_A)) * dV_ratio
    else:
        raise ValueError("boundary must be 'periodic' or 'free'")
    J_cm1 = J_eV * EV_TO_CM

    dx, dy, dz = grid_spacing(cubeA)

    # --- Comparison methods using physical separation ---
    l1 = 1.0   # dipole length for monomer A (Å)
    l2 = 1.0   # dipole length for monomer B (Å)
    Jext_cm1 = extended_dipole_coupling_formula(
        muA['mu'], muB['mu'], Rvec_phys, l1, l2, dielectric,
    )
    Jext_eV = Jext_cm1 / EV_TO_CM

    Jpd_eV, Jpd_cm1 = point_dipole_coupling(
        muA['mu'], muB['mu'], Rvec_phys, dielectric=dielectric,
    )

    return {
        'J_eV': J_eV,
        'J_cm1': J_cm1,
        'muA_eAng': muA['mu'],
        'muA_D': muA['mu_D'],
        'muB_eAng': muB['mu'],
        'muB_D': muB['mu_D'],
        'dx_dy_dz_Ang': (dx, dy, dz),
        'pad_factor': pad_factor,
        'Jpd_eV': Jpd_eV,
        'Jpd_cm1': Jpd_cm1,
        'Rvec_A_to_B_Ang': Rvec_phys,
        'R_AB_Ang': float(np.linalg.norm(Rvec_phys)),
        'Jext_cm1': Jext_cm1,
        'Jext_eV': Jext_eV,
    }


# ============================================================
# === Simplified TDC coupling (for one-dimer workflow) =======
# ============================================================

def tdc_coupling_fft_simple(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    dielectric: float = 1.0,
    pad_factor: int = 3,
    boundary: str = "free",
) -> Dict[str, float]:
    """Simplified TDC coupling for fragments sharing the same grid.

    Use this when cubeA and cubeB already share the same origin and
    voxel vectors (e.g. fragments split from a single dimer cube).
    boundary='free' (default) is the isolated problem (exact discrete sum);
    'periodic' pads the shared grid by *pad_factor* with a periodic kernel, as in
    the published workflow.
    """
    rhoA, rhoB = cubeA['rho'], cubeB['rho']
    dxA, dyA, dzA = grid_spacing(cubeA)

    if boundary == "free":
        V, V_origin = _free_space_potential(rhoB, (dxA, dyA, dzA), cubeA['origin'], cubeA)
        phi = _interpolate_potential(V, V_origin, dxA, dyA, dzA, cubeA)
        J_eV = (KE_EV_ANG / dielectric) * float(np.sum(rhoA * phi)) * voxel_volume(cubeA)
        return {'J_eV': J_eV, 'J_cm1': J_eV * EV_TO_CM}
    if boundary != "periodic":
        raise ValueError("boundary must be 'periodic' or 'free'")

    V_padded, _ = _fft_coulomb_potential(
        rhoB, dxA, dyA, dzA, cubeA['origin'], pad_factor=pad_factor,
    )

    # Same grid — extract the unpadded region directly
    nx, ny, nz = rhoB.shape
    Nx, Ny, Nz = V_padded.shape
    sx = (Nx - nx) // 2
    sy = (Ny - ny) // 2
    sz = (Nz - nz) // 2
    phi = V_padded[sx:sx+nx, sy:sy+ny, sz:sz+nz]

    J_eV = (KE_EV_ANG / dielectric) * float(np.sum(rhoA * phi))
    J_cm1 = J_eV * EV_TO_CM

    return {
        'J_eV': J_eV,
        'J_cm1': J_cm1,
    }

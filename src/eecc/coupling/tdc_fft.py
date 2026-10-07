"""FFT-based Transition Density Cube (TDC) Coulomb coupling."""

from __future__ import annotations

import warnings
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy.fft import fftn, ifftn, fftfreq, irfftn, next_fast_len, rfftn
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

# Largest free-space FFT grid (points) before tdc_coupling_fft gives up.
FREE_FFT_MAX_POINTS = 2.5e8


class FreeGridTooLarge(ValueError):
    """The free-space FFT grid would exceed FREE_FFT_MAX_POINTS."""


def boundary_label(boundary: str, pad_factor: Optional[int] = None) -> str:
    """One-line description of the FFT boundary for printed and written output."""
    if boundary == "periodic":
        return f"FFT boundary = periodic, pad factor = {3 if pad_factor is None else pad_factor}"
    return f"FFT boundary = {boundary}"


def _periodic_pad(boundary: str, pad_factor: Optional[int]) -> int:
    """Resolve *pad_factor* (default 3); warn when it is given for the free boundary, which ignores it."""
    if boundary == "free":
        if pad_factor is not None:
            warnings.warn("pad_factor applies to boundary='periodic' only and is ignored; "
                          "pass boundary='periodic' to reproduce the published FFT values", stacklevel=3)
        return 0
    if boundary != "periodic":
        raise ValueError("boundary must be 'periodic' or 'free'")
    return 3 if pad_factor is None else pad_factor


def _mean_inv_r_box(h) -> float:
    """Mean of 1/r over a box of edges *h* centred at the origin (closed form)."""
    a, b, c = (0.5 * float(x) for x in h)
    d = np.sqrt(a * a + b * b + c * c)
    F = (b * c * np.log((a + d) / np.hypot(b, c)) + a * c * np.log((b + d) / np.hypot(a, c))
         + a * b * np.log((c + d) / np.hypot(a, b))
         - 0.5 * a * a * np.arctan(b * c / (a * d)) - 0.5 * b * b * np.arctan(a * c / (b * d))
         - 0.5 * c * c * np.arctan(a * b / (c * d)))
    return float(F / (a * b * c))


def _free_target(spacing, originB, nB, target_cube):
    """Box on cubeB's lattice covering *target_cube*.

    Returns (m, nT, aligned): the box starts m lattice steps from originB, has nT points,
    and is the target grid itself when that lies on the same lattice (aligned).
    """
    h = np.asarray(spacing, float)
    tx, ty, tz = build_axes_coordinates(target_cube)
    t_lo = np.array([tx[0], ty[0], tz[0]])
    t_hi = np.array([tx[-1], ty[-1], tz[-1]])
    rel = (t_lo - originB) / h
    t_h = np.array([target_cube['vx'][0], target_cube['vy'][1], target_cube['vz'][2]])
    aligned = bool(np.allclose(t_h, h, rtol=1e-9, atol=0.0) and np.allclose(rel, np.rint(rel), atol=1e-6))
    if aligned:
        return np.rint(rel).astype(int), np.array(target_cube['rho'].shape), True
    m = np.floor(rel + 1e-9).astype(int)
    n = np.ceil((t_hi - originB) / h - 1e-9).astype(int) - m + 1
    return m, n, False


def _free_fft_shape(nB, nT):
    shape = tuple(next_fast_len(int(a + b - 1), real=True) for a, b in zip(nB, nT))
    if np.prod(np.array(shape, float)) > FREE_FFT_MAX_POINTS:
        raise FreeGridTooLarge(f"free-space FFT grid {shape} exceeds {FREE_FFT_MAX_POINTS:.3g} points")
    return shape


def check_free_grid(cubeB: Dict[str, Any], target_cube: Dict[str, Any]) -> None:
    """Raise FreeGridTooLarge if the free-space FFT of cubeB onto *target_cube* is too large."""
    nB = np.array(cubeB['rho'].shape)
    _, nT, _ = _free_target(grid_spacing(cubeB), np.asarray(cubeB['origin'], float), nB, target_cube)
    _free_fft_shape(nB, nT)


def _free_space_potential(
    rhoB: np.ndarray, spacing: Tuple[float, float, float], originB: np.ndarray,
    target_cube: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Potential of rho_B (e/Å) on cubeB's lattice over the box covering the target cube.

    Linear convolution of the voxel charges with the 1/r kernel at the lattice displacements
    between cubeB and the target box (Hockney): no periodic images, and the grid holds
    nB + nT points per axis whatever the separation. Coinciding points use the mean of 1/r
    over a voxel. Returns (V, origin of V, aligned) as _free_target defines aligned.
    """
    h = np.asarray(spacing, float)
    originB = np.asarray(originB, float)
    nB = np.array(rhoB.shape)
    m, nT, aligned = _free_target(h, originB, nB, target_cube)
    shape = _free_fft_shape(nB, nT)
    L = nB + nT - 1

    # kernel K[j] = 1/|(m + j - (nB - 1)) h|, j = 0 .. L - 1
    axes = [(mk + np.arange(Lk) - (nk - 1)) * hk for mk, Lk, nk, hk in zip(m, L, nB, h)]
    K = np.zeros(shape)
    G = K[:L[0], :L[1], :L[2]]
    G += axes[0][:, None, None] ** 2
    G += axes[1][None, :, None] ** 2
    G += axes[2][None, None, :] ** 2
    z = nB - 1 - m
    self_term = bool(np.all((z >= 0) & (z < L)))
    if self_term:
        G[tuple(z)] = 1.0
    np.sqrt(G, out=G)
    np.reciprocal(G, out=G)
    if self_term:
        G[tuple(z)] = _mean_inv_r_box(h)
    FK = rfftn(K)
    del K, G

    q = np.zeros(shape)
    q[:nB[0], :nB[1], :nB[2]] = rhoB * np.prod(h)
    FK *= rfftn(q)
    del q
    # V[t] = sum_s q[s] K[t - s + nB - 1]: full linear convolution at index t + nB - 1
    s0 = nB - 1
    V = irfftn(FK, s=shape)[s0[0]:s0[0] + nT[0], s0[1]:s0[1] + nT[1], s0[2]:s0[2] + nT[2]].copy()
    return V, originB + m * h, aligned


def _free_coupling(cubeA: Dict[str, Any], cubeB: Dict[str, Any], dielectric: float) -> float:
    """J (eV) = k_e Σ ρ_A φ_B dV_A with φ_B from the free-space FFT."""
    h = grid_spacing(cubeB)
    V, origin, aligned = _free_space_potential(cubeB['rho'], h, cubeB['origin'], cubeA)
    phi = V if aligned else _interpolate_potential(V, origin, *h, cubeA)
    return (KE_EV_ANG / dielectric) * float(np.sum(cubeA['rho'] * phi)) * voxel_volume(cubeA)


def dipole_models(cubeA: Dict[str, Any], cubeB: Dict[str, Any], dielectric: float = 1.0) -> Dict[str, Any]:
    """Transition dipoles, separation, and point- and extended-dipole couplings of two cubes."""
    # Separation between the monomers (atom centroids, or density centroids)
    Rvec_phys = monomer_separation(cubeA, cubeB)
    muA = transition_dipole_from_cube(cubeA)
    muB = transition_dipole_from_cube(cubeB)

    l1 = 1.0   # dipole length for monomer A (Å)
    l2 = 1.0   # dipole length for monomer B (Å)
    Jext_cm1 = extended_dipole_coupling_formula(
        muA['mu'], muB['mu'], Rvec_phys, l1, l2, dielectric,
    )
    Jpd_eV, Jpd_cm1 = point_dipole_coupling(
        muA['mu'], muB['mu'], Rvec_phys, dielectric=dielectric,
    )
    return {
        'muA_eAng': muA['mu'],
        'muA_D': muA['mu_D'],
        'muB_eAng': muB['mu'],
        'muB_D': muB['mu_D'],
        'dx_dy_dz_Ang': grid_spacing(cubeA),
        'Jpd_eV': Jpd_eV,
        'Jpd_cm1': Jpd_cm1,
        'Rvec_A_to_B_Ang': Rvec_phys,
        'R_AB_Ang': float(np.linalg.norm(Rvec_phys)),
        'Jext_cm1': Jext_cm1,
        'Jext_eV': Jext_cm1 / EV_TO_CM,
    }


def tdc_coupling_fft(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    dielectric: float = 1.0,
    pad_factor: Optional[int] = None,
    boundary: str = "free",
) -> Dict[str, Any]:
    """Compute TDC Coulomb coupling between two monomer cubes via FFT.

    Computes V_B on a grid, then takes V_B at cubeA's grid points.

    boundary='free' (default) solves the isolated problem at any separation. When cubeA
    lies on cubeB's lattice this is the exact discrete sum over all voxel pairs; otherwise
    V_B is interpolated linearly onto cubeA's points. Raises FreeGridTooLarge for very
    large cubes. *pad_factor* is not used (a warning says so if it is given).
    boundary='periodic' reproduces the published workflow: cubeB's own grid padded
    by *pad_factor* (default 3) with a periodic Coulomb kernel. Periodic images of B then
    add to the potential, and parts of cubeA outside the padded grid see zero potential, so
    the coupling is off by about 1 % for close neighbours and by tens of percent for
    distant pairs (BODIPY tetramer: +23 % at 16.8 Å, -52 % at 24.7 Å).

    Returns a dict with J_eV, J_cm1, transition dipoles, and point-dipole
    and extended-dipole comparison values.
    """
    pad = _periodic_pad(boundary, pad_factor)
    if boundary == "free":
        J_eV = _free_coupling(cubeA, cubeB, dielectric)
    else:
        # --- FFT Coulomb potential of rho_B on cubeB's padded grid ---
        dxB, dyB, dzB = grid_spacing(cubeB)
        V_padded, V_origin = _fft_coulomb_potential(
            cubeB['rho'], dxB, dyB, dzB, cubeB['origin'], pad_factor=pad,
        )
        VB_at_A = _interpolate_potential(V_padded, V_origin, dxB, dyB, dzB, cubeA)
        # --- TDC Coulomb coupling: J = k_e * Σ ρ_A · φ_B · dV_A ---
        # V_padded carries a factor dV_B (it is built from ρ_B·dV_B), so rescale
        # by dV_A/dV_B to integrate over cubeA's voxels.
        dV_ratio = voxel_volume(cubeA) / voxel_volume(cubeB)
        J_eV = (KE_EV_ANG / dielectric) * float(np.sum(cubeA['rho'] * VB_at_A)) * dV_ratio

    return {
        'J_eV': J_eV,
        'J_cm1': J_eV * EV_TO_CM,
        'boundary': boundary,
        'pad_factor': pad if boundary == "periodic" else None,
        **dipole_models(cubeA, cubeB, dielectric),
    }


# ============================================================
# === Simplified TDC coupling (for one-dimer workflow) =======
# ============================================================

def tdc_coupling_fft_simple(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    dielectric: float = 1.0,
    pad_factor: Optional[int] = None,
    boundary: str = "free",
) -> Dict[str, float]:
    """Simplified TDC coupling for fragments sharing the same grid.

    Use this when cubeA and cubeB already share the same origin and
    voxel vectors (e.g. fragments split from a single dimer cube).
    boundary='free' (default) is the isolated problem: for fragment densities on
    disjoint voxels (as the nearest-atom split gives) it equals the direct sum over
    all voxel pairs. Where both densities share voxels, the FFT uses the mean 1/r
    over a voxel for coinciding points, which a point-pair sum cannot reproduce.
    'periodic' pads the shared grid by *pad_factor* (default 3) with a periodic kernel,
    as in the published workflow.
    """
    rhoA, rhoB = cubeA['rho'], cubeB['rho']
    same = (rhoA.shape == rhoB.shape and np.allclose(cubeA['origin'], cubeB['origin'])
            and all(np.allclose(cubeA[v], cubeB[v]) for v in ('vx', 'vy', 'vz')))
    if not same:
        raise ValueError("tdc_coupling_fft_simple needs cubes on the same grid; use tdc_coupling_fft")
    pad = _periodic_pad(boundary, pad_factor)
    if boundary == "free":
        J_eV = _free_coupling(cubeA, cubeB, dielectric)
        return {'J_eV': J_eV, 'J_cm1': J_eV * EV_TO_CM}

    dxA, dyA, dzA = grid_spacing(cubeA)
    V_padded, _ = _fft_coulomb_potential(
        rhoB, dxA, dyA, dzA, cubeA['origin'], pad_factor=pad,
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

"""TrESP charge fitting: constrained ridge regression from transition ESP."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from eecc.constants import BOHR_TO_ANG, E2_4PI_EPS0_eVA, EV_TO_CM
from eecc.io.charges import atomic_symbol
from eecc.io.cube import read_cube as _read_cube_dict
from eecc.tresp.esp import fft_coulomb_potential, chelpg_points, trilinear_on_grid
from eecc.tresp.multipoles import (
    grid_spacing, is_axis_aligned, transition_multipoles,
)


# ============================================================
# === Cube reader (Bohr, raw) ================================
# ============================================================

def _read_cube_bohr(filename: str):
    """Read a cube file keeping Bohr units (no conversion)."""
    cube = _read_cube_dict(filename, units='angstrom')  # 'angstrom' = no conversion
    atoms = [(Z, x, y, z) for (Z, _q, x, y, z) in cube['atoms']]
    origin = cube['origin']
    axes = np.array([cube['vx'], cube['vy'], cube['vz']])
    shape = cube['nv']
    rho = cube['rho']
    return atoms, origin, axes, shape, rho


# ============================================================
# === Constrained ridge TrESP fitting ========================
# ============================================================

def fit_transition_charges(
    P: np.ndarray,
    Vp: np.ndarray,
    atom_pos_bohr: np.ndarray,
    alpha: float = 1e-3,
    target_dipole_bohr_e: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Fit atomic transition charges by constrained ridge regression.

    Parameters
    ----------
    P : (M, 3) sampling points (Bohr)
    Vp : (M,) ESP values at P (Hartree/e)
    atom_pos_bohr : (N, 3) atomic coordinates (Bohr)
    alpha : ridge regularization
    target_dipole_bohr_e : (3,) target transition dipole (Bohr·e) or None

    Returns
    -------
    (q, rms) where q is (N,) fitted charges in e, rms is ESP fit RMS.
    """
    M = P.shape[0]
    N = atom_pos_bohr.shape[0]
    diff = P[:, None, :] - atom_pos_bohr[None, :, :]
    rij = np.linalg.norm(diff, axis=2) + 1e-12
    A = 1.0 / rij
    b = Vp.copy()

    ATA = A.T @ A + alpha * np.eye(N)
    ATb = A.T @ b

    # Build constraints
    C_rows = [np.ones(N)]
    d_vals = [0.0]
    if target_dipole_bohr_e is not None:
        C_rows.append(atom_pos_bohr[:, 0])
        C_rows.append(atom_pos_bohr[:, 1])
        C_rows.append(atom_pos_bohr[:, 2])
        d_vals.extend(list(target_dipole_bohr_e))

    C = np.vstack(C_rows)
    d = np.array(d_vals, dtype=float)
    K = C.shape[0]

    # KKT system
    KKT = np.zeros((N + K, N + K), dtype=float)
    rhs = np.zeros(N + K, dtype=float)
    KKT[:N, :N] = ATA
    KKT[:N, N:] = C.T
    KKT[N:, :N] = C
    rhs[:N] = ATb
    rhs[N:] = d

    sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
    q = sol[:N]
    Vfit = A @ q
    rms = np.sqrt(np.mean((Vfit - b) ** 2))
    return q, rms


# ============================================================
# === High-level: fit TrESP from cube ========================
# ============================================================

def fit_tresp_from_cube(
    cube_path: str,
    shells_ang: Sequence[float] = (1.4, 1.8, 2.2),
    pps: int = 4000,
    alpha: float = 1e-3,
    enforce_dipole: bool = True,
    pad: int = 2,
) -> List[Tuple[str, float, float, float, float]]:
    """Extract TrESP charges from a single transition-density cube.

    Returns list of (Element, x_Å, y_Å, z_Å, q_e).
    """
    atoms, origin, axes, shape, rho = _read_cube_bohr(cube_path)
    if not is_axis_aligned(axes):
        print("[WARN] Cube axes are not strictly axis-aligned; proceeding with axis-aligned assumption.")

    # Compute transition ESP on grid
    dx, dy, dz = grid_spacing(axes, shape)
    V = fft_coulomb_potential(rho, dx, dy, dz, pad=pad)

    # Multipoles for constraints
    Q, mu, dV = transition_multipoles(rho, origin, axes)
    if abs(Q) > 1e-3:
        print(f"[WARN] Net transition charge Q={Q:+.3e} e (expected ~0).")
    target_mu = mu if enforce_dipole else None

    # Sample points and interpolate ESP
    P = chelpg_points(atoms, shells_ang=shells_ang, pps=pps, seed=1)
    Vp, mask = trilinear_on_grid(V, origin, axes, P)
    if np.count_nonzero(mask) < 100:
        print("[WARN] Very few CHELPG points fall inside the cube.")
    P_use = P[mask]
    V_use = Vp[mask]

    # Fit charges
    atom_pos_bohr = np.array([[x, y, z] for (Z, x, y, z) in atoms], dtype=float)
    q, rms = fit_transition_charges(P_use, V_use, atom_pos_bohr, alpha=alpha,
                                    target_dipole_bohr_e=target_mu)

    # Report diagnostics
    print(f"[TrESP] RMS(ESP) = {rms:.3e} Hartree/e")
    print(f"[TrESP] Sum(q)   = {np.sum(q):+.3e} e (should be 0)")
    if enforce_dipole:
        mu_q = atom_pos_bohr.T @ q
        dnorm = np.linalg.norm(mu_q - mu)
        print(f"[TrESP] ||mu_fit - mu_target|| = {dnorm:.3e} Bohr*e")

    # Prepare output in charges.txt format (Å, e)
    out: List[Tuple[str, float, float, float, float]] = []
    for (Z, x, y, z), qi in zip(atoms, q):
        out.append((atomic_symbol(Z), x * BOHR_TO_ANG, y * BOHR_TO_ANG, z * BOHR_TO_ANG, float(qi)))
    return out


# ============================================================
# === TrESP coupling from charges ============================
# ============================================================

def compute_J_from_charges(
    fragA_atoms: List[Tuple[str, float, float, float, float]],
    fragB_atoms: List[Tuple[str, float, float, float, float]],
    eps_r: float = 1.0,
) -> Tuple[float, float]:
    """Compute Coulombic exciton coupling from two sets of per-atom charges (Å, e).

    Returns (J_eV, J_cm1).
    """
    RA = np.array([[a[1], a[2], a[3]] for a in fragA_atoms], dtype=float)
    RB = np.array([[b[1], b[2], b[3]] for b in fragB_atoms], dtype=float)
    qA = np.array([a[4] for a in fragA_atoms], dtype=float)
    qB = np.array([b[4] for b in fragB_atoms], dtype=float)

    J_eV = 0.0
    for i in range(len(qA)):
        d = np.linalg.norm(RA[i][None, :] - RB, axis=1)
        valid = d > 1e-10
        if not np.any(valid):
            continue
        J_eV += np.sum((E2_4PI_EPS0_eVA / eps_r) * qA[i] * qB[valid] / d[valid])
    J_cm1 = J_eV * EV_TO_CM
    return J_eV, J_cm1

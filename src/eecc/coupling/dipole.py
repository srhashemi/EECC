"""Dipole-based coupling methods: point-dipole (cube-based) and extended-dipole."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from eecc.constants import EV_TO_CM, KE_EV_ANG, EANG_TO_DEBYE


# ============================================================
# === Fragment helpers =======================================
# ============================================================

def _fragment_arrays(
    fragment: List[Tuple],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return coordinates (Å) and charges (e) as numpy arrays from fragment list."""
    coords = np.array([[atom[1], atom[2], atom[3]] for atom in fragment], dtype=float)
    q = np.array([atom[4] for atom in fragment], dtype=float)
    return coords, q


def _choose_origin(coords: np.ndarray, mode: str = "centroid") -> np.ndarray:
    """Return origin r0 for the fragment."""
    if mode == "centroid":
        return coords.mean(axis=0)
    else:
        raise ValueError("Unsupported origin mode. Use 'centroid'.")


def neutralize_charges(
    q: np.ndarray, tol: float = 1e-8
) -> Tuple[np.ndarray, float]:
    """Enforce neutrality by removing the mean charge if residual sum is non-negligible."""
    qsum = float(q.sum())
    if abs(qsum) > tol:
        q = q - qsum / len(q)
    return q, qsum


# ============================================================
# === Transition dipole from charges =========================
# ============================================================

def compute_transition_dipole(
    fragment: List[Tuple],
    origin: Union[str, np.ndarray] = "centroid",
    enforce_neutrality: bool = True,
) -> Dict[str, object]:
    """Compute transition dipole from transition charges.

    Parameters
    ----------
    fragment : list[(elem, x, y, z, q)]
    origin : 'centroid' or explicit 3-vector
    enforce_neutrality : subtract mean charge to guarantee sum(q)=0

    Returns
    -------
    dict with keys: mu_eAng, mu_D, mu_mag_eAng, r0, qsum_before, qsum_after
    """
    R, q = _fragment_arrays(fragment)
    if enforce_neutrality:
        q, qsum_before = neutralize_charges(q)
    else:
        qsum_before = float(q.sum())

    r0 = _choose_origin(R, origin) if isinstance(origin, str) else np.asarray(origin, dtype=float)
    r = R - r0

    mu = (q[:, None] * r).sum(axis=0)
    mu_mag = float(np.linalg.norm(mu))
    mu_D = mu_mag * EANG_TO_DEBYE

    return {
        "mu_eAng": mu,
        "mu_mag_eAng": mu_mag,
        "mu_D": mu_D,
        "r0": r0,
        "qsum_before": qsum_before,
        "qsum_after": float(q.sum())
    }


# ============================================================
# === Dipole-dipole coupling =================================
# ============================================================

def dipole_dipole_coupling(
    fragment_A: List[Tuple],
    fragment_B: List[Tuple],
    origin: str = "centroid",
    enforce_neutrality: bool = True,
    dielectric: float = 1.0,
    return_details: bool = False,
) -> Union[float, Dict[str, object]]:
    """Compute excitonic coupling via dipole-dipole interaction. Returns J in cm⁻¹."""
    A = compute_transition_dipole(fragment_A, origin=origin, enforce_neutrality=enforce_neutrality)
    B = compute_transition_dipole(fragment_B, origin=origin, enforce_neutrality=enforce_neutrality)

    R = B["r0"] - A["r0"]
    R2 = float(np.dot(R, R))
    Rmag = float(np.sqrt(R2))

    if Rmag < 1e-6:
        raise ValueError("Fragments are (nearly) co-located; dipole-dipole formula breaks down.")

    muA = A["mu_eAng"]
    muB = B["mu_eAng"]

    term = (np.dot(muA, muB) / (Rmag ** 3)) - 3.0 * (np.dot(muA, R) * np.dot(muB, R)) / (Rmag ** 5)

    J_eV = (KE_EV_ANG / dielectric) * term
    J_cm1 = J_eV * EV_TO_CM

    if not return_details:
        return J_cm1

    return {
        "J_cm1": J_cm1,
        "J_eV": J_eV,
        "muA_eAng": muA,
        "muB_eAng": muB,
        "muA_D": A["mu_D"],
        "muB_D": B["mu_D"],
        "R_vec_AtoB_Ang": R,
        "R_mag_Ang": Rmag,
        "qsum_A_before": A["qsum_before"],
        "qsum_B_before": B["qsum_before"],
        "qsum_A_after": A["qsum_after"],
        "qsum_B_after": B["qsum_after"],
    }


# ============================================================
# === Extended dipole helpers ================================
# ============================================================

def fragment_dipole_length(
    fragment: List[Tuple],
    dip_info: Dict[str, object],
) -> float:
    """Effective dipole length of a fragment along its transition-dipole axis.

    Parameters
    ----------
    fragment : list of (elem, x, y, z, q)
    dip_info : dict returned by :func:`compute_transition_dipole`

    Returns
    -------
    l : float  (Å, >= 1.0 fallback if degenerate)
    """
    mu = dip_info["mu_eAng"]
    mu_mag = float(np.linalg.norm(mu))
    if mu_mag < 1e-30:
        return 1.0
    u = mu / mu_mag
    r0 = dip_info["r0"]
    coords = np.array([[a[1], a[2], a[3]] for a in fragment], dtype=float)
    l = float(np.ptp(np.dot(coords - r0, u)))
    if l < 1e-6:
        l = 1.0
    return l


# ============================================================
# === Extended dipole coupling ===============================
# ============================================================

def extended_dipole_coupling_formula(
    mu1: np.ndarray,
    mu2: np.ndarray,
    R: np.ndarray,
    l1: float,
    l2: float,
    dielectric: float = 1.0,
) -> float:
    """Compute finite-size dipole Coulomb coupling (Eq. 4.14). Returns cm⁻¹."""
    u1 = mu1 / np.linalg.norm(mu1)
    u2 = mu2 / np.linalg.norm(mu2)

    mu1_mag = np.linalg.norm(mu1)
    mu2_mag = np.linalg.norm(mu2)

    a = (l1 / 2.0) * u1
    b = (l2 / 2.0) * u2

    d1 = np.linalg.norm(R + b - a)
    d2 = np.linalg.norm(R - b + a)
    d3 = np.linalg.norm(R - b - a)
    d4 = np.linalg.norm(R + b + a)

    prefactor = (KE_EV_ANG / dielectric) * (mu1_mag * mu2_mag) / (l1 * l2)

    J_eV = prefactor * (1 / d1 + 1 / d2 - 1 / d3 - 1 / d4)
    J_cm1 = J_eV * EV_TO_CM
    return J_cm1


# ============================================================
# === Point-dipole coupling (from cube dipole vectors) =======
# ============================================================

def point_dipole_coupling(
    muA_eAng: np.ndarray,
    muB_eAng: np.ndarray,
    Rvec: np.ndarray,
    dielectric: float = 1.0,
) -> Tuple[float, float]:
    """Point-dipole Coulomb coupling in eV and cm⁻¹.

    Parameters
    ----------
    muA_eAng, muB_eAng : array
        Transition dipoles in e·Å.
    Rvec : array
        Vector from A to B in Å.

    Returns
    -------
    (J_eV, J_cm1)
    """
    Rvec = np.array(Rvec, float)
    R = np.linalg.norm(Rvec)
    if R == 0.0:
        return 0.0, 0.0

    muA = np.array(muA_eAng, float)
    muB = np.array(muB_eAng, float)
    Rhat = Rvec / R

    muA_dot_muB = np.dot(muA, muB)
    muA_dot_R = np.dot(muA, Rhat)
    muB_dot_R = np.dot(muB, Rhat)

    V_eAng = (muA_dot_muB - 3.0 * muA_dot_R * muB_dot_R) / (R ** 3)
    J_eV = (KE_EV_ANG / dielectric) * V_eAng
    J_cm1 = J_eV * EV_TO_CM
    return J_eV, J_cm1

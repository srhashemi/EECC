"""Coulomb-sum coupling from point charges and transition dipole calculation."""

from __future__ import annotations

from math import sqrt

import numpy as np

from eecc.constants import Angstrom, e_charge, KE_EV_ANG, EV_TO_CM


def transition_dipole(coords: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute transition dipole moment (C·m) from coordinates (Å) and charges (e)."""
    coords_m = coords * Angstrom
    q_C = q * e_charge
    return np.sum(q_C[:, None] * coords_m, axis=0)


def compute_J(
    fragA: list, fragB: list, dielectric: float = 1.0
) -> float:
    """Coulomb coupling from fragment lists ``[(elem, x, y, z, q), ...]``.

    Works directly in Å/e units and returns cm⁻¹.

    Parameters
    ----------
    fragA, fragB : list of (elem, x, y, z, q) tuples
    dielectric : relative dielectric constant (default 1.0)
    """
    J = 0.0

    for i, a in enumerate(fragA):
        for j, b in enumerate(fragB):
            dx = a[1] - b[1]
            dy = a[2] - b[2]
            dz = a[3] - b[3]
            r = sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-6:
                continue
            J += a[4] * b[4] / r

    J_cm = J * (KE_EV_ANG / dielectric) * EV_TO_CM
    return J_cm

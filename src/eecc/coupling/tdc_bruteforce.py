"""Brute-force TDC Coulomb coupling via direct double summation."""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from scipy.spatial.distance import cdist

from eecc.constants import KE_EV_ANG, EV_TO_CM
from eecc.io.cube import read_cube, grid_spacing, build_axes_coordinates
from eecc.coupling.tdc_fft import transition_dipole_from_cube


def _cube_charges(cube: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, charges) for all voxels: q = rho * dV."""
    xs, ys, zs = build_axes_coordinates(cube)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    q = cube['rho'].ravel() * np.prod(grid_spacing(cube))
    return pos, q


def tdc_coupling_bruteforce(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    dielectric: float = 1.0,
    threshold: float = 0.0005,
    chunk_size: int = 500,
) -> Dict[str, Any]:
    """Compute J = k_e * sum_ij q_A[i]*q_B[j] / |r_i - r_j| directly.

    Parameters
    ----------
    cubeA, cubeB : cube dicts
        Read with ``read_cube(..., units='bohr')``.
    dielectric : float
        Relative dielectric constant.
    threshold : float
        Keep voxels with ``|q| > threshold * max|q|``.  Lower = more
        accurate but slower.  0.0005 captures ~99% of |charge|.
    chunk_size : int
        Number of cubeA voxels processed per batch (controls memory).
    """
    posA, qA = _cube_charges(cubeA)
    posB, qB = _cube_charges(cubeB)

    max_qA = np.max(np.abs(qA))
    max_qB = np.max(np.abs(qB))
    maskA = np.abs(qA) > threshold * max_qA
    maskB = np.abs(qB) > threshold * max_qB

    posA_s, qA_s = posA[maskA], qA[maskA]
    posB_s, qB_s = posB[maskB], qB[maskB]

    captA = np.sum(np.abs(qA_s)) / np.sum(np.abs(qA))
    captB = np.sum(np.abs(qB_s)) / np.sum(np.abs(qB))

    nA, nB = len(qA_s), len(qB_s)

    J_sum = 0.0
    for i0 in range(0, nA, chunk_size):
        i1 = min(i0 + chunk_size, nA)
        D = cdist(posA_s[i0:i1], posB_s)
        D[D < 1e-10] = 1e-10
        J_sum += np.dot(qA_s[i0:i1], np.dot(1.0 / D, qB_s))

    J_eV = (KE_EV_ANG / dielectric) * J_sum
    J_cm1 = J_eV * EV_TO_CM

    return {
        'J_eV': J_eV,
        'J_cm1': J_cm1,
        'threshold': threshold,
        'nA': nA,
        'nB': nB,
        'capture_A': captA,
        'capture_B': captB,
    }


def run_bruteforce(
    cubeA_path: str,
    cubeB_path: str,
    dielectric: float = 1.0,
    threshold: float = 0.0005,
) -> None:
    """CLI entry: read cubes, compute brute-force J, print results."""
    print("\n=== Brute-Force TDC Coulomb Coupling ===")
    print(f"Reading: {cubeA_path}, {cubeB_path}")

    cubeA = read_cube(cubeA_path, units="bohr")
    cubeB = read_cube(cubeB_path, units="bohr")

    muA = transition_dipole_from_cube(cubeA)
    muB = transition_dipole_from_cube(cubeB)
    print(f"|mu_A| = {muA['mu_D']:.3f} D,  |mu_B| = {muB['mu_D']:.3f} D")

    import time
    t0 = time.time()
    result = tdc_coupling_bruteforce(
        cubeA, cubeB,
        dielectric=dielectric,
        threshold=threshold,
    )
    dt = time.time() - t0

    print(f"\nThreshold: {result['threshold']}")
    print(f"Voxels used: A={result['nA']:,}, B={result['nB']:,} "
          f"(pairs={result['nA']*result['nB']:.2e})")
    print(f"Density captured: A={result['capture_A']:.1%}, "
          f"B={result['capture_B']:.1%}")
    print(f"Time: {dt:.1f}s")
    print(f"\n[BF]  J = {result['J_cm1']:.2f} cm^-1  ({result['J_eV']:.6f} eV)")

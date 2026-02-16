"""Fragment index parsing and cube splitting for dimer workflows."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from eecc.io.cube import build_axes_coordinates


def parse_indices(text: str) -> List[int]:
    """Parse '1-56, 112-120, 122, 130-131' -> 0-based sorted unique indices list."""
    idx: Set[int] = set()
    for token in re.split(r"[,\s;]+", text.strip()):
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            a_int = int(a); b_int = int(b)
            if b_int < a_int:
                a_int, b_int = b_int, a_int
            idx.update(range(a_int - 1, b_int))
        else:
            idx.add(int(token) - 1)
    return sorted(idx)


def split_cube_by_nearest_atom(
    cube: Dict[str, Any],
    fragA_idx0: List[int],
    fragB_idx0: List[int],
    batch_voxels: int = 200000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dimer cube into monomer densities by nearest-atom assignment.

    Returns (maskA, maskB, rhoA, rhoB).
    """
    rho = cube['rho']
    nx, ny, nz = cube['nv']
    xs, ys, zs = build_axes_coordinates(cube)

    atom_coords = np.array([[a[2], a[3], a[4]] for a in cube['atoms']])

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    Nvox = P.shape[0]

    nearest_idx = np.empty(Nvox, dtype=int)
    for start in range(0, Nvox, batch_voxels):
        end = min(start + batch_voxels, Nvox)
        diffs = P[start:end, None, :] - atom_coords[None, :, :]
        nearest_idx[start:end] = np.argmin(np.sum(diffs ** 2, axis=2), axis=1)

    fragA_set = set(fragA_idx0)
    is_A = np.array([idx in fragA_set for idx in nearest_idx])

    maskA = is_A.reshape((nx, ny, nz))
    maskB = ~maskA

    return maskA, maskB, rho * maskA, rho * maskB


def build_monomer_cube(
    cube: Dict[str, Any],
    atom_indices: List[int],
    rho_masked: np.ndarray,
) -> Dict[str, Any]:
    """Build a monomer cube dict from a dimer cube, selected atoms, and masked density."""
    atoms = [cube['atoms'][i] for i in atom_indices]
    return {
        'origin': cube['origin'],
        'nv': cube['nv'],
        'vx': cube['vx'],
        'vy': cube['vy'],
        'vz': cube['vz'],
        'atoms': atoms,
        'rho': rho_masked
    }

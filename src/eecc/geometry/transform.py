"""Coordinate transformation utilities: centering, alignment."""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np


def center_of_mass(coords: np.ndarray) -> np.ndarray:
    """Return the geometric centroid (mean) of *coords*."""
    return np.mean(coords, axis=0)


def center_coordinates(cube: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new cube dict with atom coordinates centered at the centroid."""
    cube = cube.copy()
    atoms = cube["atoms"]

    coords = np.array([[a[2], a[3], a[4]] for a in atoms])
    centroid = coords.mean(axis=0)

    new_atoms = []
    for Z, charge, x, y, z in atoms:
        new_atoms.append([Z, charge, x - centroid[0], y - centroid[1], z - centroid[2]])

    cube["atoms"] = new_atoms
    cube["origin"] = cube["origin"] - centroid
    return cube


def align_molecules(
    cubeA: Dict[str, Any], cubeB: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Align cubeB to cubeA by translating both to their centroids (no rotation)."""
    return center_coordinates(cubeA), center_coordinates(cubeB)

"""Coordinate rotation utilities."""

from __future__ import annotations

import numpy as np


def rotate_coords(coords: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate *coords* around *axis* by *angle_deg* degrees (Rodrigues' formula)."""
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    theta = np.deg2rad(angle_deg)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return coords @ R.T


def rotate_around_point(
    coords: np.ndarray, point: np.ndarray, axis: np.ndarray, angle_deg: float
) -> np.ndarray:
    """Rotate *coords* about *point* around *axis* by *angle_deg* degrees."""
    shifted = coords - point
    rotated = rotate_coords(shifted, axis, angle_deg)
    return rotated + point


def align_to_z_axis(coords: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate *coords* so the transition dipole (from charges *q*) points along z."""
    from eecc.coupling.coulomb import transition_dipole

    mu = transition_dipole(coords, q)
    mu_mag = np.linalg.norm(mu)
    if mu_mag < 1e-30:
        return coords
    mu_norm = mu / mu_mag
    z_axis = np.array([0.0, 0.0, 1.0])

    v = np.cross(mu_norm, z_axis)
    s = np.linalg.norm(v)
    if s < 1e-10:
        return coords

    c = np.dot(mu_norm, z_axis)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return coords @ R.T

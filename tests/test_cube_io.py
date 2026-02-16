"""Tests for eecc.io.cube."""

import numpy as np
import pytest

from eecc.io.cube import (
    grid_spacing,
    voxel_volume,
    build_axes_coordinates,
    cube_atom_centroid,
    harmonize_cubes_inplace,
)


def test_grid_spacing(sample_cube):
    dx, dy, dz = grid_spacing(sample_cube)
    assert abs(dx - 0.2) < 1e-10
    assert abs(dy - 0.2) < 1e-10
    assert abs(dz - 0.2) < 1e-10


def test_voxel_volume(sample_cube):
    vol = voxel_volume(sample_cube)
    assert abs(vol - 0.008) < 1e-10


def test_build_axes_coordinates(sample_cube):
    xs, ys, zs = build_axes_coordinates(sample_cube)
    assert len(xs) == 4
    assert len(ys) == 4
    assert len(zs) == 4
    assert abs(xs[0] - 0.0) < 1e-10
    assert abs(xs[1] - 0.2) < 1e-10


def test_cube_atom_centroid(sample_cube):
    centroid = cube_atom_centroid(sample_cube)
    assert centroid.shape == (3,)
    assert abs(centroid[0] - 0.2) < 1e-10  # mean of 0.1 and 0.3


def test_harmonize_same_cubes():
    """Harmonizing identical cubes should be a no-op."""
    cube = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "nv": (3, 3, 3),
        "vx": np.array([0.1, 0.0, 0.0]),
        "vy": np.array([0.0, 0.1, 0.0]),
        "vz": np.array([0.0, 0.0, 0.1]),
        "rho": np.ones((3, 3, 3)),
    }
    cubeB = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "nv": (3, 3, 3),
        "vx": np.array([0.1, 0.0, 0.0]),
        "vy": np.array([0.0, 0.1, 0.0]),
        "vz": np.array([0.0, 0.0, 0.1]),
        "rho": np.ones((3, 3, 3)),
    }

    harmonize_cubes_inplace(cube, cubeB)
    np.testing.assert_array_almost_equal(cubeB['rho'], np.ones((3, 3, 3)))


def test_harmonize_grid_mismatch():
    """Should raise ValueError when grid sizes differ."""
    cubeA = {"nv": (3, 3, 3), "vx": np.array([0.1, 0., 0.]),
             "vy": np.array([0., 0.1, 0.]), "vz": np.array([0., 0., 0.1]),
             "origin": np.zeros(3), "rho": np.ones((3, 3, 3))}
    cubeB = {"nv": (4, 4, 4), "vx": np.array([0.1, 0., 0.]),
             "vy": np.array([0., 0.1, 0.]), "vz": np.array([0., 0., 0.1]),
             "origin": np.zeros(3), "rho": np.ones((4, 4, 4))}

    with pytest.raises(ValueError, match="Grid size mismatch"):
        harmonize_cubes_inplace(cubeA, cubeB)


def test_density_overlap_integration(sample_cube):
    """Basic overlap integral test (matches test_density_overlap.py intent)."""
    rho = sample_cube['rho']
    overlap = rho * rho
    dx, dy, dz = grid_spacing(sample_cube)
    dV = dx * dy * dz
    result = float(np.sum(overlap) * dV)
    assert result > 0  # self-overlap must be positive

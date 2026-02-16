"""Tests for eecc.geometry modules."""

import numpy as np

from eecc.geometry.rotation import rotate_coords, rotate_around_point, align_to_z_axis
from eecc.geometry.transform import center_of_mass, center_coordinates
from eecc.geometry.fragments import parse_indices, build_monomer_cube


def test_rotate_coords_identity():
    """Zero-degree rotation should be identity."""
    coords = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = rotate_coords(coords, [0, 0, 1], 0.0)
    np.testing.assert_array_almost_equal(result, coords)


def test_rotate_coords_90_z():
    """90-degree rotation around z should swap x and y."""
    coords = np.array([[1.0, 0.0, 0.0]])
    result = rotate_coords(coords, [0, 0, 1], 90.0)
    np.testing.assert_array_almost_equal(result, [[0.0, 1.0, 0.0]], decimal=10)


def test_rotate_around_point():
    """Rotating around a point should preserve distance to that point."""
    coords = np.array([[2.0, 0.0, 0.0]])
    point = np.array([1.0, 0.0, 0.0])
    result = rotate_around_point(coords, point, [0, 0, 1], 90.0)
    dist_before = np.linalg.norm(coords[0] - point)
    dist_after = np.linalg.norm(result[0] - point)
    assert abs(dist_before - dist_after) < 1e-10


def test_center_of_mass():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    com = center_of_mass(coords)
    np.testing.assert_array_almost_equal(com, [1.0, 0.0, 0.0])


def test_center_coordinates(sample_cube):
    centered = center_coordinates(sample_cube)
    # Centroid should now be at origin
    coords = np.array([[a[2], a[3], a[4]] for a in centered["atoms"]])
    centroid = coords.mean(axis=0)
    np.testing.assert_array_almost_equal(centroid, [0.0, 0.0, 0.0], decimal=10)


def test_parse_indices_simple():
    result = parse_indices("1-3")
    assert result == [0, 1, 2]


def test_parse_indices_complex():
    result = parse_indices("1-3, 5, 7-8")
    assert result == [0, 1, 2, 4, 6, 7]


def test_parse_indices_reversed():
    result = parse_indices("3-1")
    assert result == [0, 1, 2]


def test_parse_indices_semicolons():
    result = parse_indices("1-2; 4")
    assert result == [0, 1, 3]


def test_build_monomer_cube(sample_cube):
    cube = build_monomer_cube(sample_cube, [0], sample_cube['rho'])
    assert len(cube['atoms']) == 1
    assert cube['rho'].shape == sample_cube['rho'].shape


def test_align_to_z_axis_zero_charges():
    """Zero charges (no dipole) should return coords unchanged."""
    coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    q = np.array([0.0, 0.0])
    result = align_to_z_axis(coords, q)
    np.testing.assert_array_equal(result, coords)


def test_align_to_z_axis_already_along_z():
    """Dipole already along z should return coords unchanged."""
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    q = np.array([-1.0, 1.0])  # dipole points along +z
    result = align_to_z_axis(coords, q)
    np.testing.assert_array_almost_equal(result, coords, decimal=10)


def test_align_to_z_axis_dipole_along_x():
    """Dipole along x should rotate coords so dipole ends up along z."""
    coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    q = np.array([-1.0, 1.0])  # dipole points along +x
    result = align_to_z_axis(coords, q)
    # After rotation, the +charge atom should be at positive z
    assert result[1, 2] > result[0, 2]
    # x and y should be near zero for both atoms
    assert abs(result[0, 0]) < 1e-6
    assert abs(result[1, 0]) < 1e-6

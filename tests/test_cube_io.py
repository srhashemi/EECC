"""Tests for eecc.io.cube."""

import numpy as np
import pytest

from eecc.io.cube import (
    grid_spacing,
    voxel_volume,
    build_axes_coordinates,
    cube_atom_centroid,
    harmonize_cubes_inplace,
    read_cube,
)
from eecc.constants import A0_TO_ANG
from eecc.coupling.tdc_fft import transition_dipole_from_cube


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


def _write_gaussian_cube(path, rho, origin, step, atoms):
    """Write *rho* (nx, ny, nz) in Gaussian cube layout (z fastest, 6 values per line)."""
    nx, ny, nz = rho.shape
    lines = ["test cube", "written by test suite",
             f"{len(atoms):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}",
             f"{nx:5d} {step:12.6f} {0.0:12.6f} {0.0:12.6f}",
             f"{ny:5d} {0.0:12.6f} {step:12.6f} {0.0:12.6f}",
             f"{nz:5d} {0.0:12.6f} {0.0:12.6f} {step:12.6f}"]
    for Z, x, y, z in atoms:
        lines.append(f"{Z:5d} {float(Z):12.6f} {x:12.6f} {y:12.6f} {z:12.6f}")
    for ix in range(nx):
        for iy in range(ny):
            row = rho[ix, iy, :]
            for k in range(0, nz, 6):
                lines.append("".join(f"{v:13.5E}" for v in row[k:k + 6]))
    path.write_text("\n".join(lines) + "\n")


def test_read_cube_data_ordering(tmp_path):
    """Values must land at rho[ix, iy, iz] for a Gaussian (z-fastest) cube.

    Unequal grid dimensions make any axis mix-up detectable.
    """
    nx, ny, nz = 3, 4, 7
    ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    rho = (100 * ix + 10 * iy + iz + 1).astype(float)
    path = tmp_path / "order.cub"
    _write_gaussian_cube(path, rho, (0.0, 0.0, 0.0), 0.5, [(6, 0.0, 0.0, 0.0)])

    cube = read_cube(str(path), units="angstrom")
    assert cube["nv"] == (nx, ny, nz)
    np.testing.assert_allclose(cube["rho"], rho)


def test_read_cube_dipole_direction(tmp_path):
    """A density displaced along +x must give a transition dipole along +x."""
    nx, ny, nz = 21, 17, 13
    step = 0.3  # Bohr
    origin = -step * (np.array([nx, ny, nz]) - 1) / 2.0
    xs, ys, zs = (origin[i] + step * np.arange(n) for i, n in enumerate((nx, ny, nz)))
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    rho = X * np.exp(-(X**2 + Y**2 + Z**2))  # p_x-like transition density
    path = tmp_path / "px.cub"
    _write_gaussian_cube(path, rho, origin, step, [(6, 0.0, 0.0, 0.0)])

    cube = read_cube(str(path), units="bohr")
    assert abs(cube["vx"][0] - step * A0_TO_ANG) < 1e-6
    mu = transition_dipole_from_cube(cube)["mu"]
    assert mu[0] > 0
    assert abs(mu[1]) < 1e-3 * mu[0]
    assert abs(mu[2]) < 1e-3 * mu[0]

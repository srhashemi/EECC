"""Tests for eecc.tresp modules."""

import numpy as np

from eecc.tresp.multipoles import (
    is_axis_aligned,
    grid_spacing,
    grid_lengths,
    grid_coordinates,
    transition_multipoles,
)
from eecc.tresp.esp import vdW_radius_bohr, fft_coulomb_potential, chelpg_points, trilinear_on_grid
from eecc.tresp.fitting import fit_transition_charges, compute_J_from_charges


def test_is_axis_aligned():
    aligned = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    assert is_axis_aligned(aligned)

    not_aligned = np.array([[0.1, 0.01, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    assert not is_axis_aligned(not_aligned)


def test_grid_spacing_tresp():
    axes = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    shape = (10, 10, 10)
    dx, dy, dz = grid_spacing(axes, shape)
    # axes rows are step vectors; spacing = norm of step vector
    assert abs(dx - 0.3) < 1e-10


def test_grid_lengths():
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    shape = (10, 10, 10)
    Lx, Ly, Lz = grid_lengths(axes, shape)
    # grid_lengths = step * n = 1.0 * 10 = 10.0
    assert abs(Lx - 10.0) < 1e-10


def test_grid_coordinates():
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    origin = np.array([0.0, 0.0, 0.0])
    shape = (5, 5, 5)
    xs, ys, zs = grid_coordinates(origin, axes, shape)
    assert len(xs) == 5
    # step = 1.0, so xs = [0, 1, 2, 3, 4]
    assert abs(xs[1] - 1.0) < 1e-10


def test_vdW_radius_bohr():
    r_C = vdW_radius_bohr(6)
    assert r_C > 0
    # Carbon vdW ~ 1.70 Å ~ 3.21 Bohr
    assert 3.0 < r_C < 3.5


def test_chelpg_points():
    atoms = [(6, 0.0, 0.0, 0.0), (6, 2.0, 0.0, 0.0)]
    P = chelpg_points(atoms, shells_ang=(1.5,), pps=100)
    assert P.shape[1] == 3
    assert P.shape[0] > 0


def test_fft_coulomb_potential_tresp():
    rho = np.random.default_rng(42).standard_normal((6, 6, 6)) * 1e-3
    V = fft_coulomb_potential(rho, 0.2, 0.2, 0.2, pad=2)
    assert V.shape == rho.shape
    assert np.all(np.isfinite(V))


def test_fit_transition_charges():
    """Fit charges from a trivial ESP field."""
    atom_pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    # Make up some points
    P = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [-5.0, 0.0, 0.0], [0.0, -5.0, 0.0]])
    # ESP from a unit charge at atom 0
    Vp = 1.0 / np.linalg.norm(P - atom_pos[0], axis=1)
    q, rms = fit_transition_charges(P, Vp, atom_pos, alpha=0.01)
    assert len(q) == 2
    assert abs(q.sum()) < 1e-6  # neutrality enforced


def test_compute_J_from_charges():
    fragA = [("C", 0.0, 0.0, 0.0, 0.1)]
    fragB = [("C", 5.0, 0.0, 0.0, 0.1)]
    J_eV, J_cm1 = compute_J_from_charges(fragA, fragB)
    assert J_eV > 0
    assert J_cm1 > 0


def test_compute_J_from_charges_zero_distance():
    """Co-located atoms should be skipped, returning zero coupling."""
    fragA = [("C", 0.0, 0.0, 0.0, 0.1)]
    fragB = [("C", 0.0, 0.0, 0.0, 0.1)]
    J_eV, J_cm1 = compute_J_from_charges(fragA, fragB)
    assert J_eV == 0.0
    assert J_cm1 == 0.0


def test_trilinear_on_grid_at_grid_points():
    """Interpolation at grid points should return exact values."""
    V = np.arange(27, dtype=float).reshape(3, 3, 3)
    origin = np.array([0.0, 0.0, 0.0])
    # step = 0.5 Bohr per grid point
    axes = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    # Query at grid point (1, 1, 1) -> position (0.5, 0.5, 0.5)
    P = np.array([[0.5, 0.5, 0.5]])
    vals, mask = trilinear_on_grid(V, origin, axes, P)
    assert mask[0]
    assert abs(vals[0] - V[1, 1, 1]) < 1e-10


def test_trilinear_on_grid_midpoint():
    """Interpolation at midpoint should average neighbours."""
    V = np.zeros((3, 3, 3))
    V[0, 0, 0] = 0.0
    V[1, 0, 0] = 2.0
    origin = np.array([0.0, 0.0, 0.0])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Midpoint between (0,0,0) and (1,0,0) -> x=0.5
    P = np.array([[0.5, 0.0, 0.0]])
    vals, mask = trilinear_on_grid(V, origin, axes, P)
    assert mask[0]
    assert abs(vals[0] - 1.0) < 1e-10


def test_trilinear_on_grid_out_of_bounds():
    """Points outside the grid should be masked out."""
    V = np.ones((3, 3, 3))
    origin = np.array([0.0, 0.0, 0.0])
    axes = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    # Point way outside
    P = np.array([[10.0, 10.0, 10.0]])
    vals, mask = trilinear_on_grid(V, origin, axes, P)
    assert not mask[0]
    assert vals[0] == 0.0


def test_trilinear_on_grid_boundary():
    """Point at last grid point should be accepted."""
    V = np.ones((4, 4, 4)) * 7.0
    origin = np.array([0.0, 0.0, 0.0])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # Last grid point is at (3, 3, 3)
    P = np.array([[3.0, 3.0, 3.0]])
    vals, mask = trilinear_on_grid(V, origin, axes, P)
    assert mask[0]
    assert abs(vals[0] - 7.0) < 1e-10


def test_transition_multipoles_uniform():
    """Uniform density should give Q proportional to density * volume."""
    nx, ny, nz = 5, 5, 5
    rho = np.ones((nx, ny, nz)) * 2.0
    origin = np.array([0.0, 0.0, 0.0])
    # step = 0.1 Bohr
    axes = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])
    Q, mu, dV = transition_multipoles(rho, origin, axes)
    # dV = 0.1^3 = 0.001, total volume = 125 * 0.001 = 0.125
    assert abs(dV - 0.001) < 1e-12
    assert abs(Q - 2.0 * 125 * 0.001) < 1e-10


def test_transition_multipoles_dipole():
    """Asymmetric density should give nonzero dipole along the asymmetry axis."""
    nx, ny, nz = 5, 5, 5
    rho = np.zeros((nx, ny, nz))
    # Put positive density at high-x end, negative at low-x end
    rho[4, 2, 2] = 1.0
    rho[0, 2, 2] = -1.0
    origin = np.array([0.0, 0.0, 0.0])
    axes = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    Q, mu, dV = transition_multipoles(rho, origin, axes)
    # Q should be ~0 (balanced +/-)
    assert abs(Q) < 1e-10
    # Dipole should point along x (positive end at x=4)
    assert mu[0] > 0
    assert abs(mu[1]) < 1e-10
    assert abs(mu[2]) < 1e-10

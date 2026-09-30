"""Tests for eecc.coupling.tdc_fft."""

import numpy as np

from eecc.coupling.tdc_fft import (
    transition_dipole_from_cube,
    integrate_density_total_charge,
    neutralize_full_density,
    _fft_coulomb_potential,
)


def test_transition_dipole_from_cube(sample_cube):
    result = transition_dipole_from_cube(sample_cube)
    assert "mu" in result
    assert "mu_D" in result
    assert result["mu"].shape == (3,)


def test_integrate_density_total_charge(sample_cube):
    Q = integrate_density_total_charge(sample_cube)
    # Random density should integrate to near-zero (mean ~0)
    assert isinstance(Q, float)


def test_neutralize_full_density():
    rho = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    rho_neutral, mean = neutralize_full_density(rho)
    assert abs(mean - 4.5) < 1e-10
    assert abs(np.mean(rho_neutral)) < 1e-10


def test_fft_coulomb_potential_shape():
    """Output potential should have same shape as padded grid."""
    rho = np.random.default_rng(0).standard_normal((8, 8, 8)) * 1e-3
    origin = np.array([0.0, 0.0, 0.0])
    V_pad, pad_origin = _fft_coulomb_potential(rho, 0.2, 0.2, 0.2, origin, pad_factor=2)
    # Padded grid is 2x in each dimension
    assert V_pad.shape == (16, 16, 16)
    # Padded origin is shifted back
    assert pad_origin[0] < origin[0]


def test_fft_coulomb_potential_neutral_density():
    """For exactly neutral density, potential should be finite everywhere."""
    rho = np.random.default_rng(1).standard_normal((8, 8, 8))
    rho -= np.mean(rho)  # ensure neutrality
    origin = np.array([0.0, 0.0, 0.0])
    V_pad, _ = _fft_coulomb_potential(rho, 0.2, 0.2, 0.2, origin, pad_factor=2)
    assert np.all(np.isfinite(V_pad))


def _px_cube(center, step, n, grid_shift=0.0):
    """Cube dict (Å units) holding a p_x-like transition density centred at *center*.

    *grid_shift* moves the grid box (not the density or the atom) along x.
    """
    center = np.asarray(center, float)
    origin = center - step * (n - 1) / 2.0 + np.array([grid_shift, 0.0, 0.0])
    axis = np.arange(n) * step
    X, Y, Z = np.meshgrid(origin[0] + axis, origin[1] + axis, origin[2] + axis, indexing="ij")
    dx, dy, dz = X - center[0], Y - center[1], Z - center[2]
    rho = dx * np.exp(-(dx**2 + dy**2 + dz**2))
    return {
        "origin": origin,
        "nv": (n, n, n),
        "vx": np.array([step, 0.0, 0.0]),
        "vy": np.array([0.0, step, 0.0]),
        "vz": np.array([0.0, 0.0, step]),
        "atoms": [(6, 0.0, *center)],
        "rho": rho,
    }


def test_tdc_fft_independent_of_grid_spacing():
    """J must not depend on the voxel size of either cube."""
    from eecc.coupling.tdc_fft import tdc_coupling_fft

    A = _px_cube([0.0, 0.0, 0.0], 0.25, 41)
    B_same = _px_cube([8.0, 0.0, 0.0], 0.25, 41)
    B_fine = _px_cube([8.0, 0.0, 0.0], 0.20, 51)

    J_same = tdc_coupling_fft(A, B_same, pad_factor=2)["J_cm1"]
    J_fine = tdc_coupling_fft(A, B_fine, pad_factor=2)["J_cm1"]
    assert abs(J_fine - J_same) / abs(J_same) < 0.01


def test_tdc_fft_separation_from_atom_centroids():
    """The dipole-model separation is between atom centroids, not grid origins."""
    from eecc.coupling.tdc_fft import tdc_coupling_fft

    A = _px_cube([0.0, 0.0, 0.0], 0.25, 41)
    B = _px_cube([8.0, 0.0, 0.0], 0.20, 51, grid_shift=1.0)
    res = tdc_coupling_fft(A, B, pad_factor=2)
    assert abs(res["R_AB_Ang"] - 8.0) < 1e-9

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


def test_tdc_fft_free_boundary_matches_direct_sum_at_any_distance():
    """The free-space FFT reproduces the direct sum; the periodic one fails far apart."""
    from eecc.coupling.tdc_bruteforce import tdc_coupling_bruteforce
    from eecc.coupling.tdc_fft import tdc_coupling_fft

    # 7.2 Å boxes that do not overlap, so no grid points of A and B coincide
    A = _px_cube([0.0, 0.0, 0.0], 0.4, 19)
    for distance in (8.0, 24.8):  # 24.8 Å: cubeA lies outside B's periodic padded grid
        B = _px_cube([distance, 0.0, 0.0], 0.4, 19)
        direct = tdc_coupling_bruteforce(A, B, threshold=0.0)["J_cm1"]
        free = tdc_coupling_fft(A, B, boundary="free")["J_cm1"]
        assert abs(free - direct) / abs(direct) < 1e-3
    periodic = tdc_coupling_fft(A, B, pad_factor=3, boundary="periodic")["J_cm1"]
    assert abs(periodic) < 0.01 * abs(direct)


def test_monomer_separation_falls_back_to_density_centroids():
    """Cubes that both list all atoms of the dimer still give the right separation."""
    from eecc.io.cube import monomer_separation

    A = _px_cube([0.0, 0.0, 0.0], 0.25, 41)
    B = _px_cube([8.0, 0.0, 0.0], 0.25, 41)
    both = [(6, 0.0, 0.0, 0.0, 0.0), (6, 0.0, 8.0, 0.0, 0.0)]
    A["atoms"], B["atoms"] = both, list(both)
    assert abs(monomer_separation(A, B)[0] - 8.0) < 0.05
    A["atoms"], B["atoms"] = [], []
    assert abs(monomer_separation(A, B)[0] - 8.0) < 0.05


def test_tdc_fft_shared_grid_free_matches_direct_sum():
    """One dimer cube split into fragments (shared grid): the free boundary gives the exact discrete sum."""
    from eecc.coupling.tdc_bruteforce import tdc_coupling_bruteforce
    from eecc.coupling.tdc_fft import tdc_coupling_fft_simple

    A = _px_cube([0.0, 0.0, 0.0], 0.4, 41, grid_shift=4.0)  # both boxes span x = -4 ... 12 Å
    B = _px_cube([8.0, 0.0, 0.0], 0.4, 41, grid_shift=-4.0)
    assert np.allclose(A["origin"], B["origin"])
    direct = tdc_coupling_bruteforce(A, B, threshold=1e-6)["J_cm1"]
    free = tdc_coupling_fft_simple(A, B)["J_cm1"]  # free is the default
    periodic = tdc_coupling_fft_simple(A, B, pad_factor=3, boundary="periodic")["J_cm1"]
    assert abs(free - direct) / abs(direct) < 1e-3
    assert abs(periodic - direct) > abs(free - direct)


def test_cube_file_routes_default_to_free_boundary():
    """Every cube-file entry point uses the free (isolated) boundary unless told otherwise."""
    import inspect

    from eecc.coupling.tdc_fft import tdc_coupling_fft, tdc_coupling_fft_simple
    from eecc.workflows.tdc_one_dimer import run_one_dimer_tdc
    from eecc.workflows.tdc_two_monomers import run_tdc

    for f in (tdc_coupling_fft, tdc_coupling_fft_simple, run_tdc, run_one_dimer_tdc):
        assert inspect.signature(f).parameters["boundary"].default == "free", f.__name__

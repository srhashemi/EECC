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

    for opts in ({"boundary": "periodic", "pad_factor": 2}, {"boundary": "free"}):
        J_same = tdc_coupling_fft(A, B_same, **opts)["J_cm1"]
        J_fine = tdc_coupling_fft(A, B_fine, **opts)["J_cm1"]
        assert abs(J_fine - J_same) / abs(J_same) < 0.01, opts


def test_tdc_fft_separation_from_atom_centroids():
    """The dipole-model separation is between atom centroids, not grid origins."""
    from eecc.coupling.tdc_fft import tdc_coupling_fft

    A = _px_cube([0.0, 0.0, 0.0], 0.25, 41)
    B = _px_cube([8.0, 0.0, 0.0], 0.20, 51, grid_shift=1.0)
    res = tdc_coupling_fft(A, B)
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


def test_free_grid_does_not_grow_with_distance(monkeypatch):
    """The free-space grid holds nA + nB points per axis, so distant pairs cost the same as close ones."""
    from eecc.coupling import tdc_fft
    from eecc.coupling.tdc_bruteforce import tdc_coupling_bruteforce

    A = _px_cube([0.0, 0.0, 0.0], 0.4, 19)
    B = _px_cube([8.0, 0.0, 0.0], 0.4, 19)
    B_far = _px_cube([60.0, 3.2, -2.0], 0.4, 19)  # on the same lattice as A
    monkeypatch.setattr(tdc_fft, "FREE_FFT_MAX_POINTS", float(np.prod(tdc_fft._free_fft_shape((19,) * 3, (19,) * 3))))
    tdc_fft.check_free_grid(B, A)
    free = tdc_fft.tdc_coupling_fft(A, B_far)["J_cm1"]
    direct = tdc_coupling_bruteforce(A, B_far, threshold=0.0)["J_cm1"]
    assert abs(free - direct) / abs(direct) < 1e-6


def test_free_fft_misaligned_lattices_close_to_direct():
    """cubeA off cubeB's lattice: the potential is interpolated, still close to the direct sum."""
    from eecc.coupling.tdc_bruteforce import tdc_coupling_bruteforce
    from eecc.coupling.tdc_fft import tdc_coupling_fft

    A = _px_cube([0.0, 0.0, 0.0], 0.4, 19)
    B = _px_cube([8.13, 0.17, 0.0], 0.35, 21)
    free = tdc_coupling_fft(A, B)["J_cm1"]
    direct = tdc_coupling_bruteforce(A, B, threshold=0.0)["J_cm1"]
    assert abs(free - direct) / abs(direct) < 2e-3


def test_mean_inverse_distance_over_a_voxel():
    """Self-term of the kernel: the closed form matches the cube constant and a numerical average."""
    from eecc.coupling.tdc_fft import _mean_inv_r_box

    assert abs(_mean_inv_r_box((1.0, 1.0, 1.0)) - 2.3800772) < 1e-6
    h = np.array([0.2, 0.2, 0.4])
    n = 100  # even: midpoints never hit r = 0
    g = [(np.arange(n) + 0.5) / n * hk - hk / 2 for hk in h]
    r = np.sqrt(g[0][:, None, None] ** 2 + g[1][None, :, None] ** 2 + g[2][None, None, :] ** 2)
    assert abs(_mean_inv_r_box(h) - np.mean(1.0 / r)) / _mean_inv_r_box(h) < 5e-3


def test_pad_factor_with_free_boundary_warns():
    import pytest

    from eecc.coupling.tdc_fft import tdc_coupling_fft

    A = _px_cube([0.0, 0.0, 0.0], 0.4, 19)
    B = _px_cube([8.0, 0.0, 0.0], 0.4, 19)
    with pytest.warns(UserWarning, match="pad_factor"):
        tdc_coupling_fft(A, B, pad_factor=3)


def test_two_monomer_workflow_reports_nan_when_free_grid_too_large(monkeypatch, tmp_path, capsys):
    """No silent fallback to the periodic FFT: the FFT row is nan and the direct sum still runs."""
    from eecc.coupling import tdc_fft
    from eecc.workflows import tdc_two_monomers

    import os

    cubes = {"a.cub": _px_cube([0.0, 0.0, 0.0], 0.4, 11), "b.cub": _px_cube([8.0, 0.0, 0.0], 0.4, 11)}
    monkeypatch.setattr(tdc_two_monomers, "read_cube", lambda path, units: cubes[os.path.basename(path)])
    monkeypatch.setattr(tdc_fft, "FREE_FFT_MAX_POINTS", 10.0)
    monkeypatch.chdir(tmp_path)
    tdc_two_monomers.run_tdc("a.cub", "b.cub", threshold=0.0)
    out = capsys.readouterr().out
    assert "not computed (nan)" in out and "periodic" not in out
    row = [line for line in out.splitlines() if "TDC (FFT)" in line][0]
    assert "nan" in row


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


def test_one_dimer_free_coupling_does_not_depend_on_the_box():
    """A net charge leaked near the cut is removed where the density is, so the cube margin does not matter."""
    from eecc.coupling.tdc_fft import tdc_coupling_fft_simple
    from eecc.workflows.tdc_one_dimer import neutralize_fragment

    values = []
    for margin in (0.0, 2.0, 4.0):  # extra grid beyond the two densities (Å)
        n = int(round((16.0 + 2 * margin) / 0.4)) + 1
        A = _px_cube([0.0, 0.0, 0.0], 0.4, n, grid_shift=4.0)
        B = _px_cube([8.0, 0.0, 0.0], 0.4, n, grid_shift=-4.0)
        x = A["origin"][0] + 0.4 * np.arange(n)
        yz = 0.4 * (np.arange(n) - (n - 1) / 2)  # y and z axes of _px_cube, centred on 0
        X, Y, Z = np.meshgrid(x, yz, yz, indexing="ij")
        leak = 0.05 * np.exp(-((X - 4.0) ** 2 + Y ** 2 + Z ** 2))  # localized at the cut x = 4
        maskA = np.broadcast_to((x < 4.0)[:, None, None], A["rho"].shape)  # nearest-atom split at x = 4
        rhoA, _ = neutralize_fragment(np.where(maskA, A["rho"] + leak, 0.0), maskA)
        rhoB, _ = neutralize_fragment(np.where(~maskA, B["rho"] + leak, 0.0), ~maskA)
        assert abs(rhoA.sum()) < 1e-10 and abs(rhoB.sum()) < 1e-10
        values.append(tdc_coupling_fft_simple(dict(A, rho=rhoA), dict(B, rho=rhoB))["J_cm1"])
    assert max(values) - min(values) < 1e-6 * abs(values[0])


def test_simple_fft_rejects_different_grids():
    import pytest

    from eecc.coupling.tdc_fft import tdc_coupling_fft_simple

    with pytest.raises(ValueError, match="same grid"):
        tdc_coupling_fft_simple(_px_cube([0.0, 0.0, 0.0], 0.4, 19), _px_cube([8.0, 0.0, 0.0], 0.4, 19))

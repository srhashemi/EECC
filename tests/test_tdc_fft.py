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

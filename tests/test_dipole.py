"""Tests for eecc.coupling.dipole."""

import numpy as np
import pytest

from eecc.coupling.dipole import (
    compute_transition_dipole,
    dipole_dipole_coupling,
    extended_dipole_coupling_formula,
    fragment_dipole_length,
    point_dipole_coupling,
    neutralize_charges,
)


def test_neutralize_charges():
    q = np.array([0.1, 0.2, -0.1])
    q_neutral, qsum = neutralize_charges(q)
    assert abs(q_neutral.sum()) < 1e-12
    assert abs(qsum - 0.2) < 1e-12


def test_compute_transition_dipole(sample_fragment):
    result = compute_transition_dipole(sample_fragment)
    assert "mu_eAng" in result
    assert "mu_D" in result
    assert result["mu_D"] >= 0


def test_dipole_dipole_coupling_returns_float(sample_fragment):
    fragA = sample_fragment
    # Place fragB 10 Å away
    fragB = [(e, x + 10.0, y, z, q) for (e, x, y, z, q) in sample_fragment]
    J = dipole_dipole_coupling(fragA, fragB)
    assert isinstance(J, float)


def test_dipole_dipole_colocated_raises(sample_fragment):
    """Co-located fragments should raise ValueError."""
    with pytest.raises(ValueError, match="co-located"):
        dipole_dipole_coupling(sample_fragment, sample_fragment)


def test_extended_dipole_vs_point_dipole():
    """For very small l, extended dipole should approach point dipole."""
    mu1 = np.array([0.0, 0.0, 1.0])
    mu2 = np.array([0.0, 0.0, 1.0])
    R = np.array([10.0, 0.0, 0.0])

    J_ext = extended_dipole_coupling_formula(mu1, mu2, R, l1=0.001, l2=0.001)
    J_pd_eV, J_pd_cm1 = point_dipole_coupling(mu1, mu2, R)

    # Should be close for small dipole lengths
    assert abs(J_ext - J_pd_cm1) / abs(J_pd_cm1) < 0.01


def test_point_dipole_zero_distance():
    """Zero-distance should return (0, 0)."""
    mu = np.array([1.0, 0.0, 0.0])
    R = np.array([0.0, 0.0, 0.0])
    J_eV, J_cm1 = point_dipole_coupling(mu, mu, R)
    assert J_eV == 0.0
    assert J_cm1 == 0.0


def test_fragment_dipole_length():
    """Dipole length should span the fragment extent along the dipole axis."""
    # Two atoms along x, 10 Å apart, with +/- charges -> dipole along x
    frag = [("C", 0.0, 0.0, 0.0, 1.0), ("C", 10.0, 0.0, 0.0, -1.0)]
    dip = compute_transition_dipole(frag)
    l = fragment_dipole_length(frag, dip)
    # Extent along dipole axis should be ~10 Å
    assert abs(l - 10.0) < 0.1


def test_fragment_dipole_length_single_atom():
    """Single-atom fragment should return fallback length of 1.0."""
    frag = [("C", 0.0, 0.0, 0.0, 1.0)]
    dip = compute_transition_dipole(frag, enforce_neutrality=False)
    l = fragment_dipole_length(frag, dip)
    assert l == 1.0

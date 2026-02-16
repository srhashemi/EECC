"""Tests for eecc.coupling.coulomb."""

import numpy as np

from eecc.coupling.coulomb import compute_J, transition_dipole


def test_compute_J_zero_distance():
    """Coupling between co-located charges should skip zero-distance pairs."""
    fragA = [("C", 0.0, 0.0, 0.0, 1.0)]
    fragB = [("C", 0.0, 0.0, 0.0, 1.0)]
    J = compute_J(fragA, fragB)
    assert J == 0.0


def test_compute_J_two_charges():
    """Two unit charges 1 Å apart should give a known result."""
    fragA = [("C", 0.0, 0.0, 0.0, 1.0)]
    fragB = [("C", 1.0, 0.0, 0.0, 1.0)]
    J_cm = compute_J(fragA, fragB)
    # ke_eV_Ang ~ 14.4 eV, * EV_TO_CM ~ 8065.5 -> ~116,000 cm^-1
    assert J_cm > 100000


def test_compute_J_cm1():
    """compute_J should return value in cm^-1."""
    fragA = [("C", 0.0, 0.0, 0.0, 0.1)]
    fragB = [("C", 5.0, 0.0, 0.0, 0.1)]
    J_cm = compute_J(fragA, fragB)
    # Should be positive for same-sign charges
    assert J_cm > 0


def test_compute_J_dielectric():
    """compute_J with dielectric > 1 should reduce coupling."""
    fragA = [("C", 0.0, 0.0, 0.0, 0.1)]
    fragB = [("C", 5.0, 0.0, 0.0, 0.1)]
    J_vac = compute_J(fragA, fragB)
    J_eps = compute_J(fragA, fragB, dielectric=2.0)
    assert abs(J_eps - J_vac / 2.0) < 1e-6


def test_transition_dipole():
    """Dipole of a single charge at 1 Å along x should point along x."""
    coords = np.array([[1.0, 0.0, 0.0]])
    q = np.array([1.0])
    mu = transition_dipole(coords, q)
    # Should point along x (in C·m units)
    assert abs(mu[0]) > 0
    assert abs(mu[1]) < 1e-30
    assert abs(mu[2]) < 1e-30

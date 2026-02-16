"""Tests for eecc.constants."""

import numpy as np

from eecc.constants import (
    eps0, e_charge, Angstrom, A0_TO_ANG, ANG_TO_BOHR, BOHR_TO_ANG,
    J_to_eV, eV_to_cm1, EV_TO_CM, HARTREE_TO_EV,
    FOUR_PI, KE_EV_ANG, E2_4PI_EPS0_eVA,
    C_m_to_Debye, EANG_TO_DEBYE, DEBYE_PER_EANG,
)


def test_bohr_ang_consistency():
    """A0_TO_ANG and BOHR_TO_ANG should be reciprocal of ANG_TO_BOHR."""
    assert abs(A0_TO_ANG - BOHR_TO_ANG) < 1e-6
    assert abs(ANG_TO_BOHR * BOHR_TO_ANG - 1.0) < 1e-10


def test_eV_to_cm_variants():
    """Two variants of eV->cm^-1 must both be ~8065.5."""
    assert abs(eV_to_cm1 - 8065.54) < 0.01
    assert abs(EV_TO_CM - 8065.54) < 0.01


def test_four_pi():
    assert abs(FOUR_PI - 4.0 * np.pi) < 1e-12


def test_ke_ev_ang_reasonable():
    """Coulomb constant in eV*Ang should be ~14.4."""
    assert 14.3 < KE_EV_ANG < 14.5
    assert 14.3 < E2_4PI_EPS0_eVA < 14.5


def test_debye_conversion():
    assert abs(EANG_TO_DEBYE - DEBYE_PER_EANG) < 1e-8


def test_hartree_to_ev():
    assert 27.2 < HARTREE_TO_EV < 27.3

"""Shared fixtures for the eecc test suite."""

import numpy as np
import pytest


@pytest.fixture
def sample_cube():
    """Return a minimal cube dict for testing."""
    nx, ny, nz = 4, 4, 4
    rho = np.random.default_rng(42).standard_normal((nx, ny, nz)) * 1e-3
    return {
        "origin": np.array([0.0, 0.0, 0.0]),
        "nv": (nx, ny, nz),
        "vx": np.array([0.2, 0.0, 0.0]),
        "vy": np.array([0.0, 0.2, 0.0]),
        "vz": np.array([0.0, 0.0, 0.2]),
        "atoms": [
            (6, 0.0, 0.1, 0.1, 0.1),
            (6, 0.0, 0.3, 0.3, 0.3),
        ],
        "rho": rho,
    }


@pytest.fixture
def sample_fragment():
    """Return a sample fragment as list of (elem, x, y, z, q) tuples."""
    return [
        ("C", 0.0, 0.0, 0.0, 0.1),
        ("C", 1.4, 0.0, 0.0, -0.05),
        ("C", 0.7, 1.2, 0.0, -0.05),
    ]

"""XYZ file writing utilities."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def write_xyz(
    atoms: List[str],
    coords: np.ndarray,
    output_file: str,
    comment: str = "XYZ file",
) -> None:
    """Write an XYZ-format file from atom labels and coordinates (Å)."""
    N = len(atoms)
    with open(output_file, 'w') as f:
        f.write(f"{N}\n")
        f.write(f"{comment}\n")
        for atom, (x, y, z) in zip(atoms, coords):
            f.write(f"{atom} {x: .6f} {y: .6f} {z: .6f}\n")
    print(f"XYZ file written: {output_file}")


def write_dimer_xyz(
    atoms: List[str],
    coordsA: np.ndarray,
    coordsB: np.ndarray,
    output_file: str,
) -> None:
    """Write a dimer XYZ file by concatenating monomer A and B coordinates."""
    combined_atoms = atoms + atoms
    combined_coords = np.vstack([coordsA, coordsB])
    write_xyz(combined_atoms, combined_coords, output_file, comment="Dimer system (A + B)")

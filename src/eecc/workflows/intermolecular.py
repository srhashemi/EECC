"""Intermolecular coupling workflow: build oligomers and compute pairwise couplings."""

from __future__ import annotations

import os

import numpy as np

from eecc.constants import EV_TO_CM
from eecc.geometry.rotation import rotate_around_point, align_to_z_axis
from eecc.geometry.transform import center_of_mass
from eecc.io.charges import read_atoms
from eecc.io.xyz import write_xyz
from eecc.coupling.coulomb import compute_J
from eecc.coupling.dipole import (
    compute_transition_dipole,
    dipole_dipole_coupling,
    extended_dipole_coupling_formula,
    fragment_dipole_length,
)


# ============================================================
# === Placement parsing ======================================
# ============================================================

def _parse_placement(text):
    """Parse a placement string into position, axis, and angle.

    Format: ``"posX,posY,posZ axisX,axisY,axisZ angleDeg"``

    Returns
    -------
    pos : np.ndarray  (3,)
    axis : np.ndarray (3,)
    angle : float     (degrees)
    """
    parts = text.strip().split()
    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 space-separated groups (pos axis angle), got {len(parts)}: '{text}'"
        )
    pos = np.array([float(x) for x in parts[0].split(",")])
    axis = np.array([float(x) for x in parts[1].split(",")])
    angle = float(parts[2])
    return pos, axis, angle


# ============================================================
# === Geometry Preparation ===================================
# ============================================================

def _prepare_monomers(coords, charges, placements, axes, angles):
    """Generate multiple monomers from a single monomer."""
    base = coords - center_of_mass(coords)
    base = align_to_z_axis(base, charges)

    oligomer = []

    for pos, axis, ang in zip(placements, axes, angles):
        coords_i = base + pos
        coords_i = rotate_around_point(coords_i, pos, axis, ang)
        oligomer.append(coords_i)

    return oligomer


def _build_fragment(elements, coords, charges):
    """Build a fragment tuple list from arrays.

    Returns list of (elem, x, y, z, q) tuples compatible with dipole.py functions.
    """
    return [(el, c[0], c[1], c[2], q)
            for el, c, q in zip(elements, coords, charges)]


# ============================================================
# === Output Helpers =========================================
# ============================================================

def _prepare_output_dir(subfolder):
    """Create and return output directory inside /outputs."""
    base_dir = os.path.join("outputs", subfolder)
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _write_xyz_files(results_dir, atoms, oligomer_coords):
    """Write XYZ files for each monomer and the full oligomer."""
    for i, coords in enumerate(oligomer_coords):
        out_path = os.path.join(results_dir, f"monomer_{i+1}.xyz")
        write_xyz(atoms, coords, out_path, comment=f"Monomer {i+1} in oligomer")

    all_coords = np.vstack(oligomer_coords)
    all_atoms = atoms * len(oligomer_coords)

    full_path = os.path.join(results_dir, "oligomer.xyz")
    write_xyz(all_atoms, all_coords, full_path, comment="Full oligomer structure")


# ============================================================
# === Main Driver ============================================
# ============================================================

def run_intermolecular(
    input_file: str,
    placements_str: list[str],
    eps_r: float = 1.0,
) -> None:
    """Run the intermolecular coupling workflow.

    Parameters
    ----------
    input_file : str
        Monomer charge file name (looked up inside inputs/).
    placements_str : list of str
        Placement strings, each ``"posX,posY,posZ axisX,axisY,axisZ angleDeg"``.
    eps_r : float
        Relative dielectric constant.
    """
    input_path = os.path.join("inputs", input_file)

    # --- Read monomer ---
    atoms_list = read_atoms(input_path)
    elements = [a[0] for a in atoms_list]
    coords = np.array([[a[1], a[2], a[3]] for a in atoms_list])
    charges = np.array([a[4] for a in atoms_list])

    # --- Parse placements ---
    N = len(placements_str)
    if N < 2:
        print("Error: need at least 2 placements.")
        return

    placements = []
    axes = []
    angles = []
    for text in placements_str:
        pos, axis, angle = _parse_placement(text)
        placements.append(pos)
        axes.append(axis)
        angles.append(angle)

    # --- Build oligomer ---
    oligomer = _prepare_monomers(coords, charges, placements, axes, angles)

    # --- Build fragment tuples for each positioned monomer ---
    fragments = [_build_fragment(elements, oligomer[i], charges) for i in range(N)]

    # --- Compute transition dipoles ---
    dip_cache = {}
    for k, frag in enumerate(fragments):
        dip_cache[k] = compute_transition_dipole(frag, origin="centroid", enforce_neutrality=True)

    # --- Header ---
    mu_D = dip_cache[0]["mu_D"]
    base = os.path.splitext(os.path.basename(input_file))[0]
    print(f"\n=== Intermolecular Coupling (Oligomer) ===")
    print(f"  monomer: {input_path} ({len(atoms_list)} atoms)")
    print(f"  dielectric = {eps_r}")
    print(f"  N = {N} monomers")
    print()
    print("  Placements:")
    for i in range(N):
        pos = placements[i]
        ax = axes[i]
        ang = angles[i]
        print(f"    Mon {i+1}: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
              f"  axis=({ax[0]:.1f}, {ax[1]:.1f}, {ax[2]:.1f})"
              f"  angle={ang:.1f} deg")
    print()
    print(f"  Transition dipole: |mu| = {mu_D:.3f} D")
    print()

    # --- Compute all pairwise couplings ---
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            fragA = fragments[i]
            fragB = fragments[j]

            # Coulomb (TrESP / charge sum)
            J_coul = compute_J(fragA, fragB, dielectric=eps_r)

            # Point-dipole
            J_dd = dipole_dipole_coupling(
                fragA, fragB,
                origin="centroid",
                enforce_neutrality=True,
                dielectric=eps_r,
            )

            # Extended-dipole
            Rvec = dip_cache[j]["r0"] - dip_cache[i]["r0"]
            Rmag = float(np.linalg.norm(Rvec))

            mu1 = dip_cache[i]["mu_eAng"]
            mu2 = dip_cache[j]["mu_eAng"]
            l1 = fragment_dipole_length(fragA, dip_cache[i])
            l2 = fragment_dipole_length(fragB, dip_cache[j])

            J_ext = extended_dipole_coupling_formula(
                mu1, mu2, Rvec, l1, l2, dielectric=eps_r
            )

            pairs.append((i + 1, j + 1, J_coul, J_dd, J_ext, Rmag))

    # --- Print comparison table ---
    header = f"  {'Pair':>6s}  {'Coulomb':>12s}  {'Point-dipole':>14s}  {'Extended-dipole':>16s}  {'R (Ang)':>9s}"
    sep = "=" * len(header)
    print(sep)
    print(header)
    print("-" * len(header))
    for (fi, fj, Jc, Jdd, Jext, R) in pairs:
        pair = f"{fi}-{fj}"
        print(f"  {pair:>6s}  {Jc:12.2f}  {Jdd:14.2f}  {Jext:16.2f}  {R:9.2f}")
    print(sep)

    # --- Write output ---
    results_dir = _prepare_output_dir(f"intermolecular_{base}")
    _write_xyz_files(results_dir, elements, oligomer)

    out_path = os.path.join(results_dir, "results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Intermolecular Coupling Results\n")
        f.write(f"# monomer: {input_path}\n")
        f.write(f"# dielectric = {eps_r}\n")
        f.write(f"# N = {N} monomers\n")
        f.write("#\n")

        f.write("# Placements\n")
        for i in range(N):
            pos = placements[i]
            ax = axes[i]
            ang = angles[i]
            f.write(f"#   Mon {i+1}: pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})"
                    f"  axis=({ax[0]:.4f}, {ax[1]:.4f}, {ax[2]:.4f})"
                    f"  angle={ang:.4f} deg\n")
        f.write("#\n")

        f.write("# Transition dipoles\n")
        for k in range(N):
            mu = dip_cache[k]["mu_eAng"]
            mu_D_k = dip_cache[k]["mu_D"]
            f.write(
                f"#   Mon {k + 1}: "
                f"|mu| = {mu_D_k:.4f} D   "
                f"mu = ({mu[0]:.6f}, {mu[1]:.6f}, {mu[2]:.6f}) e*Ang\n"
            )
        f.write("#\n")

        for (fi, fj, Jc, Jdd, Jext, R) in pairs:
            f.write(f"# Pair {fi}-{fj}   R = {R:.4f} Ang\n")
            f.write(f"# {'Method':<20s} {'J (cm^-1)':>14s} {'J (eV)':>14s}\n")
            f.write(f"# {'-' * 50}\n")
            f.write(f"  {'Coulomb':<20s} {Jc:14.4f} {Jc / EV_TO_CM:14.8f}\n")
            f.write(f"  {'Point-dipole':<20s} {Jdd:14.4f} {Jdd / EV_TO_CM:14.8f}\n")
            f.write(f"  {'Extended-dipole':<20s} {Jext:14.4f} {Jext / EV_TO_CM:14.8f}\n")
            f.write("#\n")

    print(f"\n  Results saved to '{out_path}'")

"""Intramolecular coupling workflow: TrESP Coulomb + dipole + extended-dipole comparisons."""

from __future__ import annotations

import os
from typing import List

import numpy as np

from eecc.constants import EV_TO_CM
from eecc.io.charges import read_atoms, read_atoms_scaled
from eecc.geometry.fragments import parse_indices
from eecc.coupling.coulomb import compute_J
from eecc.coupling.dipole import (
    compute_transition_dipole,
    dipole_dipole_coupling,
    extended_dipole_coupling_formula,
    fragment_dipole_length,
)


# ============================================================
# === Helper Functions =======================================
# ============================================================

def _format_index_range(indices_0based: List[int]) -> str:
    """Format 0-based sorted indices as a compact 1-based range string.

    E.g. [0,1,2,3,55,56,57] -> '1-4,56-58'
    """
    if not indices_0based:
        return ""
    groups = []
    start = indices_0based[0]
    end = start
    for idx in indices_0based[1:]:
        if idx == end + 1:
            end = idx
        else:
            groups.append((start, end))
            start = idx
            end = idx
    groups.append((start, end))

    parts = []
    for s, e in groups:
        if s == e:
            parts.append(str(s + 1))
        else:
            parts.append(f"{s + 1}-{e + 1}")
    return ",".join(parts)


def _build_fragments(atoms, fragments_str: List[str]):
    """Parse fragment strings and extract atom tuples from the atom list.

    Returns (fragments, frag_indices) where fragments is a list of
    [(elem, x, y, z, q), ...] and frag_indices is the list of 0-based
    index lists.
    """
    fragments = []
    frag_indices = []
    used_atoms = set()

    for i, text in enumerate(fragments_str):
        idx = parse_indices(text)
        if not idx:
            raise ValueError(f"Fragment {i + 1} has no atoms (from '{text}')")
        max_idx = max(idx)
        if max_idx >= len(atoms):
            raise ValueError(
                f"Fragment {i + 1} references atom {max_idx + 1} but only "
                f"{len(atoms)} atoms loaded"
            )
        overlap = used_atoms.intersection(idx)
        if overlap:
            overlap_1based = sorted(k + 1 for k in overlap)
            print(f"  Warning: atoms {overlap_1based} appear in multiple fragments")
        used_atoms.update(idx)
        fragments.append([atoms[j] for j in idx])
        frag_indices.append(idx)

    return fragments, frag_indices


# ============================================================
# === Main Function ==========================================
# ============================================================

def run_intramolecular(
    filename: str,
    fragments_str: List[str],
    scale_factor: float = 1.0,
    dielectric: float = 1.0,
) -> None:
    """Run the intramolecular coupling workflow.

    Parameters
    ----------
    filename : str
        Charge file name (looked up inside inputs/).
    fragments_str : list of str
        Fragment atom indices as strings, e.g. ["1-56", "57-112,129-140"].
    scale_factor : float
        Charge scale factor.
    dielectric : float
        Relative dielectric constant.
    """
    filepath = os.path.join("inputs", filename)

    if scale_factor == 1.0:
        atoms = read_atoms(filepath)
    else:
        atoms = read_atoms_scaled(filepath, scale_factor)

    nfrag = len(fragments_str)
    if nfrag < 2:
        print("Error: need at least 2 fragments.")
        return

    fragments, frag_indices = _build_fragments(atoms, fragments_str)

    ORIGIN_MODE = "centroid"
    ENFORCE_NEUTRALITY = True

    # --- Compute transition dipoles ---
    dip_cache = {}
    for k, frag in enumerate(fragments):
        dip_cache[k] = compute_transition_dipole(
            frag, origin=ORIGIN_MODE, enforce_neutrality=ENFORCE_NEUTRALITY
        )

    # --- Header ---
    base = os.path.splitext(os.path.basename(filepath))[0]
    print(f"\n=== Intramolecular Coupling (TrESP Charges) ===")
    print(f"  charges: {filepath} ({len(atoms)} atoms)")
    if scale_factor != 1.0:
        print(f"  scale = {scale_factor}")
    print(f"  dielectric = {dielectric}")
    print()
    print("  Fragments:")
    for k in range(nfrag):
        idx = frag_indices[k]
        mu_D = dip_cache[k]["mu_D"]
        range_str = _format_index_range(idx)
        print(f"    Frag {k + 1}: atoms {range_str} ({len(idx)} atoms)"
              f"   |mu| = {mu_D:.3f} D")
    print()

    # --- Compute all pairwise couplings ---
    pairs = []
    for i in range(nfrag):
        for j in range(i + 1, nfrag):
            # Coulomb (TrESP)
            J_coul = compute_J(fragments[i], fragments[j], dielectric=dielectric)

            # Point-dipole
            J_dd = dipole_dipole_coupling(
                fragments[i], fragments[j],
                origin=ORIGIN_MODE,
                enforce_neutrality=ENFORCE_NEUTRALITY,
                dielectric=dielectric,
            )

            # Extended dipole
            Rvec = dip_cache[j]["r0"] - dip_cache[i]["r0"]
            Rmag = float(np.linalg.norm(Rvec))

            mu1 = dip_cache[i]["mu_eAng"]
            mu2 = dip_cache[j]["mu_eAng"]
            l1 = fragment_dipole_length(fragments[i], dip_cache[i])
            l2 = fragment_dipole_length(fragments[j], dip_cache[j])

            J_ext = extended_dipole_coupling_formula(
                mu1, mu2, Rvec, l1, l2, dielectric=dielectric
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

    # --- Write unified output file ---
    if scale_factor == 1.0:
        results_dir = os.path.join("outputs", f"intramolecular_{base}")
    else:
        results_dir = os.path.join("outputs", f"intramolecular_{base}_scaled_{scale_factor}")
    os.makedirs(results_dir, exist_ok=True)

    out_path = os.path.join(results_dir, "results.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Intramolecular Coupling Results\n")
        f.write(f"# charges: {filepath}\n")
        f.write(f"# dielectric = {dielectric}\n")
        f.write(f"# scale = {scale_factor}\n")
        f.write("#\n")

        f.write("# Transition dipoles\n")
        for k in range(nfrag):
            mu = dip_cache[k]["mu_eAng"]
            mu_D = dip_cache[k]["mu_D"]
            range_str = _format_index_range(frag_indices[k])
            f.write(
                f"#   Frag {k + 1} (atoms {range_str}): "
                f"|mu| = {mu_D:.4f} D   "
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

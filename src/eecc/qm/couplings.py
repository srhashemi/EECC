"""Pairwise excitonic couplings between fragments from the EECC coupling methods."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from eecc.constants import A0_TO_ANG, DEBYE_PER_EANG, EV_TO_CM
from eecc.coupling.coulomb import compute_J
from eecc.coupling.dipole import (
    extended_dipole_coupling_formula,
    fragment_dipole_length,
    point_dipole_coupling,
)
from eecc.io.charges import read_atoms

BOHR_TO_ANG = A0_TO_ANG

METHOD_LABELS = {
    "tdc_fft": "TDC (FFT)",
    "tdc_direct": "TDC (direct)",
    "tresp": "TrESP Coulomb",
    "mulliken": "TrMulliken Coulomb",
    "point_dipole": "Point dipole",
    "extended_dipole": "Extended dipole",
}


@dataclass
class FragmentData:
    name: str
    cube_path: str
    tresp: list  # [(el, x, y, z, q)] in Å, e
    mulliken: list
    mu_eAng: np.ndarray  # exact TDDFT transition dipole
    energy_eV: float
    f: float

    @property
    def centroid(self) -> np.ndarray:
        return np.array([a[1:4] for a in self.tresp], float).mean(axis=0)


def treat_cap_charges(atoms: list, caps: List[Tuple[int, int]], mode: str) -> list:
    """Apply the cap-charge treatment to a fragment's charge list.

    *caps* holds (cap index, capped-atom index) pairs, 0-based within the fragment.
    """
    if mode == "keep" or not caps:
        return atoms
    out = [list(a) for a in atoms]
    if mode == "merge":
        for cap, kept in caps:
            out[kept][4] += out[cap][4]
    elif mode != "drop":
        raise ValueError(f"unknown cap-charge mode '{mode}'")
    remove = {cap for cap, _ in caps}
    return [tuple(a) for i, a in enumerate(out) if i not in remove]


def load_fragment(transition_dir: str, td_dir: str, name: str,
                  caps: Optional[List[Tuple[int, int]]] = None,
                  cap_mode: str = "keep") -> FragmentData:
    with open(os.path.join(td_dir, "td.json")) as f:
        td = json.load(f)["selected"]
    caps = caps or []

    def charges(kind):
        atoms = read_atoms(os.path.join(transition_dir, f"{name}_{kind}.txt"))
        return treat_cap_charges(atoms, caps, cap_mode)

    return FragmentData(
        name=name,
        cube_path=os.path.join(transition_dir, f"{name}.cub"),
        tresp=charges("tresp"),
        mulliken=charges("mulliken"),
        mu_eAng=np.asarray(td["mu_au"]) * BOHR_TO_ANG,
        energy_eV=td["energy_eV"],
        f=td["f"],
    )


def pair_couplings(A: FragmentData, B: FragmentData, methods: List[str], ccfg,
                   cube_cache: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Couplings (cm^-1) between fragments *A* and *B* for each requested method."""
    eps = ccfg.dielectric
    out: Dict[str, float] = {}
    R = B.centroid - A.centroid

    if "tdc_fft" in methods or "tdc_direct" in methods:
        from eecc.io.cube import read_cube

        cache = cube_cache if cube_cache is not None else {}
        for frag in (A, B):
            if frag.name not in cache:
                cache[frag.name] = read_cube(frag.cube_path, units="bohr")
        cA, cB = cache[A.name], cache[B.name]
        if "tdc_fft" in methods:
            from eecc.coupling.tdc_fft import FreeGridTooLarge, tdc_coupling_fft
            pad = ccfg.tdc_pad if ccfg.tdc_boundary == "periodic" else None
            try:
                out["tdc_fft"] = tdc_coupling_fft(cA, cB, dielectric=eps, pad_factor=pad,
                                                  boundary=ccfg.tdc_boundary)["J_cm1"]
            except FreeGridTooLarge:  # grid too large for a free-space FFT: rely on the direct sum
                out["tdc_fft"] = float("nan")
        if "tdc_direct" in methods:
            from eecc.coupling.tdc_bruteforce import tdc_coupling_bruteforce
            out["tdc_direct"] = tdc_coupling_bruteforce(
                cA, cB, dielectric=eps, threshold=ccfg.tdc_direct_threshold)["J_cm1"]

    if "tresp" in methods:
        out["tresp"] = compute_J(A.tresp, B.tresp, dielectric=eps)
    if "mulliken" in methods:
        out["mulliken"] = compute_J(A.mulliken, B.mulliken, dielectric=eps)
    if "point_dipole" in methods:
        out["point_dipole"] = point_dipole_coupling(A.mu_eAng, B.mu_eAng, R, dielectric=eps)[1]
    if "extended_dipole" in methods:
        lA = fragment_dipole_length(A.tresp, {"mu_eAng": A.mu_eAng, "r0": A.centroid})
        lB = fragment_dipole_length(B.tresp, {"mu_eAng": B.mu_eAng, "r0": B.centroid})
        out["extended_dipole"] = extended_dipole_coupling_formula(
            A.mu_eAng, B.mu_eAng, R, lA, lB, dielectric=eps)
    return out


def run_couplings(fragments: List[FragmentData], ccfg, out_dir: str) -> Dict[str, Any]:
    """All pairwise couplings; writes ``couplings.txt`` and ``couplings.json``."""
    os.makedirs(out_dir, exist_ok=True)
    methods = [m for m in METHOD_LABELS if m in ccfg.methods]
    cache: Dict[str, Any] = {}
    pairs = []
    for i, A in enumerate(fragments):
        for B in fragments[i + 1:]:
            J = pair_couplings(A, B, methods, ccfg, cache)
            pairs.append({"pair": [A.name, B.name],
                          "R_Ang": float(np.linalg.norm(B.centroid - A.centroid)),
                          "J_cm-1": J})
        cache.pop(A.name, None)  # A appears in no later pair

    result = {
        "dielectric": ccfg.dielectric,
        "cap_charges": ccfg.cap_charges,
        "fragments": [{"name": f.name, "energy_eV": f.energy_eV, "f": f.f,
                       "mu_D": float(np.linalg.norm(f.mu_eAng) * DEBYE_PER_EANG),
                       "mu_eAng": f.mu_eAng.tolist()} for f in fragments],
        "pairs": pairs,
        "note": "Signs are arbitrary: the phase of each fragment's transition density is arbitrary.",
    }
    with open(os.path.join(out_dir, "couplings.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(out_dir, "couplings.txt"), "w", encoding="utf-8") as f:
        f.write(format_report(result, methods))
    return result


def format_report(result: Dict[str, Any], methods: List[str]) -> str:
    lines = ["# EECC couplings from automated TD-DFT", f"# dielectric = {result['dielectric']}",
             f"# cap charges (TrESP/TrMulliken) = {result.get('cap_charges', 'keep')}",
             "# " + result["note"], "#", "# Fragments"]
    for fr in result["fragments"]:
        lines.append(f"#   {fr['name']:<8s} E = {fr['energy_eV']:.4f} eV   f = {fr['f']:.4f}   "
                     f"|mu| = {fr['mu_D']:.3f} D")
    lines.append("#")
    header = f"  {'Pair':<14s} {'R (Å)':>8s}" + "".join(f" {METHOD_LABELS[m]:>19s}" for m in methods)
    lines += [header, "  " + "-" * (len(header) - 2)]
    for p in result["pairs"]:
        row = f"  {p['pair'][0] + '-' + p['pair'][1]:<14s} {p['R_Ang']:8.3f}"
        row += "".join(f" {p['J_cm-1'][m]:19.2f}" for m in methods)
        lines.append(row)
    lines.append("#")
    if any(np.isnan(p["J_cm-1"].get("tdc_fft", 0.0)) for p in result["pairs"]):
        lines.append("# nan: free-space FFT grid too large for this pair; use TDC (direct)")
    lines.append(f"# J in cm^-1 (1 eV = {EV_TO_CM} cm^-1)")
    return "\n".join(lines) + "\n"

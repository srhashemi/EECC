"""Representations of a transition density: cube file, transition Mulliken and TrESP charges.

Sign convention: the transition density T is treated as a charge density with
the sign of the electron density, as in the cube file. Charges from both
schemes therefore satisfy sum_A q_A R_A = trace(T r) = the TDDFT transition
dipole. Couplings do not depend on this convention because it applies to both
fragments.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from eecc.qm.excited import AU_TO_DEBYE, dipole_from_density_matrix, load_excited_states
from eecc.qm.pyscf_setup import build_mol
from eecc.qm.structure import Structure

BOHR_TO_ANG = 0.52917721092

# van der Waals radii (Å), Bondi 1964 (B, Si from Mantina et al. 2009).
VDW_RADII = {
    "H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Si": 2.10,
    "P": 1.80, "S": 1.80, "Cl": 1.75, "Br": 1.85, "I": 1.98,
}


# ============================================================
# === Cube file ==============================================
# ============================================================

def cube_grid(coords_bohr: np.ndarray, spacing: float, margin: float):
    """Axis-aligned grid with exactly *spacing* (Bohr) between points.

    Returns (origin, n, extent) with extent = (n - 1) * spacing.
    """
    lo = coords_bohr.min(axis=0) - margin
    hi = coords_bohr.max(axis=0) + margin
    n = np.ceil((hi - lo) / spacing).astype(int) + 1
    return lo, n, (n - 1) * spacing


def write_transition_cube(mol, T: np.ndarray, path: str, spacing: float = 0.25,
                          margin: float = 6.0, comment: str = "transition density") -> None:
    """Evaluate the transition density on a grid and write a Gaussian cube file."""
    from pyscf import lib
    from pyscf.dft import numint
    from pyscf.tools import cubegen

    origin, n, extent = cube_grid(mol.atom_coords(), spacing, margin)
    cc = cubegen.Cube(mol, *n, resolution=None, margin=margin, origin=origin, extent=extent)
    coords = cc.get_coords()
    ngrids = cc.get_ngrids()
    rho = np.empty(ngrids)
    blksize = min(8000, ngrids)
    ao_kind = "GTOval_cart" if mol.cart else "GTOval_sph"
    for p0, p1 in lib.prange(0, ngrids, blksize):
        ao = mol.eval_gto(ao_kind, coords[p0:p1])
        rho[p0:p1] = numint.eval_rho(mol, ao, T)
    cc.write(rho.reshape(cc.nx, cc.ny, cc.nz), path, comment=comment)


# ============================================================
# === Transition Mulliken charges ============================
# ============================================================

def mulliken_charges(mol, T: np.ndarray) -> np.ndarray:
    """Per-atom Mulliken partition of the transition density (density sign)."""
    pop = np.einsum("ij,ji->i", T, mol.intor_symmetric("int1e_ovlp"))
    q = np.zeros(mol.natm)
    for a, (_, _, p0, p1) in enumerate(mol.aoslice_by_atom()):
        q[a] = pop[p0:p1].sum()
    return q


# ============================================================
# === TrESP charges ==========================================
# ============================================================

def _fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.column_stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)])


def esp_fit_points(symbols: Sequence[str], coords_ang: np.ndarray,
                   shells: Sequence[float] = (1.4, 1.6, 1.8, 2.0),
                   density: float = 5.0) -> np.ndarray:
    """Merz-Kollman style fitting points (Å).

    For each scale factor s, points are placed on a sphere of radius s * R_vdW
    around every atom, with *density* points per Å^2. Points lying inside the
    scaled sphere of any atom are discarded.
    """
    radii = np.array([VDW_RADII.get(s, 2.0) for s in symbols])
    pts = []
    for s in shells:
        r_shell = s * radii
        for a, center in enumerate(coords_ang):
            n = max(8, int(round(density * 4.0 * np.pi * r_shell[a] ** 2)))
            cand = center + r_shell[a] * _fibonacci_sphere(n)
            d = np.linalg.norm(cand[:, None, :] - coords_ang[None, :, :], axis=2)
            keep = np.all(d >= r_shell[None, :] - 1e-8, axis=1)
            pts.append(cand[keep])
    return np.vstack(pts)


def density_potential(mol, T: np.ndarray, points_bohr: np.ndarray,
                      memory_mb: float = 2000.0) -> np.ndarray:
    """Electrostatic potential of the charge distribution T at *points_bohr* (a.u.)."""
    nao = mol.nao
    chunk = max(1, int(memory_mb * 1e6 / (8.0 * nao * nao)))
    V = np.empty(len(points_bohr))
    for p0 in range(0, len(points_bohr), chunk):
        ints = mol.intor("int1e_grids", grids=points_bohr[p0:p0 + chunk])
        V[p0:p0 + chunk] = np.einsum("gpq,pq->g", ints, T)
    return V


def tresp_charges(mol, T: np.ndarray, symbols: Sequence[str], cfg_tresp,
                  mu_au: Optional[np.ndarray] = None,
                  memory_mb: float = 2000.0) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit atomic charges to the electrostatic potential of T.

    Constraints: total charge 0 and, if requested, the transition dipole *mu_au*.
    """
    from eecc.tresp.fitting import fit_transition_charges

    atoms_bohr = mol.atom_coords()
    pts_ang = esp_fit_points(symbols, atoms_bohr * BOHR_TO_ANG, cfg_tresp.shells, cfg_tresp.density)
    pts = pts_ang / BOHR_TO_ANG
    V = density_potential(mol, T, pts, memory_mb)
    target = np.asarray(mu_au) if (cfg_tresp.dipole_constraint and mu_au is not None) else None
    q, rms = fit_transition_charges(pts, V, atoms_bohr, alpha=cfg_tresp.alpha,
                                    target_dipole_bohr_e=target)
    stats = {"n_points": int(len(pts)), "rms_au": float(rms),
             "rrms": float(rms / np.sqrt(np.mean(V ** 2)))}
    return q, stats


# ============================================================
# === Stage driver ===========================================
# ============================================================

def write_charge_file(path: str, symbols: Sequence[str], coords_ang: np.ndarray,
                      q: np.ndarray) -> None:
    """Write charges in EECC format: ``Element x(Å) y(Å) z(Å) q(e)``."""
    with open(path, "w") as f:
        for el, (x, y, z), qi in zip(symbols, coords_ang, q):
            f.write(f"{el:>2s} {x:12.6f} {y:12.6f} {z:12.6f} {qi:+14.8f}\n")


def run_transition(td_dir: str, cfg, out_dir: str, name: str, charge: int = 0) -> Dict[str, Any]:
    """Write the cube file and both charge sets for one fragment."""
    os.makedirs(out_dir, exist_ok=True)
    td = load_excited_states(td_dir)
    structure = Structure(td["symbols"], td["coords_ang"])
    mol = build_mol(structure, cfg.td.basis, cfg.td.cart, charge, 0,
                    cfg.resources.memory_mb, verbose=0)
    T = td["T"]
    k = td["state"] - 1
    mu_exact = td["tdm_au"][k]

    cube_path = os.path.join(out_dir, f"{name}.cub")
    write_transition_cube(mol, T, cube_path, cfg.transition.cube.spacing,
                          cfg.transition.cube.margin,
                          comment=f"{name} S0->S{td['state']} transition density (EECC/PySCF)")

    q_mull = mulliken_charges(mol, T)
    q_tresp, stats = tresp_charges(mol, T, structure.symbols, cfg.transition.tresp, mu_exact,
                                   memory_mb=0.25 * cfg.resources.memory_mb)
    write_charge_file(os.path.join(out_dir, f"{name}_mulliken.txt"),
                      structure.symbols, structure.coords, q_mull)
    write_charge_file(os.path.join(out_dir, f"{name}_tresp.txt"),
                      structure.symbols, structure.coords, q_tresp)

    atoms_bohr = mol.atom_coords()
    from eecc.io.cube import read_cube
    from eecc.coupling.tdc_fft import transition_dipole_from_cube
    mu_cube = transition_dipole_from_cube(read_cube(cube_path, units="bohr"))["mu"] / BOHR_TO_ANG

    def summary(mu):
        return {"mu_au": np.asarray(mu).tolist(), "mu_D": float(np.linalg.norm(mu) * AU_TO_DEBYE)}

    info = {
        "name": name,
        "state": td["state"],
        "exact": summary(mu_exact),
        "from_T": summary(dipole_from_density_matrix(mol, T)),
        "cube": summary(mu_cube),
        "mulliken": {**summary(atoms_bohr.T @ q_mull), "sum_q": float(q_mull.sum())},
        "tresp": {**summary(atoms_bohr.T @ q_tresp), "sum_q": float(q_tresp.sum()), **stats},
    }
    with open(os.path.join(out_dir, f"{name}_transition.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info

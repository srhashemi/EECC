"""Excited states of one fragment: SCF + TDDFT and the S0 -> Sn transition density matrix."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import numpy as np

from eecc.constants import AU_TO_DEBYE, HARTREE_TO_EV
from eecc.qm.pyscf_setup import build_mol, build_rks, make_td, resolve_xc, set_grid, use_cartesian
from eecc.qm.structure import Structure



def transition_density_matrix(mf, xy) -> np.ndarray:
    """AO transition density matrix T for one state of a restricted closed-shell TDDFT.

    PySCF normalizes the amplitudes of one spin to sum(X^2 - Y^2) = 1/2, so the
    spin-summed transition density is T = 2 C_occ (X + Y) C_vir^T. With this
    prefactor, the trace of T with the dipole integrals reproduces
    ``td.transition_dipole()``. T is returned symmetrized, which leaves the
    density itself unchanged.
    """
    x, y = xy
    if np.isscalar(y):
        y = np.zeros_like(x)
    mo, occ = mf.mo_coeff, mf.mo_occ
    orbo, orbv = mo[:, occ > 0], mo[:, occ == 0]
    T = 2.0 * orbo @ (x + y) @ orbv.T
    return 0.5 * (T + T.T)


def dipole_from_density_matrix(mol, T: np.ndarray) -> np.ndarray:
    """Transition dipole (a.u.) as the trace of T with the position integrals."""
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        r = mol.intor_symmetric("int1e_r", comp=3)
    return np.einsum("xpq,pq->x", r, T)


def run_excited_states(
    structure: Structure, cfg, workdir: str, name: str = "fragment", charge: int = 0,
) -> Dict[str, Any]:
    """Run SCF + TDDFT on *structure* with the ``td`` settings; write ``td.npz``/``td.json``."""
    td_cfg = cfg.td
    os.makedirs(workdir, exist_ok=True)
    mol = build_mol(structure, td_cfg.basis, td_cfg.cart, charge, 0,
                    cfg.resources.memory_mb, output=os.path.join(workdir, "td.log"))
    mf = build_rks(mol, td_cfg.xc, td_cfg.grid, td_cfg.density_fit, td_cfg.conv_tol,
                   disp=None, chkfile=os.path.join(workdir, "scf.chk"))
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"{name}: SCF did not converge")

    if td_cfg.response_grid:
        set_grid(mf, td_cfg.response_grid)  # XC kernel of the response only
    td = make_td(mf, td_cfg.method, td_cfg.nstates, td_cfg.davidson_tol)
    td.kernel()
    conv = np.atleast_1d(td.converged)
    k = td_cfg.state - 1
    if not conv[k]:
        raise RuntimeError(f"{name}: TDDFT state {td_cfg.state} did not converge")

    energies = np.asarray(td.e) * HARTREE_TO_EV
    osc = np.asarray(td.oscillator_strength())
    tdm = np.asarray(td.transition_dipole())
    T = transition_density_matrix(mf, td.xy[k])

    mu_T = dipole_from_density_matrix(mol, T)
    err = np.linalg.norm(mu_T - tdm[k])
    if err > 1e-6 * max(1.0, np.linalg.norm(tdm[k])):
        raise RuntimeError(f"{name}: transition density does not reproduce the TDDFT "
                           f"transition dipole (|diff| = {err:.2e} au)")

    np.savez_compressed(
        os.path.join(workdir, "td.npz"),
        T=T, energies_eV=energies, osc=osc, tdm_au=tdm, state=td_cfg.state,
        symbols=np.array(structure.symbols), coords_ang=structure.coords,
    )
    summary = {
        "name": name,
        "natoms": len(structure),
        "nao": int(mol.nao),
        "xc": resolve_xc(td_cfg.xc),
        "basis": td_cfg.basis,
        "cartesian": use_cartesian(td_cfg.basis, td_cfg.cart),
        "method": td_cfg.method,
        "grid": list(td_cfg.grid),
        "response_grid": list(td_cfg.response_grid) if td_cfg.response_grid else None,
        "davidson_tol": td_cfg.davidson_tol,
        "scf_energy_Eh": float(mf.e_tot),
        "state": td_cfg.state,
        "excitation_energies_eV": energies.tolist(),
        "oscillator_strengths": osc.tolist(),
        "transition_dipoles_au": tdm.tolist(),
        "selected": {
            "energy_eV": float(energies[k]),
            "f": float(osc[k]),
            "mu_au": tdm[k].tolist(),
            "mu_D": float(np.linalg.norm(tdm[k]) * AU_TO_DEBYE),
        },
    }
    with open(os.path.join(workdir, "td.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_excited_states(workdir: str) -> Dict[str, Any]:
    data = np.load(os.path.join(workdir, "td.npz"))
    out = {k: data[k] for k in data.files}
    out["symbols"] = [str(s) for s in out["symbols"]]
    out["state"] = int(out["state"])
    return out

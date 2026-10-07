"""Whole-system TDDFT and its diabatization onto the fragment states (pipeline stages 'system' and 'diabatize').

The whole oligomer is computed with the functional, basis and grids of the ``td``
settings, so its states and the fragment states are directly comparable. The
diabatization (see :mod:`eecc.qm.diabatize`) then gives the exciton Hamiltonian:
site energies and total couplings (Coulomb plus exchange, overlap, polarization
and, with CT states, charge-transfer mixing), next to the Coulomb couplings of
the fragment pipeline.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict

import numpy as np

from eecc.constants import AU_TO_DEBYE, EV_TO_CM, HARTREE_TO_EV
from eecc.qm.pyscf_setup import build_mol, build_rks
from eecc.qm.structure import Structure, read_xyz, save_xyz


def run_system_td(structure: Structure, cfg, workdir: str, nstates: int,
                  log: Callable[[str], None] = print) -> Dict[str, Any]:
    """SCF + TDDFT of the whole system; writes system.xyz, system_td.npz and system_td.json."""
    td_cfg = cfg.td
    os.makedirs(workdir, exist_ok=True)
    save_xyz(structure, os.path.join(workdir, "system.xyz"))
    mol = build_mol(structure, td_cfg.basis, td_cfg.cart, cfg.charge, 0,
                    cfg.system.memory_mb or cfg.resources.memory_mb,
                    output=os.path.join(workdir, "system_td.log"))
    mf = build_rks(mol, td_cfg.xc, td_cfg.grid, td_cfg.density_fit, td_cfg.conv_tol,
                   chkfile=os.path.join(workdir, "system_scf.chk"))
    t0 = time.time()
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("system: SCF did not converge")
    log(f"[system] {len(structure)} atoms, {mol.nao} basis functions: SCF {(time.time() - t0) / 60:.1f} min")

    if td_cfg.response_grid:
        mf.grids.atom_grid = tuple(td_cfg.response_grid)
        mf.grids.build()
    td = mf.TDA() if td_cfg.method == "tda" else mf.TDDFT()
    td.nstates = nstates
    td.conv_tol = td_cfg.davidson_tol

    # log every response call: the TDDFT of a large system runs for many hours
    progress = {"calls": 0, "vectors": 0}
    gen_vind = td.gen_vind

    def logged_gen_vind(*args, **kwargs):
        vind, hdiag = gen_vind(*args, **kwargs)

        def vind_logged(x):
            t = time.time()
            out = vind(x)
            n = np.atleast_2d(x).shape[0]
            progress["calls"] += 1
            progress["vectors"] += n
            log(f"[system] response step {progress['calls']}: {n} vectors in {time.time() - t:.0f} s "
                f"({progress['vectors']} vectors so far)")
            return out
        return vind_logged, hdiag

    td.gen_vind = logged_gen_vind
    t0 = time.time()
    td.kernel()
    converged = np.atleast_1d(td.converged).astype(bool)
    log(f"[system] TDDFT {(time.time() - t0) / 60:.1f} min, {progress['vectors']} vectors")

    energies = np.asarray(td.e) * HARTREE_TO_EV
    osc = np.asarray(td.oscillator_strength())
    tdm = np.asarray(td.transition_dipole())
    x = np.array([xy[0] for xy in td.xy])
    y = np.array([np.zeros_like(xy[0]) if np.isscalar(xy[1]) else xy[1] for xy in td.xy])
    np.savez_compressed(os.path.join(workdir, "system_td.npz"), mo_coeff=mf.mo_coeff, mo_occ=mf.mo_occ,
                        energies_eV=energies, osc=osc, tdm_au=tdm, x=x, y=y)
    summary = {
        "natoms": len(structure), "nao": int(mol.nao), "scf_energy_Eh": float(mf.e_tot),
        "states": [{"energy_eV": float(e), "f": float(f), "mu_D": float(np.linalg.norm(m) * AU_TO_DEBYE),
                    "converged": bool(c)} for e, f, m, c in zip(energies, osc, tdm, converged)],
    }
    with open(os.path.join(workdir, "system_td.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # The highest Davidson roots are the ones that most often miss the tolerance; keep the
    # (hours-long) result and report them instead of discarding everything.
    if not converged.all():
        log(f"[system] WARNING: TDDFT states {[i + 1 for i in np.flatnonzero(~converged)]} "
            f"of {nstates} did not converge")
    return summary


UNITS = {"H_eV": "eV", "adiabatic_eV": "eV", "transition_dipoles_au": "e*bohr (atomic units)",
         "J_*_cm-1": "cm^-1", "center_ang": "Angstrom", "principal_axes": "unit vectors"}
CONVENTIONS = {
    "frame": "Cartesian frame of system.xyz (the whole-system geometry)",
    "signs": "the phase of every LE state follows its fragment's transition density, so the signs of couplings "
             "and dipoles are arbitrary per state but consistent with each other",
    "ct_states": "CT(a->b): one electron from the HOMO of fragment a to the LUMO of fragment b",
    "principal_axes": "rows, ordered by increasing moment of inertia (fragment atoms without caps, atomic masses); "
                      "for a planar molecule the last axis is the plane normal; each axis has its largest "
                      "component positive",
}


def fragment_geometry(structure, n_own: int) -> Dict[str, Any]:
    """Centre of mass (Angstrom) and principal axes of a fragment's own atoms (caps excluded)."""
    from pyscf.data import elements

    xyz = np.asarray(structure.coords[:n_own], float)
    m = np.array([elements.MASSES[elements.charge(s)] for s in structure.symbols[:n_own]])
    com = m @ xyz / m.sum()
    r = xyz - com
    inertia = np.eye(3) * np.sum(m * np.sum(r ** 2, axis=1)) - (r.T * m) @ r
    _, axes = np.linalg.eigh(inertia)
    axes = axes.T
    axes *= np.sign(axes[np.arange(3), np.argmax(np.abs(axes), axis=1)])[:, None]
    return {"center_ang": com.tolist(), "principal_axes": axes.tolist()}


def state_info(label: str) -> Dict[str, Any]:
    """Type and fragment(s) of a diabatic state from its label, e.g. LE(frag1) or CT(frag1->frag2)."""
    kind, inner = label.split("(", 1)
    inner = inner.rstrip(")")
    if kind == "CT":
        donor, acceptor = inner.split("->")
        return {"label": label, "type": "CT", "donor": donor, "acceptor": acceptor}
    return {"label": label, "type": "LE", "fragment": inner}


def run_diabatization(pipe, workdir: str) -> Dict[str, Any]:
    """Diabatize the system states onto the fragments of *pipe*; writes diabatic.json/.txt."""
    from eecc.qm.diabatize import (diabatic_dipoles, diabatic_oscillator_strengths, diabatize_from_pipeline,
                                   format_dipoles, format_hamiltonian)

    cfg = pipe.cfg
    sys_dir = pipe.stage_dir("system")
    data = np.load(os.path.join(sys_dir, "system_td.npz"))
    mol = build_mol(read_xyz(os.path.join(sys_dir, "system.xyz")), cfg.td.basis, cfg.td.cart,
                    cfg.charge, 0, cfg.resources.memory_mb, verbose=0)
    names = pipe.fragment_names()
    n_frag, n_states = len(names), len(data["energies_eV"])
    include_ct = cfg.system.include_ct
    notes = []
    if include_ct and n_frag * n_frag > n_states:  # n LE + n(n-1) CT states
        include_ct = False
        notes.append(f"# NOTE: LE states only; LE + CT needs {n_frag * n_frag} system states "
                     f"(have {n_states}, increase system.nstates)")
        pipe.log("[diabatize] " + notes[-1][2:])
    res = diabatize_from_pipeline(pipe, mol, data["mo_coeff"], data["mo_occ"], data["x"], data["y"],
                                  data["energies_eV"], include_ct=include_ct)
    states = json.load(open(os.path.join(sys_dir, "system_td.json")))["states"]
    # results older than the 'converged' flag: the system stage then refused unconverged states
    unconverged = [i + 1 for i, s in enumerate(states) if not s.get("converged", True)]
    if unconverged:  # kept by the system stage; they enter the energies, couplings and dipoles below
        notes.append(f"# NOTE: system TDDFT states {unconverged} did not converge; they are used here")

    coulomb = {}
    cpath = os.path.join(pipe.stage_dir("couplings"), "couplings.json")
    if not pipe.is_done("couplings"):  # a stale file would not match these fragment states
        notes.append("# NOTE: couplings stage not finished; no Coulomb couplings to compare")
    elif os.path.exists(cpath):
        for p in json.load(open(cpath))["pairs"]:
            coulomb[tuple(p["pair"])] = p["J_cm-1"]
    pairs = []
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            entry = {"pair": [a, names[j]], "J_total_cm-1": float(res.H_eV[i, j] * EV_TO_CM)}
            entry.update({f"{m}_cm-1": v for m, v in coulomb.get((a, names[j]), {}).items()})
            pairs.append(entry)

    # transition dipoles of the diabatic states (with the Hamiltonian, a complete exciton model)
    mu = diabatic_dipoles(res, data["tdm_au"])
    osc = diabatic_oscillator_strengths(res, mu)

    fragments = []
    with open(os.path.join(pipe.stage_dir("fragments"), "fragments.json")) as f:
        meta = {fr["name"]: fr for fr in json.load(f)["fragments"]}
    for name in names:
        n_own = len(meta[name]["parent_atoms_1based"])
        fragments.append({"name": name, "natoms": n_own,
                          **fragment_geometry(pipe.load_fragment_structure(name), n_own)})

    from eecc.qm.pipeline import DIABATIZE_FORMAT
    os.makedirs(workdir, exist_ok=True)
    result = {"format_version": int(DIABATIZE_FORMAT), "units": UNITS, "conventions": CONVENTIONS,
              "labels": res.labels, "states": [state_info(l) for l in res.labels],
              "H_eV": res.H_eV.tolist(), "completeness": res.completeness.tolist(),
              "transition_dipoles_au": mu.tolist(), "oscillator_strengths": osc.tolist(),
              "fragments": fragments, "adiabatic_eV": data["energies_eV"].tolist(), "pairs": pairs}
    with open(os.path.join(workdir, "diabatic.json"), "w") as f:
        json.dump(result, f, indent=2)

    lines = ["# Exciton Hamiltonian from the whole-system TDDFT (diabatization onto fragment states)",
             "# Adiabatic states (eV): " + ", ".join(f"{e:.4f}" for e in data["energies_eV"]), "",
             *notes, format_hamiltonian(res, EV_TO_CM), "", format_dipoles(res, mu, osc), ""]
    low = [l for l, c in zip(res.labels, res.completeness) if c < 0.8]
    if low:
        lines.append(f"# WARNING: completeness < 0.8 for {low}; increase system.nstates "
                     "before using these states")
    if coulomb:
        lines.append("# Total coupling (this stage) vs Coulomb couplings (fragment pipeline), cm^-1")
        lines.append(f"  {'Pair':<14s} {'total':>10s} {'TDC direct':>11s} {'TrESP':>10s}")
        for p in pairs:
            lines.append(f"  {p['pair'][0] + '-' + p['pair'][1]:<14s} {p['J_total_cm-1']:10.1f} "
                         f"{p.get('tdc_direct_cm-1', float('nan')):11.1f} {p.get('tresp_cm-1', float('nan')):10.1f}")
    with open(os.path.join(workdir, "diabatic.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return result

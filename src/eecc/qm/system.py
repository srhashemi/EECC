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
from eecc.qm.pyscf_setup import build_mol, build_rks, make_td, set_grid
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
        set_grid(mf, td_cfg.response_grid)
    td = make_td(mf, td_cfg.method, nstates, td_cfg.davidson_tol)

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
         "oscillator_strengths": "dimensionless", "completeness": "dimensionless (0 to 1)",
         "fragments[].center_ang": "Angstrom", "fragments[].principal_axes": "unit vectors",
         "fragments[].moments_amu_ang2": "amu*Angstrom^2",
         "fragments[].vibronic": "S dimensionless; *_cm in cm^-1, *_eV in eV, temperature_K in K",
         "pairs[].J_total_cm-1, pairs[].<method>_cm-1": "cm^-1"}
CONVENTIONS = {
    "frame": "Cartesian frame of system.xyz (the whole-system geometry)",
    "signs": "the phase of every LE state follows its fragment's transition density and that of every CT state "
             "the fragment HOMO and LUMO, so the signs of couplings and dipoles are arbitrary per state but "
             "consistent with each other",
    "ct_states": "CT(a->b): one electron from the HOMO of fragment a to the LUMO of fragment b",
    "principal_axes": "rows, ordered by increasing moment of inertia (moments_amu_ang2; fragment atoms without "
                      "caps, atomic masses); for a planar molecule the last axis is the plane normal; each axis "
                      "has its largest component positive (the first one on a tie); axes with equal moments "
                      "(a symmetric top) are any orthonormal pair in their plane",
}


def fragment_geometry(structure, n_own: int) -> Dict[str, Any]:
    """Centre of mass (Angstrom) and principal axes of a fragment's own atoms (caps excluded)."""
    from pyscf.data import elements

    xyz = np.asarray(structure.coords[:n_own], float)
    m = np.array([elements.MASSES[elements.charge(s)] for s in structure.symbols[:n_own]])
    com = m @ xyz / m.sum()
    r = xyz - com
    inertia = np.eye(3) * np.sum(m * np.sum(r ** 2, axis=1)) - (r.T * m) @ r
    moments, axes = np.linalg.eigh(inertia)
    axes = axes.T
    for a in axes:  # largest component positive; a tolerance so that roundoff cannot pick the other of two equal ones
        a *= np.sign(a[np.flatnonzero(np.abs(a) >= np.abs(a).max() - 1e-6)[0]])
    return {"center_ang": com.tolist(), "principal_axes": axes.tolist(), "moments_amu_ang2": moments.tolist()}


def state_types(names, labels) -> list:
    """Type and fragment(s) of each diabatic state in *labels* (LE(name) or CT(donor->acceptor))."""
    states = [{"label": f"LE({n})", "type": "LE", "fragment": n} for n in names]
    states += [{"label": f"CT({a}->{b})", "type": "CT", "donor": a, "acceptor": b}
               for a in names for b in names if a != b]
    by_label = {s["label"]: s for s in states}
    return [by_label[l] for l in labels]


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
        fragments.append({"name": name, "natoms_own": n_own,
                          **fragment_geometry(pipe.load_fragment_structure(name), n_own)})
        if cfg.vibronic.enabled and pipe.is_done("vibronic", name):  # one-mode Holstein parameters of its LE state
            with open(os.path.join(pipe.stage_dir("vibronic", name), "vibronic.json")) as f:
                fragments[-1]["vibronic"] = json.load(f)["summary"]

    from eecc.qm.pipeline import DIABATIZE_FORMAT
    os.makedirs(workdir, exist_ok=True)
    result = {"format_version": int(DIABATIZE_FORMAT), "units": UNITS, "conventions": CONVENTIONS,
              "labels": res.labels, "states": state_types(names, res.labels),
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

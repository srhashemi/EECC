"""Ground-state geometry optimization and (optional) harmonic frequencies."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import numpy as np

from eecc.qm.pyscf_setup import build_mol, build_rks, resolve_xc
from eecc.qm.structure import Structure, read_xyz, save_xyz


def _append_frame(path: str, structure: Structure, energy: float, cycle: int) -> None:
    with open(path, "a") as f:
        f.write(f"{len(structure)}\n")
        f.write(f"cycle {cycle}  E = {energy:.10f} Eh\n")
        for s, (x, y, z) in zip(structure.symbols, structure.coords):
            f.write(f"{s} {x: .8f} {y: .8f} {z: .8f}\n")


def _last_frame(path: str) -> Structure | None:
    """Last geometry in a multi-frame XYZ trajectory (for restarts)."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lines = f.read().splitlines()
    frames, i = [], 0
    while i < len(lines):
        try:
            n = int(lines[i].split()[0])
        except (IndexError, ValueError):
            break
        frames.append(lines[i:i + n + 2])
        i += n + 2
    if not frames or len(frames[-1]) < 3:
        return None
    block = frames[-1]
    symbols = [l.split()[0] for l in block[2:]]
    coords = [[float(v) for v in l.split()[1:4]] for l in block[2:]]
    return Structure(symbols, np.array(coords), block[1])


def optimize_geometry(structure: Structure, cfg, workdir: str) -> Tuple[Structure, Dict[str, Any]]:
    """Optimize *structure* with the ``opt`` settings of a pipeline config.

    Every optimization cycle is appended to ``trajectory.xyz``; a rerun resumes
    from the last recorded geometry.
    """
    from pyscf.geomopt import geometric_solver

    opt = cfg.opt
    if resolve_xc(opt.xc) != opt.xc:
        raise ValueError(f"'{opt.xc}' lacks its dispersion term in PySCF and is not suitable "
                         "for geometry optimization; use e.g. b3lyp with disp d3bj")
    os.makedirs(workdir, exist_ok=True)
    traj = os.path.join(workdir, "trajectory.xyz")
    start = _last_frame(traj)
    if start is not None and start.symbols == structure.symbols:
        structure = start

    mol = build_mol(structure, opt.basis, opt.cart, cfg.charge, cfg.spin,
                    opt.memory_mb or cfg.resources.memory_mb, output=os.path.join(workdir, "opt.log"))
    mf = build_rks(mol, opt.xc, opt.grid, opt.density_fit, opt.conv_tol, opt.disp)
    n_prev = sum(1 for l in open(traj) if l.startswith("cycle")) if os.path.exists(traj) else 0

    def callback(envs):
        coords = envs["mol"].atom_coords(unit="Angstrom")
        _append_frame(traj, Structure(structure.symbols, coords), float(envs["energy"]),
                      n_prev + envs["self"].cycle)

    converged, mol_eq = geometric_solver.kernel(mf, maxsteps=opt.maxsteps, callback=callback)
    result = Structure(structure.symbols, mol_eq.atom_coords(unit="Angstrom"),
                       f"optimized {opt.xc}-{opt.disp or 'nodisp'}/{opt.basis}")
    save_xyz(result, os.path.join(workdir, "optimized.xyz"))
    info: Dict[str, Any] = {"converged": bool(converged)}
    if not converged:
        raise RuntimeError(f"geometry optimization did not converge in {opt.maxsteps} steps; "
                           "rerun to continue from the last geometry")

    if opt.freq:
        info.update(harmonic_frequencies(result, cfg, workdir))
    with open(os.path.join(workdir, "opt_summary.json"), "w") as f:
        json.dump(info, f, indent=2)
    return result, info


def harmonic_frequencies(structure: Structure, cfg, workdir: str) -> Dict[str, Any]:
    """Analytic Hessian at *structure*; reports frequencies and imaginary modes."""
    from pyscf.hessian import thermo

    opt = cfg.opt
    mol = build_mol(structure, opt.basis, opt.cart, cfg.charge, cfg.spin,
                    opt.memory_mb or cfg.resources.memory_mb, output=os.path.join(workdir, "freq.log"))
    mf = build_rks(mol, opt.xc, opt.grid, opt.density_fit, opt.conv_tol, opt.disp)
    mf.kernel()
    hess = mf.Hessian().kernel()
    res = thermo.harmonic_analysis(mol, hess)
    freq = np.asarray(res["freq_wavenumber"])
    imag = [float(abs(f.imag)) for f in freq if np.iscomplexobj(freq) and abs(f.imag) > 1e-6]
    real = np.real(freq)
    imag += [float(-f) for f in real if f < 0]
    np.savetxt(os.path.join(workdir, "frequencies_cm-1.txt"), np.real_if_close(freq).real)
    return {"n_imaginary": len(imag), "imaginary_cm-1": imag,
            "lowest_real_cm-1": float(np.min(real[real > 0])) if np.any(real > 0) else None}


def load_optimized(workdir: str) -> Structure:
    return read_xyz(os.path.join(workdir, "optimized.xyz"))

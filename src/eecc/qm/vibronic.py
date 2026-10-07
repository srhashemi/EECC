"""Vibronic parameters of one fragment: normal modes and Huang-Rhys factors of its excited state.

Displaced harmonic oscillator model ("vertical gradient"): ground and excited state share the
ground-state normal modes and frequencies; the excited state is shifted along each mode. At the
fragment geometry the gradient of the excitation energy, projected onto mass-weighted normal mode k
(g_k), gives the shift of the excited-state minimum, Delta Q_k = -g_k / omega_k^2, and in atomic
units (hbar = 1) the dimensionless displacement and Huang-Rhys factor

    d_k = -g_k / omega_k^(3/2),    S_k = d_k^2 / 2 = g_k^2 / (2 omega_k^3).

The gradient of the excitation energy (excited minus ground state) is used rather than that of the
excited state alone, so a geometry that is not exactly a ground-state minimum (a fragment cut from
the optimized aggregate) leaves no spurious ground-state force in the result.
"""

from __future__ import annotations

import json
import os
import zipfile
from typing import Any, Dict, Optional

import numpy as np

from eecc.constants import HARTREE_TO_EV
from eecc.qm.pyscf_setup import build_mol, build_rks, resolve_xc, use_cartesian
from eecc.qm.structure import Structure

AMU_TO_ME = 1822.888486209  # atomic mass unit -> electron masses
HARTREE_TO_CM = 219474.6313632
KB_CM = 0.6950348004  # Boltzmann constant (cm^-1 / K)
ANG_TO_BOHR = 1.0 / 0.529177210903


def _masses_amu(symbols) -> np.ndarray:
    """Masses of the most common isotopes (as Gaussian uses; e.g. 11B, not the average 10.81)."""
    from pyscf.data import elements

    return np.array([elements.COMMON_ISOTOPE_MASSES[elements.charge(s)] for s in symbols])


def normal_modes(symbols, coords_ang, hessian) -> Dict[str, np.ndarray]:
    """Harmonic frequencies and mass-weighted normal modes, translations and rotations removed.

    *hessian* is the Cartesian Hessian in Eh/Bohr^2, shaped (natm, natm, 3, 3) as PySCF returns it
    or (3N, 3N). Returns ``omega_au`` (Eh; negative for imaginary modes), ``modes`` (3N x n_modes,
    orthonormal columns in mass-weighted coordinates, atomic units) and ``sqrt_m`` (sqrt of the
    mass of each Cartesian coordinate, electron masses).
    """
    m = _masses_amu(symbols) * AMU_TO_ME
    x = np.asarray(coords_ang, float) * ANG_TO_BOHR
    n = len(m)
    H = np.asarray(hessian, float)
    if H.ndim == 4:
        H = H.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    sqrt_m = np.repeat(np.sqrt(m), 3)
    Hm = H / np.outer(sqrt_m, sqrt_m)
    Hm = 0.5 * (Hm + Hm.T)

    r = x - m @ x / m.sum()
    tr = []
    for a in range(3):
        e = np.zeros((n, 3))
        e[:, a] = 1.0
        tr.append((e * np.sqrt(m)[:, None]).ravel())
    for a in range(3):
        tr.append((np.cross(np.eye(3)[a], r) * np.sqrt(m)[:, None]).ravel())
    u, s, _ = np.linalg.svd(np.array(tr).T, full_matrices=True)
    rank = int(np.sum(s > 1e-6 * s.max()))  # 5 for a linear molecule, else 6
    B = u[:, rank:]  # orthonormal basis of the internal (vibrational) space
    w2, U = np.linalg.eigh(B.T @ Hm @ B)
    return {"omega_au": np.sign(w2) * np.sqrt(np.abs(w2)), "modes": B @ U, "sqrt_m": sqrt_m}


def huang_rhys(symbols, coords_ang, hessian, grad_excitation, min_freq_cm: float = 50.0,
               freq_scale: float = 1.0) -> Dict[str, Any]:
    """Frequencies, displacements and Huang-Rhys factors of every normal mode.

    *grad_excitation* is the Cartesian gradient of the excitation energy (Eh/Bohr, natm x 3).
    The returned frequencies are scaled by *freq_scale*, and *min_freq_cm* applies to them; d and S
    come from the unscaled harmonic model. Modes below *min_freq_cm* (including imaginary ones) get
    no S: S_k grows as 1/omega^3, so a near-zero mode of a geometry that is not a strict minimum
    would dominate every sum.
    """
    if min_freq_cm <= 0:
        raise ValueError("min_freq_cm must be positive")
    nm = normal_modes(symbols, coords_ang, hessian)
    omega = nm["omega_au"]
    g = nm["modes"].T @ (np.asarray(grad_excitation, float).ravel() / nm["sqrt_m"])
    freq = omega * HARTREE_TO_CM * freq_scale
    valid = freq >= min_freq_cm
    d = np.full_like(omega, np.nan)
    d[valid] = -g[valid] / omega[valid] ** 1.5
    return {"freq_cm": freq, "g_au": g, "d": d, "S": 0.5 * d ** 2, "valid": valid}


def summarize(freq_cm, S, valid, cutoff_cm: float = 800.0, temperature_K: float = 298.15) -> Dict[str, Any]:
    """Compress the modes for a one-mode Holstein model.

    Modes at or above *cutoff_cm* form one effective mode (S_eff = sum S_k, omega_eff = S-weighted
    mean frequency): the vibronic progression. Modes below it are too closely spaced to resolve and
    give a Gaussian width instead: sigma^2 = sum S_k omega_k^2 coth(omega_k / 2kT) (standard
    deviation; at 0 K the coth is 1). *freq_cm* as returned by :func:`huang_rhys` (already scaled).
    """
    w = np.asarray(freq_cm, float)
    S = np.asarray(S, float)
    hi = np.asarray(valid) & (w >= cutoff_cm)
    lo = np.asarray(valid) & (w < cutoff_cm)
    s_eff = float(S[hi].sum())
    w_eff = float((S[hi] * w[hi]).sum() / s_eff) if s_eff > 0 else None
    coth = 1.0 / np.tanh(w[lo] / (2 * KB_CM * temperature_K)) if temperature_K > 0 else np.ones(lo.sum())
    sigma0 = float(np.sqrt((S[lo] * w[lo] ** 2).sum()))
    sigma_t = float(np.sqrt((S[lo] * w[lo] ** 2 * coth).sum()))
    lam_hi, lam_lo = float((S[hi] * w[hi]).sum()), float((S[lo] * w[lo]).sum())
    cm_to_ev = 1.0 / 8065.544006
    strong_low = [{"freq_cm": float(a), "S": float(b)} for a, b in zip(w[lo], S[lo]) if b > 1.0]
    return {
        "cutoff_cm": cutoff_cm, "temperature_K": temperature_K,
        "S_eff": s_eff, "omega_eff_cm": w_eff, "omega_eff_eV": w_eff * cm_to_ev if w_eff else None,
        "reorganization_cm": lam_hi + lam_lo, "reorganization_eV": (lam_hi + lam_lo) * cm_to_ev,
        "reorganization_high_cm": lam_hi, "reorganization_low_cm": lam_lo,
        "sigma_low_0K_cm": sigma0, "sigma_low_cm": sigma_t, "sigma_low_eV": sigma_t * cm_to_ev,
        "n_modes": int(len(w)), "n_high": int(hi.sum()), "n_low": int(lo.sum()),
        "n_excluded": int((~np.asarray(valid)).sum()),
        # a low-frequency mode with S > 1 is a large-amplitude distortion (e.g. a torsion): one
        # effective mode plus a width may then be too crude, and the harmonic model itself is doubtful
        "strong_low_modes": strong_low,
    }


def format_modes(res: Dict[str, Any], summary: Dict[str, Any], top: int = 15) -> str:
    """Text table of the summary and the most displaced modes."""
    s = summary
    lines = ["# Vibronic parameters (displaced harmonic oscillator, vertical gradient)",
             f"# effective mode (>= {s['cutoff_cm']:.0f} cm-1): S_eff = {s['S_eff']:.3f}, "
             + (f"omega_eff = {s['omega_eff_cm']:.0f} cm-1 ({s['omega_eff_eV']:.4f} eV)"
                if s["omega_eff_cm"] else "no modes"),
             f"# reorganization energy {s['reorganization_cm']:.0f} cm-1 ({s['reorganization_eV']:.4f} eV): "
             f"high {s['reorganization_high_cm']:.0f}, low {s['reorganization_low_cm']:.0f} cm-1",
             f"# width from low-frequency modes (Gaussian sigma): {s['sigma_low_cm']:.0f} cm-1 at "
             f"{s['temperature_K']:g} K, {s['sigma_low_0K_cm']:.0f} cm-1 at 0 K",
             f"# {s['n_modes']} modes: {s['n_high']} high, {s['n_low']} low, {s['n_excluded']} excluded "
             "(below the minimum frequency or imaginary)"]
    if s["strong_low_modes"]:
        lines.append("# WARNING: low-frequency modes with S > 1: "
                     + ", ".join(f"{m['freq_cm']:.0f} cm-1 (S {m['S']:.2f})" for m in s["strong_low_modes"]))
    lines += ["", f"{'mode':>5s} {'freq/cm-1':>10s} {'S':>8s} {'d':>8s}"]
    S = np.where(res["valid"], res["S"], -1.0)
    for k in sorted(np.argsort(-S)[:top]):
        if res["valid"][k]:
            lines.append(f"{k + 1:5d} {res['freq_cm'][k]:10.1f} {res['S'][k]:8.4f} {res['d'][k]:8.4f}")
    return "\n".join(lines)


def run_vibronic(structure: Structure, cfg, workdir: str, name: str = "fragment", charge: int = 0,
                 guess_chk: Optional[str] = None, log=None, force: bool = False) -> Dict[str, Any]:
    """Ground-state Hessian and excitation-energy gradient of the ``td.state`` state; writes
    ``vibronic.npz`` (raw data), ``vibronic.json`` and ``vibronic.txt``.

    Uses the ``td`` method (functional, basis, TDDFT or TDA, state) so that the parameters belong
    to the same state as the couplings. Only the summary is recomputed when ``vibronic.npz`` from
    the same calculation already exists (e.g. after changing the cutoff), unless *force* is set.
    """
    td_cfg, vc = cfg.td, cfg.vibronic
    os.makedirs(workdir, exist_ok=True)
    raw = os.path.join(workdir, "vibronic.npz")
    partial = os.path.join(workdir, PARTIAL)
    key = calc_key(cfg, structure, charge)
    data = None if force else _load_npz(raw, key)
    if data is None:
        if force and os.path.exists(partial):
            os.remove(partial)
        data = _compute(structure, cfg, workdir, name, charge, guess_chk, log, key)
        data["calc_key"] = np.array(key)
        _save_npz(raw, data)
        if os.path.exists(partial):
            os.remove(partial)

    res = huang_rhys(structure.symbols, structure.coords, data["hessian"],
                     data["grad_excited"] - data["grad_ground"], vc.min_frequency, vc.freq_scale)
    summary = summarize(res["freq_cm"], res["S"], res["valid"], vc.cutoff, vc.temperature)
    summary["freq_scale"] = vc.freq_scale
    out = {
        "name": name, "state": td_cfg.state, "method": td_cfg.method, "xc": resolve_xc(td_cfg.xc),
        "basis": td_cfg.basis, "cartesian": use_cartesian(td_cfg.basis, td_cfg.cart),
        "excitation_energy_eV": float(data["excitation_eV"]),
        "min_frequency_cm": vc.min_frequency, "freq_scale": vc.freq_scale, "summary": summary,
        "modes": {"freq_cm": res["freq_cm"].tolist(),
                  "S": [None if not v else float(s) for s, v in zip(res["S"], res["valid"])],
                  "d": [None if not v else float(s) for s, v in zip(res["d"], res["valid"])]},
    }
    with open(os.path.join(workdir, "vibronic.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(workdir, "vibronic.txt"), "w") as f:
        f.write(format_modes(res, summary) + "\n")
    return out


# td settings that the vibronic calculation does not use (it runs its own response on the SCF grid)
TD_UNUSED = ("response_grid", "davidson_tol")

# gradients and excitation energy, saved before the Hessian so a killed job does not repeat them
PARTIAL = "vibronic_partial.npz"


def _load_npz(path: str, key: str) -> Optional[Dict[str, np.ndarray]]:
    """Contents of *path* if it holds the calculation *key*, else None."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            if str(z["calc_key"]) == key:
                return {k: z[k] for k in z.files}
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):  # unreadable (e.g. truncated): compute again
        pass
    return None


def _save_npz(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)  # never leave a half-written file under the final name


def eri_incore(mode: str, nao: int, memory_mb: float, hessian: bool) -> Optional[bool]:
    """Keep the two-electron integrals (nao^4 bytes with 8-fold symmetry) in memory?

    True/False forces it; None leaves PySCF's choice (in memory if they fit next to the memory
    already in use). auto: PySCF's choice, but for the Hessian, which needs much memory of its
    own, never if they take more than half of *memory_mb*.
    """
    if mode == "auto":
        return False if hessian and nao ** 4 / 1e6 > 0.5 * memory_mb else None
    return mode == "incore"


def _set_eri(mf, incore: Optional[bool]) -> None:
    """Apply eri_incore's choice to a (non-density-fitted) PySCF SCF object.

    PySCF's RHF.get_jk builds the in-memory array when mf._eri exists, mol.incore_anyway is set,
    or mf._is_mem_enough(); all three are set here. _compute checks afterwards that it worked.
    """
    if incore is False:
        mf._eri = None  # release the array if the SCF built one
        mf._is_mem_enough = lambda: False
        mf.mol.incore_anyway = False  # may come from the PySCF configuration
    elif incore:
        mf.mol.incore_anyway = True


def calc_key(cfg, structure: Structure, charge: int) -> str:
    """Identifies the quantum-chemical calculation (not the summary settings).

    Built from the same td hash as the stage (non-default settings, changed defaults, the Cartesian
    rule) plus the resolved functional and d-function choice, so every td option that changes the
    calculation is covered without listing it here.
    """
    import hashlib

    td = cfg.td
    payload = [cfg.section_hash("td", ignore=TD_UNUSED), resolve_xc(td.xc), use_cartesian(td.basis, td.cart),
               cfg.vibronic.davidson_tol, charge, list(structure.symbols),
               np.round(np.asarray(structure.coords, float), 8).tolist()]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


def _compute(structure, cfg, workdir, name, charge, guess_chk, log, key: str = "") -> Dict[str, np.ndarray]:
    td_cfg = cfg.td
    log = log or (lambda *a: None)
    partial_path = os.path.join(workdir, PARTIAL)
    partial = _load_npz(partial_path, key) if key else None
    log_path = os.path.join(workdir, "vibronic.log")
    if partial is not None and os.path.exists(log_path):  # keep the killed run's output (TDDFT, gradients)
        os.replace(log_path, os.path.join(workdir, "vibronic_previous.log"))
    mol = build_mol(structure, td_cfg.basis, td_cfg.cart, charge, 0, cfg.resources.memory_mb,
                    output=log_path)
    own_chk = os.path.join(workdir, "scf.chk")
    mf = build_rks(mol, td_cfg.xc, td_cfg.grid, td_cfg.density_fit, td_cfg.conv_tol,
                   disp=None, chkfile=own_chk)
    # with density fitting PySCF holds the 3-index tensor instead; vibronic.eri does not apply
    use_eri = not td_cfg.density_fit
    if use_eri:
        _set_eri(mf, eri_incore(cfg.vibronic.eri, mol.nao, cfg.resources.memory_mb, hessian=False))
    dm0 = None
    # a killed run of the same calculation left its own SCF checkpoint: the best guess
    for chk in ([own_chk] if partial is not None else []) + [guess_chk]:
        if chk and os.path.exists(chk):
            try:  # only a guess: any problem with the file just means the next or default guess
                dm0 = mf.from_chk(chk)
                break
            except Exception as exc:  # noqa: BLE001 (h5py raises various errors on a busy or partial file)
                log(f"[vibronic] {name}: ignoring {chk} as a guess ({exc})")
    mf.kernel(dm0=dm0)
    if not mf.converged:
        raise RuntimeError(f"{name}: SCF did not converge")
    if key and partial is None:  # marks scf.chk as this calculation's converged SCF
        _save_npz(partial_path, {"calc_key": np.array(key)})

    if partial is not None and "grad_excited" in partial:
        g0, g1, e_exc = partial["grad_ground"], partial["grad_excited"], float(partial["excitation_eV"])
        log(f"[vibronic] {name}: S{td_cfg.state} = {e_exc:.4f} eV and gradients from {PARTIAL}")
    else:
        g0 = mf.nuc_grad_method().kernel()
        # The SCF grid also for the response: gradients need the response on the grid of the SCF.
        td = mf.TDA() if td_cfg.method == "tda" else mf.TDDFT()
        td.nstates = td_cfg.nstates
        td.conv_tol = cfg.vibronic.davidson_tol
        td.kernel()
        k = td_cfg.state - 1
        if not np.atleast_1d(td.converged)[k]:
            raise RuntimeError(f"{name}: TDDFT state {td_cfg.state} did not converge")
        e_exc = float(td.e[k]) * HARTREE_TO_EV
        log(f"[vibronic] {name}: S{td_cfg.state} = {e_exc:.4f} eV; excited-state gradient")
        g1 = td.nuc_grad_method().kernel(state=td_cfg.state)
        if key:
            _save_npz(partial_path, {"calc_key": np.array(key), "grad_ground": np.asarray(g0),
                                     "grad_excited": np.asarray(g1), "excitation_eV": np.array(e_exc)})
    incore = eri_incore(cfg.vibronic.eri, mol.nao, cfg.resources.memory_mb, hessian=True) if use_eri else None
    if use_eri:
        _set_eri(mf, incore)
    how = ("density fitting" if not use_eri else "integrals in memory" if incore or mf._eri is not None
           else "integrals direct" if incore is False else "integrals as PySCF chooses")
    log(f"[vibronic] {name}: ground-state Hessian ({mol.natm} atoms, {mol.nao} basis functions, {how})")
    hess = mf.Hessian().kernel()
    if incore is False and getattr(mf, "_eri", None) is not None:
        log(f"[vibronic] {name}: WARNING PySCF held the two-electron integrals in memory despite "
            "direct mode (changed PySCF internals?)")
    return {"hessian": np.asarray(hess), "grad_ground": np.asarray(g0), "grad_excited": np.asarray(g1),
            "excitation_eV": np.array(e_exc), "symbols": np.array(structure.symbols),
            "coords_ang": np.asarray(structure.coords, float), "e_scf_Eh": np.array(float(mf.e_tot))}

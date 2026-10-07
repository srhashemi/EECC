"""Diabatization of supermolecule TDDFT states by projection onto fragment states.

The adiabatic states I of the whole system (energies E_I) are combined into
diabatic states that resemble given fragment states: locally excited (LE) states
from the fragment TDDFT, and optionally charge-transfer (CT) states built from
the fragments' frontier orbitals.

Every state is represented by its transition amplitudes Z = X + Y in the
occupied-virtual orbital space of the whole system (for a fragment state, its
transition density projected through the fragment-system overlap integrals).
Transition densities are linear in the state coefficients, so a diabatic state
k is the combination sum_I C_Ik |I> whose amplitudes best match Z_k in the
least-squares sense. The coefficient vectors are Löwdin-orthonormalized and

    H_diab = D^T diag(E) D,

whose diagonal holds the diabatic (site and CT) energies and whose off-diagonal
elements are the total couplings, including exchange, overlap and polarization
effects. ``completeness`` (0..1) reports how well the adiabatic states span each
diabatic state; values well below 1 mean more adiabatic states are needed.

Signs follow the phases of the fragment transition densities, as for the
Coulomb couplings of the fragment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from eecc.constants import AU_TO_DEBYE, HARTREE_TO_EV


@dataclass
class DiabaticResult:
    labels: List[str]
    H_eV: np.ndarray  # diabatic Hamiltonian (eV)
    completeness: np.ndarray  # per diabatic state, 0..1
    D: np.ndarray  # adiabatic -> diabatic coefficients (n_adiabatic x n_diabatic)


def cross_overlap(mol_system, mol_fragment) -> np.ndarray:
    """AO overlap <system AO | fragment AO>."""
    from pyscf import gto

    return gto.intor_cross("int1e_ovlp", mol_system, mol_fragment)


def project_transition_density(T_frag: np.ndarray, S_sf: np.ndarray,
                               C_occ: np.ndarray, C_vir: np.ndarray) -> np.ndarray:
    """Amplitudes Z (nocc x nvir) of a fragment transition density in the system orbitals.

    *T_frag* is the symmetrized AO transition density of the fragment, with the
    convention of :func:`eecc.qm.excited.transition_density_matrix`, so that for
    the fragment itself C_occ^T S T S C_vir = X + Y.
    """
    return C_occ.T @ S_sf @ T_frag @ S_sf.T @ C_vir


def ct_amplitudes(occ_donor: np.ndarray, vir_acceptor: np.ndarray, S_sd: np.ndarray,
                  S_sa: np.ndarray, C_occ: np.ndarray, C_vir: np.ndarray) -> np.ndarray:
    """Amplitudes of the singlet excitation donor orbital -> acceptor orbital in the system orbitals.

    Uses PySCF's spin convention for singlets (a pure single excitation has
    X = 1/sqrt(2)), as for the TDDFT amplitudes of the system.
    """
    o = C_occ.T @ S_sd @ occ_donor
    v = C_vir.T @ S_sa @ vir_acceptor
    return np.outer(o, v) / np.sqrt(2.0)


def diabatize(Z_adiabatic: np.ndarray, E_adiabatic: np.ndarray, Z_diabatic: np.ndarray,
              labels: Sequence[str]) -> DiabaticResult:
    """Diabatic Hamiltonian from adiabatic amplitudes/energies and target diabatic amplitudes."""
    n_ad, n_dia = len(E_adiabatic), len(Z_diabatic)
    if n_dia > n_ad:
        raise ValueError(f"{n_dia} diabatic states need at least as many adiabatic states (got {n_ad})")
    A = np.asarray(Z_adiabatic).reshape(n_ad, -1).T
    B = np.asarray(Z_diabatic).reshape(n_dia, -1).T
    C, *_ = np.linalg.lstsq(A, B, rcond=None)
    completeness = np.sum((A @ C) ** 2, axis=0) / np.sum(B ** 2, axis=0)
    w, V = np.linalg.eigh(C.T @ C)
    if w.min() < 1e-10:
        raise ValueError("diabatic states are linearly dependent within the adiabatic space")
    D = C @ V @ np.diag(w ** -0.5) @ V.T
    H = D.T @ np.diag(E_adiabatic) @ D
    return DiabaticResult(list(labels), 0.5 * (H + H.T), completeness, D)


def diabatic_dipoles(res: DiabaticResult, tdm_adiabatic: np.ndarray) -> np.ndarray:
    """Transition dipoles of the diabatic states, mu_k = sum_I D_Ik mu_I (n_diabatic x 3).

    *tdm_adiabatic* holds the transition dipoles of the adiabatic states that were
    diabatized, in the same order. A state's phase enters both D_Ik and mu_I, so the
    result is independent of the adiabatic phases; its sign follows the fragment
    state, like the couplings.
    """
    tdm = np.asarray(tdm_adiabatic)
    if tdm.shape != (res.D.shape[0], 3):
        raise ValueError(f"need one transition dipole (3 components) per adiabatic state: expected "
                         f"{(res.D.shape[0], 3)}, got {tdm.shape}")
    return res.D.T @ tdm


def diabatic_oscillator_strengths(res: DiabaticResult, mu: np.ndarray) -> np.ndarray:
    """f_k = 2/3 E_k |mu_k|^2 (atomic units), with E_k the diabatic energy (diagonal of H)."""
    return 2.0 / 3.0 * np.diag(res.H_eV) / HARTREE_TO_EV * np.sum(np.asarray(mu) ** 2, axis=1)


def format_dipoles(res: DiabaticResult, mu: np.ndarray, osc: np.ndarray) -> str:
    """Readable table of the diabatic transition dipoles (au), their length (Debye) and f.

    States with completeness below 0.8 are marked: their dipoles are not reliable.
    """
    w = _label_width(res.labels)
    lines = ["Transition dipoles of the diabatic states (au) and oscillator strengths:"]
    for lab, m, f, c in zip(res.labels, mu, osc, res.completeness):
        lines.append(f"  {lab:<{w}s} {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.4f}   |mu| {np.linalg.norm(m) * AU_TO_DEBYE:7.3f} D"
                     f"   f {f:7.4f}" + ("   (completeness < 0.8)" if c < 0.8 else ""))
    return "\n".join(lines)


def _label_width(labels: Sequence[str]) -> int:
    return max(10, *(len(l) for l in labels))


def format_hamiltonian(res: DiabaticResult, ev_to_cm: float = 8065.544006) -> str:
    """Readable table: diagonal in eV, off-diagonal couplings in cm^-1."""
    n = len(res.labels)
    w = _label_width(res.labels)
    lines = ["Diabatic energies (eV) and completeness:"]
    for k, lab in enumerate(res.labels):
        lines.append(f"  {lab:<{w}s} {res.H_eV[k, k]:9.4f}   {res.completeness[k]:6.3f}")
    lines.append("Couplings (cm^-1):")
    for i in range(n):
        for j in range(i + 1, n):
            lines.append(f"  {res.labels[i]:<{w}s} - {res.labels[j]:<{w}s} {res.H_eV[i, j] * ev_to_cm:10.1f}")
    return "\n".join(lines)


def diabatize_from_pipeline(pipe, mol_system, mo_coeff: np.ndarray, mo_occ: np.ndarray,
                            x: np.ndarray, y: np.ndarray, energies_eV: np.ndarray,
                            include_ct: bool = True) -> DiabaticResult:
    """Diabatize system TDDFT states onto the fragments of a finished fragment pipeline.

    *pipe* is a :class:`eecc.qm.pipeline.Pipeline` whose fragments and td stages are
    done (same TD settings as the system calculation). *x*, *y* are the system TDDFT
    amplitudes (n_states x nocc x nvir), *energies_eV* their excitation energies.
    Diabatic states: the selected LE state of every fragment and, with *include_ct*,
    the HOMO(A) -> LUMO(B) CT state of every ordered fragment pair.
    """
    import os

    from pyscf import scf

    from eecc.qm.excited import load_excited_states
    from eecc.qm.pyscf_setup import build_mol

    cfg = pipe.cfg
    C_occ, C_vir = mo_coeff[:, mo_occ > 0], mo_coeff[:, mo_occ == 0]
    frags = []
    for name in pipe.fragment_names():
        td = load_excited_states(pipe.stage_dir("td", name))
        mol_f = build_mol(pipe.load_fragment_structure(name), cfg.td.basis, cfg.td.cart,
                          pipe._fragment_charge(name), 0, cfg.resources.memory_mb, verbose=0)
        chk = scf.chkfile.load(os.path.join(pipe.stage_dir("td", name), "scf.chk"), "scf")
        occ = chk["mo_occ"] > 0
        frags.append(dict(name=name, T=td["T"], S=cross_overlap(mol_system, mol_f),
                          homo=chk["mo_coeff"][:, occ][:, -1], lumo=chk["mo_coeff"][:, ~occ][:, 0]))

    labels, Z = [], []
    for f in frags:
        labels.append(f"LE({f['name']})")
        Z.append(project_transition_density(f["T"], f["S"], C_occ, C_vir))
    if include_ct:
        for a in frags:
            for b in frags:
                if a is not b:
                    labels.append(f"CT({a['name']}->{b['name']})")
                    Z.append(ct_amplitudes(a["homo"], b["lumo"], a["S"], b["S"], C_occ, C_vir))
    return diabatize(np.asarray(x) + np.asarray(y), np.asarray(energies_eV), np.array(Z), labels)

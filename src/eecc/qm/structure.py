"""Molecular structures: XYZ I/O, connectivity, fragment selection and link-atom capping."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from eecc.geometry.fragments import parse_indices
from eecc.io.xyz import write_xyz


# Covalent radii (Å), Cordero et al., Dalton Trans. 2008, 2832.
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "I": 1.39,
}


@dataclass
class Structure:
    symbols: List[str]
    coords: np.ndarray  # (N, 3) in Å
    comment: str = ""

    def __post_init__(self):
        self.coords = np.asarray(self.coords, dtype=float).reshape(-1, 3)
        if len(self.symbols) != len(self.coords):
            raise ValueError("symbols and coords have different lengths")

    def __len__(self) -> int:
        return len(self.symbols)

    def pyscf_atoms(self) -> List[Tuple[str, Tuple[float, float, float]]]:
        return [(s, tuple(c)) for s, c in zip(self.symbols, self.coords)]


@dataclass
class Fragment:
    name: str
    structure: Structure
    parent_indices: List[int]  # 0-based indices into the parent structure (caps excluded)
    caps: List[Tuple[int, int]] = field(default_factory=list)  # (kept atom, removed atom), parent indices


# ============================================================
# === XYZ I/O ================================================
# ============================================================

def _normalize_symbol(s: str) -> str:
    s = "".join(ch for ch in s if ch.isalpha())
    return s[:1].upper() + s[1:].lower()


def read_xyz(path: str) -> Structure:
    """Read a standard XYZ file (count line, comment line, then ``El x y z``)."""
    with open(path) as f:
        lines = f.read().splitlines()
    try:
        natoms = int(lines[0].split()[0])
    except (IndexError, ValueError):
        raise ValueError(f"{path}: first line must hold the number of atoms")
    symbols, coords = [], []
    for line in lines[2:2 + natoms]:
        parts = line.split()
        symbols.append(_normalize_symbol(parts[0]))
        coords.append([float(v) for v in parts[1:4]])
    if len(symbols) != natoms:
        raise ValueError(f"{path}: expected {natoms} atoms, found {len(symbols)}")
    return Structure(symbols, np.array(coords), comment=lines[1] if len(lines) > 1 else "")


def save_xyz(structure: Structure, path: str, comment: Optional[str] = None) -> None:
    write_xyz(structure.symbols, structure.coords, path,
              comment=comment if comment is not None else structure.comment)


# ============================================================
# === Connectivity ===========================================
# ============================================================

def bonds(structure: Structure, scale: float = 1.2) -> List[Tuple[int, int]]:
    """Return bonded pairs (i < j): d_ij < scale * (r_i + r_j)."""
    r = np.array([COVALENT_RADII.get(s, 1.5) for s in structure.symbols])
    X = structure.coords
    d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    cutoff = scale * (r[:, None] + r[None, :])
    i, j = np.where(np.triu(d < cutoff, k=1))
    return list(zip(i.tolist(), j.tolist()))


def connected_components(n: int, bond_list: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Connected components of the bond graph, sorted by their smallest atom index."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in bond_list:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=min)


# ============================================================
# === Fragments ==============================================
# ============================================================

def _cap_position(kept: np.ndarray, removed: np.ndarray, length: float) -> np.ndarray:
    u = removed - kept
    return kept + length * u / np.linalg.norm(u)


def make_fragment(
    parent: Structure,
    indices: Sequence[int],
    name: str,
    bond_list: Sequence[Tuple[int, int]],
    cap_element: str = "H",
    cap_lengths: Optional[dict] = None,
    default_cap_length: float = 1.09,
) -> Fragment:
    """Extract atoms *indices* and cap every bond to an atom outside the fragment.

    Each cap is placed on the cut bond, at the element-specific bond length
    from the kept atom (link-atom scheme).
    """
    cap_lengths = cap_lengths or {}
    idx = sorted(set(int(i) for i in indices))
    inside = set(idx)
    symbols = [parent.symbols[i] for i in idx]
    coords = [parent.coords[i] for i in idx]
    caps = []
    for a, b in bond_list:
        if (a in inside) == (b in inside):
            continue
        kept, removed = (a, b) if a in inside else (b, a)
        length = cap_lengths.get(parent.symbols[kept], default_cap_length)
        symbols.append(cap_element)
        coords.append(_cap_position(parent.coords[kept], parent.coords[removed], length))
        caps.append((kept, removed))
    caps_note = f", {len(caps)} cap(s)" if caps else ""
    comment = f"{name}: {len(idx)} atoms from parent{caps_note}"
    return Fragment(name, Structure(symbols, np.array(coords), comment), idx, caps)


def build_fragments(parent: Optional[Structure], cfg,
                    topology: Optional[Structure] = None) -> List[Fragment]:
    """Build fragments according to a :class:`~eecc.qm.config.FragmentsConfig`.

    Bonds are detected on *topology* (same atoms as *parent*, e.g. the geometry
    before optimization) when given, otherwise on *parent*.
    """
    if cfg.mode == "files":
        raise ValueError("mode 'files' is handled by load_fragment_files()")
    assert parent is not None
    if topology is not None and topology.symbols != parent.symbols:
        raise ValueError("topology structure does not match the parent atoms")
    bl = bonds(topology if topology is not None else parent, cfg.bond_scale)
    if cfg.mode == "auto":
        groups = connected_components(len(parent), bl)
        if len(groups) < 2:
            raise ValueError("fragments.mode 'auto' found a single connected molecule; "
                             "give explicit fragments.ranges for covalent systems")
    else:
        groups = [parse_indices(r) for r in cfg.ranges]
        seen = set()
        for k, g in enumerate(groups):
            if not g:
                raise ValueError(f"fragment {k + 1} is empty")
            if max(g) >= len(parent):
                raise ValueError(f"fragment {k + 1} refers to atom {max(g) + 1}, "
                                 f"but the structure has {len(parent)} atoms")
            overlap = seen.intersection(g)
            if overlap:
                raise ValueError(f"atoms {sorted(i + 1 for i in overlap)} are in more than one fragment")
            seen.update(g)
    return [
        make_fragment(parent, g, f"frag{k + 1}", bl, cfg.cap.element,
                      cfg.cap.bond_lengths, cfg.cap.default_bond_length)
        for k, g in enumerate(groups)
    ]


def load_fragment_files(paths: Sequence[str]) -> List[Fragment]:
    frags = []
    for k, p in enumerate(paths):
        s = read_xyz(p)
        frags.append(Fragment(f"frag{k + 1}", s, list(range(len(s)))))
    return frags

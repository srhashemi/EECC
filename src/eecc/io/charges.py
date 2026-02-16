"""Reading and writing atomic transition-charge files."""

from __future__ import annotations

from typing import Dict, List, Tuple


# ---------- Minimal periodic table for output symbols ----------
Z2SYM: Dict[int, str] = {
    1: "H", 6: "C", 7: "N", 8: "O", 9: "F",
    15: "P", 16: "S", 17: "Cl", 35: "Br"
}


def atomic_symbol(Z: int) -> str:
    """Return the element symbol for atomic number *Z*."""
    return Z2SYM.get(int(Z), f"Z{int(Z)}")


def read_atoms(filename: str) -> List[Tuple[str, float, float, float, float]]:
    """Read a charge file with columns: Element X Y Z Charge."""
    atoms: List[Tuple[str, float, float, float, float]] = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                element = parts[0]
                x, y, z = map(float, parts[1:4])
                q = float(parts[4])
                atoms.append((element, x, y, z, q))
    return atoms


def read_atoms_scaled(
    filename: str, scale_factor: float
) -> List[Tuple[str, float, float, float, float]]:
    """Read a charge file, dividing each charge by *scale_factor*."""
    atoms: List[Tuple[str, float, float, float, float]] = []
    with open(filename, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 5:
                element = parts[0]
                x, y, z = map(float, parts[1:4])
                q = float(parts[4]) / scale_factor
                atoms.append((element, x, y, z, q))
    return atoms


def write_charges_txt(
    atoms_with_q: List[Tuple[str, float, float, float, float]],
    out_path: str = "charges_trEsp.txt",
) -> None:
    """Write charges in the format: Element  x(Å)  y(Å)  z(Å)  q(e)."""
    with open(out_path, "w") as f:
        for el, x, y, z, q in atoms_with_q:
            f.write(f"{el:>2s} {x:12.6f} {y:12.6f} {z:12.6f} {q:+14.8f}\n")
    print(f"[IO] Wrote {len(atoms_with_q)} TrESP charges to '{out_path}'")

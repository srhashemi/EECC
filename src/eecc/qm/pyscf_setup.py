"""Construction of PySCF molecules and Kohn-Sham objects from pipeline settings."""

from __future__ import annotations

import os
import re
from typing import Optional, Sequence

from eecc.qm.structure import Structure


# Functionals whose common name PySCF rejects, mapped to the libxc name.
# PySCF blacklists 'wb97x-d' because the Chai-Head-Gordon dispersion term is not
# implemented. The libxc functional below is the exchange-correlation part of
# wB97X-D. The missing dispersion term is a geometry-only energy shift, so it
# changes neither vertical excitation energies nor transition densities, but it
# must not be used for geometry optimization.
XC_ALIASES = {
    "wb97x-d": "HYB_GGA_XC_WB97X_D",
    "wb97xd": "HYB_GGA_XC_WB97X_D",
    "wb97x_d": "HYB_GGA_XC_WB97X_D",
}

_POPLE = re.compile(r"^\d+-\d+\+*g", re.IGNORECASE)


def resolve_xc(xc: str) -> str:
    return XC_ALIASES.get(xc.strip().lower(), xc)


def use_cartesian(basis: str, cart: Optional[bool]) -> bool:
    """Follow Gaussian's defaults unless set explicitly.

    Gaussian uses Cartesian d functions (6D) for Pople bases such as 3-21G and
    6-31G, but spherical ones (5D) for the 6-311G family and for all other bases.
    """
    if cart is not None:
        return bool(cart)
    b = basis.strip().lower()
    return bool(_POPLE.match(b)) and not b.startswith("6-311")


def set_threads(threads: int) -> int:
    """Set the PySCF/OpenMP thread count (0 keeps the environment default)."""
    from pyscf import lib

    if threads and threads > 0:
        lib.num_threads(threads)
    return lib.num_threads()


def set_tmpdir(tmpdir: Optional[str]) -> None:
    from pyscf import lib

    if tmpdir:
        tmpdir = os.path.expandvars(tmpdir)
        os.makedirs(tmpdir, exist_ok=True)
        lib.param.TMPDIR = tmpdir


def build_mol(
    structure: Structure,
    basis: str,
    cart: Optional[bool] = None,
    charge: int = 0,
    spin: int = 0,
    memory_mb: int = 16000,
    verbose: int = 4,
    output: Optional[str] = None,
):
    from pyscf import gto

    return gto.M(
        atom=structure.pyscf_atoms(),
        unit="Angstrom",
        basis=basis,
        cart=use_cartesian(basis, cart),
        charge=charge,
        spin=spin,
        max_memory=memory_mb,
        verbose=verbose,
        output=output,
    )


def build_rks(
    mol,
    xc: str,
    grid: Sequence[int] = (99, 590),
    density_fit: bool = False,
    conv_tol: float = 1e-9,
    disp: Optional[str] = None,
    chkfile: Optional[str] = None,
):
    from pyscf import dft

    if mol.spin != 0:
        raise NotImplementedError("only closed-shell (spin 0) systems are supported")
    mf = dft.RKS(mol, xc=resolve_xc(xc))
    mf.grids.atom_grid = tuple(grid)
    mf.conv_tol = conv_tol
    if disp:
        mf.disp = disp
    if chkfile:
        mf.chkfile = chkfile
    if density_fit:
        mf = mf.density_fit()
    return mf

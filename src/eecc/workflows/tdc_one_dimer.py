"""TDC workflow: compute coupling from a single dimer cube by fragment splitting."""

from __future__ import annotations

import os

import numpy as np

from eecc.io.cube import read_cube, grid_spacing
from eecc.coupling.tdc_fft import (
    transition_dipole_from_cube,
    integrate_density_total_charge,
    neutralize_full_density,
    tdc_coupling_fft_simple,
)
from eecc.coupling.dipole import point_dipole_coupling
from eecc.geometry.fragments import (
    parse_indices,
    split_cube_by_nearest_atom,
    build_monomer_cube,
)


def run_one_dimer_tdc(
    dimer_name: str, fragA: list, fragB: list,
    dielectric: float = 1.0, pad_factor: int = 3,
) -> None:
    """Run TDC coupling from a single dimer cube file."""
    inputs_dir = "inputs"
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    cube_path = os.path.join(inputs_dir, dimer_name)
    cube = read_cube(cube_path, units="bohr")

    maskA, maskB, rhoA_raw, rhoB_raw = split_cube_by_nearest_atom(cube, fragA, fragB)

    rhoA, offA = neutralize_full_density(rhoA_raw)
    rhoB, offB = neutralize_full_density(rhoB_raw)

    cubeA = build_monomer_cube(cube, fragA, rhoA)
    cubeB = build_monomer_cube(cube, fragB, rhoB)

    # --- Diagnostics ---
    print("\n=== Diagnostics ===")

    muA = transition_dipole_from_cube(cubeA)
    muB = transition_dipole_from_cube(cubeB)

    print(f"mu_A (e*Ang) = {muA['mu']}")
    print(f"|mu_A| (D) = {muA['mu_D']:.6f}")
    print(f"mu_B (e*Ang) = {muB['mu']}")
    print(f"|mu_B| (D) = {muB['mu_D']:.6f}")

    QA = integrate_density_total_charge(cubeA)
    QB = integrate_density_total_charge(cubeB)
    print(f"Total charge A = {QA:.6e} e")
    print(f"Total charge B = {QB:.6e} e")
    print(f"Neutralization offsets: offA = {offA:.6e}, offB = {offB:.6e} (e/Ang^3)")

    rhoA_rms = np.sqrt(np.mean(cubeA['rho']**2))
    rhoB_rms = np.sqrt(np.mean(cubeB['rho']**2))
    print(f"RMS(rho_A) = {rhoA_rms:.6e}")
    print(f"RMS(rho_B) = {rhoB_rms:.6e}")

    dx, dy, dz = grid_spacing(cubeA)
    print(f"Grid spacing (Ang): dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")

    atomsA = np.array([[a[2], a[3], a[4]] for a in cubeA['atoms']])
    atomsB = np.array([[a[2], a[3], a[4]] for a in cubeB['atoms']])
    centA = atomsA.mean(axis=0)
    centB = atomsB.mean(axis=0)
    distAB = np.linalg.norm(centA - centB)
    print(f"Centroid distance A-B = {distAB:.4f} Ang")

    Rvec = centB - centA
    Jpd_eV, Jpd_cm1 = point_dipole_coupling(muA['mu'], muB['mu'], Rvec, dielectric=dielectric)

    print(f"  dielectric = {dielectric}")
    print(f"  pad factor = {pad_factor}")
    print(f"Point-dipole J ~ {Jpd_cm1:.2f} cm^-1 ({Jpd_eV:.6f} eV)")

    tdc = tdc_coupling_fft_simple(cubeA, cubeB, dielectric=dielectric, pad_factor=pad_factor)

    out_path = os.path.join(outputs_dir, "TDC_from_one_dimer.txt")
    with open(out_path, "w") as f:
        f.write(f"# dielectric = {dielectric}\n")
        f.write(f"# pad_factor = {pad_factor}\n")
        f.write("J_eV  J_cm^-1\n")
        f.write(f"{tdc['J_eV']:.8f}  {tdc['J_cm1']:.4f}\n\n")

        f.write("# Transition dipoles\n")
        f.write(f"muA (e*Ang): {muA['mu']}\n")
        f.write(f"|muA| (D): {muA['mu_D']:.6f}\n")
        f.write(f"muB (e*Ang): {muB['mu']}\n")
        f.write(f"|muB| (D): {muB['mu_D']:.6f}\n\n")

        f.write("# Total charges\n")
        f.write(f"Q_A = {QA:.6e} e\n")
        f.write(f"Q_B = {QB:.6e} e\n\n")

        f.write("# RMS densities\n")
        f.write(f"RMS(rho_A) = {rhoA_rms:.6e}\n")
        f.write(f"RMS(rho_B) = {rhoB_rms:.6e}\n\n")

        f.write("# Centroid distance\n")
        f.write(f"Centroid distance A-B = {distAB:.4f} Ang\n")

    print(f"TDC coupling saved to {out_path}")

"""TDC workflow: compute coupling from two monomer cube files.

Runs four methods — point-dipole, extended-dipole, TDC (FFT), and
TDC (direct summation) — and prints a comparison table.
"""

from __future__ import annotations

import os
import time

from eecc.io.cube import read_cube, grid_spacing
from eecc.coupling.tdc_fft import tdc_coupling_fft, transition_dipole_from_cube
from eecc.coupling.tdc_bruteforce import tdc_coupling_bruteforce


def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def run_tdc(
    monomerA_name: str,
    monomerB_name: str,
    dielectric: float = 1.0,
    pad_factor: int = 3,
    threshold: float = 0.0005,
) -> None:
    """Run all TDC coupling methods between two monomer cubes."""
    inputs_dir = "inputs"
    outputs_dir = "outputs"
    _ensure_dir(outputs_dir)

    cubeA_path = os.path.join(inputs_dir, monomerA_name)
    cubeB_path = os.path.join(inputs_dir, monomerB_name)

    print("\n=== Transition Density Cube (TDC) Coupling ===")
    print(f"  cubeA: {cubeA_path}")
    print(f"  cubeB: {cubeB_path}")

    cubeA = read_cube(cubeA_path, units="bohr")
    cubeB = read_cube(cubeB_path, units="bohr")

    # --- Transition dipoles ---
    muA = transition_dipole_from_cube(cubeA)
    muB = transition_dipole_from_cube(cubeB)

    # --- Grid info ---
    dxA, dyA, dzA = grid_spacing(cubeA)
    dxB, dyB, dzB = grid_spacing(cubeB)

    print(f"\n  Transition dipoles:")
    print(f"    |mu_A| = {muA['mu_D']:.4f} D    mu_A = ({muA['mu'][0]:.6f}, {muA['mu'][1]:.6f}, {muA['mu'][2]:.6f}) e*Ang")
    print(f"    |mu_B| = {muB['mu_D']:.4f} D    mu_B = ({muB['mu'][0]:.6f}, {muB['mu'][1]:.6f}, {muB['mu'][2]:.6f}) e*Ang")
    print(f"\n  Grid A: {cubeA['nv']},  spacing ({dxA:.6f}, {dyA:.6f}, {dzA:.6f}) Ang")
    print(f"  Grid B: {cubeB['nv']},  spacing ({dxB:.6f}, {dyB:.6f}, {dzB:.6f}) Ang")

    # --- FFT TDC ---
    t0 = time.time()
    fft = tdc_coupling_fft(cubeA, cubeB, dielectric=dielectric, pad_factor=pad_factor)
    t_fft = time.time() - t0

    # --- Direct TDC ---
    t0 = time.time()
    direct = tdc_coupling_bruteforce(cubeA, cubeB, dielectric=dielectric, threshold=threshold)
    t_direct = time.time() - t0

    # --- Print table ---
    w = 62
    print("\n" + "=" * w)
    print(f"  {'Method':<22} {'J (cm^-1)':>12} {'J (eV)':>12} {'Time':>10}")
    print("-" * w)
    print(f"  {'Point-dipole':<22} {fft['Jpd_cm1']:>12.2f} {fft['Jpd_eV']:>12.6f} {'---':>10}")
    print(f"  {'Extended-dipole':<22} {fft['Jext_cm1']:>12.2f} {fft['Jext_eV']:>12.6f} {'---':>10}")
    print(f"  {'TDC (FFT)':<22} {fft['J_cm1']:>12.2f} {fft['J_eV']:>12.6f} {t_fft:>9.1f}s")
    print(f"  {'TDC (direct)':<22} {direct['J_cm1']:>12.2f} {direct['J_eV']:>12.6f} {t_direct:>9.1f}s")
    print("=" * w)

    print(f"\n  R_AB = {fft['R_AB_Ang']:.4f} Ang")
    print(f"  FFT pad factor = {pad_factor}")
    print(f"  Direct threshold = {threshold}  "
          f"(voxels: A={direct['nA']:,}, B={direct['nB']:,},  "
          f"capture: A={direct['capture_A']:.1%}, B={direct['capture_B']:.1%})")

    # --- Write output file ---
    out_file = os.path.join(outputs_dir, "TDC_two_monomer_results.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# TDC Coupling Results — Two Monomer Cubes\n")
        f.write(f"# cubeA: {cubeA_path}\n")
        f.write(f"# cubeB: {cubeB_path}\n")
        f.write(f"# dielectric = {dielectric}\n")
        f.write("#\n")

        f.write("# Transition dipoles\n")
        f.write(f"#   |mu_A| = {muA['mu_D']:.4f} D    "
                f"mu_A = ({muA['mu'][0]:.6f}, {muA['mu'][1]:.6f}, {muA['mu'][2]:.6f}) e*Ang\n")
        f.write(f"#   |mu_B| = {muB['mu_D']:.4f} D    "
                f"mu_B = ({muB['mu'][0]:.6f}, {muB['mu'][1]:.6f}, {muB['mu'][2]:.6f}) e*Ang\n")
        f.write(f"#   R_AB = {fft['R_AB_Ang']:.6f} Ang\n")
        f.write("#\n")

        f.write(f"# Grid A: {cubeA['nv']},  spacing ({dxA:.6f}, {dyA:.6f}, {dzA:.6f}) Ang\n")
        f.write(f"# Grid B: {cubeB['nv']},  spacing ({dxB:.6f}, {dyB:.6f}, {dzB:.6f}) Ang\n")
        f.write(f"# FFT pad factor = {pad_factor}\n")
        f.write(f"# Direct threshold = {threshold}  "
                f"(voxels: A={direct['nA']:,}, B={direct['nB']:,},  "
                f"capture: A={direct['capture_A']:.1%}, B={direct['capture_B']:.1%})\n")
        f.write("#\n")

        hdr = f"# {'Method':<22} {'J (cm^-1)':>14} {'J (eV)':>14} {'Time (s)':>10}\n"
        sep = f"# {'-'*60}\n"
        f.write(hdr)
        f.write(sep)
        f.write(f"  {'Point-dipole':<22} {fft['Jpd_cm1']:>14.4f} {fft['Jpd_eV']:>14.8f} {'---':>10}\n")
        f.write(f"  {'Extended-dipole':<22} {fft['Jext_cm1']:>14.4f} {fft['Jext_eV']:>14.8f} {'---':>10}\n")
        f.write(f"  {'TDC (FFT)':<22} {fft['J_cm1']:>14.4f} {fft['J_eV']:>14.8f} {t_fft:>10.1f}\n")
        f.write(f"  {'TDC (direct)':<22} {direct['J_cm1']:>14.4f} {direct['J_eV']:>14.8f} {t_direct:>10.1f}\n")

    print(f"\n  Results saved to '{out_file}'")

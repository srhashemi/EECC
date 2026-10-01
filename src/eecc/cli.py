"""Unified CLI for the Exciton-Exciton Coupling Calculator (eecc)."""

from __future__ import annotations

import argparse
import sys


def _cmd_intramolecular(args):
    from eecc.workflows.intramolecular import run_intramolecular
    run_intramolecular(
        filename=args.charges_file,
        fragments_str=args.frags,
        scale_factor=args.scale,
        dielectric=args.dielectric,
    )


def _cmd_intermolecular(args):
    from eecc.workflows.intermolecular import run_intermolecular
    run_intermolecular(
        input_file=args.monomer_file,
        placements_str=args.placement,
        eps_r=args.dielectric,
    )


def _cmd_tdc_two_monomers(args):
    from eecc.workflows.tdc_two_monomers import run_tdc
    run_tdc(args.cubeA, args.cubeB,
            dielectric=args.dielectric, pad_factor=args.pad,
            threshold=args.threshold)


def _cmd_tdc_one_dimer(args):
    from eecc.geometry.fragments import parse_indices
    from eecc.workflows.tdc_one_dimer import run_one_dimer_tdc

    fragA = parse_indices(args.fragA)
    fragB = parse_indices(args.fragB)
    run_one_dimer_tdc(args.dimer_cube, fragA, fragB,
                      dielectric=args.dielectric, pad_factor=args.pad)


def _cmd_tresp(args):
    from eecc.tresp.fitting import fit_tresp_from_cube
    from eecc.io.charges import write_charges_txt

    shells = tuple(float(x) for x in args.shells.split(","))
    atoms_q = fit_tresp_from_cube(
        cube_path=args.cube,
        shells_ang=shells,
        pps=args.pps,
        alpha=args.alpha,
        enforce_dipole=(not args.no_dipole),
        pad=args.pad,
    )
    write_charges_txt(atoms_q, out_path=args.out)


def _cmd_check_charge(args):
    from eecc.io.cube import read_cube, grid_spacing
    import numpy as np

    cube = read_cube(args.cube_file, units="bohr")
    rho = cube['rho']
    dx, dy, dz = grid_spacing(cube)
    dV = dx * dy * dz
    Q = float(np.sum(rho) * dV)
    print(f"Total charge = {Q:.3e} e")


def _cmd_view(args):
    from eecc.viz.viewer import main as viewer_main
    # Re-pack sys.argv for viewer's own argparse
    argv = [args.cubeA]
    if args.cubeB:
        argv.extend(["--cubeB", args.cubeB])
    if args.save:
        argv.extend(["--save", args.save])
    if args.center:
        argv.append("--center")
    if args.align:
        argv.append("--align")

    sys.argv = ["eecc view"] + argv
    viewer_main()


def _cmd_plot_density(args):
    from eecc.io.cube import read_cube
    from eecc.viz.cube_plot import plot_two_density_slices, print_grid_summary

    cubeA = read_cube(args.cubeA, units="bohr")
    cubeB = read_cube(args.cubeB, units="bohr")

    print_grid_summary("cubeA", cubeA)
    print_grid_summary("cubeB", cubeB)

    plot_two_density_slices(cubeA, cubeB, plane=args.plane, index=args.index)


def _cmd_run(args):
    try:
        from eecc.qm.config import load_config
        from eecc.qm.pipeline import Pipeline
    except ImportError as exc:
        sys.exit(f"eecc run needs the optional QM dependencies: pip install 'eecc[qm]' ({exc})")

    cfg = load_config(args.config)
    pipe = Pipeline(cfg)
    if args.restamp:
        marked = pipe.restamp()
        print("Marked as up to date: " + (", ".join(marked) if marked else "nothing"))
        return
    if args.slurm or args.dry_run:
        from eecc.qm.slurm import submit
        out = submit(pipe, args.config, dry_run=args.dry_run)
        if args.dry_run:
            print("Job scripts written (not submitted):")
            print("\n".join(f"  {p}" for p in out))
        else:
            print("Submitted jobs: " + ", ".join(out))
        return
    from eecc.qm.pyscf_setup import set_threads, set_tmpdir
    set_threads(cfg.resources.threads)
    set_tmpdir(cfg.resources.tmpdir)
    pipe.run(stage=args.stage, fragment=args.fragment, force=args.force)


def main() -> None:
    """Entry point for the ``eecc`` command."""
    parser = argparse.ArgumentParser(
        prog="eecc",
        description="Exciton-Exciton Coupling Calculator",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0.0")
    sub = parser.add_subparsers(dest="command", help="Available subcommands")

    # --- intramolecular ---
    p = sub.add_parser("intramolecular", help="Intramolecular coupling from charge file")
    p.add_argument("charges_file", help="Charge file name inside inputs/")
    p.add_argument("--frags", nargs="+", required=True,
                   help='Fragment atom indices (1-based), e.g. --frags "1-56" "57-112,129-140"')
    p.add_argument("--scale", type=float, default=1.0, help="Charge scale factor")
    p.add_argument("--dielectric", type=float, default=1.0, help="Dielectric constant")
    p.set_defaults(func=_cmd_intramolecular)

    # --- intermolecular ---
    p = sub.add_parser("intermolecular", help="Intermolecular coupling from monomer file")
    p.add_argument("monomer_file", help="Monomer file name inside inputs/")
    p.add_argument("--placement", nargs="+", required=True,
                   help='Monomer placements: "posX,posY,posZ axisX,axisY,axisZ angleDeg"')
    p.add_argument("--dielectric", type=float, default=1.0, help="Dielectric constant")
    p.set_defaults(func=_cmd_intermolecular)

    # --- tdc-two-monomers ---
    p = sub.add_parser("tdc-two-monomers", help="TDC coupling from two monomer cubes")
    p.add_argument("cubeA", help="Monomer A cube filename inside inputs/")
    p.add_argument("cubeB", help="Monomer B cube filename inside inputs/")
    p.add_argument("--pad", type=int, default=3, help="FFT padding factor")
    p.add_argument("--threshold", type=float, default=0.0005,
                   help="Direct-sum threshold: keep voxels with |q| > threshold*max|q|")
    p.add_argument("--dielectric", type=float, default=1.0, help="Dielectric constant")
    p.set_defaults(func=_cmd_tdc_two_monomers)

    # --- tdc-one-dimer ---
    p = sub.add_parser("tdc-one-dimer", help="TDC coupling from one dimer cube")
    p.add_argument("dimer_cube", help="Dimer cube filename inside inputs/")
    p.add_argument("--fragA", type=str, required=True, help="Atom indices for fragment A")
    p.add_argument("--fragB", type=str, required=True, help="Atom indices for fragment B")
    p.add_argument("--pad", type=int, default=3, help="FFT padding factor")
    p.add_argument("--dielectric", type=float, default=1.0, help="Dielectric constant")
    p.set_defaults(func=_cmd_tdc_one_dimer)

    # --- tresp ---
    p = sub.add_parser("tresp", help="Fit TrESP charges from a transition-density cube")
    p.add_argument("--cube", required=True, help="Transition-density cube file")
    p.add_argument("--out", default="charges.txt", help="Output charges file")
    p.add_argument("--shells", type=str, default="1.6,2.0,2.4", help="CHELPG shell offsets (Å)")
    p.add_argument("--pps", type=int, default=6000, help="Points per shell")
    p.add_argument("--alpha", type=float, default=1e-3, help="Ridge regularization")
    p.add_argument("--no_dipole", action="store_true", help="Skip dipole constraint")
    p.add_argument("--pad", type=int, default=2, help="FFT padding factor")
    p.set_defaults(func=_cmd_tresp)

    # --- check-charge ---
    p = sub.add_parser("check-charge", help="Print total transition charge from a cube")
    p.add_argument("cube_file", help="Cube file path")
    p.set_defaults(func=_cmd_check_charge)

    # --- view ---
    p = sub.add_parser("view", help="Visualize molecules from cube files")
    p.add_argument("cubeA", help="Monomer A cube file")
    p.add_argument("--cubeB", help="Optional monomer B cube file")
    p.add_argument("--save", help="Save figure to PNG")
    p.add_argument("--center", action="store_true", help="Center before plotting")
    p.add_argument("--align", action="store_true", help="Align B to A")
    p.set_defaults(func=_cmd_view)

    # --- plot-density ---
    p = sub.add_parser("plot-density", help="Plot density slices from two cubes")
    p.add_argument("cubeA", help="Monomer A cube file")
    p.add_argument("cubeB", help="Monomer B cube file")
    p.add_argument("--plane", choices=["x", "y", "z"], default="z", help="Slice plane")
    p.add_argument("--index", type=int, default=None, help="Slice index")
    p.set_defaults(func=_cmd_plot_density)

    # --- run (automated QM pipeline) ---
    p = sub.add_parser("run", help="Automated pipeline: geometry -> TD-DFT -> couplings (needs eecc[qm])")
    p.add_argument("config", help="Pipeline YAML config file")
    p.add_argument("--stage", default="all",
                   choices=["all", "opt", "fragments", "td", "transition", "couplings"],
                   help="Run only this stage")
    p.add_argument("--fragment", type=int, default=None,
                   help="1-based fragment index (td/transition stages only)")
    p.add_argument("--force", action="store_true", help="Rerun stages even if up to date")
    p.add_argument("--slurm", action="store_true", help="Submit the pipeline as Slurm jobs")
    p.add_argument("--dry-run", action="store_true", help="Write Slurm scripts without submitting")
    p.add_argument("--restamp", action="store_true",
                   help="Mark stages with existing outputs as up to date (after upgrading EECC "
                        "with unchanged settings)")
    p.set_defaults(func=_cmd_run)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

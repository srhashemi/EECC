"""Molecule viewer: load cube files and render molecules."""

from __future__ import annotations

import argparse

from eecc.io.cube import read_cube
from eecc.geometry.transform import center_coordinates, align_molecules
from eecc.viz.mol_plot import plot_molecule_from_cube, plot_two_molecules


def parse_arguments():
    """Parse command-line arguments for the molecule viewer."""
    parser = argparse.ArgumentParser(
        description="Visualize molecules from cube atom blocks."
    )

    parser.add_argument("cubeA", help="Path to monomer A cube file (Bohr units)")
    parser.add_argument("--cubeB", help="Optional path to monomer B cube file (Bohr units)")

    parser.add_argument("--nobox", action="store_true", help="Disable cube grid box")
    parser.add_argument("--noaxes", action="store_true", help="Disable XYZ axes")

    parser.add_argument("--save", help="Save figure to PNG instead of only showing it")
    parser.add_argument("--dpi", type=int, default=200, help="Resolution for saved PNG (default: 200)")
    parser.add_argument("--transparent", action="store_true",
                        help="Export PNG with transparent background")

    parser.add_argument("--center", action="store_true",
                        help="Center monomer A (and B if provided) before plotting")
    parser.add_argument("--align", action="store_true",
                        help="Align monomer B to monomer A (centroid alignment)")

    return parser.parse_args()


def main():
    """Entry point for the molecule viewer."""
    args = parse_arguments()

    cubeA = read_cube(args.cubeA, units="bohr")

    if args.center:
        cubeA = center_coordinates(cubeA)

    if args.cubeB:
        cubeB = read_cube(args.cubeB, units="bohr")

        if args.align:
            cubeA, cubeB = align_molecules(cubeA, cubeB)
        elif args.center:
            cubeB = center_coordinates(cubeB)

        fig, ax = plot_two_molecules(
            cubeA,
            cubeB,
            show_box=not args.nobox,
            show_axes=not args.noaxes,
            title="Two Monomers"
        )
    else:
        fig, ax = plot_molecule_from_cube(
            cubeA,
            show_box=not args.nobox,
            show_axes=not args.noaxes,
            title="Monomer A"
        )

    if args.save:
        fig.savefig(
            args.save,
            dpi=args.dpi,
            transparent=args.transparent,
            bbox_inches="tight"
        )
        print(f"Saved figure to {args.save}")

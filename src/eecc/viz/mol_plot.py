"""Molecular visualization from cube atom blocks."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ============================================================
#  Covalent radii (Å) and element colors
# ============================================================

COVALENT_RADII = {
    1: 0.31,  # H
    5: 0.85,  # B
    6: 0.76,  # C
    7: 0.71,  # N
    8: 0.66,  # O
    9: 0.57,  # F
}

ELEMENT_COLORS = {
    1: "#FFFFFF",  # H white
    5: "#00BFFF",  # B deep sky
    6: "#909090",  # C gray
    7: "#3050F8",  # N blue
    8: "#FF0D0D",  # O red
    9: "#90E050",  # F green
}

DEFAULT_COLOR = "#FFD700"
DEFAULT_RADIUS = 0.77


# ============================================================
#  Internal helpers
# ============================================================

def _get_atom_arrays(cube):
    """Extract atomic numbers (Z) and coordinates (R) from cube['atoms']."""
    atoms = cube.get("atoms", [])
    if not atoms:
        raise ValueError("cube['atoms'] is missing or empty.")

    Z = np.array([int(a[0]) for a in atoms], dtype=int)
    R = np.array([[a[2], a[3], a[4]] for a in atoms], dtype=float)
    return Z, R


def _infer_bonds(Z, R, scale=1.15, max_dist=1.9):
    """Infer bonds using covalent radii and distance thresholds."""
    n = len(Z)
    if n == 0:
        return []

    radii = np.array([COVALENT_RADII.get(int(z), DEFAULT_RADIUS) for z in Z])
    bonds = []

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(R[i] - R[j])
            cutoff = scale * (radii[i] + radii[j])
            if dist <= max_dist and dist <= cutoff:
                bonds.append((i, j))

    return bonds


def _axes_from_cube_extent(cube):
    """Convert cube grid definition to total extents."""
    nx, ny, nz = cube["nv"]
    VX = (nx - 1) * cube["vx"]
    VY = (ny - 1) * cube["vy"]
    VZ = (nz - 1) * cube["vz"]
    return cube["origin"], VX, VY, VZ


def _box_segments(origin, VX, VY, VZ):
    """Return line segments for the cube box edges."""
    O = np.asarray(origin)
    VX, VY, VZ = map(np.asarray, (VX, VY, VZ))

    corners = np.array([
        O,
        O + VX,
        O + VY,
        O + VZ,
        O + VX + VY,
        O + VX + VZ,
        O + VY + VZ,
        O + VX + VY + VZ,
    ])

    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]

    return [(corners[i], corners[j]) for i, j in edges], corners


def _hide_axes_3d(ax):
    """Hide all axis elements for a clean molecular view."""
    ax.set_axis_off()
    ax.grid(False)


# ============================================================
#  Public plotting functions
# ============================================================

def plot_molecule_from_cube(
    cube: Dict[str, Any],
    show_box: bool = True,
    show_axes: bool = True,
    atom_scale: float = 250.0,
    linewidth: float = 1.6,
    bond_scale: float = 1.15,
    bond_max: float = 1.9,
    title: str = "Molecule from cube",
) -> Tuple[Any, Any]:
    """Plot a single molecule from cube data."""
    Z, R = _get_atom_arrays(cube)
    bonds = _infer_bonds(Z, R, scale=bond_scale, max_dist=bond_max)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if bonds:
        segs = [(R[i], R[j]) for i, j in bonds]
        ax.add_collection3d(Line3DCollection(segs, colors="k", linewidths=linewidth, alpha=0.8))

    colors = [ELEMENT_COLORS.get(int(z), DEFAULT_COLOR) for z in Z]
    ax.scatter(
        R[:, 0], R[:, 1], R[:, 2],
        s=atom_scale, c=colors,
        edgecolors="k", linewidths=0.5, alpha=0.95
    )

    pts = R.copy()
    if show_box:
        origin, VX, VY, VZ = _axes_from_cube_extent(cube)
        segs_box, corners = _box_segments(origin, VX, VY, VZ)
        ax.add_collection3d(Line3DCollection(segs_box, colors="C1", linewidths=1.0, alpha=0.5))
        ax.scatter([origin[0]], [origin[1]], [origin[2]], c="C1", s=30)
        pts = np.vstack([pts, corners])

    if show_axes:
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title(title)
    else:
        _hide_axes_3d(ax)

    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    margin = 0.5
    ax.set_xlim(mins[0] - margin, maxs[0] + margin)
    ax.set_ylim(mins[1] - margin, maxs[1] + margin)
    ax.set_zlim(mins[2] - margin, maxs[2] + margin)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_two_molecules(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    show_box: bool = True,
    show_axes: bool = True,
    title: str = "Two molecules from cubes",
) -> Tuple[Any, Any]:
    """Overlay two molecules (atoms + bonds) from two cube files."""
    ZA, RA = _get_atom_arrays(cubeA)
    bondsA = _infer_bonds(ZA, RA)
    colorsA = [ELEMENT_COLORS.get(int(z), DEFAULT_COLOR) for z in ZA]

    ZB, RB = _get_atom_arrays(cubeB)
    bondsB = _infer_bonds(ZB, RB)
    colorsB = [ELEMENT_COLORS.get(int(z), DEFAULT_COLOR) for z in ZB]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if bondsA:
        ax.add_collection3d(Line3DCollection(
            [(RA[i], RA[j]) for i, j in bondsA],
            colors="k", linewidths=1.6, alpha=0.8
        ))
    if bondsB:
        ax.add_collection3d(Line3DCollection(
            [(RB[i], RB[j]) for i, j in bondsB],
            colors="#444444", linewidths=1.4, alpha=0.8
        ))

    ax.scatter(RA[:, 0], RA[:, 1], RA[:, 2],
               s=240, c=colorsA, edgecolors="k", linewidths=0.5, alpha=0.95)
    ax.scatter(RB[:, 0], RB[:, 1], RB[:, 2],
               s=240, c=colorsB, edgecolors="#333", linewidths=0.5, alpha=0.85)

    pts = np.vstack([RA, RB])

    if show_box:
        for cube, color in [(cubeA, "C0"), (cubeB, "C1")]:
            origin, VX, VY, VZ = _axes_from_cube_extent(cube)
            segs, corners = _box_segments(origin, VX, VY, VZ)
            ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=0.8, alpha=0.4))
            pts = np.vstack([pts, corners])

    if show_axes:
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.set_zlabel("Z (Å)")
        ax.set_title(title)
    else:
        _hide_axes_3d(ax)

    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    margin = 0.5
    ax.set_xlim(mins[0] - margin, maxs[0] + margin)
    ax.set_ylim(mins[1] - margin, maxs[1] + margin)
    ax.set_zlim(mins[2] - margin, maxs[2] + margin)

    plt.tight_layout()
    plt.show()
    return fig, ax

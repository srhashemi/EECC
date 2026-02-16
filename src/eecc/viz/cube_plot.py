"""Visualization of cube grids and transition-density slices."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from eecc.io.cube import grid_spacing


# ============================================================
#  Core geometric helpers
# ============================================================

def _cube_corners(origin, vx, vy, vz):
    """Return the 8 cube corners given origin and voxel vectors."""
    O = np.asarray(origin, float)
    vx = np.asarray(vx, float)
    vy = np.asarray(vy, float)
    vz = np.asarray(vz, float)

    return np.array([
        O,
        O + vx,
        O + vy,
        O + vz,
        O + vx + vy,
        O + vx + vz,
        O + vy + vz,
        O + vx + vy + vz,
    ])


def _cube_edges_from_corners(P):
    """Return line segments (pairs of points) for cube edges."""
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    return [(P[i], P[j]) for i, j in edges]


# ============================================================
#  Plotting: cube grid and axes
# ============================================================

def plot_cube_grid(ax: Any, origin: np.ndarray, VX: np.ndarray, VY: np.ndarray,
                   VZ: np.ndarray, color: str = 'C0', label: str = 'cube',
                   alpha: float = 1.0, lw: float = 1.5) -> None:
    """Plot a cube wireframe defined by origin and total extents VX, VY, VZ."""
    corners = _cube_corners(origin, VX, VY, VZ)
    segs = _cube_edges_from_corners(corners)

    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha))
    ax.scatter([origin[0]], [origin[1]], [origin[2]], color=color, s=30, label=label)


def plot_axes_quivers(ax: Any, origin: np.ndarray, vx: np.ndarray, vy: np.ndarray,
                      vz: np.ndarray, scale: float = 5.0,
                      colors: Tuple[str, ...] = ('r', 'g', 'b')) -> None:
    """Plot voxel axes (scaled) as quivers from the cube origin."""
    O = np.asarray(origin, float)
    vx = np.asarray(vx, float) * scale
    vy = np.asarray(vy, float) * scale
    vz = np.asarray(vz, float) * scale

    ax.quiver(O[0], O[1], O[2], vx[0], vx[1], vx[2], color=colors[0])
    ax.quiver(O[0], O[1], O[2], vy[0], vy[1], vy[2], color=colors[1])
    ax.quiver(O[0], O[1], O[2], vz[0], vz[1], vz[2], color=colors[2])


# ============================================================
#  Plotting: cube overlays
# ============================================================

def plot_two_cubes_overlaid(cubeA: Dict[str, Any], cubeB: Dict[str, Any],
                            title: str = "Cube grids (origin & axes)",
                            show: bool = True) -> Tuple[Any, Any]:
    """Overlay two cube grids with their voxel axes."""
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    nx, ny, nz = cubeA['nv']
    VX_A = (nx - 1) * cubeA['vx']
    VY_A = (ny - 1) * cubeA['vy']
    VZ_A = (nz - 1) * cubeA['vz']

    VX_B = (nx - 1) * cubeB['vx']
    VY_B = (ny - 1) * cubeB['vy']
    VZ_B = (nz - 1) * cubeB['vz']

    plot_cube_grid(ax, cubeA['origin'], VX_A, VY_A, VZ_A, color='C0', label='cubeA', alpha=0.9)
    plot_axes_quivers(ax, cubeA['origin'], cubeA['vx'], cubeA['vy'], cubeA['vz'],
                      scale=5.0, colors=('r', 'g', 'b'))

    plot_cube_grid(ax, cubeB['origin'], VX_B, VY_B, VZ_B, color='C1', label='cubeB', alpha=0.9)
    plot_axes_quivers(ax, cubeB['origin'], cubeB['vx'], cubeB['vy'], cubeB['vz'],
                      scale=5.0, colors=('m', 'c', 'y'))

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.set_title(title)
    ax.legend()

    all_pts = np.vstack([
        _cube_corners(cubeA['origin'], VX_A, VY_A, VZ_A),
        _cube_corners(cubeB['origin'], VX_B, VY_B, VZ_B),
    ])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    if show:
        plt.show()

    return fig, ax


# ============================================================
#  Plotting: density slices
# ============================================================

def plot_density_slice(cube: Dict[str, Any], plane: str = 'z',
                       index: Optional[int] = None, vmin: Optional[float] = None,
                       vmax: Optional[float] = None, cmap: str = 'seismic',
                       title: str = '') -> None:
    """Plot a 2D slice of the 3D density cube."""
    rho = cube['rho']
    nx, ny, nz = cube['nv']
    ox, oy, oz = cube['origin']
    vx, vy, vz = cube['vx'], cube['vy'], cube['vz']

    if plane == 'x':
        index = nx // 2 if index is None else index
        slice2d = rho[index, :, :]
        xs = oy + np.arange(ny) * vy[1]
        ys = oz + np.arange(nz) * vz[2]
        xlabel, ylabel = "Y (Å)", "Z (Å)"
    elif plane == 'y':
        index = ny // 2 if index is None else index
        slice2d = rho[:, index, :].T
        xs = ox + np.arange(nx) * vx[0]
        ys = oz + np.arange(nz) * vz[2]
        xlabel, ylabel = "X (Å)", "Z (Å)"
    else:  # 'z'
        index = nz // 2 if index is None else index
        slice2d = rho[:, :, index]
        xs = ox + np.arange(nx) * vx[0]
        ys = oy + np.arange(ny) * vy[1]
        xlabel, ylabel = "X (Å)", "Y (Å)"

    extent = [xs.min(), xs.max(), ys.min(), ys.max()]

    plt.figure(figsize=(6, 5))
    plt.imshow(slice2d.T, origin='lower', extent=extent, aspect='auto',
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label="ρ (e/Å³)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title or f"{plane}-slice @ index {index}")
    plt.tight_layout()
    plt.show()


def plot_two_density_slices(cubeA: Dict[str, Any], cubeB: Dict[str, Any],
                            plane: str = 'z', index: Optional[int] = None,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None) -> None:
    """Plot the same slice plane/index for two cubes."""
    plot_density_slice(cubeA, plane=plane, index=index, vmin=vmin, vmax=vmax,
                       title=f"A: {plane}-slice")
    plot_density_slice(cubeB, plane=plane, index=index, vmin=vmin, vmax=vmax,
                       title=f"B: {plane}-slice")


# ============================================================
#  Grid summary
# ============================================================

def print_grid_summary(label: str, cube: Dict[str, Any]) -> None:
    """Print grid dimensions, origin, spacing, and voxel vectors."""
    dx, dy, dz = grid_spacing(cube)
    nx, ny, nz = cube['nv']
    ox, oy, oz = cube['origin']
    vx, vy, vz = cube['vx'], cube['vy'], cube['vz']

    print(
        f"[{label}] nv=({nx},{ny},{nz}) "
        f"origin=({ox:.6f},{oy:.6f},{oz:.6f}) Å "
        f"dx,dy,dz=({dx:.6f},{dy:.6f},{dz:.6f}) Å "
        f"vx={vx} vy={vy} vz={vz}"
    )

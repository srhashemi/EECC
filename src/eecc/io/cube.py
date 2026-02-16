"""Gaussian cube file I/O, grid utilities, harmonization, and diagnostics."""

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from eecc.constants import A0_TO_ANG


# ============================================================
# === Cube I/O ===============================================
# ============================================================

def read_cube(path: str, units: str = "bohr") -> Dict[str, Any]:
    """Read a Gaussian .cube file and return a cube dict.

    Parameters
    ----------
    path : str
        Path to the cube file.
    units : str
        ``'bohr'`` (default) converts Bohr -> Å; ``'angstrom'`` keeps raw values.
    """
    with open(path, 'r') as f:
        lines = f.read().strip().splitlines()

    header3 = lines[2].split()
    natoms = int(header3[0])
    origin = np.array(list(map(float, header3[1:4])), dtype=float)

    lx = lines[3].split()
    nx = int(lx[0]); vx = np.array(list(map(float, lx[1:4])), dtype=float)

    ly = lines[4].split()
    ny = int(ly[0]); vy = np.array(list(map(float, ly[1:4])), dtype=float)

    lz = lines[5].split()
    nz = int(lz[0]); vz = np.array(list(map(float, lz[1:4])), dtype=float)

    atoms = []
    atom_start = 6
    atom_end = atom_start + natoms

    for line in lines[atom_start:atom_end]:
        parts = line.split()
        Z = int(float(parts[0]))
        q = float(parts[1])
        x, y, z = map(float, parts[2:5])
        atoms.append((Z, q, x, y, z))

    data_strs = " ".join(lines[atom_end:]).split()
    data = np.array(list(map(float, data_strs)), dtype=float)

    if data.size != nx * ny * nz:
        raise ValueError(f"Data size mismatch: got {data.size}, expected {nx*ny*nz}")

    rho = data.reshape((nz, ny, nx), order='C').transpose(2, 1, 0).copy()

    if units.lower().startswith('bohr'):
        origin = origin * A0_TO_ANG
        vx = vx * A0_TO_ANG
        vy = vy * A0_TO_ANG
        vz = vz * A0_TO_ANG
        atoms = [(Z, q, x*A0_TO_ANG, y*A0_TO_ANG, z*A0_TO_ANG) for (Z, q, x, y, z) in atoms]
        rho = rho / (A0_TO_ANG**3)
    elif units.lower().startswith('ang'):
        pass
    else:
        raise ValueError("Unsupported units. Use 'bohr' or 'angstrom'.")

    return {
        'origin': origin,
        'nv': (nx, ny, nz),
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'atoms': atoms,
        'rho': rho
    }


# ============================================================
# === Grid Utilities =========================================
# ============================================================

def grid_spacing(cube: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return ``(dx, dy, dz)`` grid spacing in Å from a cube dict."""
    dx = float(np.linalg.norm(cube['vx']))
    dy = float(np.linalg.norm(cube['vy']))
    dz = float(np.linalg.norm(cube['vz']))
    return dx, dy, dz


def voxel_volume(cube: Dict[str, Any]) -> float:
    """Return the volume of a single voxel in Å³."""
    dx, dy, dz = grid_spacing(cube)
    return dx * dy * dz


def build_axes_coordinates(
    cube: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 1-D coordinate arrays ``(xs, ys, zs)`` for the cube grid."""
    nx, ny, nz = cube['nv']
    ox, oy, oz = cube['origin']
    vx, vy, vz = cube['vx'], cube['vy'], cube['vz']

    xs = ox + np.arange(nx) * vx[0]
    ys = oy + np.arange(ny) * vy[1]
    zs = oz + np.arange(nz) * vz[2]
    return xs, ys, zs


def cube_atom_centroid(cube: Dict[str, Any]) -> np.ndarray:
    """Return the geometric centroid of the atoms in a cube dict."""
    coords = np.array([[a[2], a[3], a[4]] for a in cube['atoms']], float)
    return coords.mean(axis=0)


# ============================================================
# === Diagnostics ============================================
# ============================================================

def print_cube_grid_info(label: str, cube: Dict[str, Any]) -> None:
    """Print grid dimensions, origin, and spacing for a cube."""
    nx, ny, nz = cube['nv']
    ox, oy, oz = cube['origin']
    vx, vy, vz = cube['vx'], cube['vy'], cube['vz']
    dx, dy, dz = grid_spacing(cube)

    print(
        f"[{label}] nv=({nx},{ny},{nz}) "
        f"origin=({ox:.6f},{oy:.6f},{oz:.6f}) Å "
        f"dx,dy,dz=({dx:.6f},{dy:.6f},{dz:.6f}) Å "
        f"vx={vx} vy={vy} vz={vz}"
    )


def print_cube_mismatch_details(
    cubeA: Dict[str, Any], cubeB: Dict[str, Any]
) -> None:
    """Print a side-by-side comparison of two cube grids."""
    print_cube_grid_info("cubeA", cubeA)
    print_cube_grid_info("cubeB", cubeB)
    print("--- diffs ---")
    print("Δorigin =", cubeB['origin'] - cubeA['origin'])
    print("Δvx     =", cubeB['vx'] - cubeA['vx'])
    print("Δvy     =", cubeB['vy'] - cubeA['vy'])
    print("Δvz     =", cubeB['vz'] - cubeA['vz'])
    print("nv A,B  =", cubeA['nv'], cubeB['nv'])


def check_density_tail(cube: Dict[str, Any]) -> None:
    """Print the sum of |rho| in the 3-voxel boundary shell."""
    rho = cube['rho']
    abs_rho = np.abs(rho)

    tail = (
        abs_rho[:3,:,:].sum() +
        abs_rho[-3:,:,:].sum() +
        abs_rho[:,:3,:].sum() +
        abs_rho[:,-3:,:].sum() +
        abs_rho[:,:,:3].sum() +
        abs_rho[:,:,-3:].sum()
    )

    print(f"Tail density sum = {tail:.3e}")


def check_density_tail_fraction(cube: Dict[str, Any]) -> None:
    """Print the tail density sum, total |rho|, and their ratio."""
    rho = cube['rho']
    abs_rho = np.abs(rho)

    tail = (
        abs_rho[:3,:,:].sum() +
        abs_rho[-3:,:,:].sum() +
        abs_rho[:,:3,:].sum() +
        abs_rho[:,-3:,:].sum() +
        abs_rho[:,:,:3].sum() +
        abs_rho[:,:,-3:].sum()
    )

    total = abs_rho.sum()
    print(f"Tail sum      = {tail:.3e}")
    print(f"Total |rho|   = {total:.3e}")
    print(f"Tail fraction = {tail/total:.3e}")


def print_cube_padding(cube: Dict[str, Any]) -> None:
    """Print the padding (Å) between atom extent and grid boundary."""
    coords = np.array([[a[2], a[3], a[4]] for a in cube['atoms']])
    mins_atoms = coords.min(axis=0)
    maxs_atoms = coords.max(axis=0)

    nx, ny, nz = cube['nv']
    ox, oy, oz = cube['origin']
    vx, vy, vz = cube['vx'], cube['vy'], cube['vz']

    xs = ox + np.arange(nx) * vx[0]
    ys = oy + np.arange(ny) * vy[1]
    zs = oz + np.arange(nz) * vz[2]

    mins_cube = np.array([xs.min(), ys.min(), zs.min()])
    maxs_cube = np.array([xs.max(), ys.max(), zs.max()])

    print("Cube padding (Å):")
    print("  lower:", mins_atoms - mins_cube)
    print("  upper:", maxs_cube - maxs_atoms)


# ============================================================
# === Grid Harmonization Helpers =============================
# ============================================================

def _snap_axes_if_close(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """Snap cubeB axis vectors to cubeA if they are close enough."""
    ok = True
    for key in ['vx', 'vy', 'vz']:
        a = np.array(cubeA[key], float)
        b = np.array(cubeB[key], float)
        if np.allclose(a, b, rtol=rtol, atol=atol):
            cubeB[key] = a.copy()
        else:
            ok = False
    return ok


def _fourier_shift_density(
    rho: np.ndarray,
    dx: float, dy: float, dz: float,
    delta: np.ndarray,
) -> np.ndarray:
    """Shift density by *delta* (Å) using Fourier phase multiplication."""
    nx, ny, nz = rho.shape

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)

    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')

    phase = np.exp(-1j * (KX * delta[0] + KY * delta[1] + KZ * delta[2]))

    rho_k = np.fft.fftn(rho)
    rho_shifted = np.fft.ifftn(rho_k * phase).real

    return rho_shifted


def harmonize_cubes_inplace(
    cubeA: Dict[str, Any],
    cubeB: Dict[str, Any],
    rtol_axes: float = 1e-5,
    atol_axes: float = 1e-8,
    atol_origin_snap: float = 1e-12,
) -> None:
    """Snap cubeB grid to cubeA and Fourier-shift density if origins differ."""
    if cubeA['nv'] != cubeB['nv']:
        raise ValueError(f"Grid size mismatch: A {cubeA['nv']} vs B {cubeB['nv']}")

    axes_ok = _snap_axes_if_close(cubeA, cubeB, rtol=rtol_axes, atol=atol_axes)
    if not axes_ok:
        raise ValueError("Axis vectors differ beyond tolerance; re-export cubes with identical grid.")

    originA = np.array(cubeA['origin'], float)
    originB = np.array(cubeB['origin'], float)
    delta = originB - originA

    if np.linalg.norm(delta) > atol_origin_snap:
        dx, dy, dz = grid_spacing(cubeA)
        rhoB_shifted = _fourier_shift_density(cubeB['rho'], dx, dy, dz, delta=delta)
        cubeB['rho'] = rhoB_shifted
        cubeB['origin'] = originA.copy()
    else:
        cubeB['origin'] = originA.copy()


# ============================================================
# === Cube origin shifter (text-level) =======================
# ============================================================

def shift_cube_origin_only(
    in_path: str, out_path: str, shift_Ang: np.ndarray
) -> None:
    """Create a shifted cube by modifying ONLY the origin line in the text."""
    with open(in_path, "r") as f:
        lines = f.readlines()

    header2 = lines[2].split()
    natoms = header2[0]
    ox, oy, oz = map(float, header2[1:4])

    shift = np.array(shift_Ang, float)
    ox_new = ox + shift[0]
    oy_new = oy + shift[1]
    oz_new = oz + shift[2]

    lines[2] = f"{int(natoms):5d} {ox_new:12.6f} {oy_new:12.6f} {oz_new:12.6f}\n"

    with open(out_path, "w") as f:
        f.writelines(lines)

    print(f"Shifted cube written to {out_path}")

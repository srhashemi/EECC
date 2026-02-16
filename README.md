# EECC — Exciton-Exciton Coupling Calculator

A modular Python package for computing intermolecular and intramolecular excitonic couplings using:

- **TDC** (Transition Density Cube) via FFT Poisson solver
- **TDC direct** — brute-force Coulomb double summation for cross-checking
- **TrESP** (Transition ElectroStatic Potential) charge fitting
- **Point-dipole** and **extended-dipole** approximations
- **Coulomb** coupling from Mulliken or ESP-derived transition charges

## Installation

Requires Python 3.9+ with NumPy and SciPy.

```bash
pip install -e .
```

For visualization tools (matplotlib):

```bash
pip install -e ".[viz]"
```

For development (pytest):

```bash
pip install -e ".[dev]"
```

## CLI Usage

After installation the `eecc` command is available with these subcommands:

### TDC coupling from two monomer cube files

```bash
eecc tdc-two-monomers monomerA.cub monomerB.cub [--pad 3] [--threshold 0.0005] [--dielectric 1.0]
```

Runs four coupling methods and prints a comparison table:
- **Point-dipole** and **extended-dipole** approximations
- **TDC (FFT)** — Coulomb coupling via FFT Poisson solver
- **TDC (direct)** — brute-force double summation (cross-check)

Results with timing are printed and saved to `outputs/TDC_two_monomer_results.txt`.

### TDC coupling from a single dimer cube

```bash
eecc tdc-one-dimer dimer.cub --fragA "1-24" --fragB "25-48"
```

Splits a dimer transition-density cube into two fragment cubes by assigning
each grid point to the nearest atom in each fragment, then computes TDC coupling.

### Intramolecular coupling from a charge file

```bash
eecc intramolecular tresp.txt --frags "1-56,113-120" "57-112,121-128" --scale 1.4
```

Computes pairwise Coulomb (TrESP), point-dipole, and extended-dipole couplings
between fragments defined by atom index ranges. Prints a comparison table and
saves results to `outputs/intramolecular_tresp/results.txt`. Note: Following the Multiwfn manual, a scale of 1.4 is essential for TrESP charges.

```bash
eecc intramolecular trmulliken.txt --frags "1-56,113-120" "57-112,121-128" 
```

Computes pairwise Coulomb (TrMulliken), point-dipole, and extended-dipole couplings
between fragments defined by atom index ranges. Prints a comparison table and
saves results to `outputs/intramolecular_trmulliken/results.txt`. 
Fragment indices are 1-based and support ranges, commas, and mixed notation:

```bash
# Dimer — each fragment combines two discontinuous ranges
eecc intramolecular trmulliken.txt --frags "1-56,113-120" "57-112,121-128"

# Trimer — three fragments, three coupling pairs
eecc intramolecular tresp.txt --frags "1-40,121-128" "41-90" "91-120"

# With dielectric screening and charge scaling
eecc intramolecular tresp.txt --frags "1-56,113-120" "57-112,121-128" --dielectric 2.0 --scale 0.72
```

### Intermolecular coupling from a monomer file

```bash
# Dimer — place two copies of a monomer at different positions/orientations
# Each --placement string: "posX,posY,posZ axisX,axisY,axisZ angleDeg"
eecc intermolecular monomer.txt \
  --placement "0,0,0 0,0,1 0" "7,0,0 0,0,1 180"

# Trimer — three monomers, three coupling pairs
eecc intermolecular monomer.txt \
  --placement "0,0,0 0,0,1 0" "7,0,0 0,0,1 180" "14,0,0 0,0,1 0"

# With dielectric screening
eecc intermolecular monomer.txt \
  --placement "0,0,0 0,0,1 0" "7,0,0 0,0,1 180" --dielectric 2.0
```

Computes pairwise Coulomb, point-dipole, and extended-dipole couplings between
positioned/rotated copies of a monomer. Prints a comparison table and saves
results to `outputs/intermolecular_monomer/results.txt`.

### TrESP charge fitting from a transition-density cube

```bash
eecc tresp --cube density.cub [--out charges.txt] [--shells "1.6,2.0,2.4"] [--pps 6000] [--alpha 0.001]
```

Fits atomic transition charges by matching the electrostatic potential on
CHELPG-style sampling shells around the molecule.

### Check total transition charge of a cube file

```bash
eecc check-charge density.cub
```

### Visualize molecules from cube files

```bash
eecc view cubeA.cub [--cubeB cubeB.cub] [--center] [--align] [--save image.png]
```

### Plot density slices

```bash
eecc plot-density cubeA.cub cubeB.cub [--plane z] [--index 50]
```

## Package Structure

```
src/eecc/
├── __init__.py           # Version and top-level exports
├── constants.py          # Physical constants (Bohr, Hartree, eV, etc.)
├── cli.py                # Unified argparse CLI
│
├── io/                   # File I/O
│   ├── cube.py           # Gaussian cube read/write, grid utilities
│   ├── charges.py        # Transition charge file readers and writers
│   └── xyz.py            # XYZ coordinate file writers
│
├── coupling/             # Coupling calculations
│   ├── coulomb.py        # Coulomb coupling from point charges
│   ├── dipole.py         # Point-dipole and extended-dipole models
│   ├── tdc_fft.py        # TDC coupling via FFT Poisson solver
│   ├── tdc_bruteforce.py # TDC coupling via direct double summation
│   └── tdc_kdtree.py     # TDC coupling via KD-tree
│
├── tresp/                # TrESP charge fitting pipeline
│   ├── esp.py            # FFT Coulomb potential, CHELPG sampling, vdW radii
│   ├── multipoles.py     # Transition multipole moments from cube data
│   └── fitting.py        # Constrained ridge regression for charge fitting
│
├── geometry/             # Molecular geometry operations
│   ├── rotation.py       # Axis rotations (Rodrigues' formula)
│   ├── transform.py      # Center of mass, centering, alignment
│   └── fragments.py      # Index parsing, cube splitting by fragment
│
├── viz/                  # Visualization (requires matplotlib)
│   ├── cube_plot.py      # Grid overlays and density slice plots
│   ├── mol_plot.py       # 3D molecule rendering with bond inference
│   └── viewer.py         # Interactive molecule viewer
│
└── workflows/            # High-level calculation workflows
    ├── intramolecular.py # Intramolecular coupling workflow
    ├── intermolecular.py # Intermolecular coupling workflow
    ├── tdc_two_monomers.py  # TDC from two separate monomer cubes
    └── tdc_one_dimer.py     # TDC from a single dimer cube
```

## Python API

### TDC coupling from two monomer cubes

```python
from eecc.io.cube import read_cube
from eecc.coupling.tdc_fft import tdc_coupling_fft

cubeA = read_cube("monomerA.cub", units="bohr")
cubeB = read_cube("monomerB.cub", units="bohr")

result = tdc_coupling_fft(cubeA, cubeB, dielectric=1.0, pad_factor=3)

print(f"J = {result['J_cm1']:.2f} cm^-1  ({result['J_eV']:.6f} eV)")
print(f"Point-dipole:     {result['Jpd_cm1']:.2f} cm^-1")
print(f"Extended-dipole:  {result['Jext_cm1']:.2f} cm^-1")
```

### TrESP charge fitting

```python
from eecc.tresp.fitting import fit_tresp_from_cube

atoms_q = fit_tresp_from_cube(
    "density.cub",
    shells_ang=(1.6, 2.0, 2.4),
    pps=6000,
    alpha=1e-3,
)
# atoms_q is a list of (element, x, y, z, charge)
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Dependencies

| Package    | Version  | Required for     |
|------------|----------|------------------|
| numpy      | >= 1.21  | Core             |
| scipy      | >= 1.7   | Core             |
| matplotlib | >= 3.5   | Visualization    |
| pytest     | >= 7.0   | Testing          |

## Author

S. Rasoul Hashemi

## License

This project is licensed under the [MIT License](LICENSE).

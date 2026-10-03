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

For the automated TD-DFT pipeline (`eecc run`; PySCF, pyscf-dispersion, geomeTRIC, PyYAML):

```bash
pip install -e ".[qm]"
```

For development (pytest):

```bash
pip install -e ".[dev]"
```

## CLI Usage

After installation the `eecc` command is available with these subcommands.
Workflow commands look up input files inside `inputs/` and write results to
`outputs/`, relative to the current directory, so run them from the repository root.

### TDC coupling from two monomer cube files

```bash
eecc tdc-two-monomers monomerA.cub monomerB.cub [--pad 3] [--threshold 0.0005] [--dielectric 1.0]
```

Runs four coupling methods and prints a comparison table:
- **Point-dipole** and **extended-dipole** approximations
- **TDC (FFT)** — Coulomb coupling via FFT Poisson solver
- **TDC (direct)** — brute-force double summation (cross-check)

Results with timing are printed and saved to `outputs/TDC_two_monomer_results.txt`.

Example with the bundled BODIPY dimer fragment cubes (see [Example data](#example-data)):

```bash
eecc tdc-two-monomers BOPIDY_DIM-s-TD-WB97XD-631Gd-f1.cub BOPIDY_DIM-s-TD-WB97XD-631Gd-f2.cub --pad 3 --threshold 0.0005
```

This gives |J| ≈ 990 cm⁻¹ (TDC, FFT). The sign is arbitrary because the phase
of each fragment's transition density is arbitrary.

### TDC coupling from a single dimer cube

```bash
eecc tdc-one-dimer dimer.cub --fragA "1-24" --fragB "25-48"
```

Splits a dimer transition-density cube into two fragment cubes by assigning
each grid point to the nearest atom in each fragment, then computes TDC coupling.

### Intramolecular coupling from a charge file

```bash
eecc intramolecular Combinedfragmentschargef1-1-65-f2-66-130.txt --frags "1-65" "66-130" --scale 1.4
```

Computes pairwise Coulomb (TrESP), point-dipole, and extended-dipole couplings
between fragments defined by atom index ranges. Prints a comparison table and
saves results to `outputs/intramolecular_<file>_scaled_<scale>/results.txt`.
`--scale` divides every charge by the given factor. A scale of 1.4 (≈√2) is used
for Multiwfn TrESP charges, whose magnitude Multiwfn overestimates by √2.

```bash
eecc intramolecular MullikenCombinedfragmentschargef1-1-65-f2-66-130.txt --frags "1-65" "66-130"
```

Computes pairwise Coulomb (TrMulliken), point-dipole, and extended-dipole couplings
between fragments defined by atom index ranges. Prints a comparison table and
saves results to `outputs/intramolecular_<file>/results.txt`.
Fragment indices are 1-based and support ranges, commas, and mixed notation
(`charges.txt` below stands for any charge file in `inputs/`):

```bash
# Dimer — each fragment combines two discontinuous ranges
eecc intramolecular charges.txt --frags "1-56,113-120" "57-112,121-128"

# Trimer — three fragments, three coupling pairs
eecc intramolecular charges.txt --frags "1-40,121-128" "41-90" "91-120"

# With dielectric screening and charge scaling
eecc intramolecular charges.txt --frags "1-56,113-120" "57-112,121-128" --dielectric 2.0 --scale 1.4
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

## Automated pipeline: from geometry to couplings

`eecc run` automates the whole fragment workflow with [PySCF](https://pyscf.org),
without Gaussian or Multiwfn:

1. **opt**: optimize the oligomer ground state (default B3LYP-D3(BJ)/def2-SVP; optional frequencies).
2. **fragments**: split it into chromophores. Bonds between fragments are cut and
   capped with H along the cut bond. Separate molecules (`mode: auto`) need no capping.
3. **td**: TD-DFT on each fragment (default TD-ωB97X-D/6-31G(d) with Cartesian d
   functions, as in Gaussian) and its S0→Sn transition density matrix.
4. **transition**: transition-density cube, transition Mulliken charges and TrESP
   charges fitted to the exact electrostatic potential of the transition density.
5. **couplings**: TDC (FFT and/or direct), TrESP, TrMulliken, point-dipole and
   extended-dipole couplings for every fragment pair.

```bash
eecc run config.yaml                      # run all stages locally
eecc run config.yaml --stage td --fragment 2
eecc run config.yaml --slurm              # prep job -> one array task per fragment -> analysis
eecc run config.yaml --dry-run            # write the Slurm scripts only
```

Results go to `<workdir>/05_couplings/couplings.txt` (and `.json`). Each stage
records a hash of the settings it depends on, so a rerun skips finished stages
and resumes an interrupted optimization. Changing, for example, the TD settings
reruns only the TD stage and the stages after it, and rerunning a stage with
`--force` also reruns the stages that use its results. Only settings that differ
from their defaults are hashed, so upgrading EECC keeps finished stages; when a
default that affects results changes, stages computed under the old default are
rerun. After an upgrade that should not change results, `eecc run config.yaml
--restamp` marks existing results as current.

A fully commented configuration for the BODIPY dimer is in
[`examples/qm/bodipy_dimer/config.yaml`](examples/qm/bodipy_dimer/config.yaml).
A minimal configuration for two separate molecules:

```yaml
geometry: pair.xyz
fragments:
  mode: auto
opt:
  enabled: false
```

Notes:

- The transition charges are fitted to the exact transition density, so no
  `--scale` correction is needed for them.
- PySCF does not implement the dispersion term of ωB97X-D. It is a
  geometry-only energy shift, so excitation energies and transition densities are
  unaffected. For that reason ωB97X-D is refused for optimizations.
- `opt.convergence` picks a geomeTRIC criteria set (default `gau`, Gaussian's
  force and displacement thresholds plus a 10⁻⁶ Eh energy change), and
  `opt.thresholds` overrides single criteria. Loosening them saves little on
  flexible oligomers: there the step size, not the force, keeps the optimization
  going, and stopping early leaves soft modes (such as substituent twists) unrelaxed.
- The default optimization grid `[99, 590]` matches Gaussian's UltraFine. On the
  BODIPY dimer (128 atoms), `opt.grid: [75, 302]` makes each step about 35% faster
  but changes the gradient by up to 2×10⁻⁴ Eh/Bohr, enough to shift the minimum
  along soft modes. Use it only for rigid molecules.
- TDC (FFT) solves the isolated Coulomb problem on a grid that covers both
  fragments (`couplings.tdc_boundary: free`) and agrees with the direct sum at any
  separation. `tdc_boundary: periodic` reproduces the published workflow, whose
  periodic images make distant pairs inaccurate (BODIPY tetramer, 24.7 Å apart:
  −14 cm⁻¹ instead of −29 cm⁻¹).
- TDDFT speed: the TDDFT step, not the SCF, dominates the cost. The defaults evaluate
  the XC kernel of the response on a `[75, 302]` grid (`td.response_grid`, SCF on
  `td.grid`), stop the Davidson solver at `td.davidson_tol: 1e-4`, and run Slurm jobs
  on physical cores (`slurm.physical_cores`). For a BODIPY fragment (624 basis
  functions) this cuts the TDDFT from about 5.4 h to 1.5 h on 32 cores with identical
  S1 energy, oscillator strength and transition density. TDA is faster but moves the
  bright BODIPY state above a dark one, and density fitting does not speed up the
  TDDFT step.
- Only closed-shell systems are supported. Signs of the couplings are arbitrary
  because the phase of each fragment's transition density is arbitrary.

### Whole-system TDDFT and the exciton Hamiltonian

The fragment couplings above are Coulomb couplings. With `system.enabled: true`
the pipeline also computes the whole system with the same TDDFT settings (stage
`system`, one node) and diabatizes its states onto the fragment states (stage
`diabatize`): the transition densities of the fragments' locally excited (LE)
states and of HOMO→LUMO charge-transfer (CT) states between fragments are
projected into the orbitals of the whole system and fitted as combinations of
its states. The result, `<workdir>/07_diabatize/diabatic.txt`, is the exciton
Hamiltonian: site and CT energies, and *total* couplings, which include
exchange, overlap, polarization and CT mixing, listed next to the Coulomb
couplings. A completeness below about 0.8 means `system.nstates` (default: four
per fragment) is too small to describe that state.

```yaml
system:
  enabled: true
  include_ct: true
slurm:
  system_partition: main     # whole node; use a longer partition for trimers and larger
  time_system: "24:00:00"
```

On Dardel (128 cores) the BODIPY dimer (1244 basis functions) took 11 h and the
trimer (1864) an estimated day.

| System | Whole-system result | Total coupling | Coulomb (TDC direct) |
|---|---|---|---|
| BODIPY dimer | S1 2.8118 eV (bright), S2 3.0736 eV; Davydov J = −1056 cm⁻¹ (published Gaussian: −1057) | 1050 cm⁻¹ | 906 cm⁻¹ |
| PDI dimer, 3.5 Å cofacial | S1 2.065 eV: 63 % LE + 37 % CT; CT 0.22 eV above LE; LE–CT coupling about 2750 cm⁻¹ | 1275 cm⁻¹ | 1160 cm⁻¹ |

For the covalent BODIPY dimer the total coupling is 10–15 % larger than the
Coulomb coupling and reproduces the Davydov splitting. In the π-stacked PDI
dimer the direct LE–LE coupling is close to the Coulomb value, but CT mixing
dominates the low-energy states, which no Coulomb coupling can describe.

### Validation against the published BODIPY oligomers

Run with the defaults on the fragment geometries of the published dataset
(doi:10.5878/pyja-rd94; Gaussian 16 + Multiwfn), the pipeline reproduces the
published couplings (cm⁻¹; published / `eecc run`, with the arbitrary signs of
`eecc run` aligned to the published ones):

| System | Pair | R (Å) | TDC (direct) | TrESP | TrMulliken |
|---|---|---|---|---|---|
| Dimer | 1–2 | 8.58 | 967 / 952 | 1141 / 1177 | 816 / 836 |
| Trimer¹ | 1–2 | 8.65 | 954 / 937 | 1117 / 1152 | 803 / 820 |
| | 1–3 | 16.78 | 104 / 102 | 106 / 104 | 95 / 97 |
| | 2–3 | 9.68 | −937 / −923 | −1111 / −1141 | −784 / −800 |
| Tetramer | 1–2 | 9.68 | −937 / −924 | −1112 / −1142 | −784 / −799 |
| | 1–3 | 16.81 | −103 / −102 | −106 / −104 | −94 / −95 |
| | 1–4 | 24.71 | 30 / 29 | 30 / 29 | 28 / 28 |
| | 2–3 | 8.66 | −935 / −932 | −1092 / −1127 | −791 / −803 |
| | 2–4 | 17.11 | 102 / 101 | 105 / 103 | 94 / 96 |
| | 3–4 | 8.66 | 953 / 944 | 1119 / 1151 | 802 / 819 |

¹ The published trimer used 6-311G(d,p); `eecc run` used the default 6-31G(d).

TDC agrees within 2 %, TrESP within 3.5 % and TrMulliken within 2.5 %. Compare
against the published *direct* TDC values: the published FFT values used periodic
boundaries and depend on the cube grid (dimer: 990 FFT vs 967 direct; tetramer
pair 1–4: 12 FFT vs 30 direct).

Two further checks on the dimer:

- **Automatic cutting and capping** (`mode: ranges`) gives couplings about 5 %
  smaller (TDC 906 cm⁻¹). The dataset fragments carry a pyramidal cap
  hydrogen (H–C–C 110°, 55° out of the ring plane) on an aromatic carbon;
  `eecc run` places it in the ring plane along the cut bond.
- **Re-optimizing the dimer** (B3LYP-D3(BJ)/def2-SVP in PySCF, 26 steps) changes
  the couplings by about 1 % (TDC 899 cm⁻¹).

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
├── workflows/            # High-level calculation workflows
│   ├── intramolecular.py # Intramolecular coupling workflow
│   ├── intermolecular.py # Intermolecular coupling workflow
│   ├── tdc_two_monomers.py  # TDC from two separate monomer cubes
│   └── tdc_one_dimer.py     # TDC from a single dimer cube
│
└── qm/                   # Automated TD-DFT pipeline (requires eecc[qm])
    ├── config.py         # YAML configuration and validation
    ├── structure.py      # XYZ I/O, connectivity, fragments, H capping
    ├── pyscf_setup.py    # PySCF molecule and Kohn-Sham construction
    ├── ground.py         # Geometry optimization and frequencies
    ├── excited.py        # TDDFT and transition density matrices
    ├── transition.py     # Cube files, transition Mulliken and TrESP charges
    ├── couplings.py      # Pairwise couplings via the EECC methods
    ├── system.py         # Whole-system TDDFT and diabatization stages
    ├── diabatize.py      # Projection diabatization onto fragment LE/CT states
    ├── pipeline.py       # Restartable stage runner
    └── slurm.py          # Slurm job submission
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

## Example data

The files in `inputs/` (except `monomer.txt`) are the BODIPY dimer "fragment
method" data from the dataset accompanying the article cited below:

Ringström, R.; Hashemi, S. R.; Liang, Y.; Hestand, N. J.; Börjesson, K.
*Supporting data for "Strong exciton coupling: a practical toolbox for computing
interaction energies, wavefunctions, and optical spectra".* University of
Gothenburg, 2026. https://doi.org/10.5878/pyja-rd94 (CC BY 4.0)

Each dimer fragment (monomer capped with H at the dimer geometry) was computed
separately with TD-ωB97X-D/6-31G(d) in Gaussian. Multiwfn was then used to
generate the transition-density cube and the transition charges.

| File | Content |
|------|---------|
| `BOPIDY_DIM-s-TD-WB97XD-631Gd-f1.cub`, `-f2.cub` | S0→S1 transition density of fragments 1 and 2 |
| `Combinedfragmentschargef1-1-65-f2-66-130.txt` | TrESP charges, fragment 1 = atoms 1–65, fragment 2 = atoms 66–130 |
| `MullikenCombinedfragmentschargef1-1-65-f2-66-130.txt` | Transition Mulliken charges, same atom order |

Expected couplings (cm⁻¹; sign arbitrary):

| Method | J |
|--------|---|
| TDC (FFT / direct) | 990 / 967 |
| TrESP Coulomb (`--scale 1.4`) | 1141 |
| TrMulliken Coulomb | 816 |

For comparison, the Davydov splitting of the full dimer at the same level
(S1 2.812 eV, bright; S2 3.074 eV, dark) gives J ≈ −1057 cm⁻¹.

`monomer.txt` is a single-monomer charge file used for the `intermolecular` example.

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
| pyscf, pyscf-dispersion, geomeTRIC, PyYAML | see `pyproject.toml` | `eecc run` |

## Citation

If you use this software in your research, please cite:

Ringström, R.; Hashemi, S. R.; Liang, Y.; Hestand, N. J.; Börjesson, K.
*Strong exciton coupling: a practical toolbox for computing interaction energies, wavefunctions, and optical spectra.*
**Chem. Soc. Rev.** 2026, **55** (12), 6462–6499.
https://doi.org/10.1039/d6cs00157b

```bibtex
@article{ringstrom2026strong,
  title={Strong exciton coupling: a practical toolbox for computing interaction energies, wavefunctions, and optical spectra},
  author={Ringstr{\"o}m, Rasmus and Hashemi, S Rasoul and Liang, Yuanxin and Hestand, Nicholas J and B{\"o}rjesson, Karl},
  journal={Chemical Society Reviews},
  volume={55},
  number={12},
  pages={6462--6499},
  year={2026},
  publisher={The Royal Society of Chemistry}
}
```

## Author

S. Rasoul Hashemi

## License

This project is licensed under the [MIT License](LICENSE).

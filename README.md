# EECC — Exciton-Exciton Coupling Calculator

EECC computes excitonic couplings between chromophores, within one molecule
(covalent oligomers) or between molecules, in two ways:

- **From a geometry, fully automated** (`eecc run`, [PySCF](https://pyscf.org)):
  optimization, fragments, TD-DFT, transition densities and couplings, locally
  or as Slurm jobs, without Gaussian or Multiwfn. Optionally the whole system is
  computed as well and diabatized into an exciton Hamiltonian with site energies,
  *total* couplings and charge-transfer states.
- **From existing transition densities or charges** (for example Gaussian cube
  files and Multiwfn charges), with the commands under
  [Working from cube or charge files](#working-from-cube-or-charge-files).

Coupling methods:

- **TDC** (Transition Density Cube) via an FFT Poisson solver, and **TDC direct**,
  a brute-force Coulomb double sum for cross-checking
- **TrESP** (Transition ElectroStatic Potential) and **TrMulliken** transition charges
- **Point-dipole** and **extended-dipole** approximations

## In short

```bash
pip install -e ".[qm]"
eecc init pair.xyz           # writes config.yaml: every option, its default and a comment
eecc run config.yaml         # runs here; add --slurm on a cluster
```

The couplings of every fragment pair are in `work/05_couplings/couplings.txt`.
Separate molecules are split automatically; for one covalent molecule, give
the atoms of each chromophore with `--ranges` (see [Getting started](#getting-started)).

Contents: [Installation](#installation) ·
[Getting started](#getting-started) ·
[Automated pipeline](#automated-pipeline-from-geometry-to-couplings) ·
[Validation](#validation) ·
[Working from cube or charge files](#working-from-cube-or-charge-files) ·
[Python API](#python-api) ·
[Configuration reference](#configuration-reference) ·
[Citation](#citation)

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

| Package    | Version  | Required for     |
|------------|----------|------------------|
| numpy      | >= 1.21  | Core             |
| scipy      | >= 1.7   | Core             |
| matplotlib | >= 3.5   | Visualization    |
| pytest     | >= 7.0   | Testing          |
| pyscf, pyscf-dispersion, geomeTRIC, PyYAML | see `pyproject.toml` | `eecc run` |

## Getting started

The BODIPY dimer of the validation, from its geometry to the couplings:

```bash
cd examples/qm/bodipy_dimer
eecc init dimer.xyz --ranges "57-112,121-128" "1-56,113-120" --no-opt \
          --account <allocation> -o my_config.yaml
eecc run my_config.yaml --slurm    # on a cluster; without --slurm it runs here (a few hours)
cat work/05_couplings/couplings.txt
```

For Slurm, put the lines that load Python and EECC in the jobs (modules, virtual
environment) under `slurm.setup` in the file.

`eecc init` writes a `config.yaml` that lists every option with its default and
a comment, so the whole configuration is in one file to edit. The most common
choices can be set directly:

```bash
eecc init pair.xyz                        # two separate molecules: fragments found automatically
eecc init dimer.xyz --ranges ... --functional cam-b3lyp --basis def2-svp --nstates 6 \
          --no-opt --account my-project-123
eecc init --help                          # all choices (fragment files, TDA, state, whole system, ...)
```

For a single covalent molecule without `--ranges`, the file marks
`fragments.ranges` as required. `eecc init` and `eecc run` check the
configuration before anything runs: misspelled keys (with a suggestion), values
of the wrong type (such as an unquoted `12:00:00`, which YAML reads as seconds),
and functionals or basis sets that PySCF does not know or that lack an element
of the molecule. All options are listed under
[Configuration reference](#configuration-reference); a configuration for the
BODIPY dimer is in [`examples/qm/bodipy_dimer/config.yaml`](examples/qm/bodipy_dimer/config.yaml).

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

### Notes on methods and settings

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
couplings. It also lists the transition dipole and oscillator strength of each
diabatic state, so `diabatic.json` is a complete exciton model, for example for
spectra. Each fragment state is built as a mixture of the computed whole-system
states; its *completeness* (0–1) is the fraction of it that this mixture
reproduces. LE + CT needs n² states for n fragments (the default covers up to
four); with fewer, the stage fits the LE states only and says so in
`diabatic.txt`.

```yaml
system:
  enabled: true
  include_ct: true
slurm:
  system_partition: <name>   # a partition with whole nodes and long enough time limits
  time_system: "24:00:00"    # enough for a dimer; the tetramer took 47 h
```

`diabatic.json` is the exciton model in a fixed, versioned format (`format_version`),
meant to be read by other programs, for example for spectra:

| Key | Content |
|---|---|
| `format_version`, `units`, `conventions` | format number; units of every quantity; frame, sign and CT conventions |
| `labels`, `states` | diabatic states in Hamiltonian order; `states` gives each one's type (`LE` with its `fragment`, or `CT` with `donor` and `acceptor`) |
| `H_eV` | exciton Hamiltonian: site and CT energies on the diagonal, total couplings off it (eV) |
| `transition_dipoles_au`, `oscillator_strengths` | transition dipole (x, y, z; atomic units) and oscillator strength of each diabatic state |
| `completeness` | how well the computed whole-system states reproduce each diabatic state (0–1) |
| `fragments` | per fragment: `name`, `natoms_own`, `center_ang` (centre of mass, Å), `principal_axes` (unit vectors, by increasing moment of inertia; for a planar molecule the last is the plane normal) and `moments_amu_ang2`, from the fragment's own atoms without caps; two equal moments (a symmetric top) leave the two axes free to rotate in their plane |
| `adiabatic_eV`, `pairs` | whole-system excitation energies; per fragment pair, the total and the Coulomb couplings (cm⁻¹) |
| `fragments[].vibronic` | with `vibronic.enabled`: the fragment's Holstein parameters (`S_eff`, `omega_eff_cm`, `sigma_low_cm`, reorganization energies; see [Vibronic parameters](#vibronic-parameters)) |

All positions and vectors are in the frame of `system.xyz`. The phase of each
diabatic state (LE or CT) is arbitrary, so signs of couplings and dipoles are
arbitrary per state but consistent with each other. A `diabatic.json` in an older
format is rewritten the next time the pipeline runs (only the diabatize stage reruns;
the whole-system TDDFT is reused).

On one 128-core node (see [Hardware](#hardware)) the BODIPY dimer (1244 basis
functions) took 11 h, the trimer (1864) 20 h and the tetramer (2484) 47 h.

The results for the BODIPY dimer, trimer and tetramer and a PDI dimer are under
[Whole-system results](#whole-system-results).

**How many states?** Below a completeness of 0.8, `diabatic.txt` warns: increase
`system.nstates` (default: four per fragment). Above 0.8 the result can still be
off, and completeness can stay near 0.9 even when it is fine, because a capped
fragment is not exactly part of the whole system. To check a result you rely on,
run again with about twice as many states in a separate `workdir` and compare
the couplings in the two `diabatic.txt` files. If they agree, the first run had
enough states.

### Vibronic parameters

With `vibronic.enabled: true` (or `eecc init --vibronic`) the optional **vibronic**
stage gives each fragment the Huang–Rhys factors and frequencies of its `td.state`,
the vibronic input of Frenkel–Holstein spectrum models. It uses the displaced
harmonic oscillator model with the vertical-gradient method: at the fragment
geometry it computes the ground-state Hessian (normal modes, frequencies ω_k) and
the gradient of the excitation energy, with the `td` functional, basis and method.
Projected onto mass-weighted mode k (component g_k), the gradient gives the
dimensionless displacement and Huang–Rhys factor (atomic units)

d_k = −g_k / ω_k^(3/2),  S_k = d_k² / 2,  reorganization energy λ = Σ S_k ω_k.

The gradient of the excitation energy (excited minus ground state) is used rather
than that of the excited state alone, so the fragment, cut from the optimized
aggregate and not exactly at its own minimum, gets no spurious ground-state force,
and the parameters belong to the geometry of the couplings. Modes below
`vibronic.min_frequency` (and imaginary ones) get no S, as S_k grows as 1/ω³.

A one-mode Holstein model needs one vibration, so the modes are also summarized:

- modes at or above `vibronic.cutoff` (default 800 cm⁻¹; for π-conjugated dyes
  mostly C=C/C–N ring stretches at 1200–1600 cm⁻¹, too closely spaced to resolve)
  form one effective mode: S_eff = Σ S_k, ω_eff = Σ S_k ω_k / Σ S_k;
- modes below it broaden the bands instead: Gaussian σ² = Σ S_k ω_k² coth(ω_k / 2kT).

`<workdir>/08_vibronic/<fragment>/vibronic.json` lists every mode (ω_k, S_k, d_k)
and the summary; `vibronic.txt` lists the summary and the most displaced modes and
warns about low-frequency modes with S > 1 (large-amplitude distortions such as
torsions, where one effective mode is too crude). With `system.enabled` the summary
is also written to `diabatic.json` (`fragments[].vibronic`). Changing only the
summary settings (`cutoff`, `temperature`, `freq_scale`, `min_frequency`) reuses the
Hessian and gradient stored in `vibronic.npz`.

The Hessian dominates the cost (for a 65-atom BODIPY fragment 7.5 h of 10.9 h on a
full node, see [Validation](#vibronic-parameters-against-experiment)); with
`--slurm` the vibronic stage runs as its own array job (`slurm.time_vibronic`) in
parallel with the fragment TDDFT. `vibronic.response_grid: [75, 302]` halves the
TDDFT part for a small change in the weakest modes.

Memory: holding the two-electron integrals in memory takes about nao⁴ bytes (168 GB
at 640 basis functions), and the Hessian needs more on top. `vibronic.eri: auto`
keeps them in memory when they fit, but drops them before the Hessian unless they
take at most half of `resources.memory_mb`; the Hessian then recomputes them
(`direct`: slower, little memory; `incore`: always in memory; with
`td.density_fit` the option does not apply). The converged SCF and then the gradients
are recorded in `vibronic_partial.npz`, so a rerun after a killed job restarts from
its own SCF and, if the gradients were done, goes straight to the Hessian (the killed
run's log is kept as `vibronic_previous.log`).

## Validation

### Hardware

All timings in this README were measured on CPU compute nodes with two 64-core
AMD EPYC 7742 processors (128 cores, 256 GB memory) under Slurm, one thread per
physical core: fragment jobs on 32 cores, whole-system jobs on a full node.

### Fragment couplings against the published BODIPY oligomers

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

### Whole-system results

Whole-system TDDFT and its diabatization (the exciton Hamiltonian), with the
total couplings next to the Coulomb couplings of the fragment pipeline:

| System | Whole-system result | Total coupling | Coulomb (TDC direct) |
|---|---|---|---|
| BODIPY dimer | S1 2.8118 eV (bright), S2 3.0736 eV; Davydov J = −1056 cm⁻¹ (published Gaussian: −1057) | 1050 cm⁻¹ | 906 cm⁻¹ |
| PDI dimer, 3.5 Å cofacial | S1 2.065 eV: 63 % LE + 37 % CT; CT 0.22 eV above LE; LE–CT coupling about 2750 cm⁻¹ | 1275 cm⁻¹ | 1160 cm⁻¹ |
| BODIPY trimer | S1 2.6682 eV (f 2.26), S2 2.9603 eV (f 0.03); site energies 2.944 / 2.839 / 2.938 eV | 1078, 1248; 192 cm⁻¹ | 903, 887; 104 cm⁻¹ |
| BODIPY tetramer | S1 2.6093 eV (f 3.22), S2 2.8432 eV (f 0.04); end sites 2.94 eV, inner sites 2.84 eV | 1081, 1122, 1247; 220, 212; 68 cm⁻¹ | 908, 907, 885; 104, 105; 30 cm⁻¹ |

For the covalent BODIPY dimer the total coupling is 10–15 % larger than the
Coulomb coupling and reproduces the Davydov splitting. In the π-stacked PDI
dimer the direct LE–LE coupling is close to the Coulomb value, but CT mixing
dominates the low-energy states, which no Coulomb coupling can describe.

For the trimer and tetramer the couplings are listed along the chain: nearest
neighbours, then second neighbours, then (tetramer) the two ends. Only LE states
were fitted, since LE + CT would need 9 and 16 states (the LE completeness of
0.86–0.92 is limited by the capped fragments). The nearest-neighbour total
couplings are 20–40 % larger than the Coulomb couplings, and the more distant
ones about twice as large, although these are small. The inner sites lie 0.1 eV
below the end sites. The 4 × 4 tetramer Hamiltonian reproduces the four lowest
whole-system states within 4 meV. With the isolated-fragment site energies,
Coulomb couplings place the trimer S1 0.2 eV too high, so the site energies
matter more than the couplings.

### Vibronic parameters against experiment

BODIPY monomer (fragment 1 of the dimer with its cap, 65 atoms; ωB97X-D/6-31G(d),
full TDDFT S1) against its absorption and emission in toluene (Schaefer et al.,
Nat. Commun. 15 (2024), source data). The band shapes are computed from the modes
(displaced oscillators at 295 K); only the 0–0 energy and one Gaussian width, which
stands in for the modes left out, are fitted. Scale 1 means the computed S fit as
they are:

| Modes taken from the calculation | Absorption: best scale on all S | Emission: best scale on all S |
|---|---|---|
| all | 0.43 | 0.60 |
| ≥ 150 cm⁻¹ | 0.76 | 1.05 |
| ≥ 400 cm⁻¹ | 0.99 | 1.35 |

The high-frequency factors reproduce the measured vibronic structure without
scaling (S_eff 0.16 at ω_eff 1321 cm⁻¹; the absorption shoulder at 2.6 eV). Two
low-frequency torsions of the meso aryl group (62 and 129 cm⁻¹, S 1.1 and 1.7,
flagged in `vibronic.txt`) are overestimated by the harmonic vertical-gradient
model: they give a width of 610 cm⁻¹ where the spectra need about 230–290 cm⁻¹. When
`vibronic.txt` warns about strong low-frequency modes, use S_eff and ω_eff with a
width from experiment, or take `sigma_low` as an upper bound.

Cost on a full node: 10.9 h (TDDFT and gradients 3.4 h with Davidson 1e-6, Hessian
7.5 h). With the integrals in memory the job peaked at 307 GB, on a 512 GB node;
on 256 GB nodes `vibronic.eri: auto` recomputes them for the Hessian. The default
Davidson tolerance 1e-4 gives the same S (to 2e-4) and a 30 % faster TDDFT;
`vibronic.response_grid: [75, 302]` halves the TDDFT and gradient time again, with
the summary values within 0.4 % and single weakly displaced modes within about 10 %.

## Working from cube or charge files

These subcommands work on transition densities and transition charges computed
elsewhere, for example with Gaussian and Multiwfn. They look up input files
inside `inputs/` and write results to `outputs/`, relative to the current
directory, so run them from the repository root. The bundled BODIPY dimer files
are described under [Example data](#example-data).

### TDC coupling from two monomer cube files

```bash
eecc tdc-two-monomers monomerA.cub monomerB.cub [--boundary free] [--threshold 0.0005] [--dielectric 1.0]
```

Runs four coupling methods and prints a comparison table:
- **Point-dipole** and **extended-dipole** approximations
- **TDC (FFT)** — Coulomb coupling via FFT Poisson solver
- **TDC (direct)** — brute-force double summation (cross-check)

The FFT solves the isolated problem (`--boundary free`, the default): the exact sum
over all voxels at any separation, in seconds (when the two cubes lie on different
grids, the potential is interpolated onto cube A's points). For cubes too large for
its grid the FFT row reads nan and the direct sum is the result. `--boundary periodic [--pad 3]`
reproduces the published workflow, whose periodic images distort the coupling,
mildly for neighbours and strongly for distant pairs (BODIPY oligomer cubes):

| Pair | R (Å) | FFT, periodic | FFT, free | Direct sum (threshold 0.0005) |
|---|---|---|---|---|
| Dimer 1–2 | 8.6 | 990.1 | 978.8 | 967.2 |
| Tetramer 1–2 | 9.7 | −948.3 | −934.8 | −924.2 |
| Tetramer 1–3 | 16.8 | −126.6 | −103.2 | −101.7 |
| Tetramer 1–4 | 24.7 | −14.1 | −29.4 | −29.1 |

The free FFT took about 1 s per pair (16 cores), the direct sum about 5 min. The direct sum
drops voxels below the threshold and so comes out slightly low: on the dimer,
thresholds of 0.0005, 0.0002, 0.0001 and 0.00005 give 967.2, 974.9, 977.4 and
978.5 cm⁻¹, approaching the free FFT value.

Results with timing are printed and saved to `outputs/TDC_two_monomer_results.txt`.

Example with the bundled BODIPY dimer fragment cubes (see [Example data](#example-data)):

```bash
eecc tdc-two-monomers BOPIDY_DIM-s-TD-WB97XD-631Gd-f1.cub BOPIDY_DIM-s-TD-WB97XD-631Gd-f2.cub --threshold 0.0005
```

This gives |J| ≈ 979 cm⁻¹ (TDC, FFT; 990 cm⁻¹ with `--boundary periodic`, the
published value). The sign is arbitrary because the phase of each fragment's
transition density is arbitrary.

### TDC coupling from a single dimer cube

```bash
eecc tdc-one-dimer dimer.cub --fragA "1-24" --fragB "25-48" [--boundary free]
```

Splits a dimer transition-density cube into two fragment cubes by assigning
each grid point to the nearest atom in each fragment, then computes TDC coupling
(FFT on the shared grid; free boundary by default, `--boundary periodic` as published).

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

### Example data

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
| TDC (FFT free / periodic / direct) | 979 / 990 / 967 |
| TrESP Coulomb (`--scale 1.4`) | 1141 |
| TrMulliken Coulomb | 816 |

For comparison, the Davydov splitting of the full dimer at the same level
(S1 2.812 eV, bright; S2 3.074 eV, dark) gives J ≈ −1057 cm⁻¹.

`monomer.txt` is a single-monomer charge file used for the `intermolecular` example.

## Python API

### TDC coupling from two monomer cubes

```python
from eecc.io.cube import read_cube
from eecc.coupling.tdc_fft import tdc_coupling_fft

cubeA = read_cube("monomerA.cub", units="bohr")
cubeB = read_cube("monomerB.cub", units="bohr")

result = tdc_coupling_fft(cubeA, cubeB, dielectric=1.0)  # free boundary; boundary="periodic", pad_factor=3 as published

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

## Configuration reference

Every option of `config.yaml`, with its default. Only `geometry` (or
`fragments.files`) and the fragment definition are required.

<!-- options:start (generated: python -m eecc.qm.options README.md) -->
| Option | Default | Description |
|---|---|---|
| `geometry` | `''` | XYZ file of the whole system (Å); relative to this file |
| `name` | `eecc_run` | run name, used in Slurm job names |
| `workdir` | `work` | output directory; relative to this file |
| `charge` | `0` | total charge |
| `spin` | `0` | 2S; only closed-shell (0) is supported |
| `fragments.mode` | `ranges` | ranges: atom indices per chromophore (cut and capped); auto: one fragment per separate molecule; files: ready-made fragment XYZ files |
| `fragments.ranges` | `[]` | mode ranges: 1-based atom indices per fragment, e.g. ["1-56,113-120", "57-112"] |
| `fragments.files` | `[]` | mode files: one XYZ file per fragment (no cutting or capping) |
| `fragments.charges` | `[]` | charge of each fragment; empty: all neutral |
| `fragments.cap.element` | `H` | element of the cap atom |
| `fragments.cap.bond_lengths` | `{B: 1.19, C: 1.09, N: 1.01, O: 0.96, P: 1.42, S: 1.34, Si: 1.48}` | cap bond length (Å) by element of the capped atom |
| `fragments.cap.default_bond_length` | `1.09` | cap bond length (Å) for other elements |
| `fragments.bond_scale` | `1.2` | atoms are bonded if closer than this × the sum of covalent radii |
| `opt.enabled` | `true` | optimize the geometry first; false if it is already optimized |
| `opt.xc` | `b3lyp` | functional |
| `opt.basis` | `def2-svp` | basis set |
| `opt.cart` | `null` | Cartesian d functions; null: as Gaussian (Cartesian for 6-31G-type bases) |
| `opt.disp` | `d3bj` | dispersion correction (d3bj, d4, ...); null: none |
| `opt.grid` | `[99, 590]` | DFT grid [radial, angular] |
| `opt.density_fit` | `true` | density fitting (faster, negligible error for geometries) |
| `opt.conv_tol` | `1.0e-09` | SCF convergence (Eh) |
| `opt.maxsteps` | `200` | maximum optimization steps |
| `opt.convergence` | `gau` | geomeTRIC criteria set: gau, gau_loose, gau_tight, gau_verytight, nwchem_loose, turbomole, interfrag_tight |
| `opt.thresholds` | `{}` | override single criteria: energy (Eh), grms/gmax (Eh/Bohr), drms/dmax (Å) |
| `opt.freq` | `false` | frequencies and imaginary-mode check after the optimization (expensive) |
| `opt.memory_mb` | `null` | PySCF memory for this stage (MB); null: resources.memory_mb |
| `td.xc` | `wb97x-d` | functional (any PySCF/libxc name, e.g. wb97x-d, cam-b3lyp, b3lyp, pbe0, m06-2x) |
| `td.basis` | `6-31g(d)` | basis set (any PySCF name, e.g. 6-31g(d), 6-311g(d,p), def2-svp, def2-tzvp) |
| `td.cart` | `null` | Cartesian d functions; null: as Gaussian (Cartesian for 6-31G-type bases) |
| `td.nstates` | `10` | excited states computed per fragment |
| `td.method` | `tddft` | tddft (full, as Gaussian TD) or tda (faster; check that the state order is right) |
| `td.state` | `1` | excited state used for the couplings (1 = S1) |
| `td.grid` | `[99, 590]` | DFT grid of the SCF [radial, angular] |
| `td.response_grid` | `[75, 302]` | grid of the TDDFT response only (about 1.6× faster, same results); null: td.grid |
| `td.density_fit` | `false` | density fitting (does not speed up the TDDFT step) |
| `td.conv_tol` | `1.0e-09` | SCF convergence (Eh) |
| `td.davidson_tol` | `0.0001` | TDDFT convergence; 1e-5 gives the same states about 1.3× slower |
| `transition.cube.spacing` | `0.25` | grid spacing (Bohr), identical for all fragments |
| `transition.cube.margin` | `6.0` | grid extent beyond the outermost atoms (Bohr) |
| `transition.tresp.shells` | `[1.4, 1.6, 1.8, 2.0]` | fit-point shells at these multiples of the vdW radius |
| `transition.tresp.density` | `5.0` | fit points per Å² on each shell |
| `transition.tresp.dipole_constraint` | `true` | fitted charges reproduce the TDDFT transition dipole |
| `transition.tresp.alpha` | `0.0` | ridge regularization of the fit |
| `couplings.methods` | `[tdc_fft, tresp, mulliken, point_dipole, extended_dipole]` | any of tdc_fft, tdc_direct (slow, exact cross-check), tresp, mulliken, point_dipole, extended_dipole |
| `couplings.dielectric` | `1.0` | relative permittivity screening the couplings |
| `couplings.cap_charges` | `merge` | charges on cap atoms: merge into the capped atom, drop, or keep |
| `couplings.tdc_pad` | `3` | FFT padding factor (periodic boundary only) |
| `couplings.tdc_boundary` | `free` | TDC FFT boundary: free (exact at any distance) or periodic (as the published workflow; wrong for distant pairs) |
| `couplings.tdc_direct_threshold` | `0.0005` | tdc_direct keeps voxels with |q| > threshold × max|q| |
| `system.enabled` | `false` | also compute the whole system and diabatize it (one full node, hours to days) |
| `system.nstates` | `0` | whole-system states; 0: four per fragment. LE + CT needs n² for n fragments |
| `system.include_ct` | `true` | add HOMO→LUMO charge-transfer states between every fragment pair |
| `system.memory_mb` | `null` | PySCF memory for the whole-system run (MB); null: resources.memory_mb |
| `vibronic.enabled` | `false` | compute vibronic parameters (ground-state Hessian + excitation-energy gradient per fragment; about as long as the td stage or longer) |
| `vibronic.cutoff` | `800.0` | cm⁻¹; modes above form the effective Holstein mode, modes below a Gaussian width |
| `vibronic.temperature` | `298.15` | K; temperature of the width from the low-frequency modes |
| `vibronic.freq_scale` | `1.0` | scales the reported frequencies (e.g. 0.95 for hybrid functionals); S is unscaled |
| `vibronic.min_frequency` | `50.0` | cm⁻¹; lower and imaginary modes get no S (S grows as 1/ω³) |
| `vibronic.davidson_tol` | `0.0001` | TDDFT convergence for the excited-state gradient (1e-4 gives the S of 1e-6) |
| `vibronic.response_grid` | `null` | grid of the excitation-energy gradient (TDDFT and both gradients; the Hessian keeps td.grid), e.g. [75, 302]: about 2× faster TDDFT, summary within 0.4 %, single weakly displaced modes up to ~10 %; null: td.grid |
| `vibronic.eri` | `auto` | two-electron integrals in memory (incore: fast, ~nao⁴ bytes, 168 GB at 640 basis functions) or recomputed (direct); auto: in memory if they fit, but for the Hessian only if they take at most half of resources.memory_mb; not used with td.density_fit |
| `resources.threads` | `0` | threads; 0: OMP_NUM_THREADS or all cores |
| `resources.memory_mb` | `16000` | PySCF memory (MB) |
| `resources.tmpdir` | `null` | PySCF scratch directory; null: default |
| `slurm.account` | `null` | allocation (sbatch -A), e.g. my-project-123 |
| `slurm.partition` | `null` | partition of the optimization, fragment and analysis jobs; null: the cluster's default |
| `slurm.physical_cores` | `true` | one thread per physical core (--hint=nomultithread; about 2× faster) |
| `slurm.cpus` | `32` | CPUs per optimization or fragment job |
| `slurm.mem` | `28G` | memory per optimization or fragment job; where cores are allocated by memory, keep ≤ cpus × memory per core |
| `slurm.analysis_cpus` | `8` | CPUs of the prep, analysis and diabatize jobs |
| `slurm.analysis_mem` | `16G` | memory of the prep, analysis and diabatize jobs |
| `slurm.opt_partition` | `null` | partition of the optimization job; null: slurm.partition |
| `slurm.opt_cpus` | `null` | CPUs of the optimization job; null: slurm.cpus |
| `slurm.opt_mem` | `null` | memory of the optimization job ("0": whole node); null: slurm.mem |
| `slurm.system_partition` | `null` | partition of the whole-system job (one full node; trimers and larger need more than 24 h); null: slurm.partition |
| `slurm.system_cpus` | `128` | CPUs of the whole-system job |
| `slurm.system_mem` | `'0'` | memory of the whole-system job ("0": whole node) |
| `slurm.time_system` | `'24:00:00'` | time limit of the whole-system job (BODIPY dimer 11 h, trimer 20 h, tetramer 47 h) |
| `slurm.time_opt` | `'24:00:00'` | time limit of the optimization job |
| `slurm.time_td` | `'12:00:00'` | time limit of one fragment TDDFT job (a 65-atom BODIPY takes about 1.5 h on 32 cores) |
| `slurm.time_vibronic` | `'24:00:00'` | time limit of one fragment vibronic job (Hessian + excited-state gradient) |
| `slurm.time_analysis` | `'02:00:00'` | time limit of the analysis and diabatize jobs |
| `slurm.setup` | `[]` | shell lines run before eecc, e.g. ["module load python", "source venv/bin/activate"] |
<!-- options:end -->

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
    ├── options.py        # Option descriptions: config template (eecc init), README reference
    ├── structure.py      # XYZ I/O, connectivity, fragments, H capping
    ├── pyscf_setup.py    # PySCF molecule and Kohn-Sham construction
    ├── ground.py         # Geometry optimization and frequencies
    ├── excited.py        # TDDFT and transition density matrices
    ├── transition.py     # Cube files, transition Mulliken and TrESP charges
    ├── couplings.py      # Pairwise couplings via the EECC methods
    ├── system.py         # Whole-system TDDFT and diabatization stages
    ├── diabatize.py      # Projection diabatization onto fragment LE/CT states
    ├── vibronic.py       # Normal modes and Huang-Rhys factors (vibronic stage)
    ├── pipeline.py       # Restartable stage runner
    └── slurm.py          # Slurm job submission
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

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

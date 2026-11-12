# Changelog

All notable changes to EECC. Versions follow [Semantic Versioning](https://semver.org).

## [2.0.1] - 2026-11-12

Installation on Windows and macOS.

### Fixed

- `pip install "eecc[qm]"` failed on Windows and on Macs with Apple silicon: pyscf-dispersion (the
  D3/D4 correction of the geometry optimization) has pip packages for Linux and Intel macOS only. It
  is now required only there. Without it, a run that would optimize with `opt.disp` set stops
  before anything runs and says what to do (WSL on Windows, conda-forge on Apple silicon,
  `opt.disp: null`, or no optimization); `eecc init` notes it in the file. Runs that do not
  optimize (fragment files, later stages, `--restamp`) are not affected.
- Text files (`config.yaml`, JSON and text results, cube, charge and XYZ files) are read and written
  as UTF-8 on every system; Windows used its local code page. Input files may start with a
  byte-order mark (Notepad) and have titles in another code page.
- Output redirected to a file or pipe is written as UTF-8: on Windows, characters such as Δ stopped
  the program.
- `eecc init` with the geometry on another Windows drive than the configuration file.
- Slurm job scripts have Unix line endings on every system.
- Install hints use double quotes (`pip install "eecc[qm]"`), which also work in the Windows
  Command Prompt.

### Added

- `python -m eecc`, the same as `eecc`, for when pip's script folder is not on PATH.
- README: installation on Windows (natively, with uv, and the pipeline in WSL) and macOS.
- GitHub Actions: the core and pipeline tests also on Windows and macOS.

## [2.0.0] - 2026-11-05

### Added

- **Automated pipeline** `eecc run config.yaml` (extra `eecc[qm]`: PySCF, pyscf-dispersion, geomeTRIC,
  PyYAML): geometry optimization, fragments, fragment TD-DFT, transition densities and charges, and
  couplings, without Gaussian or Multiwfn.
  - Fragments from atom ranges (cut bonds capped with hydrogens), one per separate molecule, or from
    ready-made fragment files.
  - Couplings of every fragment pair: TDC (FFT and direct sum), TrESP, transition Mulliken charges,
    point dipole and extended dipole.
  - Restartable stages: each stage records a hash of its settings and input, so a rerun skips what is
    still valid and recomputes what changed (and the stages that depend on it).
  - Slurm: `--slurm` submits each stage as a job (fragments as an array job, resources per stage) and
    resubmits only the unfinished parts.
- **Whole system and exciton Hamiltonian** (`system.enabled: true`): whole-system TD-DFT, diabatized
  by projection onto the fragment locally excited (LE) and charge-transfer (CT) states. Gives site
  energies, total couplings (beyond Coulomb), CT states, the completeness of each diabatic state, and
  diabatic transition dipoles and oscillator strengths.
- **`diabatic.json`**, a documented, versioned exciton-model file (format 5) with units, sign and frame
  conventions, state types (LE fragment, CT donor/acceptor), fragment centres and principal axes, and
  the vibronic parameters. `eecc.formats` (`DIABATIC_FORMAT`, `check_diabatic`) lets other programs
  check the file they read.
- **Vibronic parameters** (`vibronic.enabled: true`): Huang–Rhys factors and frequencies of each
  fragment's excited state (vertical gradient: ground-state Hessian + excitation-energy gradient), an
  effective Holstein mode, up to `vibronic.groups` grouped effective modes, and a Gaussian width from
  the low-frequency modes. Memory control for the Hessian (`vibronic.eri`) and a restart point before it.
- `eecc init`: writes a `config.yaml` that lists every option with its default and a comment.
- `eecc paths CONFIG`: the output locations as JSON, and whether `diabatic.json` is current.
- Configuration checks before anything runs: misspelled keys (with a suggestion), wrong value types,
  functionals and basis sets unknown to PySCF or missing an element of the molecule.
- `--boundary free|periodic` on `tdc-two-monomers` and `tdc-one-dimer`.
- README: getting started, configuration reference, and validation against the published BODIPY
  oligomers, two published diabatic models (naphthalene–TCNE, DBC dimer) and measured spectra (BODIPY,
  quaterrylene).
- GitHub Actions: core tests on Python 3.10–3.13 and the pipeline tests on small molecules; package
  build check.

### Changed

- **Requires Python 3.10 or newer** (Python 3.9 reached its end of life in October 2025).
- **TDC from cube files: free-space FFT by default.** The FFT now solves the isolated problem (exact at
  any separation) instead of a periodic box. This changes the default results of `tdc-two-monomers`,
  `tdc-one-dimer`, `tdc_coupling_fft` and `tdc_coupling_fft_simple`: the published BODIPY dimer gives
  979 cm⁻¹ instead of 990 cm⁻¹, and distant pairs change by tens of percent. `--boundary periodic`
  (`boundary="periodic"`) reproduces the earlier numbers.
- `pad_factor` defaults to None and is used only with the periodic boundary.
- TDC from two cubes integrates over cube A's grid (`dV_A`), so the result no longer depends on the
  voxel sizes of the two cubes.
- The point-dipole and extended-dipole comparisons use the vector between the atom centroids as R,
  instead of the vector between the cube origins.
- The cube and charge commands (`tdc-two-monomers`, `tdc-one-dimer`, `intramolecular`,
  `intermolecular`) take a file path as well: each input is `inputs/<name>` when that file exists
  (as before), otherwise the path as given.
- `eecc view` and `eecc plot-density` say to install `eecc[viz]` when matplotlib is missing.
- The example data are the published BODIPY dimer fragment files (doi:10.5878/pyja-rd94, CC BY 4.0).
- Packaging: `pyproject.toml` only (`setup.py` and `setup.cfg` removed); the version is defined once,
  in `eecc.__version__`; the source distribution includes the tests and this changelog.

### Fixed

- `read_cube` read the volumetric data in the wrong order (x fastest instead of z), which scrambled
  the density whenever the grid dimensions differ. This affected every cube-based result of 1.0.0:
  TDC (FFT and direct), `tdc-one-dimer`, TrESP fitting and dipoles from cubes.

## [1.0.0] - 2026-02-16

First version: couplings from Gaussian cube files and transition charges (TDC via FFT and direct sum,
TrESP fitting, transition Mulliken charges, point and extended dipoles), intra- and intermolecular
workflows, and visualization tools.

[2.0.1]: https://github.com/srhashemi/EECC/releases/tag/v2.0.1
[2.0.0]: https://github.com/srhashemi/EECC/releases/tag/v2.0.0
[1.0.0]: https://github.com/srhashemi/EECC/tree/ee0470b

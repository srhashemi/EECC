"""Descriptions of every pipeline option: the commented config template and the README reference.

``HELP`` is the single source of the user-facing text. A test checks that it covers
every field of :class:`eecc.qm.config.PipelineConfig`, so new options cannot go
undocumented. Regenerate the README table with ``python -m eecc.qm.options README.md``.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

from eecc.qm.config import PipelineConfig

SECTION_HELP: Dict[str, str] = {
    "fragments": "how the system is split into chromophores",
    "fragments.cap": "link atoms on bonds cut between fragments",
    "opt": "ground-state geometry optimization of the whole system",
    "td": "excited states of each fragment (and of the whole system)",
    "transition": "transition densities and charges",
    "transition.cube": "transition-density cube grid",
    "transition.tresp": "TrESP charge fit",
    "couplings": "Coulomb couplings between every fragment pair",
    "system": "whole-system TDDFT and exciton Hamiltonian (Phase 2)",
    "resources": "local run (and inside each Slurm job)",
    "slurm": "Slurm jobs (eecc run --slurm)",
}

HELP: Dict[str, str] = {
    "geometry": "XYZ file of the whole system (Å); relative to this file",
    "name": "run name, used in Slurm job names",
    "workdir": "output directory; relative to this file",
    "charge": "total charge",
    "spin": "2S; only closed-shell (0) is supported",
    "fragments.mode": "ranges: atom indices per chromophore (cut and capped); auto: one fragment per "
                      "separate molecule; files: ready-made fragment XYZ files",
    "fragments.ranges": "mode ranges: 1-based atom indices per fragment, e.g. [\"1-56,113-120\", \"57-112\"]",
    "fragments.files": "mode files: one XYZ file per fragment (no cutting or capping)",
    "fragments.charges": "charge of each fragment; empty: all neutral",
    "fragments.cap.element": "element of the cap atom",
    "fragments.cap.bond_lengths": "cap bond length (Å) by element of the capped atom",
    "fragments.cap.default_bond_length": "cap bond length (Å) for other elements",
    "fragments.bond_scale": "atoms are bonded if closer than this × the sum of covalent radii",
    "opt.enabled": "optimize the geometry first; false if it is already optimized",
    "opt.xc": "functional",
    "opt.basis": "basis set",
    "opt.cart": "Cartesian d functions; null: as Gaussian (Cartesian for 6-31G-type bases)",
    "opt.disp": "dispersion correction (d3bj, d4, ...); null: none",
    "opt.grid": "DFT grid [radial, angular]",
    "opt.density_fit": "density fitting (faster, negligible error for geometries)",
    "opt.conv_tol": "SCF convergence (Eh)",
    "opt.maxsteps": "maximum optimization steps",
    "opt.convergence": "geomeTRIC criteria set: gau, gau_loose, gau_tight, gau_verytight, nwchem_loose, "
                       "turbomole, interfrag_tight",
    "opt.thresholds": "override single criteria: energy (Eh), grms/gmax (Eh/Bohr), drms/dmax (Å)",
    "opt.freq": "frequencies and imaginary-mode check after the optimization (expensive)",
    "opt.memory_mb": "PySCF memory for this stage (MB); null: resources.memory_mb",
    "td.xc": "functional (any PySCF/libxc name, e.g. wb97x-d, cam-b3lyp, b3lyp, pbe0, m06-2x)",
    "td.basis": "basis set (any PySCF name, e.g. 6-31g(d), 6-311g(d,p), def2-svp, def2-tzvp)",
    "td.cart": "Cartesian d functions; null: as Gaussian (Cartesian for 6-31G-type bases)",
    "td.nstates": "excited states computed per fragment",
    "td.method": "tddft (full, as Gaussian TD) or tda (faster; check that the state order is right)",
    "td.state": "excited state used for the couplings (1 = S1)",
    "td.grid": "DFT grid of the SCF [radial, angular]",
    "td.response_grid": "grid of the TDDFT response only (about 1.6× faster, same results); null: td.grid",
    "td.density_fit": "density fitting (does not speed up the TDDFT step)",
    "td.conv_tol": "SCF convergence (Eh)",
    "td.davidson_tol": "TDDFT convergence; 1e-5 gives the same states about 1.3× slower",
    "transition.cube.spacing": "grid spacing (Bohr), identical for all fragments",
    "transition.cube.margin": "grid extent beyond the outermost atoms (Bohr)",
    "transition.tresp.shells": "fit-point shells at these multiples of the vdW radius",
    "transition.tresp.density": "fit points per Å² on each shell",
    "transition.tresp.dipole_constraint": "fitted charges reproduce the TDDFT transition dipole",
    "transition.tresp.alpha": "ridge regularization of the fit",
    "couplings.methods": "any of tdc_fft, tdc_direct (slow, exact cross-check), tresp, mulliken, "
                         "point_dipole, extended_dipole",
    "couplings.dielectric": "relative permittivity screening the couplings",
    "couplings.cap_charges": "charges on cap atoms: merge into the capped atom, drop, or keep",
    "couplings.tdc_pad": "FFT padding factor (periodic boundary only)",
    "couplings.tdc_boundary": "TDC FFT boundary: free (exact at any distance) or periodic "
                              "(as the published workflow; wrong for distant pairs)",
    "couplings.tdc_direct_threshold": "tdc_direct keeps voxels with |q| > threshold × max|q|",
    "system.enabled": "also compute the whole system and diabatize it (one full node, hours to days)",
    "system.nstates": "whole-system states; 0: four per fragment. LE + CT needs n² for n fragments",
    "system.include_ct": "add HOMO→LUMO charge-transfer states between every fragment pair",
    "system.memory_mb": "PySCF memory for the whole-system run (MB); null: resources.memory_mb",
    "resources.threads": "threads; 0: OMP_NUM_THREADS or all cores",
    "resources.memory_mb": "PySCF memory (MB)",
    "resources.tmpdir": "PySCF scratch directory; null: default",
    "slurm.account": "allocation (sbatch -A), e.g. naiss2025-5-749",
    "slurm.partition": "partition of the optimization, fragment and analysis jobs",
    "slurm.physical_cores": "one thread per physical core (--hint=nomultithread; about 2× faster)",
    "slurm.cpus": "CPUs per optimization or fragment job",
    "slurm.mem": "memory per optimization or fragment job; on Dardel shared keep ≤ cpus × 0.87 GB",
    "slurm.analysis_cpus": "CPUs of the prep, analysis and diabatize jobs",
    "slurm.analysis_mem": "memory of the prep, analysis and diabatize jobs",
    "slurm.opt_partition": "partition of the optimization job; null: slurm.partition",
    "slurm.opt_cpus": "CPUs of the optimization job; null: slurm.cpus",
    "slurm.opt_mem": "memory of the optimization job (\"0\": whole node); null: slurm.mem",
    "slurm.system_partition": "partition of the whole-system job; use long for trimers and larger",
    "slurm.system_cpus": "CPUs of the whole-system job",
    "slurm.system_mem": "memory of the whole-system job (\"0\": whole node)",
    "slurm.time_system": "time limit of the whole-system job (BODIPY dimer 11 h, trimer 20 h, tetramer 47 h)",
    "slurm.time_opt": "time limit of the optimization job",
    "slurm.time_td": "time limit of one fragment TDDFT job (a 65-atom BODIPY takes about 1.5 h on 32 cores)",
    "slurm.time_analysis": "time limit of the analysis and diabatize jobs",
    "slurm.setup": "shell lines run before eecc, e.g. [\"module load cray-python\", \"source venv/bin/activate\"]",
}


def iter_options(obj: Any = None, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """(dotted key, value) for every leaf option, in definition order."""
    obj = PipelineConfig() if obj is None else obj
    for f in dataclasses.fields(obj):
        if f.metadata.get("internal"):
            continue
        value = getattr(obj, f.name)
        if dataclasses.is_dataclass(value):
            yield from iter_options(value, f"{prefix}{f.name}.")
        else:
            yield prefix + f.name, value


def format_value(value: Any) -> str:
    """YAML flow representation (quotes strings YAML would misread, such as 24:00:00)."""
    import yaml

    return yaml.safe_dump([value], default_flow_style=True, width=10 ** 6, allow_unicode=True).strip()[1:-1]


def config_template(cfg: Optional[PipelineConfig] = None, notes: Optional[Dict[str, str]] = None) -> str:
    """Commented YAML with every option of *cfg* (default: all defaults).

    *notes* adds a remark to single options, e.g. {"fragments.ranges": "fill in"}.
    """
    cfg = PipelineConfig() if cfg is None else cfg
    notes = notes or {}
    lines = ["# EECC pipeline configuration. Run:  eecc run <this file>   (or --slurm for Slurm jobs)",
             "# Every option is listed with its default unless set by 'eecc init'; delete what you",
             "# do not need. Reference: README, section 'Configuration reference'.", ""]
    opened: List[str] = []
    for key, value in iter_options(cfg):
        parts = key.split(".")
        sections, leaf = parts[:-1], parts[-1]
        for depth in range(len(sections)):
            sec = ".".join(sections[:depth + 1])
            if depth < len(opened) and opened[depth] == sec:
                continue
            del opened[depth:]
            if depth == 0:
                lines.append("")
            head = "  " * depth + f"{sections[depth]}:"
            lines.append(f"{head:<34s} # {SECTION_HELP[sec]}" if sec in SECTION_HELP else head)
            opened.append(sec)
        del opened[len(sections):]
        text = "  " * len(sections) + f"{leaf}: {format_value(value)}"
        comment = HELP[key] + (f"  <-- {notes[key]}" if key in notes else "")
        lines.append(f"{text:<34s} # {comment}")
    return "\n".join(lines) + "\n"


def reference_markdown() -> str:
    """Markdown table of every option, its default and its description."""
    rows = ["| Option | Default | Description |", "|---|---|---|"]
    for key, value in iter_options():
        default = format_value(value).replace("|", "\\|")
        rows.append(f"| `{key}` | `{default}` | {HELP[key]} |")
    return "\n".join(rows)


README_START = "<!-- options:start (generated: python -m eecc.qm.options README.md) -->"
README_END = "<!-- options:end -->"


def update_readme(path: str) -> bool:
    """Replace the generated reference table in *path*; True if it changed."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    i, j = text.index(README_START) + len(README_START), text.index(README_END)
    new = text[:i] + "\n" + reference_markdown() + "\n" + text[j:]
    if new != text:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new)
    return new != text


if __name__ == "__main__":
    if len(sys.argv) == 2:
        print("updated" if update_readme(sys.argv[1]) else "up to date")
    else:
        print(reference_markdown())

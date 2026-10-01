"""Pipeline configuration: YAML file -> nested dataclasses with validation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# === Configuration sections =================================
# ============================================================

@dataclass
class CapConfig:
    """Link-atom capping of bonds cut between fragments."""
    element: str = "H"
    # Bond length (Å) from the capped atom to the cap, by element of the capped atom.
    bond_lengths: Dict[str, float] = field(default_factory=lambda: {
        "C": 1.09, "N": 1.01, "O": 0.96, "B": 1.19, "S": 1.34, "Si": 1.48, "P": 1.42,
    })
    default_bond_length: float = 1.09


@dataclass
class FragmentsConfig:
    # 'ranges': 1-based atom index strings; 'auto': connected components (non-covalent);
    # 'files': ready-made fragment XYZ files (no cutting/capping).
    mode: str = "ranges"
    ranges: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)
    charges: List[int] = field(default_factory=list)  # per fragment; empty = all neutral
    cap: CapConfig = field(default_factory=CapConfig)
    bond_scale: float = 1.2  # bonded if d < bond_scale * (r_cov,i + r_cov,j)


@dataclass
class OptConfig:
    enabled: bool = True
    xc: str = "b3lyp"
    basis: str = "def2-svp"
    cart: Optional[bool] = None  # None: Cartesian for Pople bases, spherical otherwise
    disp: Optional[str] = "d3bj"
    grid: List[int] = field(default_factory=lambda: [99, 590])
    density_fit: bool = True
    conv_tol: float = 1e-9
    maxsteps: int = 200
    # geomeTRIC convergence: a named criteria set (see OPT_CONVERGENCE_SETS) plus
    # optional overrides of single criteria: energy (Eh), grms/gmax (Eh/Bohr),
    # drms/dmax (Angstrom). All five criteria must be met.
    convergence: str = "gau"
    thresholds: Dict[str, float] = field(default_factory=dict)
    freq: bool = False
    memory_mb: Optional[int] = None  # PySCF memory for this stage; default resources.memory_mb


@dataclass
class TDConfig:
    xc: str = "wb97x-d"
    basis: str = "6-31g(d)"
    cart: Optional[bool] = None
    nstates: int = 10
    method: str = "tddft"  # 'tddft' (full, as Gaussian TD) or 'tda'
    state: int = 1  # 1-based excited state used for the couplings
    grid: List[int] = field(default_factory=lambda: [99, 590])
    density_fit: bool = False
    conv_tol: float = 1e-9


@dataclass
class CubeConfig:
    spacing: float = 0.25  # Bohr; identical for all fragments
    margin: float = 6.0  # Bohr beyond the outermost atoms


@dataclass
class TrESPConfig:
    # Merz-Kollman style shells at these multiples of the vdW radius.
    shells: List[float] = field(default_factory=lambda: [1.4, 1.6, 1.8, 2.0])
    density: float = 5.0  # points per Å^2 on each shell
    dipole_constraint: bool = True
    alpha: float = 0.0  # ridge regularization


@dataclass
class TransitionConfig:
    cube: CubeConfig = field(default_factory=CubeConfig)
    tresp: TrESPConfig = field(default_factory=TrESPConfig)


@dataclass
class CouplingsConfig:
    methods: List[str] = field(default_factory=lambda: [
        "tdc_fft", "tresp", "mulliken", "point_dipole", "extended_dipole",
    ])
    dielectric: float = 1.0
    # Charges on link-atom caps sit close to the neighbouring fragment and distort
    # charge-based couplings. 'merge' adds each cap's charge to the atom it caps,
    # 'drop' removes it, 'keep' uses the charges as fitted.
    cap_charges: str = "merge"
    tdc_pad: int = 3
    tdc_direct_threshold: float = 0.0005


@dataclass
class ResourcesConfig:
    threads: int = 0  # 0: take OMP_NUM_THREADS / all available
    memory_mb: int = 16000
    tmpdir: Optional[str] = None


@dataclass
class SlurmConfig:
    account: Optional[str] = None
    partition: str = "shared"
    # CPUs/memory for optimization and fragment TDDFT jobs. On partitions that
    # allocate cores by memory (e.g. Dardel 'shared', ~0.87 GB per core), keep
    # mem <= cpus x that ratio to avoid being billed for extra cores.
    cpus: int = 32
    mem: str = "28G"
    analysis_cpus: int = 8
    analysis_mem: str = "16G"
    # Optional overrides for the optimization job, which needs far more memory
    # than the fragment jobs (e.g. a whole node: opt_partition: main, opt_mem: "0").
    opt_partition: Optional[str] = None
    opt_cpus: Optional[int] = None
    opt_mem: Optional[str] = None
    time_opt: str = "24:00:00"
    time_td: str = "04:00:00"
    time_analysis: str = "02:00:00"
    setup: List[str] = field(default_factory=list)  # shell lines run before eecc


@dataclass
class PipelineConfig:
    geometry: str = ""
    name: str = "eecc_run"
    workdir: str = "work"
    charge: int = 0
    spin: int = 0  # 2S
    fragments: FragmentsConfig = field(default_factory=FragmentsConfig)
    opt: OptConfig = field(default_factory=OptConfig)
    td: TDConfig = field(default_factory=TDConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    couplings: CouplingsConfig = field(default_factory=CouplingsConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    # Directory of the config file; relative paths are resolved against it.
    base_dir: str = field(default=".", metadata={"internal": True})

    # --------------------------------------------------------
    def path(self, p: str) -> str:
        """Resolve *p* relative to the config file directory."""
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(self.base_dir, p))

    @property
    def workdir_path(self) -> str:
        return self.path(self.workdir)

    def fragment_charge(self, k: int) -> int:
        """Charge of fragment *k* (0-based)."""
        ch = self.fragments.charges
        return int(ch[k]) if k < len(ch) else 0

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("base_dir", None)
        return d

    def section_hash(self, *names: str) -> str:
        """Stable hash of the named sections (used to decide whether a stage is stale).

        Only settings that differ from their defaults enter the hash, and settings
        that do not affect results (see ``HASH_IGNORED``) are skipped. Adding a new
        option with a default value therefore leaves existing stages up to date,
        while changing any setting that matters invalidates them.
        """
        payload = {}
        for n in names:
            value = getattr(self, n)
            if dataclasses.is_dataclass(value):
                value = _non_default(value, type(value)())
            payload[n] = value
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# Settings that never change results and are left out of stage hashes.
HASH_IGNORED = {"memory_mb"}


def _non_default(obj, default) -> Dict[str, Any]:
    """Nested dict of the fields of dataclass *obj* that differ from *default*."""
    out: Dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        if f.name in HASH_IGNORED:
            continue
        v, d = getattr(obj, f.name), getattr(default, f.name)
        if dataclasses.is_dataclass(v):
            sub = _non_default(v, d)
            if sub:
                out[f.name] = sub
        elif v != d:
            out[f.name] = v
    return out


# ============================================================
# === Loading ================================================
# ============================================================

VALID_FRAGMENT_MODES = ("ranges", "auto", "files")
VALID_TD_METHODS = ("tddft", "tda")
OPT_CONVERGENCE_SETS = ("gau", "gau_loose", "gau_tight", "gau_verytight",
                        "nwchem_loose", "turbomole", "interfrag_tight")
OPT_THRESHOLDS = ("energy", "grms", "gmax", "drms", "dmax")
VALID_METHODS = ("tdc_fft", "tdc_direct", "tresp", "mulliken", "point_dipole", "extended_dipole")


def _build(cls, data: Optional[Dict[str, Any]], where: str):
    """Instantiate dataclass *cls* from *data*, recursing and rejecting unknown keys."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ValueError(f"'{where}' must be a mapping, got {type(data).__name__}")
    fields = {f.name: f for f in dataclasses.fields(cls) if not f.metadata.get("internal")}
    unknown = set(data) - set(fields)
    if unknown:
        raise ValueError(f"Unknown key(s) in '{where}': {sorted(unknown)}; "
                         f"allowed: {sorted(fields)}")
    kwargs = {}
    for name, value in data.items():
        f = fields[name]
        sub = f.default_factory if f.default_factory is not dataclasses.MISSING else None
        if dataclasses.is_dataclass(sub):
            kwargs[name] = _build(sub, value, f"{where}.{name}" if where else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def validate(cfg: PipelineConfig) -> None:
    if not cfg.geometry and cfg.fragments.mode != "files":
        raise ValueError("'geometry' is required unless fragments.mode is 'files'")
    if cfg.fragments.mode not in VALID_FRAGMENT_MODES:
        raise ValueError(f"fragments.mode must be one of {VALID_FRAGMENT_MODES}")
    if cfg.fragments.mode == "ranges" and len(cfg.fragments.ranges) < 2:
        raise ValueError("fragments.ranges needs at least two fragments")
    if cfg.fragments.mode == "files" and len(cfg.fragments.files) < 2:
        raise ValueError("fragments.files needs at least two fragment files")
    n_frag = len(cfg.fragments.ranges if cfg.fragments.mode == "ranges" else cfg.fragments.files)
    if cfg.fragments.charges and cfg.fragments.mode != "auto" and len(cfg.fragments.charges) != n_frag:
        raise ValueError("fragments.charges must list one charge per fragment")
    if cfg.opt.convergence.lower() not in OPT_CONVERGENCE_SETS:
        raise ValueError(f"opt.convergence must be one of {OPT_CONVERGENCE_SETS}")
    bad = set(cfg.opt.thresholds) - set(OPT_THRESHOLDS)
    if bad:
        raise ValueError(f"Unknown opt.thresholds key(s) {sorted(bad)}; allowed: {OPT_THRESHOLDS}")
    if any(float(v) <= 0 for v in cfg.opt.thresholds.values()):
        raise ValueError("opt.thresholds must be positive")
    if cfg.td.method not in VALID_TD_METHODS:
        raise ValueError(f"td.method must be one of {VALID_TD_METHODS}")
    if not 1 <= cfg.td.state <= cfg.td.nstates:
        raise ValueError("td.state must be between 1 and td.nstates")
    bad = set(cfg.couplings.methods) - set(VALID_METHODS)
    if bad:
        raise ValueError(f"Unknown coupling method(s) {sorted(bad)}; allowed: {VALID_METHODS}")
    if cfg.couplings.cap_charges not in ("merge", "drop", "keep"):
        raise ValueError("couplings.cap_charges must be 'merge', 'drop' or 'keep'")
    if cfg.transition.cube.spacing <= 0 or cfg.transition.cube.margin <= 0:
        raise ValueError("transition.cube spacing and margin must be positive")


def config_from_dict(data: Dict[str, Any], base_dir: str = ".") -> PipelineConfig:
    cfg = _build(PipelineConfig, data, "")
    cfg.base_dir = os.path.abspath(base_dir)
    validate(cfg)
    return cfg


def load_config(path: str) -> PipelineConfig:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data, base_dir=os.path.dirname(os.path.abspath(path)))

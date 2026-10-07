"""Restartable pipeline: geometry -> optimization -> fragments -> TDDFT -> transition data -> couplings.

Each stage writes into its own directory under the work directory and records a
hash of everything it depends on in a ``.done`` file. A stage is skipped when its
hash is current, so reruns continue where a previous run stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import List, Optional, Tuple

from eecc.qm.config import PipelineConfig
from eecc.qm.structure import (
    Fragment, Structure, build_fragments, load_fragment_files, read_xyz, save_xyz,
)


STAGES = ("opt", "fragments", "td", "transition", "couplings", "system", "diabatize")
# Stages whose results each stage uses (a rerun invalidates everything that depends on it).
DEPENDS = {
    "fragments": ("opt",), "td": ("fragments",), "transition": ("td",), "couplings": ("transition",),
    "system": ("opt",), "diabatize": ("system", "td"),
}
# Output format of the diabatize stage; raising it reruns older results (2: diabatic transition dipoles).
DIABATIZE_FORMAT = "2"
PER_FRAGMENT = ("td", "transition")


def _file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _combine(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class Pipeline:
    def __init__(self, cfg: PipelineConfig, log=print):
        self.cfg = cfg
        self.root = cfg.workdir_path
        self.log = log

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    def stage_dir(self, stage: str, fragment: Optional[str] = None) -> str:
        num = STAGES.index(stage) + 1
        d = os.path.join(self.root, f"{num:02d}_{stage}")
        return os.path.join(d, fragment) if fragment else d

    # --------------------------------------------------------
    # Hashes (derived from the config and input files only)
    # --------------------------------------------------------
    def _input_hash(self) -> str:
        cfg = self.cfg
        if cfg.fragments.mode == "files":
            return _combine(*[_file_hash(cfg.path(p)) for p in cfg.fragments.files])
        return _file_hash(cfg.path(cfg.geometry))

    def expected_hash(self, stage: str) -> str:
        cfg = self.cfg
        h = self._input_hash()
        if stage == "opt":
            return _combine(h, cfg.section_hash("opt", "charge", "spin"))
        # Frequencies do not change the geometry, so they do not invalidate later stages.
        h = _combine(h, cfg.section_hash("opt", "charge", "spin", ignore=("freq",)))
        if stage == "system":  # whole system with the td method; independent of the fragments
            if cfg.fragments.mode == "files":  # the input hash covers only the fragment files
                h = _combine(h, _file_hash(cfg.path(cfg.geometry)))
            return _combine(h, cfg.section_hash("td", ignore=("nstates", "state")),
                            cfg.section_hash("system", ignore=("include_ct", "enabled")))
        if stage == "diabatize":
            return _combine(self.expected_hash("system"), self.expected_hash("td"),
                            cfg.section_hash("system", ignore=("enabled",)), "format " + DIABATIZE_FORMAT)
        h = _combine(h, cfg.section_hash("fragments"))
        if stage == "fragments":
            return h
        h = _combine(h, cfg.section_hash("td"))
        if stage == "td":
            return h
        h = _combine(h, cfg.section_hash("transition"))
        if stage == "transition":
            return h
        return _combine(h, cfg.section_hash("couplings"))

    def is_done(self, stage: str, fragment: Optional[str] = None) -> bool:
        marker = os.path.join(self.stage_dir(stage, fragment), ".done")
        return os.path.exists(marker) and open(marker).read().strip() == self.expected_hash(stage)

    def _mark_done(self, stage: str, fragment: Optional[str] = None) -> None:
        self._invalidate_downstream(stage, fragment)
        with open(os.path.join(self.stage_dir(stage, fragment), ".done"), "w") as f:
            f.write(self.expected_hash(stage) + "\n")

    def _invalidate_downstream(self, stage: str, fragment: Optional[str] = None) -> None:
        """Remove the markers of every stage that depends on the results of *stage*.

        A rerun (for example with --force) may produce different results under the
        same settings, so stages built on the previous results must run again.
        """
        dependents, grew = {stage}, True
        while grew:
            new = {s for s, deps in DEPENDS.items() if dependents.intersection(deps)} - dependents
            dependents |= new
            grew = bool(new)
        for later in [s for s in STAGES if s in dependents and s != stage]:
            if later in PER_FRAGMENT:
                d = self.stage_dir(later)
                names = [fragment] if fragment else (os.listdir(d) if os.path.isdir(d) else [])
                markers = [os.path.join(d, n, ".done") for n in names]
            else:
                markers = [os.path.join(self.stage_dir(later), ".done")]
            for m in markers:
                if os.path.exists(m):
                    os.remove(m)

    # --------------------------------------------------------
    # Fragment bookkeeping
    # --------------------------------------------------------
    def fragment_names(self) -> List[str]:
        path = os.path.join(self.stage_dir("fragments"), "fragments.json")
        if not os.path.exists(path):
            raise RuntimeError("fragments stage has not run yet")
        with open(path) as f:
            return [fr["name"] for fr in json.load(f)["fragments"]]

    def load_fragment_structure(self, name: str) -> Structure:
        return read_xyz(os.path.join(self.stage_dir("fragments"), f"{name}.xyz"))

    def planned_fragment_count(self) -> int:
        """Number of fragments, known before running anything (used for Slurm arrays)."""
        cfg = self.cfg
        if cfg.fragments.mode == "files":
            return len(cfg.fragments.files)
        if cfg.fragments.mode == "ranges":
            return len(cfg.fragments.ranges)
        return len(build_fragments(read_xyz(cfg.path(cfg.geometry)), cfg.fragments))

    # --------------------------------------------------------
    # Stages
    # --------------------------------------------------------
    def run_opt(self, force: bool = False) -> None:
        d = self.stage_dir("opt")
        if self.is_done("opt") and not force:
            self.log("[opt] up to date")
            return
        if self.cfg.fragments.mode == "files":
            self.log("[opt] fragments given as files: nothing to optimize")
            os.makedirs(d, exist_ok=True)
            self._mark_done("opt")
            return
        os.makedirs(d, exist_ok=True)
        structure = read_xyz(self.cfg.path(self.cfg.geometry))
        save_xyz(structure, os.path.join(d, "input.xyz"))
        if self.cfg.opt.enabled:
            from eecc.qm.ground import optimize_geometry
            self._prepare_trajectory(d, force)
            self.log(f"[opt] optimizing {len(structure)} atoms "
                     f"({self.cfg.opt.xc}-{self.cfg.opt.disp}/{self.cfg.opt.basis})")
            optimize_geometry(structure, self.cfg, d)
        else:
            self.log("[opt] disabled: using the input geometry")
            shutil.copyfile(os.path.join(d, "input.xyz"), os.path.join(d, "optimized.xyz"))
        self._mark_done("opt")

    def _prepare_trajectory(self, d: str, force: bool) -> None:
        """Keep trajectory.xyz for resuming only if it belongs to the current input and settings."""
        traj, stamp = os.path.join(d, "trajectory.xyz"), os.path.join(d, "trajectory.hash")
        current = self.expected_hash("opt")
        stale = force or not os.path.exists(stamp) or open(stamp).read().strip() != current
        if stale and os.path.exists(traj):
            self.log("[opt] discarding trajectory.xyz from a different input or settings")
            os.remove(traj)
        with open(stamp, "w") as f:
            f.write(current + "\n")

    def run_fragments(self, force: bool = False) -> None:
        d = self.stage_dir("fragments")
        if self.is_done("fragments") and not force:
            self.log("[fragments] up to date")
            return
        if not self.is_done("opt"):
            raise RuntimeError("run the 'opt' stage first")
        os.makedirs(d, exist_ok=True)
        cfg = self.cfg
        if cfg.fragments.mode == "files":
            frags: List[Fragment] = load_fragment_files([cfg.path(p) for p in cfg.fragments.files])
        else:
            parent = read_xyz(os.path.join(self.stage_dir("opt"), "optimized.xyz"))
            # Bonds come from the input geometry, so the fragments (and the Slurm
            # array size) do not depend on how far the optimization moved atoms.
            frags = build_fragments(parent, cfg.fragments, topology=read_xyz(cfg.path(cfg.geometry)))
        meta = []
        for k, fr in enumerate(frags):
            save_xyz(fr.structure, os.path.join(d, f"{fr.name}.xyz"))
            meta.append({"name": fr.name, "natoms": len(fr.structure),
                         "parent_atoms_1based": [i + 1 for i in fr.parent_indices],
                         "caps": [{"kept": a + 1, "removed": b + 1} for a, b in fr.caps],
                         "charge": cfg.fragment_charge(k)})
            self.log(f"[fragments] {fr.name}: {len(fr.structure)} atoms, {len(fr.caps)} cap(s)")
        with open(os.path.join(d, "fragments.json"), "w") as f:
            json.dump({"fragments": meta}, f, indent=2)
        self._mark_done("fragments")

    def fragment_caps(self, name: str) -> List[Tuple[int, int]]:
        """(cap index, capped-atom index) pairs, 0-based within the fragment structure.

        Caps are appended after the fragment's own atoms in the order recorded.
        """
        with open(os.path.join(self.stage_dir("fragments"), "fragments.json")) as f:
            fr = next(x for x in json.load(f)["fragments"] if x["name"] == name)
        own = fr["parent_atoms_1based"]
        pos = {p: i for i, p in enumerate(own)}
        return [(len(own) + c, pos[cap["kept"]]) for c, cap in enumerate(fr["caps"])]

    def _fragment_charge(self, name: str) -> int:
        with open(os.path.join(self.stage_dir("fragments"), "fragments.json")) as f:
            for fr in json.load(f)["fragments"]:
                if fr["name"] == name:
                    return int(fr["charge"])
        raise KeyError(name)

    def run_td(self, fragment: str, force: bool = False) -> None:
        if self.is_done("td", fragment) and not force:
            self.log(f"[td] {fragment} up to date")
            return
        if not self.is_done("fragments"):
            raise RuntimeError("run the 'fragments' stage first")
        from eecc.qm.excited import run_excited_states
        d = self.stage_dir("td", fragment)
        t0 = time.time()
        s = run_excited_states(self.load_fragment_structure(fragment), self.cfg, d,
                               fragment, self._fragment_charge(fragment))
        sel = s["selected"]
        self.log(f"[td] {fragment}: S{s['state']} = {sel['energy_eV']:.4f} eV, f = {sel['f']:.4f}, "
                 f"|mu| = {sel['mu_D']:.3f} D ({time.time() - t0:.0f} s)")
        self._mark_done("td", fragment)

    def run_transition(self, fragment: str, force: bool = False) -> None:
        if self.is_done("transition", fragment) and not force:
            self.log(f"[transition] {fragment} up to date")
            return
        if not self.is_done("td", fragment):
            raise RuntimeError(f"run the 'td' stage for {fragment} first")
        from eecc.qm.transition import run_transition
        d = self.stage_dir("transition", fragment)
        t0 = time.time()
        info = run_transition(self.stage_dir("td", fragment), self.cfg, d, fragment,
                              self._fragment_charge(fragment))
        self.log(f"[transition] {fragment}: |mu| exact {info['exact']['mu_D']:.3f} D, "
                 f"cube {info['cube']['mu_D']:.3f} D, Mulliken {info['mulliken']['mu_D']:.3f} D, "
                 f"TrESP {info['tresp']['mu_D']:.3f} D (ESP rrms {info['tresp']['rrms']:.3f}) "
                 f"({time.time() - t0:.0f} s)")
        self._mark_done("transition", fragment)

    def run_couplings(self, force: bool = False) -> None:
        if self.is_done("couplings") and not force:
            self.log("[couplings] up to date")
            return
        names = self.fragment_names()
        missing = [n for n in names if not self.is_done("transition", n)]
        if missing:
            raise RuntimeError(f"transition stage not finished for {missing}")
        from eecc.qm.couplings import load_fragment, run_couplings, format_report, METHOD_LABELS
        frags = [load_fragment(self.stage_dir("transition", n), self.stage_dir("td", n), n,
                               self.fragment_caps(n), self.cfg.couplings.cap_charges)
                 for n in names]
        d = self.stage_dir("couplings")
        result = run_couplings(frags, self.cfg.couplings, d)
        methods = [m for m in METHOD_LABELS if m in self.cfg.couplings.methods]
        self.log(format_report(result, methods))
        self._mark_done("couplings")

    def run_system(self, force: bool = False) -> None:
        if self.is_done("system") and not force:
            self.log("[system] up to date")
            return
        if not self.is_done("opt"):
            raise RuntimeError("run the 'opt' stage first")
        from eecc.qm.system import run_system_td
        opt_xyz = os.path.join(self.stage_dir("opt"), "optimized.xyz")
        structure = read_xyz(opt_xyz if os.path.exists(opt_xyz) else self.cfg.path(self.cfg.geometry))
        nstates = self.cfg.system.nstates or 4 * self.planned_fragment_count()
        run_system_td(structure, self.cfg, self.stage_dir("system"), nstates, self.log)
        self._mark_done("system")

    def run_diabatize(self, force: bool = False) -> None:
        if self.is_done("diabatize") and not force:
            self.log("[diabatize] up to date")
            return
        if not self.is_done("system"):
            raise RuntimeError("run the 'system' stage first")
        missing = [n for n in self.fragment_names() if not self.is_done("td", n)]
        if missing:
            raise RuntimeError(f"td stage not finished for {missing}")
        from eecc.qm.system import run_diabatization
        d = self.stage_dir("diabatize")
        run_diabatization(self, d)
        self.log(open(os.path.join(d, "diabatic.txt"), encoding="utf-8").read())
        self._mark_done("diabatize")

    # --------------------------------------------------------
    # Marking existing results as current
    # --------------------------------------------------------
    def _outputs_present(self, stage: str, fragment: Optional[str] = None) -> bool:
        d = self.stage_dir(stage, fragment)
        files = {
            "opt": ["optimized.xyz"] if self.cfg.fragments.mode != "files" else [],
            "fragments": ["fragments.json"],
            "td": ["td.npz", "td.json"],
            "transition": [f"{fragment}.cub", f"{fragment}_tresp.txt",
                           f"{fragment}_mulliken.txt", f"{fragment}_transition.json"],
            "couplings": ["couplings.json", "couplings.txt"],
            "system": ["system_td.npz", "system_td.json", "system.xyz"],
            "diabatize": ["diabatic.json", "diabatic.txt"],
        }[stage]
        ok = os.path.isdir(d) and all(os.path.exists(os.path.join(d, f)) for f in files)
        if ok and stage == "diabatize":  # a result from before the diabatic transition dipoles is not current
            try:
                with open(os.path.join(d, "diabatic.json")) as f:
                    ok = "transition_dipoles_au" in json.load(f)
            except (OSError, ValueError):
                ok = False
        return ok

    def restamp(self) -> List[str]:
        """Mark stages whose outputs exist as up to date, in pipeline order.

        Use this only when the results are known to match the current settings,
        e.g. after upgrading EECC with an unchanged configuration. Stops at the
        first stage whose outputs are missing.
        """
        marked = []
        for st in STAGES:
            if st in ("td", "transition"):
                if not os.path.exists(os.path.join(self.stage_dir("fragments"), "fragments.json")):
                    break
                names = self.fragment_names()
                if not all(self._outputs_present(st, n) for n in names):
                    break
                for n in names:
                    self._mark_done(st, n)
            else:
                if not self._outputs_present(st):
                    break
                os.makedirs(self.stage_dir(st), exist_ok=True)
                self._mark_done(st)
            marked.append(st)
        return marked

    def run(self, stage: str = "all", fragment: Optional[int] = None, force: bool = False) -> None:
        """Run one stage (optionally for one 1-based fragment index) or everything."""
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "config.resolved.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)
        todo = STAGES if stage == "all" else (stage,)
        for st in todo:
            if st == "opt":
                self.run_opt(force)
            elif st == "fragments":
                self.run_fragments(force)
            elif st in ("td", "transition"):
                names = self.fragment_names()
                selected = names if fragment is None else [names[fragment - 1]]
                for n in selected:
                    (self.run_td if st == "td" else self.run_transition)(n, force)
            elif st == "couplings":
                self.run_couplings(force)
            elif st in ("system", "diabatize"):
                if stage == st or self.cfg.system.enabled:
                    (self.run_system if st == "system" else self.run_diabatize)(force)
            else:
                raise ValueError(f"unknown stage '{st}'")

"""Submit the pipeline as a chain of Slurm jobs.

1. prep: optimization + fragments (one job)
2. fragments: TDDFT + transition data, one array task per fragment
3. analysis: couplings, after all array tasks succeeded
4. with system.enabled: system (whole-system TDDFT, one node) after prep, in
   parallel with the fragments, and diabatize after both
5. with vibronic.enabled: vibronic (Hessian + excited-state gradient), one array
   task per fragment after prep; diabatize waits for it to end, but runs even if it
   failed (then without the vibronic parameters of the failed fragments)

Stages that are already complete are skipped, so resubmitting after a failure
reruns only the missing jobs and array tasks.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import List, Optional

from eecc.qm.pipeline import Pipeline


def _script(pipe: Pipeline, job: str, time: str, commands: List[str],
            cpus: int, mem: str, array: Optional[str] = None,
            partition: Optional[str] = None) -> str:
    s = pipe.cfg.slurm
    log_dir = os.path.join(pipe.root, "slurm")
    lines = ["#!/bin/bash"]
    if s.account:
        lines.append(f"#SBATCH -A {s.account}")
    if partition or s.partition:
        lines.append(f"#SBATCH -p {partition or s.partition}")
    lines += [
        f"#SBATCH -J {pipe.cfg.name}-{job}",
        "#SBATCH -n 1",
        f"#SBATCH -c {cpus}",
    ]
    if s.physical_cores:
        lines.append("#SBATCH --hint=nomultithread")
    lines += [
        f"#SBATCH --mem={mem}",
        f"#SBATCH -t {time}",
    ]
    out = f"{log_dir}/%x-%A_%a.out" if array else f"{log_dir}/%x-%j.out"
    lines.append(f"#SBATCH -o {out}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    # No 'set -u': environment activation scripts (conda, venv) often read unset variables.
    lines += ["", "set -eo pipefail"]
    lines += s.setup
    lines.append("export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}")
    tmp = pipe.cfg.resources.tmpdir
    if tmp:
        lines.append(f"export PYSCF_TMPDIR={shlex.quote(tmp)}")
    lines += commands
    return "\n".join(lines) + "\n"


def _array_spec(tasks: List[int]) -> str:
    """Slurm array spec for 1-based *tasks*, e.g. [1, 2, 3, 5] -> '1-3,5'."""
    parts, start = [], None
    for i, t in enumerate(tasks):
        if start is None:
            start = t
        if i + 1 == len(tasks) or tasks[i + 1] != t + 1:
            parts.append(f"{start}-{t}" if t > start else str(t))
            start = None
    return ",".join(parts)


def submit(pipe: Pipeline, config_path: str, dry_run: bool = False) -> List[str]:
    """Write the job scripts and submit them with dependencies.

    Returns the job ids (or script paths when *dry_run* is set).
    """
    s = pipe.cfg.slurm
    if not s.account and not dry_run:
        raise ValueError("slurm.account must be set to submit jobs")
    log_dir = os.path.join(pipe.root, "slurm")
    os.makedirs(log_dir, exist_ok=True)
    run = f"{shlex.quote(sys.executable)} -m eecc.cli run {shlex.quote(os.path.abspath(config_path))}"
    n = pipe.planned_fragment_count()
    prep_done = pipe.is_done("opt") and pipe.is_done("fragments")
    todo = list(range(1, n + 1))
    if prep_done:
        todo = [k for k, nm in enumerate(pipe.fragment_names(), 1) if not pipe.is_done("transition", nm)]

    optimizing = pipe.cfg.opt.enabled and pipe.cfg.fragments.mode != "files"
    if optimizing:
        prep_res = (s.time_opt, s.opt_cpus or s.cpus, s.opt_mem or s.mem, s.opt_partition)
    else:
        prep_res = ("00:15:00", 1, "2G", None)
    scripts = {
        "prep": _script(pipe, "prep", prep_res[0],
                        [f"{run} --stage opt", f"{run} --stage fragments"],
                        cpus=prep_res[1], mem=prep_res[2], partition=prep_res[3]),
        "frag": _script(pipe, "frag", s.time_td,
                        [f"{run} --stage td --fragment $SLURM_ARRAY_TASK_ID",
                         f"{run} --stage transition --fragment $SLURM_ARRAY_TASK_ID"],
                        cpus=s.cpus, mem=s.mem, array=_array_spec(todo)),
        "analysis": _script(pipe, "analysis", s.time_analysis, [f"{run} --stage couplings"],
                            cpus=s.analysis_cpus, mem=s.analysis_mem),
    }
    vib_todo = []
    if pipe.cfg.vibronic.enabled:
        vib_todo = list(range(1, n + 1))
        if prep_done:
            vib_todo = [k for k, nm in enumerate(pipe.fragment_names(), 1) if not pipe.is_done("vibronic", nm)]
        if vib_todo:
            scripts["vib"] = _script(pipe, "vib", s.time_vibronic,
                                     [f"{run} --stage vibronic --fragment $SLURM_ARRAY_TASK_ID"],
                                     cpus=s.vibronic_cpus, mem=s.vibronic_mem, partition=s.vibronic_partition,
                                     array=_array_spec(vib_todo))
    if pipe.cfg.system.enabled:
        scripts["system"] = _script(pipe, "system", s.time_system, [f"{run} --stage system"],
                                    cpus=s.system_cpus, mem=s.system_mem, partition=s.system_partition)
        scripts["diabatize"] = _script(pipe, "diabatize", s.time_analysis, [f"{run} --stage diabatize"],
                                       cpus=s.analysis_cpus, mem=s.analysis_mem)
        if pipe.is_done("system"):
            scripts.pop("system")
        if pipe.is_done("diabatize"):
            scripts.pop("diabatize")
    # Skip jobs whose stages are already complete (e.g. when resubmitting).
    if prep_done:
        scripts.pop("prep")
    if not todo:
        scripts.pop("frag")
    paths = {}
    for key, text in scripts.items():
        paths[key] = os.path.join(log_dir, f"{key}.sh")
        with open(paths[key], "w") as f:
            f.write(text)
    if dry_run:
        return list(paths.values())

    def sbatch(path: str, deps: List[str], any_deps: List[str]) -> str:
        cmd = ["sbatch", "--parsable"]
        cond = [f"{kind}:" + ":".join(ids_) for kind, ids_ in (("afterok", deps), ("afterany", any_deps)) if ids_]
        if cond:
            cmd.append("--dependency=" + ",".join(cond))
        cmd.append(path)
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]

    # job -> jobs it waits for (only those submitted now; finished stages need no wait)
    waits = {"prep": [], "frag": ["prep"], "vib": ["prep"], "analysis": ["frag"], "system": ["prep"],
             "diabatize": ["system", "frag", "analysis"]}
    ids: dict = {}
    for key in ("prep", "frag", "vib", "analysis", "system", "diabatize"):
        if key in paths:
            # analysis also needs prep when the fragment jobs are skipped
            deps = [ids[w] for w in waits[key] if w in ids] or ([ids["prep"]] if "prep" in ids and key != "prep" else [])
            # diabatize waits for the optional vibronic jobs to end, not to succeed
            any_deps = [ids["vib"]] if key == "diabatize" and "vib" in ids else []
            ids[key] = sbatch(paths[key], deps, any_deps)
    return [f"{key} {job}" for key, job in ids.items()]

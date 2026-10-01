"""Submit the pipeline as a chain of Slurm jobs.

1. prep: optimization + fragments (one job)
2. fragments: TDDFT + transition data, one array task per fragment
3. analysis: couplings, after all array tasks succeeded
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
    lines += [
        f"#SBATCH -p {partition or s.partition}",
        f"#SBATCH -J {pipe.cfg.name}-{job}",
        "#SBATCH -n 1",
        f"#SBATCH -c {cpus}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH -t {time}",
    ]
    out = f"{log_dir}/%x-%A_%a.out" if array else f"{log_dir}/%x-%j.out"
    lines.append(f"#SBATCH -o {out}")
    if array:
        lines.append(f"#SBATCH --array={array}")
    lines += ["", "set -euo pipefail"]
    lines += s.setup
    lines.append("export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}")
    tmp = pipe.cfg.resources.tmpdir
    if tmp:
        lines.append(f"export PYSCF_TMPDIR={shlex.quote(tmp)}")
    lines += commands
    return "\n".join(lines) + "\n"


def submit(pipe: Pipeline, config_path: str, dry_run: bool = False) -> List[str]:
    """Write the three job scripts and submit them with dependencies.

    Returns the job ids (or script paths when *dry_run* is set).
    """
    s = pipe.cfg.slurm
    if not s.account and not dry_run:
        raise ValueError("slurm.account must be set to submit jobs")
    log_dir = os.path.join(pipe.root, "slurm")
    os.makedirs(log_dir, exist_ok=True)
    run = f"{shlex.quote(sys.executable)} -m eecc.cli run {shlex.quote(os.path.abspath(config_path))}"
    n = pipe.planned_fragment_count()

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
                        cpus=s.cpus, mem=s.mem, array=f"1-{n}"),
        "analysis": _script(pipe, "analysis", s.time_analysis, [f"{run} --stage couplings"],
                            cpus=s.analysis_cpus, mem=s.analysis_mem),
    }
    # Skip jobs whose stages are already complete (e.g. when resubmitting).
    prep_done = pipe.is_done("opt") and pipe.is_done("fragments")
    frag_done = prep_done and all(pipe.is_done("transition", nm) for nm in pipe.fragment_names())
    if prep_done:
        scripts.pop("prep")
    if frag_done:
        scripts.pop("frag")
    paths = {}
    for key, text in scripts.items():
        paths[key] = os.path.join(log_dir, f"{key}.sh")
        with open(paths[key], "w") as f:
            f.write(text)
    if dry_run:
        return list(paths.values())

    def sbatch(path: str, dep: Optional[str] = None) -> str:
        cmd = ["sbatch", "--parsable"]
        if dep:
            cmd.append(f"--dependency=afterok:{dep}")
        cmd.append(path)
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().split(";")[0]

    jobs, dep = [], None
    for key in ("prep", "frag", "analysis"):
        if key in paths:
            dep = sbatch(paths[key], dep)
            jobs.append(f"{key} {dep}")
    return jobs

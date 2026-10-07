"""Tests for the automated QM pipeline (eecc.qm). Skipped without PySCF."""

import json
import os

import numpy as np
import pytest

pytest.importorskip("pyscf")
pytest.importorskip("yaml")

from eecc.qm.config import config_from_dict, load_config
from eecc.qm.structure import (
    Structure, bonds, build_fragments, connected_components, read_xyz, save_xyz,
)

ETHYLENE = Structure(
    ["C", "C", "H", "H", "H", "H"],
    np.array([[0, 0, 0.667], [0, 0, -0.667], [0, 0.923, 1.238], [0, -0.923, 1.238],
              [0, 0.923, -1.238], [0, -0.923, -1.238]], float),
)

# s-trans-butadiene (C1=C2-C3=C4), approximate geometry
BUTADIENE = Structure(
    ["C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
    np.array([
        [-1.829, 0.395, 0.0], [-0.620, -0.169, 0.0], [0.620, 0.169, 0.0], [1.829, -0.395, 0.0],
        [-2.726, -0.215, 0.0], [-1.964, 1.475, 0.0], [-0.534, -1.255, 0.0],
        [0.534, 1.255, 0.0], [2.726, 0.215, 0.0], [1.964, -1.475, 0.0],
    ]),
)


def _stacked_dimer(distance=4.0):
    B = ETHYLENE.coords + np.array([distance, 0.0, 0.0])
    return Structure(ETHYLENE.symbols * 2, np.vstack([ETHYLENE.coords, B]))


def _small_config(tmp_path, geometry, **over):
    data = {
        "geometry": str(geometry),
        "workdir": str(tmp_path / "work"),
        "fragments": {"mode": "auto"},
        "opt": {"enabled": False},
        "td": {"basis": "sto-3g", "nstates": 2, "state": 1, "grid": [50, 194]},
        "transition": {"cube": {"spacing": 0.4, "margin": 4.0},
                       "tresp": {"density": 2.0}},
        "couplings": {"tdc_pad": 2},
        "resources": {"threads": 4, "memory_mb": 2000},
    }
    for key, value in over.items():
        data[key] = {**data.get(key, {}), **value} if isinstance(value, dict) else value
    return config_from_dict(data)


# ============================================================
# === Config =================================================
# ============================================================

def test_config_defaults_and_unknown_key(tmp_path):
    cfg = config_from_dict({"geometry": "x.xyz", "fragments": {"ranges": ["1-3", "4-6"]}})
    assert cfg.td.xc == "wb97x-d" and cfg.opt.disp == "d3bj"
    with pytest.raises(ValueError, match="Unknown key"):
        config_from_dict({"geometry": "x.xyz", "td": {"bogus": 1}})
    with pytest.raises(ValueError, match="at least two"):
        config_from_dict({"geometry": "x.xyz", "fragments": {"ranges": ["1-3"]}})


def test_config_paths_relative_to_file(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("geometry: mol.xyz\nfragments:\n  mode: auto\n")
    cfg = load_config(str(p))
    assert cfg.path(cfg.geometry) == str(tmp_path / "mol.xyz")
    assert cfg.section_hash("td") == load_config(str(p)).section_hash("td")


def test_config_errors_are_explained():
    with pytest.raises(ValueError, match="did you mean 'basis'"):
        config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"}, "td": {"basis_set": "sto-3g"}})
    with pytest.raises(ValueError, match="must be quoted"):  # YAML reads unquoted 12:00:00 as 43200
        config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"}, "slurm": {"time_td": 43200}})
    cfg = config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"},
                            "slurm": {"system_mem": 0, "mem": 28000, "time_td": "720"}})
    assert (cfg.slurm.system_mem, cfg.slurm.mem, cfg.slurm.time_td) == ("0", "28000", "720")
    with pytest.raises(ValueError, match="'td.nstates' must be an integer"):
        config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"}, "td": {"nstates": "ten"}})
    with pytest.raises(ValueError, match="'opt.enabled' must be true or false"):
        config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"}, "opt": {"enabled": "no"}})
    with pytest.raises(ValueError, match="cannot be null"):
        config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"}, "td": {"basis": None}})
    # YAML 1.1 reads 1e-5 as a string; numbers are accepted, and optional settings take null
    cfg = config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"},
                            "td": {"davidson_tol": "1e-5", "response_grid": None}})
    assert cfg.td.davidson_tol == 1e-5 and cfg.td.response_grid is None


def test_check_names(tmp_path):
    from eecc.qm.config import check_names

    geo = tmp_path / "mol.xyz"
    save_xyz(Structure(["C", "I"], np.array([[0.0, 0, 0], [2.1, 0, 0]])), str(geo))
    base = {"geometry": str(geo), "fragments": {"mode": "auto"}, "opt": {"enabled": False}}
    check_names(config_from_dict({**base, "td": {"basis": "def2-svp", "xc": "cam-b3lyp"}}))
    with pytest.raises(ValueError, match="has no functions for I"):
        check_names(config_from_dict({**base, "td": {"basis": "6-31g(d)"}}))
    with pytest.raises(ValueError, match="unknown functional 'b3lpy'.*unknown basis set '6-31gg'"):
        check_names(config_from_dict({**base, "td": {"basis": "6-31gg", "xc": "b3lpy"}}))
    check_names(config_from_dict({**base, "td": {"basis": "unc-def2-svp"}}))  # prefixed names, as PySCF
    with pytest.raises(ValueError, match="unknown basis set 'nofile.nw'"):
        check_names(config_from_dict({**base, "td": {"basis": "nofile.nw"}}))


def test_options_are_documented_and_template_roundtrips(tmp_path):
    import dataclasses
    import yaml
    from eecc.qm.config import PipelineConfig, _build
    from eecc.qm.options import HELP, SECTION_HELP, config_template, iter_options, reference_markdown

    keys = [k for k, _ in iter_options()]
    assert sorted(keys) == sorted(HELP), "every option needs exactly one entry in eecc.qm.options.HELP"
    sections = {".".join(k.split(".")[:i]) for k in keys for i in range(1, k.count(".") + 1)}
    assert set(SECTION_HELP) <= sections
    assert _build(PipelineConfig, yaml.safe_load(config_template()), "") == PipelineConfig()
    cfg = config_from_dict({"geometry": "x.xyz", "fragments": {"mode": "auto"},
                            "td": {"basis": "def2-svp", "nstates": 4}, "slurm": {"time_td": "03:00:00"}})
    back = _build(PipelineConfig, yaml.safe_load(config_template(cfg)), "")
    assert dataclasses.replace(back, base_dir=cfg.base_dir) == cfg
    readme = os.path.join(os.path.dirname(__file__), os.pardir, "README.md")
    assert reference_markdown() in open(readme, encoding="utf-8").read(), \
        "README reference is out of date: python -m eecc.qm.options README.md"


def test_cli_init(tmp_path, monkeypatch, capsys):
    import sys
    from eecc.cli import main

    def init(*argv):
        monkeypatch.setattr(sys, "argv", ["eecc", "init", *argv])
        main()

    save_xyz(_stacked_dimer(), str(tmp_path / "pair.xyz"))
    save_xyz(ETHYLENE, str(tmp_path / "mono.xyz"))
    monkeypatch.chdir(tmp_path)
    init("pair.xyz", "--basis", "sto-3g", "--nstates", "4", "--account", "p", "--no-opt")
    cfg = load_config(str(tmp_path / "config.yaml"))
    assert (cfg.fragments.mode, cfg.td.basis, cfg.td.nstates, cfg.slurm.account, cfg.opt.enabled) == \
        ("auto", "sto-3g", 4, "p", False)
    assert cfg.path(cfg.geometry) == str(tmp_path / "pair.xyz") and cfg.name == "pair"
    with pytest.raises(SystemExit, match="exists"):
        init("pair.xyz")
    # one covalent molecule without ranges: written, with ranges marked as required
    init("mono.xyz", "-o", "mono.yaml")
    text = open(tmp_path / "mono.yaml").read()
    assert "REQUIRED" in text and "mode: ranges" in text
    init("mono.xyz", "-o", "mono2.yaml", "--ranges", "1-3", "4-6", "--method", "tda")
    cfg = load_config(str(tmp_path / "mono2.yaml"))
    assert cfg.fragments.ranges == ["1-3", "4-6"] and cfg.td.method == "tda"
    with pytest.raises(SystemExit, match="unknown functional"):
        init("pair.xyz", "-o", "bad.yaml", "--functional", "b3lpy")
    assert not os.path.exists(tmp_path / "bad.yaml")
    (tmp_path / "broken.xyz").write_text("not an xyz file\n")
    with pytest.raises(SystemExit, match="cannot read broken.xyz"):
        init("broken.xyz", "-o", "bad.yaml")


# ============================================================
# === Structures and fragments ===============================
# ============================================================

def test_xyz_roundtrip(tmp_path):
    path = tmp_path / "e.xyz"
    save_xyz(ETHYLENE, str(path), comment="ethylene")
    s = read_xyz(str(path))
    assert s.symbols == ETHYLENE.symbols
    np.testing.assert_allclose(s.coords, ETHYLENE.coords, atol=1e-6)


def test_auto_fragments_non_covalent():
    dimer = _stacked_dimer()
    assert len(connected_components(len(dimer), bonds(dimer))) == 2
    cfg = config_from_dict({"geometry": "x", "fragments": {"mode": "auto"}}).fragments
    frags = build_fragments(dimer, cfg)
    assert [len(f.structure) for f in frags] == [6, 6]
    assert all(not f.caps for f in frags)


def test_covalent_fragments_are_capped_along_cut_bond():
    cfg = config_from_dict({"geometry": "x",
                            "fragments": {"ranges": ["1-2,5-7", "3-4,8-10"]}}).fragments
    frags = build_fragments(BUTADIENE, cfg)
    for fr, (kept, removed) in zip(frags, [(1, 2), (2, 1)]):
        assert fr.caps == [(kept, removed)]
        assert len(fr.structure) == 6 and fr.structure.symbols[-1] == "H"
        cap = fr.structure.coords[-1]
        c_kept, c_removed = BUTADIENE.coords[kept], BUTADIENE.coords[removed]
        assert abs(np.linalg.norm(cap - c_kept) - 1.09) < 1e-9
        u = (c_removed - c_kept) / np.linalg.norm(c_removed - c_kept)
        assert np.dot((cap - c_kept) / 1.09, u) > 1 - 1e-12


def test_overlapping_ranges_rejected():
    cfg = config_from_dict({"geometry": "x", "fragments": {"ranges": ["1-4", "4-10"]}}).fragments
    with pytest.raises(ValueError, match="more than one fragment"):
        build_fragments(BUTADIENE, cfg)


# ============================================================
# === Excited states and transition data =====================
# ============================================================

@pytest.mark.parametrize("method", ["tda", "tddft"])
def test_transition_density_reproduces_tdm(tmp_path, method):
    from eecc.qm.excited import run_excited_states

    cfg = _small_config(tmp_path, "x.xyz", td={"method": method, "nstates": 3})
    s = run_excited_states(ETHYLENE, cfg, str(tmp_path / "td"))  # raises if the check fails
    assert len(s["excitation_energies_eV"]) == 3
    assert s["cartesian"] is False  # sto-3g is not a Pople split-valence basis


def test_transition_representations_consistent(tmp_path):
    from eecc.qm.excited import run_excited_states
    from eecc.qm.transition import run_transition

    cfg = _small_config(tmp_path, "x.xyz", td={"nstates": 3, "state": 2})
    run_excited_states(ETHYLENE, cfg, str(tmp_path / "td"))
    info = run_transition(str(tmp_path / "td"), cfg, str(tmp_path / "tr"), "eth")
    exact = np.array(info["exact"]["mu_au"])
    assert np.linalg.norm(exact) > 0.1  # bright state
    np.testing.assert_allclose(info["from_T"]["mu_au"], exact, atol=1e-8)
    np.testing.assert_allclose(info["tresp"]["mu_au"], exact, atol=1e-6)
    assert np.linalg.norm(np.array(info["cube"]["mu_au"]) - exact) < 0.02 * np.linalg.norm(exact)
    # Mulliken charges only approximate the dipole (about 12% off in STO-3G)
    assert np.linalg.norm(np.array(info["mulliken"]["mu_au"]) - exact) < 0.2 * np.linalg.norm(exact)
    assert abs(info["mulliken"]["sum_q"]) < 1e-10 and abs(info["tresp"]["sum_q"]) < 1e-10


def test_cube_grid_spacing_exact():
    from eecc.qm.transition import cube_grid

    origin, n, extent = cube_grid(np.array([[0.0, 0.0, 0.0], [1.3, 2.9, 0.4]]), 0.25, 3.0)
    np.testing.assert_allclose(extent / (n - 1), 0.25)
    assert np.all(origin + extent >= np.array([1.3, 2.9, 0.4]) + 3.0 - 1e-12)


# ============================================================
# === Full pipeline ==========================================
# ============================================================

def test_pipeline_non_covalent_dimer(tmp_path):
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(4.0), str(geo))
    cfg = _small_config(tmp_path, geo, td={"state": 2, "nstates": 3},
                        couplings={"methods": ["tdc_fft", "tresp", "mulliken",
                                               "point_dipole", "extended_dipole"]})
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run()

    res = json.load(open(os.path.join(pipe.stage_dir("couplings"), "couplings.json")))
    assert len(res["fragments"]) == 2 and len(res["pairs"]) == 1
    J = res["pairs"][0]["J_cm-1"]
    assert abs(res["pairs"][0]["R_Ang"] - 4.0) < 1e-6
    # identical monomers side by side (H-type): all methods agree in sign, TrESP ~ TDC
    signs = {np.sign(v) for v in J.values()}
    assert len(signs) == 1
    assert abs(J["tresp"] - J["tdc_fft"]) < 0.25 * abs(J["tdc_fft"])
    for stage in ("opt", "fragments", "couplings"):
        assert pipe.is_done(stage)
    assert all(pipe.is_done("td", n) for n in pipe.fragment_names())


def test_pipeline_covalent_with_optimization(tmp_path):
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "butadiene.xyz"
    save_xyz(BUTADIENE, str(geo))
    cfg = _small_config(
        tmp_path, geo,
        fragments={"mode": "ranges", "ranges": ["1-2,5-7", "3-4,8-10"]},
        opt={"enabled": True, "basis": "sto-3g", "grid": [50, 194], "density_fit": False},
        couplings={"methods": ["tresp", "point_dipole"]},
    )
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run()
    opt = read_xyz(os.path.join(pipe.stage_dir("opt"), "optimized.xyz"))
    assert len(opt) == 10 and os.path.exists(os.path.join(pipe.stage_dir("opt"), "trajectory.xyz"))
    frags = json.load(open(os.path.join(pipe.stage_dir("fragments"), "fragments.json")))["fragments"]
    assert [f["natoms"] for f in frags] == [6, 6]
    assert os.path.exists(os.path.join(pipe.stage_dir("couplings"), "couplings.txt"))

    # a changed TD setting invalidates td and later stages, but not opt/fragments
    cfg.td.nstates = 4
    assert pipe.is_done("opt") and pipe.is_done("fragments")
    assert not pipe.is_done("td", "frag1") and not pipe.is_done("couplings")


def test_slurm_dry_run(tmp_path):
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.slurm import submit

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, slurm={"account": "proj-1", "setup": ["module load x"]})
    scripts = submit(Pipeline(cfg), str(tmp_path / "cfg.yaml"), dry_run=True)
    assert len(scripts) == 3
    frag = open(scripts[1]).read()
    assert "#SBATCH --array=1-2" in frag and "#SBATCH -A proj-1" in frag
    assert "--stage td --fragment $SLURM_ARRAY_TASK_ID" in frag and "module load x" in frag
    prep = open(scripts[0]).read()
    assert "#SBATCH -c 1" in prep  # no optimization: the prep job only splits fragments
    assert "#SBATCH -c 32" in frag and "#SBATCH --mem=28G" in frag


def test_slurm_opt_job_overrides(tmp_path):
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.slurm import submit

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, opt={"enabled": True},
                        slurm={"account": "p", "partition": "shared", "opt_partition": "main", "opt_cpus": 128,
                               "opt_mem": "0"})
    prep, frag, _ = (open(p).read() for p in submit(Pipeline(cfg), str(tmp_path / "c.yaml"),
                                                    dry_run=True))
    assert "#SBATCH -p main" in prep and "#SBATCH -c 128" in prep and "#SBATCH --mem=0" in prep
    assert "#SBATCH -p shared" in frag and "#SBATCH -c 32" in frag


def test_cap_charge_treatment():
    from eecc.qm.couplings import treat_cap_charges

    atoms = [("C", 0, 0, 0, 0.10), ("C", 1.4, 0, 0, -0.08), ("H", 2.5, 0, 0, -0.02)]
    caps = [(2, 1)]
    merged = treat_cap_charges(atoms, caps, "merge")
    assert len(merged) == 2 and abs(merged[1][4] - (-0.10)) < 1e-12
    assert abs(sum(a[4] for a in merged)) < 1e-12
    assert len(treat_cap_charges(atoms, caps, "drop")) == 2
    assert treat_cap_charges(atoms, caps, "keep") == atoms


def test_pipeline_fragment_caps_mapping(tmp_path):
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "butadiene.xyz"
    save_xyz(BUTADIENE, str(geo))
    cfg = _small_config(tmp_path, geo, fragments={"mode": "ranges",
                                                  "ranges": ["1-2,5-7", "3-4,8-10"]})
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run(stage="opt")
    pipe.run(stage="fragments")
    # fragment 1 = parent atoms 1,2,5,6,7 (+cap on atom 2 at index 5); atom 2 is index 1
    assert pipe.fragment_caps("frag1") == [(5, 1)]
    # fragment 2 = parent atoms 3,4,8,9,10 (+cap on atom 3); atom 3 is index 0
    assert pipe.fragment_caps("frag2") == [(5, 0)]


def test_stage_hash_ignores_defaults_and_resources():
    base = {"geometry": "x.xyz", "fragments": {"mode": "auto"}}
    h = config_from_dict(base).section_hash("opt", "td")
    # spelling out a default value, or changing memory, keeps the hash
    same = config_from_dict({**base, "td": {"nstates": 10}, "opt": {"memory_mb": 9000}})
    assert same.section_hash("opt", "td") == h
    # a real change invalidates it
    assert config_from_dict({**base, "td": {"nstates": 6}}).section_hash("opt", "td") != h


def test_restamp_and_resubmit_skips_finished_stages(tmp_path):
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.slurm import submit

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, couplings={"methods": ["tresp"]}, slurm={"account": "p"})
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run()
    # simulate markers written by an older version
    for root, _, files in os.walk(pipe.root):
        if ".done" in files:
            open(os.path.join(root, ".done"), "w").write("stale\n")
    assert not pipe.is_done("opt")
    assert pipe.restamp() == ["opt", "fragments", "td", "transition", "couplings"]
    assert pipe.is_done("couplings") and pipe.is_done("td", "frag2")

    scripts = submit(pipe, str(tmp_path / "c.yaml"), dry_run=True)
    assert [os.path.basename(s) for s in scripts] == ["analysis.sh"]

    # an unfinished fragment: only its array task is resubmitted
    open(os.path.join(pipe.stage_dir("transition", "frag2"), ".done"), "w").write("stale\n")
    scripts = submit(pipe, str(tmp_path / "c.yaml"), dry_run=True)
    assert [os.path.basename(s) for s in scripts] == ["frag.sh", "analysis.sh"]
    assert "#SBATCH --array=2\n" in open(scripts[0]).read()


def test_slurm_array_spec():
    from eecc.qm.slurm import _array_spec

    assert _array_spec([1, 2, 3]) == "1-3"
    assert _array_spec([3]) == "3"
    assert _array_spec([1, 2, 4, 6, 7]) == "1-2,4,6-7"


def test_opt_convergence_settings(tmp_path):
    from eecc.qm.ground import convergence_params

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    base = {"geometry": str(geo), "fragments": {"mode": "auto"}}
    assert convergence_params(config_from_dict(base).opt) == {"convergence_set": "GAU"}
    cfg = config_from_dict({**base, "opt": {"convergence": "gau_loose", "thresholds": {"energy": 1e-5}}})
    assert convergence_params(cfg.opt) == {"convergence_set": "GAU_LOOSE", "convergence_energy": 1e-5}
    # the default leaves existing optimizations up to date; a change invalidates them
    assert config_from_dict(base).section_hash("opt") == config_from_dict(
        {**base, "opt": {"convergence": "gau"}}).section_hash("opt")
    assert cfg.section_hash("opt") != config_from_dict(base).section_hash("opt")
    with pytest.raises(ValueError, match="opt.convergence"):
        config_from_dict({**base, "opt": {"convergence": "sloppy"}})
    with pytest.raises(ValueError, match="opt.thresholds"):
        config_from_dict({**base, "opt": {"thresholds": {"force": 1e-3}}})


def test_cartesian_rule_follows_gaussian():
    from eecc.qm.pyscf_setup import use_cartesian

    assert use_cartesian("6-31g(d)", None) and use_cartesian("6-31+G*", None)
    assert not use_cartesian("6-311g(d,p)", None)
    assert not use_cartesian("def2-svp", None) and not use_cartesian("sto-3g", None)
    assert use_cartesian("def2-svp", True)


def test_changed_defaults_invalidate_old_results():
    """Stages computed under an old default must not count as current."""
    import hashlib

    def stored_before_change(section, values):  # hash written by the code before the change
        payload = json.dumps({section: values}, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    base = {"geometry": "x.xyz", "fragments": {"mode": "auto"}}
    # couplings computed with the old defaults (cap charges kept, periodic FFT)
    assert config_from_dict(base).section_hash("couplings") != stored_before_change("couplings", {})
    explicit_old = {**base, "couplings": {"cap_charges": "keep", "tdc_boundary": "periodic"}}
    assert config_from_dict(explicit_old).section_hash("couplings") == stored_before_change("couplings", {})
    # 6-311G used Cartesian d functions under the original rule, spherical ones now
    # (other TD settings pinned to their old defaults to isolate the basis rule)
    old_td = {"response_grid": None, "davidson_tol": 1e-5}
    tz = config_from_dict({**base, "td": {"basis": "6-311g(d,p)", **old_td}})
    assert tz.section_hash("td") != stored_before_change("td", {"basis": "6-311g(d,p)"})
    pople = config_from_dict({**base, "td": {"basis": "6-31g(d)", **old_td}})
    assert pople.section_hash("td") == stored_before_change("td", {})


def test_frequencies_and_maxsteps_do_not_invalidate_later_stages(tmp_path):
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    base = _small_config(tmp_path, geo, opt={"enabled": True})
    other = _small_config(tmp_path, geo, opt={"enabled": True, "freq": True, "maxsteps": 7})
    p, q = Pipeline(base), Pipeline(other)
    assert p.expected_hash("td") == q.expected_hash("td")
    assert p.expected_hash("opt") != q.expected_hash("opt")  # the opt stage itself reruns for frequencies


def test_rerun_invalidates_dependent_stages(tmp_path):
    from eecc.qm.pipeline import PER_FRAGMENT, STAGES, Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    pipe = Pipeline(_small_config(tmp_path, geo), log=lambda *a: None)
    for st in STAGES:
        for frag in (["frag1", "frag2"] if st in ("td", "transition") else [None]):
            os.makedirs(pipe.stage_dir(st, frag), exist_ok=True)
            open(os.path.join(pipe.stage_dir(st, frag), ".done"), "w").write(pipe.expected_hash(st) + "\n")
    pipe._mark_done("td", "frag1")  # e.g. after --stage td --fragment 1 --force
    assert pipe.is_done("td", "frag1") and pipe.is_done("td", "frag2")
    assert not pipe.is_done("transition", "frag1") and pipe.is_done("transition", "frag2")
    assert not pipe.is_done("couplings")
    pipe._mark_done("fragments")
    assert not any(pipe.is_done(st, f) for st in ("td", "transition") for f in ("frag1", "frag2"))


def test_optimization_restarts_only_from_matching_trajectory(tmp_path):
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    pipe = Pipeline(_small_config(tmp_path, geo, opt={"enabled": True}), log=lambda *a: None)
    d = pipe.stage_dir("opt")
    os.makedirs(d)
    traj = os.path.join(d, "trajectory.xyz")
    open(traj, "w").write("frames of another run\n")
    pipe._prepare_trajectory(d, force=False)  # no stamp: unknown origin
    assert not os.path.exists(traj)
    open(traj, "w").write("frames of this run\n")
    pipe._prepare_trajectory(d, force=False)  # stamp matches: resume
    assert os.path.exists(traj)
    save_xyz(_stacked_dimer(4.5), str(geo))  # new input geometry
    pipe._prepare_trajectory(d, force=False)
    assert not os.path.exists(traj)
    open(traj, "w").write("frames\n")
    pipe._prepare_trajectory(d, force=True)
    assert not os.path.exists(traj)


def test_unconverged_optimization_leaves_no_optimized_geometry(tmp_path):
    from eecc.qm.ground import optimize_geometry

    cfg = _small_config(tmp_path, tmp_path / "x.xyz",
                        opt={"enabled": True, "basis": "sto-3g", "grid": [50, 194],
                             "density_fit": False, "maxsteps": 1, "disp": None})
    distorted = Structure(BUTADIENE.symbols, BUTADIENE.coords * 1.1)
    with pytest.raises(RuntimeError, match="did not converge"):
        optimize_geometry(distorted, cfg, str(tmp_path / "opt"))
    assert not os.path.exists(tmp_path / "opt" / "optimized.xyz")


def test_auto_fragments_use_input_connectivity():
    """Molecules pushed within bonding distance by an optimization stay separate."""
    from eecc.qm.config import FragmentsConfig

    far, close = _stacked_dimer(4.0), _stacked_dimer(1.2)
    cfg = FragmentsConfig(mode="auto")
    assert len(build_fragments(close, cfg, topology=far)) == 2
    with pytest.raises(ValueError, match="single connected"):
        build_fragments(close, cfg)


def test_slurm_scripts_tolerate_unset_variables_in_setup(tmp_path):
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.slurm import submit

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, slurm={"account": "p", "setup": ["conda activate eecc"]})
    for path in submit(Pipeline(cfg), str(tmp_path / "c.yaml"), dry_run=True):
        text = open(path).read()
        assert "set -eo pipefail" in text and "set -u" not in text and "-euo" not in text


def test_fast_tddft_settings_are_new_defaults():
    """Response grid, Davidson tolerance and physical cores; TD results from before stay distinguishable."""
    import hashlib

    base = {"geometry": "x.xyz", "fragments": {"mode": "auto"}}
    legacy = hashlib.sha256(json.dumps({"td": {}}, sort_keys=True).encode()).hexdigest()[:16]
    assert config_from_dict(base).section_hash("td") != legacy
    old = {**base, "td": {"response_grid": None, "davidson_tol": 1e-5}}
    assert config_from_dict(old).section_hash("td") == legacy
    with pytest.raises(ValueError, match="response_grid"):
        config_from_dict({**base, "td": {"response_grid": [75]}})


def test_slurm_physical_cores_hint(tmp_path):
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.slurm import submit

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    on = submit(Pipeline(_small_config(tmp_path, geo, slurm={"account": "p"})), str(tmp_path / "c.yaml"), dry_run=True)
    assert all("#SBATCH --hint=nomultithread" in open(p).read() for p in on)
    cfg = _small_config(tmp_path, geo, slurm={"account": "p", "physical_cores": False})
    off = submit(Pipeline(cfg), str(tmp_path / "c.yaml"), dry_run=True)
    assert not any("nomultithread" in open(p).read() for p in off)


def test_diabatization_recovers_fragment_states_and_davydov_coupling(tmp_path):
    """Stacked ethylene dimer: LE diabats match the fragment S1 and the Davydov splitting."""
    from eecc.qm.diabatize import diabatize_from_pipeline
    from eecc.qm.excited import HARTREE_TO_EV
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.pyscf_setup import build_mol, build_rks

    dimer = _stacked_dimer(4.0)
    geo = tmp_path / "dimer.xyz"
    save_xyz(dimer, str(geo))
    cfg = _small_config(tmp_path, geo, td={"nstates": 3}, couplings={"methods": ["tresp"]})
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run()

    mol = build_mol(dimer, "sto-3g", None, 0, 0, 2000, verbose=0)
    mf = build_rks(mol, cfg.td.xc, cfg.td.grid, False, 1e-9)
    mf.kernel()
    mf.grids.atom_grid = tuple(cfg.td.response_grid); mf.grids.build()
    td = mf.TDDFT(); td.nstates = 6; td.conv_tol = 1e-6; td.kernel()
    E = np.asarray(td.e) * HARTREE_TO_EV
    x = np.array([xy[0] for xy in td.xy]); y = np.array([xy[1] for xy in td.xy])

    res = diabatize_from_pipeline(pipe, mol, mf.mo_coeff, mf.mo_occ, x, y, E, include_ct=False)
    e_frag = json.load(open(os.path.join(pipe.stage_dir("td", "frag1"), "td.json")))["selected"]["energy_eV"]
    assert np.all(res.completeness > 0.9)
    assert abs(res.H_eV[0, 0] - res.H_eV[1, 1]) < 1e-3  # symmetric dimer
    assert abs(res.H_eV[0, 0] - e_frag) < 0.05
    # the LE block reproduces the energies of the two adiabatic states carrying the LE character
    # (the Davydov pair), so its coupling is half their splitting
    pair = np.argsort(-np.sum(res.D ** 2, axis=1))[:2]
    assert np.allclose(np.linalg.eigvalsh(res.H_eV), np.sort(E[pair]), atol=0.02)
    # with the adiabatic states themselves as targets, dipoles and f are PySCF's (checks units and prefactor)
    from eecc.qm.diabatize import diabatic_dipoles, diabatic_oscillator_strengths, diabatize
    ident = diabatize(x + y, E, x + y, [str(i) for i in range(len(E))])
    mu_id = diabatic_dipoles(ident, td.transition_dipole())
    assert np.allclose(np.abs(mu_id), np.abs(td.transition_dipole()), atol=1e-6)
    assert np.allclose(diabatic_oscillator_strengths(ident, mu_id), td.oscillator_strength(), atol=1e-6)
    res_ct = diabatize_from_pipeline(pipe, mol, mf.mo_coeff, mf.mo_occ, x, y, E, include_ct=True)
    assert res_ct.H_eV.shape == (4, 4) and res_ct.labels[2] == "CT(frag1->frag2)"


def test_diabatic_dipoles_rotate_with_the_states():
    """mu_k = sum_I D_Ik mu_I: a permutation of the states permutes the dipoles, a rotation keeps sum |mu|^2."""
    from eecc.qm.diabatize import diabatic_dipoles, diabatize

    rng = np.random.default_rng(1)
    A = np.linalg.qr(rng.normal(size=(12, 4)))[0].T.reshape(4, 3, 4)  # orthonormal adiabatic amplitudes
    E, mu = np.array([1.0, 1.2, 1.5, 2.0]), rng.normal(size=(4, 3))
    res = diabatize(A, E, A[[2, 0, 3, 1]], list("abcd"))
    assert np.allclose(diabatic_dipoles(res, mu), mu[[2, 0, 3, 1]])
    R = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    res = diabatize(A, E, np.einsum("kI,Iab->kab", R, A), list("abcd"))
    mu_d = diabatic_dipoles(res, mu)
    assert np.allclose(mu_d, R @ mu) and np.isclose(np.sum(mu_d ** 2), np.sum(mu ** 2))
    assert np.allclose(np.linalg.eigvalsh(res.H_eV), E)
    with pytest.raises(ValueError, match="one transition dipole"):
        diabatic_dipoles(res, mu.T[:, :3])


def test_fragment_geometry_and_state_types():
    from eecc.qm.system import fragment_geometry, state_types

    g = fragment_geometry(ETHYLENE, len(ETHYLENE))
    xyz = np.asarray(ETHYLENE.coords)
    assert np.allclose(g["center_ang"], xyz[:2].mean(axis=0), atol=1e-9)  # symmetric: between the carbons
    axes = np.array(g["principal_axes"])
    assert np.allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    cc = (xyz[1] - xyz[0]) / np.linalg.norm(xyz[1] - xyz[0])
    assert abs(abs(axes[0] @ cc) - 1) < 1e-9  # smallest moment: along the C=C bond
    normal = np.cross(xyz[2] - xyz[0], xyz[3] - xyz[0])
    assert abs(abs(axes[2] @ normal / np.linalg.norm(normal)) - 1) < 1e-9  # largest moment: the plane normal
    assert np.all(np.diff(g["moments_amu_ang2"]) > 0)
    # caps are excluded: a heavy atom appended after the own atoms changes nothing
    capped = Structure(ETHYLENE.symbols + ["Cl"], np.vstack([xyz, [3.0, 1.0, 2.0]]))
    assert np.allclose(fragment_geometry(capped, len(ETHYLENE))["center_ang"], g["center_ang"])
    assert np.allclose(fragment_geometry(capped, len(ETHYLENE))["principal_axes"], axes)
    assert not np.allclose(fragment_geometry(capped, len(capped))["center_ang"], g["center_ang"])
    # sign rule on a tie: an axis at 45 degrees gets its first component positive whatever the roundoff
    square = Structure(["C"] * 4, np.array([[1, 1, 0], [-1, -1, 0], [2, -2, 0], [-2, 2, 0]], float))
    assert np.all(np.array(fragment_geometry(square, 4)["principal_axes"])[:2, 0] > 0)
    assert state_types(["a", "b"], ["LE(a)", "LE(b)", "CT(b->a)"]) == [
        {"label": "LE(a)", "type": "LE", "fragment": "a"}, {"label": "LE(b)", "type": "LE", "fragment": "b"},
        {"label": "CT(b->a)", "type": "CT", "donor": "b", "acceptor": "a"}]


def test_system_stage_dependencies_and_hashes(tmp_path):
    """The whole-system run depends only on the geometry and TD method, not on the fragments."""
    from eecc.qm.pipeline import PER_FRAGMENT, STAGES, Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    pipe = Pipeline(_small_config(tmp_path, geo, system={"enabled": True}), log=lambda *a: None)
    def stamp_all():
        for st in STAGES:
            for frag in (["frag1", "frag2"] if st in PER_FRAGMENT else [None]):
                os.makedirs(pipe.stage_dir(st, frag), exist_ok=True)
                open(os.path.join(pipe.stage_dir(st, frag), ".done"), "w").write(pipe.expected_hash(st) + "\n")
    stamp_all()
    pipe._mark_done("couplings")
    assert pipe.is_done("system") and pipe.is_done("diabatize")
    pipe._mark_done("td", "frag1")
    assert pipe.is_done("system") and not pipe.is_done("diabatize")
    stamp_all()
    pipe._mark_done("fragments")
    assert pipe.is_done("system") and not pipe.is_done("diabatize")
    stamp_all()
    pipe._mark_done("opt")
    assert not pipe.is_done("system")

    h_sys, h_dia = pipe.expected_hash("system"), pipe.expected_hash("diabatize")
    pipe.cfg.td.nstates = 4  # fragment states only
    pipe.cfg.system.include_ct = False  # diabatization only
    assert pipe.expected_hash("system") == h_sys and pipe.expected_hash("diabatize") != h_dia
    pipe.cfg.td.xc = "b3lyp"
    assert pipe.expected_hash("system") != h_sys
    # system.enabled does not change results, so toggling it keeps finished stages
    h_sys, h_dia = pipe.expected_hash("system"), pipe.expected_hash("diabatize")
    pipe.cfg.system.enabled = False
    assert pipe.expected_hash("system") == h_sys and pipe.expected_hash("diabatize") == h_dia

    # with fragment files the whole-system geometry is a separate input and must be hashed
    for name in ("a.xyz", "b.xyz"):
        save_xyz(ETHYLENE, str(tmp_path / name))
    frag_files = [str(tmp_path / "a.xyz"), str(tmp_path / "b.xyz")]
    files = Pipeline(_small_config(tmp_path, geo, fragments={"mode": "files", "files": frag_files},
                                   system={"enabled": True}), log=lambda *a: None)
    h_sys = files.expected_hash("system")
    save_xyz(_stacked_dimer(5.0), str(geo))
    assert files.expected_hash("system") != h_sys


def test_slurm_jobs_for_system_stage(tmp_path):
    from eecc.qm.pipeline import Pipeline
    from eecc.qm.slurm import submit

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, system={"enabled": True}, slurm={"account": "p"})
    scripts = submit(Pipeline(cfg), str(tmp_path / "c.yaml"), dry_run=True)
    assert [os.path.basename(p) for p in scripts] == ["prep.sh", "frag.sh", "analysis.sh", "system.sh", "diabatize.sh"]
    system = open(scripts[3]).read()
    assert "#SBATCH -p " not in system and "#SBATCH -c 128" in system and "--stage system" in system
    assert "--stage diabatize" in open(scripts[4]).read()
    with pytest.raises(ValueError, match="system.enabled"):
        config_from_dict({"fragments": {"mode": "files", "files": ["a.xyz", "b.xyz"]}, "system": {"enabled": True}})


def test_slurm_diabatize_waits_for_couplings(tmp_path, monkeypatch):
    """The diabatization reads couplings.json, so its job must wait for the analysis job."""
    import subprocess
    from eecc.qm import slurm
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, system={"enabled": True}, slurm={"account": "p"})
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{100 + len(calls)}\n")
    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    jobs = dict(line.split() for line in slurm.submit(Pipeline(cfg), str(tmp_path / "c.yaml")))
    dia = next(c for c in calls if c[-1].endswith("diabatize.sh"))
    deps = set(dia[2].split(":", 1)[1].split(":"))
    assert {jobs["system"], jobs["frag"], jobs["analysis"]} <= deps


def test_pipeline_with_system_stage(tmp_path):
    """End to end: fragments, whole-system TDDFT and the exciton Hamiltonian next to the Coulomb couplings."""
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(4.0), str(geo))
    cfg = _small_config(tmp_path, geo, td={"nstates": 3}, couplings={"methods": ["tresp", "tdc_direct"]},
                        system={"enabled": True, "nstates": 6, "include_ct": False})
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run()
    res = json.load(open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.json")))
    assert res["labels"] == ["LE(frag1)", "LE(frag2)"] and min(res["completeness"]) > 0.9
    # documented exciton-model format: version, units, state types, fragment centres and axes
    from eecc.qm.pipeline import DIABATIZE_FORMAT
    assert res["format_version"] == int(DIABATIZE_FORMAT) and "H_eV" in res["units"]
    assert res["states"] == [{"label": "LE(frag1)", "type": "LE", "fragment": "frag1"},
                             {"label": "LE(frag2)", "type": "LE", "fragment": "frag2"}]
    c1, c2 = (np.array(f["center_ang"]) for f in res["fragments"])
    assert np.allclose(c2 - c1, [4.0, 0.0, 0.0], atol=1e-6)  # the stacking vector of _stacked_dimer
    # diabatic transition dipoles: equal for the two sites of the symmetric dimer and close to the
    # fragment's (here a dark state, so an absolute tolerance)
    mu = np.linalg.norm(res["transition_dipoles_au"], axis=1)
    mu_frag = np.linalg.norm(json.load(open(os.path.join(pipe.stage_dir("td", "frag1"), "td.json")))["selected"]["mu_au"])
    assert abs(mu[0] - mu[1]) < 1e-3 * mu[0] + 1e-4 and abs(mu[0] - mu_frag) < 0.15 * mu_frag + 0.01
    assert len(res["oscillator_strengths"]) == 2 and "Transition dipoles of the diabatic states" in \
        open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.txt")).read()
    pair = res["pairs"][0]
    assert {"J_total_cm-1", "tresp_cm-1", "tdc_direct_cm-1"} <= set(pair)
    assert "Total coupling (this stage) vs Coulomb" in open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.txt")).read()
    assert pipe.is_done("system") and pipe.is_done("diabatize")
    assert all(s["converged"] for s in json.load(open(os.path.join(pipe.stage_dir("system"), "system_td.json")))["states"])

    # results of the previous output format (no diabatic dipoles; system_td.json without 'converged'):
    # not current anywhere (is_done, restamp), and rerun without errors
    import eecc.qm.pipeline as pipeline_module
    path = os.path.join(pipe.stage_dir("diabatize"), "diabatic.json")
    old = {k: v for k, v in json.load(open(path)).items()
           if k not in ("transition_dipoles_au", "oscillator_strengths", "format_version")}
    json.dump(old, open(path, "w"))
    spath = os.path.join(pipe.stage_dir("system"), "system_td.json")
    sys_td = json.load(open(spath))
    for st in sys_td["states"]:
        del st["converged"]
    json.dump(sys_td, open(spath, "w"))
    fmt = pipeline_module.DIABATIZE_FORMAT
    try:
        pipeline_module.DIABATIZE_FORMAT = "1"
        pipe._mark_done("diabatize")
    finally:
        pipeline_module.DIABATIZE_FORMAT = fmt
    assert not pipe.is_done("diabatize") and not pipe._outputs_present("diabatize")
    assert "diabatize" not in pipe.restamp()
    pipe.run_diabatize()
    assert "transition_dipoles_au" in json.load(open(path)) and pipe.is_done("diabatize")

    # too few system states for LE + CT: fall back to LE only instead of failing after the system run
    from eecc.qm.diabatize import diabatize_from_pipeline
    from eecc.qm.pyscf_setup import build_mol

    npz = os.path.join(pipe.stage_dir("system"), "system_td.npz")
    d = dict(np.load(npz))
    mol = build_mol(read_xyz(os.path.join(pipe.stage_dir("system"), "system.xyz")), cfg.td.basis, cfg.td.cart,
                    0, 0, 2000, verbose=0)
    le = diabatize_from_pipeline(pipe, mol, d["mo_coeff"], d["mo_occ"], d["x"], d["y"], d["energies_eV"],
                                 include_ct=False)
    keep = np.sort(np.argsort(-np.sum(le.D ** 2, axis=1))[:3])  # the 3 states that carry the LE states
    np.savez(npz, **{k: (v[keep] if k in ("energies_eV", "osc", "tdm_au", "x", "y") else v) for k, v in d.items()})
    pipe.cfg.system.include_ct = True
    # stale couplings (marker removed, file left behind) must not be listed as the Coulomb couplings
    os.remove(os.path.join(pipe.stage_dir("couplings"), ".done"))
    pipe.run_diabatize(force=True)
    res = json.load(open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.json")))
    txt = open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.txt")).read()
    assert res["labels"] == ["LE(frag1)", "LE(frag2)"] and "LE states only" in txt
    assert "tresp_cm-1" not in res["pairs"][0] and "couplings stage not finished" in txt


# ============================================================
# === Vibronic parameters ====================================
# ============================================================

FORMALDEHYDE = Structure(["C", "O", "H", "H"],
                         np.array([[0, 0, 0], [0, 0, 1.21], [0, 0.94, -0.59], [0, -0.94, -0.59]], float))


def test_huang_rhys_of_a_displaced_oscillator():
    """Exact harmonic model: an excited state displaced by Delta gives S_k = omega_k (L_k . M^1/2 Delta)^2 / 2."""
    from eecc.qm.vibronic import AMU_TO_ME, _masses_amu, huang_rhys, normal_modes

    rng = np.random.default_rng(1)
    st = FORMALDEHYDE
    m = np.repeat(_masses_amu(st.symbols) * AMU_TO_ME, 3)
    # a translation- and rotation-free mass-weighted Hessian with known frequencies (atomic units)
    probe = normal_modes(st.symbols, st.coords, np.eye(12))  # any Hessian: gives the internal space
    L0 = probe["modes"]
    w_true = np.array([1200.0, 1300.0, 1600.0, 1800.0, 2900.0, 3000.0]) / 219474.6313632
    Hm = L0 @ np.diag(w_true ** 2) @ L0.T
    H = Hm * np.sqrt(np.outer(m, m))
    nm = normal_modes(st.symbols, st.coords, H)
    assert np.allclose(nm["omega_au"], w_true, rtol=1e-8)

    delta = rng.normal(size=12) * 0.02  # Bohr; translations/rotations in it must not matter
    delta += np.tile([0.3, -0.2, 0.1], 4)  # a pure translation
    grad = -(H @ delta)  # gradient at x = 0 of 1/2 (x - delta)^T H (x - delta) - 1/2 x^T H x
    res = huang_rhys(st.symbols, st.coords, H, grad.reshape(4, 3))
    d_true = np.sqrt(w_true) * (nm["modes"].T @ (np.sqrt(m) * delta))
    assert np.allclose(np.abs(res["d"]), np.abs(d_true), rtol=1e-8)
    assert np.allclose(res["S"], d_true ** 2 / 2, rtol=1e-8)
    assert np.allclose(res["freq_cm"], w_true * 219474.6313632)
    # freq_scale scales the frequencies (and the threshold applies to them), not S
    scaled = huang_rhys(st.symbols, st.coords, H, grad.reshape(4, 3), min_freq_cm=1250.0, freq_scale=0.95)
    assert np.allclose(scaled["freq_cm"], 0.95 * res["freq_cm"])
    assert list(scaled["valid"]) == [False, False, True, True, True, True]  # 1235 cm-1 < 1250 after scaling
    assert np.allclose(scaled["S"][2:], res["S"][2:])
    assert _masses_amu(["B"])[0] == pytest.approx(11.0093, abs=1e-4)  # most common isotope, as Gaussian


def test_vibronic_summary():
    from eecc.qm.vibronic import KB_CM, summarize

    freq = np.array([30.0, 200.0, 400.0, 1200.0, 1600.0])
    S = np.array([5.0, 0.5, 0.2, 0.3, 0.6])
    valid = freq >= 50
    s = summarize(freq, S, valid, cutoff_cm=800, temperature_K=0.0)
    assert s["S_eff"] == pytest.approx(0.9) and s["omega_eff_cm"] == pytest.approx((0.3 * 1200 + 0.6 * 1600) / 0.9)
    assert s["reorganization_high_cm"] == pytest.approx(0.3 * 1200 + 0.6 * 1600)
    assert s["reorganization_low_cm"] == pytest.approx(0.5 * 200 + 0.2 * 400)  # the 30 cm-1 mode is excluded
    assert s["sigma_low_cm"] == pytest.approx(np.sqrt(0.5 * 200 ** 2 + 0.2 * 400 ** 2))
    assert (s["n_high"], s["n_low"], s["n_excluded"]) == (2, 2, 1) and s["strong_low_modes"] == []
    hot = summarize(freq, S, valid, cutoff_cm=800, temperature_K=300.0)
    coth = 1 / np.tanh(freq[1:3] / (2 * KB_CM * 300))
    assert hot["sigma_low_cm"] == pytest.approx(np.sqrt((S[1:3] * freq[1:3] ** 2 * coth).sum()))
    assert summarize(freq, S * 4, valid)["strong_low_modes"] == [{"freq_cm": 200.0, "S": 2.0}]


@pytest.mark.parametrize("method", ["tda", "tddft"])
def test_vibronic_gradient_matches_finite_differences(tmp_path, method):
    """The projected excitation-energy gradient equals the slope of the excitation energy along each mode."""
    from pyscf import dft, gto
    from pyscf.hessian import thermo

    from eecc.qm.vibronic import _compute, _masses_amu, huang_rhys

    geo = tmp_path / "x.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, td={"basis": "6-31g", "xc": "b3lyp", "method": method, "nstates": 3})
    data = _compute(FORMALDEHYDE, cfg, str(tmp_path), "h2co", 0, None, None)
    res = huang_rhys(FORMALDEHYDE.symbols, FORMALDEHYDE.coords, data["hessian"],
                     data["grad_excited"] - data["grad_ground"])
    mol = gto.M(atom=FORMALDEHYDE.pyscf_atoms(), basis="6-31g", unit="Angstrom", verbose=0)
    ref = np.sort(np.real(thermo.harmonic_analysis(mol, data["hessian"],
                                                   mass=_masses_amu(FORMALDEHYDE.symbols))["freq_wavenumber"]))
    assert np.allclose(res["freq_cm"], ref, atol=0.1)

    from eecc.qm.vibronic import normal_modes
    nm = normal_modes(FORMALDEHYDE.symbols, FORMALDEHYDE.coords, data["hessian"])

    def excitation(coords_ang):
        m = gto.M(atom=list(zip(FORMALDEHYDE.symbols, coords_ang)), basis="6-31g", unit="Angstrom", verbose=0)
        mf = dft.RKS(m, xc="b3lyp")
        mf.grids.atom_grid = (50, 194)
        mf.conv_tol = 1e-11
        mf.kernel()
        td = mf.TDA() if method == "tda" else mf.TDDFT()
        td.nstates, td.conv_tol = 3, 1e-9
        td.kernel()
        return td.e[0]

    h = 0.5  # mass-weighted step (sqrt(me) Bohr); small next to the zero-point amplitude (~20)
    k = int(np.nanargmax(res["S"]))  # the C=O stretch
    step = (nm["modes"][:, k] / nm["sqrt_m"]).reshape(4, 3) * h * 0.529177210903  # Angstrom
    fd = (excitation(FORMALDEHYDE.coords + step) - excitation(FORMALDEHYDE.coords - step)) / (2 * h)
    assert res["g_au"][k] == pytest.approx(fd, rel=2e-3)
    assert res["S"][k] > 0.5  # n -> pi*: the C=O stretch is strongly displaced
    out_of_plane = [i for i in range(6) if np.allclose(nm["modes"][:, i].reshape(4, 3)[:, 1:], 0, atol=1e-6)]
    assert out_of_plane and all(res["S"][i] < 1e-8 for i in out_of_plane)  # symmetry: not displaced


def test_vibronic_eri_choice():
    from eecc.qm.vibronic import eri_incore

    assert eri_incore("auto", 640, 200000, hessian=False) is None  # PySCF's choice before the Hessian
    assert eri_incore("auto", 100, 16000, hessian=True) is None  # 0.1 GB: PySCF's choice
    assert eri_incore("auto", 640, 200000, hessian=True) is False  # 168 GB of 200 GB: the Hessian would not fit
    assert eri_incore("incore", 640, 1000, hessian=True) and eri_incore("direct", 10, 1e9, hessian=False) is False


def test_vibronic_unreadable_npz_means_recompute(tmp_path):
    from eecc.qm.vibronic import _load_npz, _save_npz

    p = str(tmp_path / "x.npz")
    _save_npz(p, {"calc_key": np.array("k"), "a": np.arange(1000.0)})
    assert _load_npz(p, "k") is not None and _load_npz(p, "other") is None
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) // 2])  # truncated, e.g. a file system problem
    assert _load_npz(p, "k") is None


def test_vibronic_resumes_after_a_killed_hessian(tmp_path, monkeypatch):
    """Gradients are saved before the Hessian; a rerun (here with direct integrals) skips them."""
    from pyscf.grad import rks as grad_rks
    from pyscf.hessian import rks as hess_rks

    from eecc.qm import vibronic

    geo = tmp_path / "x.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, td={"basis": "6-31g", "xc": "b3lyp", "method": "tda", "nstates": 3})
    from pyscf.grad import tdrks as grad_tdrks
    from pyscf.tdscf import rhf as td_rhf

    hess_kernel = hess_rks.Hessian.kernel
    held = []

    def checked_hessian(self, *a, **k):  # whether the integrals are in memory during the Hessian
        held.append(self.base._eri is not None)
        return hess_kernel(self, *a, **k)

    monkeypatch.setattr(hess_rks.Hessian, "kernel", checked_hessian)
    cfg.vibronic.eri = "incore"
    ref = vibronic.run_vibronic(FORMALDEHYDE, cfg, str(tmp_path / "ref"), "h2co")
    ref_raw = dict(np.load(tmp_path / "ref" / "vibronic.npz"))
    assert not os.path.exists(tmp_path / "ref" / vibronic.PARTIAL)
    assert held == [True]

    work = str(tmp_path / "run")
    cfg.vibronic.eri = "direct"
    monkeypatch.setattr(hess_rks.Hessian, "kernel", lambda self, *a, **k: (_ for _ in ()).throw(MemoryError("killed")))
    with pytest.raises(MemoryError):
        vibronic.run_vibronic(FORMALDEHYDE, cfg, work, "h2co")
    assert os.path.exists(os.path.join(work, vibronic.PARTIAL))
    assert not os.path.exists(os.path.join(work, "vibronic.npz"))

    monkeypatch.setattr(hess_rks.Hessian, "kernel", checked_hessian)
    for cls in (grad_rks.Gradients, grad_tdrks.Gradients, td_rhf.TDA):
        monkeypatch.setattr(cls, "kernel", lambda *a, **k: pytest.fail("recomputed a step before the Hessian"))
    out = vibronic.run_vibronic(FORMALDEHYDE, cfg, work, "h2co")
    assert held == [True, False]  # direct: no in-memory integrals during the Hessian
    assert os.path.exists(os.path.join(work, "vibronic_previous.log"))
    raw = dict(np.load(os.path.join(work, "vibronic.npz")))
    assert not os.path.exists(os.path.join(work, vibronic.PARTIAL))
    # direct == in-memory integrals, up to integral screening and SCF convergence (~1e-5 relative)
    assert np.allclose(raw["hessian"], ref_raw["hessian"], rtol=1e-4, atol=1e-5)
    assert np.allclose(raw["grad_excited"], ref_raw["grad_excited"], atol=1e-6)
    assert np.allclose(out["modes"]["freq_cm"], ref["modes"]["freq_cm"], atol=0.1)
    assert out["summary"]["S_eff"] == pytest.approx(ref["summary"]["S_eff"], rel=1e-3)


def test_pipeline_with_vibronic_stage(tmp_path, monkeypatch):
    """vibronic stage per fragment, its parameters in diabatic.json, and cheap reruns of the summary."""
    from eecc.qm import vibronic
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(4.0), str(geo))
    cfg = _small_config(tmp_path, geo, td={"nstates": 3}, couplings={"methods": ["tresp"]},
                        system={"enabled": True, "nstates": 6, "include_ct": False}, vibronic={"enabled": True})
    pipe = Pipeline(cfg, log=lambda *a: None)
    pipe.run()
    for name in ("frag1", "frag2"):
        assert pipe.is_done("vibronic", name)
        out = json.load(open(os.path.join(pipe.stage_dir("vibronic", name), "vibronic.json")))
        assert len(out["modes"]["freq_cm"]) == 12 and out["summary"]["n_modes"] == 12
    res = json.load(open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.json")))
    assert [set(f["vibronic"]) >= {"S_eff", "omega_eff_cm", "sigma_low_cm"} for f in res["fragments"]] == [True] * 2
    s1, s2 = (f["vibronic"] for f in res["fragments"])
    assert s1["S_eff"] == pytest.approx(s2["S_eff"], rel=1e-4)  # identical monomers
    assert "08_vibronic" in pipe.stage_dir("vibronic")  # existing work dirs keep their numbers

    # a new cutoff reruns the summary (and the diabatization), not the quantum chemistry
    monkeypatch.setattr(vibronic, "_compute", lambda *a, **k: pytest.fail("recomputed the Hessian"))
    pipe.cfg.vibronic.cutoff = 100.0
    assert not pipe.is_done("vibronic", "frag1") and not pipe.is_done("diabatize")
    pipe.run()
    res = json.load(open(os.path.join(pipe.stage_dir("diabatize"), "diabatic.json")))
    assert res["fragments"][0]["vibronic"]["cutoff_cm"] == 100.0
    assert pipe.is_done("diabatize")
    assert pipe.restamp()[-2:] == ["vibronic", "diabatize"] and pipe.is_done("diabatize")

    # --force recomputes the quantum chemistry instead of reusing vibronic.npz
    calls = []
    monkeypatch.setattr(vibronic, "_compute", lambda *a, **k: calls.append(1) or dict(
        np.load(os.path.join(pipe.stage_dir("vibronic", "frag1"), "vibronic.npz"))))
    pipe.run(stage="vibronic", fragment=1, force=True)
    assert calls == [1] and not pipe.is_done("diabatize")
    # with the stage disabled, running it by hand leaves the diabatization alone
    pipe.run_diabatize()
    marker = os.path.join(pipe.stage_dir("diabatize"), ".done")
    assert os.path.exists(marker)
    pipe.cfg.vibronic.enabled = False
    pipe.run(stage="vibronic", fragment=1, force=True)
    assert os.path.exists(marker)
    with pytest.raises(ValueError, match="min_frequency"):
        _small_config(tmp_path, geo, vibronic={"min_frequency": 0.0})


def test_slurm_jobs_for_vibronic_stage(tmp_path, monkeypatch):
    import subprocess
    from eecc.qm import slurm
    from eecc.qm.pipeline import Pipeline

    geo = tmp_path / "dimer.xyz"
    save_xyz(_stacked_dimer(), str(geo))
    cfg = _small_config(tmp_path, geo, system={"enabled": True}, vibronic={"enabled": True},
                        slurm={"account": "p"})
    scripts = slurm.submit(Pipeline(cfg), str(tmp_path / "c.yaml"), dry_run=True)
    vib = next(p for p in scripts if p.endswith("vib.sh"))
    text = open(vib).read()
    assert "--array=1-2" in text and "--stage vibronic --fragment $SLURM_ARRAY_TASK_ID" in text
    assert "#SBATCH -t 24:00:00" in text
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=f"{100 + len(calls)}\n")
    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    jobs = dict(line.split() for line in slurm.submit(Pipeline(cfg), str(tmp_path / "c.yaml")))
    dia = next(c for c in calls if c[-1].endswith("diabatize.sh"))
    assert jobs["vib"] in dia[2] and jobs["prep"] in next(c for c in calls if c[-1].endswith("vib.sh"))[2]

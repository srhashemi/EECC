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
                        slurm={"account": "p", "opt_partition": "main", "opt_cpus": 128,
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

"""The diabatic.json format (eecc.formats): a real file passes, broken ones are caught, and 'eecc paths'."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest

from eecc.formats import DIABATIC_FORMAT, OLDEST_FORMAT, check_diabatic

DATA = Path(__file__).resolve().parent / "data"


def real():
    """diabatic.json of the BODIPY dimer from 'eecc run' (format 5, LE + CT, vibronic stage)."""
    return json.loads((DATA / "diabatic_bodipy_dimer.json").read_text())


def test_real_file_passes():
    d = real()
    assert d["format_version"] == DIABATIC_FORMAT
    assert check_diabatic(d) == []


def test_pipeline_writes_this_format():
    from eecc.qm.pipeline import DIABATIZE_FORMAT
    assert int(DIABATIZE_FORMAT) == DIABATIC_FORMAT


def test_older_formats():
    d = real()
    for f in d["fragments"]:  # format 4: no grouped modes
        del f["vibronic"]["modes_grouped"]
    assert any("modes_grouped" in p for p in check_diabatic(d))
    d["format_version"] = 4
    assert check_diabatic(d) == []
    assert any("format 4" in p for p in check_diabatic({**d, "format_version": 3}))  # vibronic is format 4
    d3 = copy.deepcopy(d)
    for f in d3["fragments"]:
        del f["vibronic"]
    assert check_diabatic({**d3, "format_version": OLDEST_FORMAT}) == []


@pytest.mark.parametrize("change, message", [
    (lambda d: d.pop("H_eV"), "missing key 'H_eV'"),
    (lambda d: d.__setitem__("format_version", 2), "outside"),
    (lambda d: d.__setitem__("format_version", DIABATIC_FORMAT + 1), "outside"),
    (lambda d: d.__setitem__("format_version", "5"), "not an integer"),
    (lambda d: d["H_eV"].pop(), "H_eV must be"),
    (lambda d: d["H_eV"][0].__setitem__(1, float("nan")), "H_eV must be"),
    (lambda d: d["transition_dipoles_au"][0].pop(), "transition_dipoles_au"),
    (lambda d: d["completeness"].pop(), "completeness"),
    (lambda d: d["states"][0].__setitem__("type", "XX"), "not 'LE' or 'CT'"),
    (lambda d: d["states"][0].__setitem__("fragment", "nope"), "not in fragments"),
    (lambda d: d["states"][2].__setitem__("donor", "nope"), "donor"),
    (lambda d: d["states"][1].__setitem__("label", "LE(x)"), "differs from labels"),
    (lambda d: d["fragments"][0].pop("center_ang"), "lacks 'center_ang'"),
    (lambda d: d["fragments"][0]["vibronic"].pop("S_eff"), "lacks 'S_eff'"),
    (lambda d: d["fragments"][0]["vibronic"].__setitem__("omega_eff_eV", -1.0), "omega_eff_eV"),
    (lambda d: d["fragments"][0]["vibronic"]["modes_grouped"][0].pop("S"), "modes_grouped[0]"),
    (lambda d: d["fragments"][0]["vibronic"]["modes_grouped"][0].__setitem__("omega_eV", 0.0), "omega_eV > 0"),
])
def test_broken_files_are_caught(change, message):
    d = real()
    change(d)
    problems = check_diabatic(d)
    assert any(message in p for p in problems), problems


def test_not_an_object():
    assert check_diabatic([]) == ["not a JSON object"]


def test_paths_command(tmp_path, capsys, monkeypatch):
    pytest.importorskip("yaml")
    from eecc.cli import main
    cfg = tmp_path / "config.yaml"
    cfg.write_text("geometry: x.xyz\nworkdir: out\nfragments:\n  ranges: ['1-3', '4-6']\n")
    monkeypatch.setattr(sys, "argv", ["eecc", "paths", str(cfg)])
    main()
    out = json.loads(capsys.readouterr().out)
    assert out["workdir"] == os.path.join(str(tmp_path), "out")
    assert out["diabatic_json"] == os.path.join(out["stages"]["diabatize"], "diabatic.json")
    assert out["diabatic_json_exists"] is False and out["diabatize_done"] is False
    monkeypatch.setattr(sys, "argv", ["eecc", "paths", str(tmp_path / "missing.yaml")])
    with pytest.raises(SystemExit, match="error in"):
        main()

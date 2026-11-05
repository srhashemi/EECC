"""Tests for eecc.cli argument parsing."""

import os
import sys
import pytest


def test_cli_help(capsys, monkeypatch):
    """--help should succeed and print subcommands."""
    from eecc.cli import main

    monkeypatch.setattr(sys, "argv", ["eecc", "--help"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "tdc-two-monomers" in captured.out
    assert "intramolecular" in captured.out
    assert "check-charge" in captured.out


def test_cli_version(capsys, monkeypatch):
    """--version should print the package version and exit."""
    from eecc import __version__
    from eecc.cli import main

    monkeypatch.setattr(sys, "argv", ["eecc", "--version"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == f"eecc {__version__}"


def test_cli_no_args(capsys, monkeypatch):
    """No args should print help and exit 1."""
    from eecc.cli import main

    monkeypatch.setattr(sys, "argv", ["eecc"])
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1


def test_find_input_prefers_inputs_dir_then_path_as_given(tmp_path, monkeypatch):
    """Cube/charge commands: inputs/<name> first (as in 1.x), else the path as given."""
    from eecc.workflows import find_input

    monkeypatch.chdir(tmp_path)
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "a.cub").write_text("")
    (tmp_path / "a.cub").write_text("")
    (tmp_path / "b.cub").write_text("")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "c.cub").write_text("")

    assert find_input("a.cub") == os.path.join("inputs", "a.cub")  # inputs/ wins, as before
    assert find_input("b.cub") == "b.cub"
    assert find_input(os.path.join("data", "c.cub")) == os.path.join("data", "c.cub")
    absolute = str(tmp_path / "b.cub")
    assert find_input(absolute) == absolute
    with pytest.raises(FileNotFoundError, match="missing.cub.*inputs/"):
        find_input("missing.cub")


@pytest.mark.parametrize("command", ["view", "plot-density"])
def test_viz_commands_without_matplotlib_say_what_to_install(command, monkeypatch):
    from eecc.cli import main

    monkeypatch.setitem(sys.modules, "matplotlib", None)  # import matplotlib -> ImportError
    argv = ["eecc", command, "a.cub"] + (["b.cub"] if command == "plot-density" else [])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert "pip install 'eecc[viz]'" in str(exc_info.value.code)

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
    (tmp_path / "inputs" / "a.cub").write_text("", encoding="utf-8")
    (tmp_path / "a.cub").write_text("", encoding="utf-8")
    (tmp_path / "b.cub").write_text("", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "c.cub").write_text("", encoding="utf-8")

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
    assert 'pip install "eecc[viz]"' in str(exc_info.value.code)  # double quotes work in cmd.exe too


def test_python_m_eecc():
    """python -m eecc works where pip's script folder is not on PATH (common on Windows)."""
    import subprocess
    from eecc import __version__

    out = subprocess.run([sys.executable, "-m", "eecc", "--version"], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == f"eecc {__version__}"


def test_redirected_output_is_utf8(monkeypatch):
    """Output redirected to a file on Windows would use e.g. cp1252, which has no Delta: UTF-8 instead."""
    import io
    from eecc.cli import _utf8_output

    out, err = (io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n") for _ in range(2))
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    _utf8_output()
    print("\u0394origin = 1 \u00c5")  # Delta, Angstrom
    out.flush()
    assert out.buffer.getvalue().decode("utf-8") == "\u0394origin = 1 \u00c5\n"


def test_text_files_are_opened_as_utf8():
    """Every text file EECC reads or writes is UTF-8 (Windows would use its local code page)."""
    import ast
    import pathlib
    import eecc

    missing = []
    for path in sorted(pathlib.Path(eecc.__file__).parent.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call) or any(
                    k.arg == "encoding" and getattr(k.value, "value", None) in ("utf-8", "utf-8-sig")
                    for k in node.keywords):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id == "open":
                mode = node.args[1] if len(node.args) > 1 else next(
                    (k.value for k in node.keywords if k.arg == "mode"), None)
                text = mode is None or "b" not in getattr(mode, "value", "")
            else:
                text = isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text")
            if text:
                missing.append(f"{path.name}:{node.lineno}")
    assert not missing, f"open() without encoding='utf-8': {missing}"

"""Tests for eecc.cli argument parsing."""

import sys
import pytest


def test_cli_help(capsys):
    """--help should succeed and print subcommands."""
    from eecc.cli import main

    sys.argv = ["eecc", "--help"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "tdc-two-monomers" in captured.out
    assert "intramolecular" in captured.out
    assert "check-charge" in captured.out


def test_cli_version(capsys):
    """--version should print version and exit."""
    from eecc.cli import main

    sys.argv = ["eecc", "--version"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "1.0.0" in captured.out


def test_cli_no_args(capsys):
    """No args should print help and exit 1."""
    from eecc.cli import main

    sys.argv = ["eecc"]
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 1

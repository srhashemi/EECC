"""Workflow runners for intramolecular, intermolecular, and TDC calculations."""

from __future__ import annotations

import os

INPUTS_DIR = "inputs"


def find_input(name: str) -> str:
    """Path of an input file of the cube and charge commands.

    ``inputs/<name>`` when that file exists (the repository layout, as in EECC 1.x), otherwise
    ``name`` itself (a path relative to the current directory, or absolute).
    """
    in_dir = os.path.join(INPUTS_DIR, name)
    if os.path.isfile(in_dir):
        return in_dir
    if os.path.isfile(name):
        return name
    raise FileNotFoundError(f"{name}: no such file (looked inside {INPUTS_DIR}/ and as the path given)")

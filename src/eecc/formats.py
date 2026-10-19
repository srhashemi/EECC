"""The ``diabatic.json`` file written by the diabatize stage of ``eecc run``: its format version and a check.

Other programs read this file (VESPA builds its exciton model from it), so its layout is defined here, in one
place: ``DIABATIC_FORMAT`` is the version EECC writes and ``check_diabatic`` lists what a file of any supported
version (3 to ``DIABATIC_FORMAT``) lacks. EECC checks every file before writing it; a reader can call
``check_diabatic`` on a file it got, or compare its own supported versions with ``DIABATIC_FORMAT``.

Format history: 3 adds format_version, units, conventions, state types, fragment centres and axes;
4 adds fragments[].vibronic; 5 adds fragments[].vibronic.modes_grouped.

Needs only the standard library.
"""

import math
from numbers import Real

DIABATIC_FORMAT = 5
OLDEST_FORMAT = 3  # files before this lack the state types and units

TOP_KEYS = ("format_version", "units", "conventions", "labels", "states", "H_eV", "completeness",
            "transition_dipoles_au", "oscillator_strengths", "fragments", "adiabatic_eV", "pairs")
FRAGMENT_KEYS = ("name", "natoms_own", "center_ang", "principal_axes", "moments_amu_ang2")
VIBRONIC_KEYS = ("cutoff_cm", "temperature_K", "S_eff", "omega_eff_cm", "omega_eff_eV", "reorganization_cm",
                 "reorganization_eV", "reorganization_low_cm", "sigma_low_cm", "sigma_low_eV")
GROUP_KEYS = ("omega_cm", "omega_eV", "S", "n_modes")


def _number(x) -> bool:
    return isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x)


def _matrix(x, rows, cols) -> bool:
    return (isinstance(x, list) and len(x) == rows
            and all(isinstance(r, list) and len(r) == cols and all(_number(v) for v in r) for r in x))


def _vector(x, n) -> bool:
    return isinstance(x, list) and len(x) == n and all(_number(v) for v in x)


def check_diabatic(d) -> list:
    """Problems of a parsed ``diabatic.json`` (empty when it is a valid file of format 3 to DIABATIC_FORMAT)."""
    if not isinstance(d, dict):
        return ["not a JSON object"]
    v = d.get("format_version")
    if not isinstance(v, int) or isinstance(v, bool):
        return [f"format_version is {v!r}, not an integer"]
    if not OLDEST_FORMAT <= v <= DIABATIC_FORMAT:
        return [f"format_version {v} is outside {OLDEST_FORMAT}-{DIABATIC_FORMAT} (this EECC writes "
                f"{DIABATIC_FORMAT})"]
    problems = [f"missing key '{k}'" for k in TOP_KEYS if k not in d]
    if problems:
        return problems
    labels, states = d["labels"], d["states"]
    if not (isinstance(labels, list) and labels and all(isinstance(x, str) for x in labels)):
        return ["labels must be a non-empty list of strings"]
    n = len(labels)
    if not _matrix(d["H_eV"], n, n):
        problems.append(f"H_eV must be a {n} x {n} matrix of numbers")
    if not _matrix(d["transition_dipoles_au"], n, 3):
        problems.append(f"transition_dipoles_au must be {n} rows of 3 numbers")
    for k in ("completeness", "oscillator_strengths"):
        if not _vector(d[k], n):
            problems.append(f"{k} must be {n} numbers")
    if not (isinstance(d["adiabatic_eV"], list) and all(_number(x) for x in d["adiabatic_eV"])):
        problems.append("adiabatic_eV must be a list of numbers")
    for k in ("units", "conventions"):
        if not isinstance(d[k], dict):
            problems.append(f"{k} must be an object")
    if not isinstance(d["pairs"], list):
        problems.append("pairs must be a list")

    frags = d["fragments"]
    if not (isinstance(frags, list) and all(isinstance(f, dict) for f in frags)):
        return problems + ["fragments must be a list of objects"]
    names = [f.get("name") for f in frags]
    if not isinstance(states, list) or len(states) != n:
        problems.append(f"states must be {n} objects, one per label")
    else:
        for i, (s, lab) in enumerate(zip(states, labels)):
            if not isinstance(s, dict):
                problems.append(f"states[{i}] is not an object")
            elif s.get("label") != lab:
                problems.append(f"states[{i}].label {s.get('label')!r} differs from labels[{i}] {lab!r}")
            elif s.get("type") == "LE":
                if s.get("fragment") not in names:
                    problems.append(f"states[{i}] (LE) names fragment {s.get('fragment')!r}, not in fragments")
            elif s.get("type") == "CT":
                for role in ("donor", "acceptor"):
                    if s.get(role) not in names:
                        problems.append(f"states[{i}] (CT) {role} {s.get(role)!r} is not in fragments")
            else:
                problems.append(f"states[{i}].type is {s.get('type')!r}, not 'LE' or 'CT'")
    for i, f in enumerate(frags):
        problems += [f"fragments[{i}] lacks '{k}'" for k in FRAGMENT_KEYS if k not in f]
        vib = f.get("vibronic")
        if vib is None:
            continue
        if v < 4:
            problems.append(f"fragments[{i}].vibronic in a format {v} file (added in format 4)")
            continue
        if not isinstance(vib, dict):
            problems.append(f"fragments[{i}].vibronic is not an object")
            continue
        problems += [f"fragments[{i}].vibronic lacks '{k}'" for k in VIBRONIC_KEYS if k not in vib]
        for k in ("S_eff", "sigma_low_eV"):  # null when the fragment has no modes above the cutoff
            if vib.get(k) is not None and not (_number(vib[k]) and vib[k] >= 0):
                problems.append(f"fragments[{i}].vibronic.{k} must be a number >= 0 or null")
        w = vib.get("omega_eff_eV")
        if w is not None and not (_number(w) and w > 0):
            problems.append(f"fragments[{i}].vibronic.omega_eff_eV must be a number > 0 or null")
        if v >= 5:
            groups = vib.get("modes_grouped")
            if not isinstance(groups, list):
                problems.append(f"fragments[{i}].vibronic.modes_grouped must be a list (format 5)")
                continue
            for j, g in enumerate(groups):
                if not (isinstance(g, dict) and all(k in g for k in GROUP_KEYS)):
                    problems.append(f"fragments[{i}].vibronic.modes_grouped[{j}] needs {', '.join(GROUP_KEYS)}")
                elif not (_number(g["omega_eV"]) and g["omega_eV"] > 0 and _number(g["S"]) and g["S"] >= 0):
                    problems.append(f"fragments[{i}].vibronic.modes_grouped[{j}] needs omega_eV > 0 and S >= 0")
    return problems

# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Shared constants + NeoN-dict helpers for the incompressibleVoFNeon models.

``ALPHA1_FIELD`` is the single definition of the phase-fraction field name the
solver reads/advects — interFoam derives it from the ``phases`` entry in
``transportProperties``; this solver is (for now) fixed to the water phase, and
everything that needs the name (field reads, scheme keys, solver-dict lookups)
takes it from here so a future generalisation has one seam.

The ``read_*`` helpers tolerate the NeoN dictionary's strict typed getters
(``cAlpha 1`` parses as an int, so ``get_double`` raises ``bad any_cast``;
OpenFOAM switches may be stored as words). The alpha solver-dict lookups
encode the regex-key contract in one place: the user's fvSolution subdict is
keyed by an OpenFOAM regex (damBreak: ``"alpha.water.*"``) which the NeoN
dictionary stores verbatim (no regex matching), while ``create_fields``
registers a MULES-key-stripped copy under the exact key for the MULESCorr
predictor's linear solver.
"""

from typing import Any, Optional

# Phase-1 volume fraction field (interFoam's alpha1 == "alpha.water").
ALPHA1_FIELD = "alpha.water"


def read_int(d: Any, key: str, default: int) -> int:
    return int(d.get_int(key)) if d.contains(key) else default


def read_float(d: Any, key: str, default: float) -> float:
    """Read a scalar, tolerating int storage (``cAlpha 1`` parses as an int, so
    ``get_double`` would raise ``bad any_cast``)."""
    if not d.contains(key):
        return default
    try:
        return float(d.get_double(key))
    except Exception:
        return float(d.get_int(key))


def read_switch(d: Any, key: str, default: bool) -> bool:
    """Read an OpenFOAM on/off switch, tolerating word or bool storage."""
    if d is None or not d.contains(key):
        return default
    try:
        return bool(d.get_bool(key))
    except Exception:
        return d.get_string(key).strip().lower() in ("yes", "true", "on", "1")


def prefixed_alpha_solver_key(solvers: Any) -> Optional[str]:
    """The user's regex-keyed alpha solver key (e.g. ``"alpha.water.*"``).

    Any key that starts with ``ALPHA1_FIELD`` but is not the bare exact name —
    the exact key is the stripped predictor copy ``create_fields`` registers.
    Returns ``None`` when the case has no such key.
    """
    return next(
        (k for k in solvers.keys() if k.startswith(ALPHA1_FIELD) and k != ALPHA1_FIELD),
        None,
    )


def alpha_solver_dict(rt: Any) -> Any:
    """The alpha-controls subdict carrying the MULES controls, or ``None``.

    PREFERS the regex-keyed user subdict (which still carries
    ``MULESCorr``/``nAlphaCorr``/``cAlpha``/``nLimiterIter``), falling back to
    the exact ``ALPHA1_FIELD`` key only if it is the sole match.
    """
    solvers = rt.fv_solution_dict.subDict("solvers")
    key = prefixed_alpha_solver_key(solvers)
    if key is None and solvers.contains(ALPHA1_FIELD):
        key = ALPHA1_FIELD
    return solvers.subDict(key) if key is not None else None

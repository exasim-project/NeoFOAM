# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence test bindings over the shared case-discovery helpers.

One turbulence family (``momentumTransportModel``) now holds every model; a
single ``fallback`` flag picks the backend
(:func:`~neofoam.turbulence.selection.select_turbulence_model`). Each shipped
case declares in its ``expected.yaml`` whether its model is ``native``
(registered — a dual native+fallback model) or ``unregistered`` (not in the
family, so selection raises). The generic case discovery lives in
``test/_case_discovery.py``; only the domain constants are bound here.

OpenFOAM / pybFoam is a hard requirement of NeoFOAM, so there is no "skip if
OpenFOAM missing" gating.
"""

from pathlib import Path

# Importing the package registers the bundled models (laminar / kEpsilon / … ).
import neofoam.turbulence  # noqa: F401

from _case_discovery import Case, discover_cases
from _case_discovery import case_for as _case_for

#: Self-contained OpenFOAM cases shipped with the turbulence tests (each holds a
#: real ``constant/turbulenceProperties`` plus an ``expected.yaml`` manifest).
CASES = discover_cases(Path(__file__).resolve().parent / "cases")


def case_for(model_name: str) -> Case:
    """Return the native turbulence case for *model_name* (see :func:`_case_discovery.case_for`)."""
    return _case_for(CASES, model_name)

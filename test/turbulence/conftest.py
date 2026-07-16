# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Turbulence test bindings over the shared case-discovery helpers.

The discovery / selection / assertion machinery lives in
``test/_case_discovery.py``; here it is bound to the turbulence domain. Models
are built exactly the way ``create_fields.create_turbulence`` builds them: select
from the case's real ``constant/turbulenceProperties`` and wrap a native model in
the small :class:`SpecMomentumTransport` read interface over its ``ModelRuntime``
(which owns ``nut`` and registers a stress computer).

OpenFOAM / pybFoam is a hard requirement of NeoFOAM, so there is no
"skip if OpenFOAM missing" gating here and the registry is used as-is (no
throwaway registration to clean up between tests).
"""

from pathlib import Path

from neofoam.turbulence.config import TurbulencePropertiesConfig
from neofoam.turbulence.fallback import OpenFOAMTurbulenceModel
from neofoam.turbulence.momentumTransport import SpecMomentumTransport
from neofoam.turbulence.selection import select_turbulence_model

# Importing the package registers the bundled native models (laminar).
import neofoam.turbulence  # noqa: F401

from _case_discovery import (
    Case,
    discover_cases,
    make_assert_selection,
    make_build_as_solver,
)
from _case_discovery import case_for as _case_for

#: Self-contained OpenFOAM cases shipped with the turbulence tests (each holds a
#: real ``constant/turbulenceProperties`` plus an ``expected.yaml`` manifest).
CASES = discover_cases(Path(__file__).resolve().parent / "cases")

build_as_solver = make_build_as_solver(
    TurbulencePropertiesConfig, select_turbulence_model, SpecMomentumTransport
)
assert_selection = make_assert_selection(OpenFOAMTurbulenceModel)


def case_for(model_name: str) -> Case:
    """Return the native turbulence case for *model_name* (see :func:`_case_discovery.case_for`)."""
    return _case_for(CASES, model_name)

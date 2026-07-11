# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Viscosity test bindings over the shared case-discovery helpers.

The discovery / selection / assertion machinery lives in
``test/_case_discovery.py``; here it is bound to the viscosity domain. Models are
built exactly the way ``create_fields.create_viscosity`` builds them: select from
the case's real ``constant/transportProperties`` and read ``nu`` from that dict.
A native model is a plain ``ModelRuntime`` that owns its ``nu`` field, so the
native-wrap is the identity.

OpenFOAM / pybFoam is a hard requirement of NeoFOAM, so there is no
"skip if OpenFOAM missing" gating here and the registry is used as-is (no
throwaway registration to clean up between tests).
"""

from pathlib import Path

from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.fallback import OpenFOAMViscosityModel
from neofoam.viscosity.selection import select_viscosity_model

# Importing the package registers the bundled native models (Newtonian).
import neofoam.viscosity  # noqa: F401

from _case_discovery import (
    Case,
    discover_cases,
    make_assert_selection,
    make_build_as_solver,
)
from _case_discovery import case_for as _case_for

#: Self-contained OpenFOAM cases shipped with the viscosity tests (each holds a
#: real ``constant/transportProperties`` plus an ``expected.yaml`` manifest).
CASES = discover_cases(Path(__file__).resolve().parent / "cases")

#: A native viscosity model is a plain ``ModelRuntime`` used as-is (no native-wrap).
build_as_solver = make_build_as_solver(
    TransportPropertiesConfig, select_viscosity_model
)
assert_selection = make_assert_selection(OpenFOAMViscosityModel)


def case_for(model_name: str) -> Case:
    """Return the native viscosity case for *model_name* (see :func:`_case_discovery.case_for`)."""
    return _case_for(CASES, model_name)

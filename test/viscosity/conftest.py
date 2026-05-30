# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Local helpers for the viscosity test suite — case discovery + solver-faithful build.

Everything here is concrete to viscosity (no generic plugin abstraction yet).
Cases are *discovered* by globbing ``cases/*/`` for an ``expected.yaml``
manifest, so adding a case directory extends coverage with zero test-module
edits. Expected values come from those manifests, never from literals baked into
test bodies.

Models are built exactly the way the incompressibleFluid solver builds them in
``create_fields.create_viscosity``: select from the case's real
``constant/transportProperties`` and read ``nu`` from that dict — never by
hand-injecting a fabricated constructor argument.

OpenFOAM / pybFoam is a hard requirement of NeoFOAM, so there is no
"skip if OpenFOAM missing" gating here and the registry is used as-is (no
throwaway registration to clean up between tests).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from neofoam.framework.model import ModelSpec
from neofoam.viscosity.config import TransportPropertiesConfig
from neofoam.viscosity.fallback import OpenFOAMViscosityModel
from neofoam.viscosity.models.newtonian import NewtonianModel
from neofoam.viscosity.selection import select_viscosity_model

# Importing the package registers the bundled native models (Newtonian).
import neofoam.viscosity  # noqa: F401

#: Self-contained OpenFOAM cases shipped with the viscosity tests. Each holds a
#: real ``constant/transportProperties`` dictionary plus an ``expected.yaml``
#: manifest (no dict content is encoded in the test modules).
CASES_DIR = Path(__file__).resolve().parent / "cases"

#: Kinematic-viscosity dimensions [0 2 -1 0 0 0 0] — mirrors
#: ``create_fields._NU_DIMENSIONS``.
_NU_DIMENSIONS = (0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Case:
    """A discovered viscosity case and its expectation manifest."""

    name: str  # directory name → parametrize id
    path: Path  # case dir (holds constant/transportProperties + expected.yaml)
    config: dict[str, Any]  # manifest["config"]    — expected parsed dict
    selection: dict[str, Any]  # manifest["selection"] — {model_name, resolves_to}
    model: Optional[dict[str, Any]]  # manifest.get("model") — per-model runtime values


def discover_cases() -> list[Case]:
    """Glob ``cases/*/expected.yaml`` and load each into a :class:`Case`."""
    cases: list[Case] = []
    for manifest in sorted(CASES_DIR.glob("*/expected.yaml")):
        data = yaml.safe_load(manifest.read_text())
        cases.append(
            Case(
                name=manifest.parent.name,
                path=manifest.parent,
                config=data["config"],
                selection=data["selection"],
                model=data.get("model"),
            )
        )
    return cases


#: Module-level so it can be used directly in ``@pytest.mark.parametrize``.
CASES = discover_cases()


def case_for(model_name: str) -> Case:
    """Return the native case whose ``selection.model_name`` is ``model_name``.

    Points a registered native model at the case the solver would feed it.
    """
    for case in CASES:
        if (
            case.selection["resolves_to"] == "native"
            and case.selection["model_name"] == model_name
        ):
            return case
    raise LookupError(f"no native case for model {model_name!r}")


def _read_nu(case_dir: Path) -> Any:
    """Build ``nu`` from ``constant/transportProperties`` (replica of the solver).

    Self-contained replica of the solver-private ``create_fields._read_nu``
    (plan Q3a): the pybFoam transport binding exposes no ``nu()``, so the native
    Newtonian model is fed a ``dimensionedScalar`` built from the same dict
    entry OpenFOAM reads. Folds back into a shared helper once
    ``newtonian.load`` is implemented.
    """
    import pybFoam as pyf

    transport_dict = pyf.dictionary.read(
        str(case_dir / "constant" / "transportProperties")
    )
    return pyf.dimensionedScalar(
        pyf.Word("nu"), pyf.dimensionSet(*_NU_DIMENSIONS), transport_dict
    )


def build_as_solver(case: Case) -> Any:
    """Initialize the viscosity model exactly as ``create_viscosity`` does.

    Loads the real config, selects the model, and — for the native Newtonian —
    builds ``NewtonianModel`` from ``nu`` read out of the case dict. For
    non-native cases the selector's fallback adapter is returned (unbuilt), the
    same object the solver wires its raw transport behind.
    """
    cfg = TransportPropertiesConfig.load(case_dir=case.path)
    selected = select_viscosity_model(cfg)
    if isinstance(selected, ModelSpec) and selected.name == "Newtonian":
        return NewtonianModel(_read_nu(case.path))
    return selected


def assert_selection(selected: Any, case: Case) -> None:
    """Assert ``selected`` matches the case manifest's ``selection`` block."""
    resolves_to = case.selection["resolves_to"]
    if resolves_to == "native":
        assert isinstance(selected, ModelSpec)
        assert selected.name == case.selection["model_name"]
    elif resolves_to == "fallback":
        assert isinstance(selected, OpenFOAMViscosityModel)
    else:
        raise AssertionError(f"unknown resolves_to: {resolves_to!r}")

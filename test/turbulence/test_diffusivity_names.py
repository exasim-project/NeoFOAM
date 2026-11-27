# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""The native NeoN diffusivities are named after the laplacian keys the operations declare.

NeoN's laplacian looks its scheme up as ``laplacian(<gamma.name>,<field.name>)``. An
unnamed ``interpolate`` result is called ``interpolated_<source>`` (``interpolated_tmp``
for an arithmetic source), a key no ``fvSchemes`` declares, so each model passes the
declared name at the call site. A mismatch only shows on a case whose
``laplacianSchemes`` has no ``default``; this checks it without a NeoN runtime, by reading
the ``name=`` literals from the model sources. The run under ``default none`` is
``test/solver/incompressibleFluidNeoN/test_laplacian_default_none.py``.
"""

import ast
import importlib
import inspect
from typing import Any

import pytest

from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.pimpleAlgorithm import (
    PimpleNeoNFvSchemes,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.simpleAlgorithm import (
    SimpleNeoNFvSchemes,
)
from neofoam.turbulence.models.kEpsilon import KEpsilonFvSchemes
from neofoam.turbulence.models.kOmegaSST import KOmegaSSTFvSchemes
from neofoam.turbulence.models.spalartAllmaras import SpalartAllmarasFvSchemes

#: Each model's ``interpolate`` result names and the laplacian key each one feeds.
MODELS = [
    ("laminar", None, {"nuEff": "laplacian(nuEff,U)"}),
    (
        "kEpsilon",
        KEpsilonFvSchemes,
        {
            "nuEff": "laplacian(nuEff,U)",
            "DkEff": "laplacian(DkEff,k)",
            "DepsilonEff": "laplacian(DepsilonEff,epsilon)",
        },
    ),
    (
        "kOmegaSST",
        KOmegaSSTFvSchemes,
        {
            "nuEff": "laplacian(nuEff,U)",
            "DkEff": "laplacian(DkEff,k)",
            "DomegaEff": "laplacian(DomegaEff,omega)",
        },
    ),
    (
        "spalartAllmaras",
        SpalartAllmarasFvSchemes,
        {"nuEff": "laplacian(nuEff,U)", "DnuTildaEff": "laplacian(DnuTildaEff,nuTilda)"},
    ),
]
IDS = [module for module, _, _ in MODELS]

#: The momentum predictors that consume ``nuEff``.
MOMENTUM_SCHEMES = (PimpleNeoNFvSchemes, SimpleNeoNFvSchemes)


def _interpolation_names(module: str) -> set[Any]:
    """The ``name=`` literal of every ``.interpolate(...)`` call; ``None`` for an unnamed one."""
    source = inspect.getsource(importlib.import_module(f"neofoam.turbulence.models.{module}"))
    names: set[Any] = set()
    for node in ast.walk(ast.parse(source)):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "interpolate"
        ):
            continue
        named = [kw.value for kw in node.keywords if kw.arg == "name"]
        names.add(named[0].value if named and isinstance(named[0], ast.Constant) else None)
    return names


def _laplacian_keys(schemes: Any) -> set[str]:
    section = schemes.model_fields["laplacianSchemes"].annotation
    return {field.alias for field in section.model_fields.values()}


@pytest.mark.parametrize(("module", "schemes", "expected"), MODELS, ids=IDS)
def test_model_names_every_interpolation(
    module: str, schemes: Any, expected: dict[str, str]
) -> None:
    assert _interpolation_names(module) == set(expected)


@pytest.mark.parametrize(("module", "schemes", "expected"), MODELS, ids=IDS)
def test_interpolation_names_are_declared_laplacian_keys(
    module: str, schemes: Any, expected: dict[str, str]
) -> None:
    declared = set.intersection(*(_laplacian_keys(m) for m in MOMENTUM_SCHEMES))
    if schemes is not None:
        declared |= _laplacian_keys(schemes)
    assert set(expected.values()) <= declared

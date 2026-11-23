# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Spec for how the application solvers declare their typed ``fvSchemes`` and ``fvSolution``."""

from typing import Any, Optional

import pytest

from neofoam.solver.incompressibleFluid.models.pressure_velocity.pimpleAlgorithm import (
    PimpleFvSolution,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.pimpleAlgorithm import (
    PimpleNeoNFvSchemes,
    PimpleNeoNFvSolution,
)
from neofoam.solver.incompressibleFluidNeoN.models.pressure_velocity.simpleAlgorithm import (
    SimpleNeoNFvSchemes,
    SimpleNeoNFvSolution,
)


@pytest.mark.parametrize(
    "spec_slice", [PimpleNeoNFvSolution, SimpleNeoNFvSolution], ids=lambda c: c.__name__
)
def test_neon_slice_declares_the_ginkgo_backend(spec_slice: Any) -> None:
    assert (spec_slice._backend, spec_slice.__name__.endswith("NeoN_fvSolution")) == (
        "ginkgo",
        True,
    )


@pytest.mark.parametrize(
    ("spec_slice", "section", "expected"),
    [
        (
            PimpleNeoNFvSchemes,
            "divSchemes",
            {"div(phi,U)": ("U", "main"), "div((nuEff*dev2(T(grad(U)))))": ("U", "advanced")},
        ),
        (PimpleNeoNFvSchemes, "ddtSchemes", {"ddt(U)": ("U", "main")}),
        (PimpleNeoNFvSchemes, "gradSchemes", {"grad(U)": ("U", "main"), "grad(p)": ("p", "main")}),
        (
            PimpleNeoNFvSchemes,
            "laplacianSchemes",
            {"laplacian(nuEff,U)": ("U", "main"), "laplacian(rAUf,p)": ("p", "main")},
        ),
        (PimpleNeoNFvSchemes, "snGradSchemes", {"snGrad(p)": ("p", "advanced")}),
        (
            PimpleNeoNFvSchemes,
            "interpolationSchemes",
            {
                "flux(U)": ("U", "advanced"),
                "flux(HbyA)": ("p", "advanced"),
                "interpolate(rAU)": ("p", "advanced"),
                "dotInterpolate(S,U_0)": ("p", "advanced"),
            },
        ),
        (
            SimpleNeoNFvSchemes,
            "divSchemes",
            {"div(phi,U)": ("U", "main"), "div((nuEff*dev2(T(grad(U)))))": ("U", "advanced")},
        ),
        (SimpleNeoNFvSchemes, "gradSchemes", {"grad(U)": ("U", "main"), "grad(p)": ("p", "main")}),
        (
            SimpleNeoNFvSchemes,
            "laplacianSchemes",
            {"laplacian(nuEff,U)": ("U", "main"), "laplacian(rAUf,p)": ("p", "main")},
        ),
        (SimpleNeoNFvSchemes, "snGradSchemes", {"snGrad(p)": ("p", "advanced")}),
        (
            SimpleNeoNFvSchemes,
            "interpolationSchemes",
            {"flux(HbyA)": ("p", "advanced"), "interpolate(rAU)": ("p", "advanced")},
        ),
    ],
)
def test_scheme_keys_declare_field_and_level(
    spec_slice: Any, section: str, expected: dict[str, tuple[str, str]]
) -> None:
    schema = spec_slice.model_json_schema()
    ref = schema["properties"][section]["$ref"].rsplit("/", 1)[1]
    props = schema["$defs"][ref]["properties"]
    assert {key: (prop["nfField"], prop["nfLevel"]) for key, prop in props.items()} == expected


@pytest.mark.parametrize(
    ("spec_slice", "alias", "expected"),
    [
        (PimpleNeoNFvSolution, "U", {"nfField": "U"}),
        (PimpleNeoNFvSolution, "UFinal", {"nfField": "U", "nfFinalOf": "U"}),
        (PimpleNeoNFvSolution, "p", {"nfField": "p"}),
        (PimpleNeoNFvSolution, "pFinal", {"nfField": "p", "nfFinalOf": "p"}),
        (SimpleNeoNFvSolution, "UFinal", {"nfField": "U", "nfFinalOf": "U"}),
        (SimpleNeoNFvSolution, "pFinal", {"nfField": "p", "nfFinalOf": "p"}),
    ],
)
def test_solver_entries_declare_their_field_group(
    spec_slice: Any, alias: str, expected: dict[str, str]
) -> None:
    schema = spec_slice.model_json_schema()
    ref = schema["properties"]["solvers"]["$ref"].rsplit("/", 1)[1]
    prop = schema["$defs"][ref]["properties"][alias]
    assert {key: value for key, value in prop.items() if key.startswith("nf")} == expected


@pytest.mark.parametrize(
    ("spec_slice", "section", "levels"),
    [
        (
            PimpleNeoNFvSolution,
            "PIMPLE",
            {
                "nOuterCorrectors": None,
                "nCorrectors": None,
                "nNonOrthogonalCorrectors": "advanced",
                "pRefCell": "advanced",
                "pRefValue": "advanced",
            },
        ),
        (
            SimpleNeoNFvSolution,
            "SIMPLE",
            {
                "nNonOrthogonalCorrectors": "advanced",
                "pRefCell": "advanced",
                "pRefValue": "advanced",
            },
        ),
    ],
)
def test_control_entries_declare_their_level(
    spec_slice: Any, section: str, levels: dict[str, Optional[str]]
) -> None:
    schema = spec_slice.model_json_schema()
    ref = schema["properties"][section]["$ref"].rsplit("/", 1)[1]
    props = schema["$defs"][ref]["properties"]
    assert {key: prop.get("nfLevel") for key, prop in props.items()} == levels


def test_pybfoam_slice_uses_the_openfoam_plugins() -> None:
    entry = PimpleFvSolution._pending_sections["solvers"]["p"][0]
    block = entry.model_validate({"solver": "PCG", "preconditioner": "DIC"})
    assert type(block.root).__name__ == "FoamPCG"

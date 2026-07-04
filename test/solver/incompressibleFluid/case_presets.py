# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Predefined, parameterized case data for incompressibleFluid solver tests.

Test-only helper (not shipped): a single source of "known-good, complete" config
data so tests don't re-spell the cavity config set inline. A preset returns the
full ``BaseConfig`` set a case needs — ``controlDict``, ``fvSchemes`` /
``fvSolution``, ``transport`` / ``turbulence``, and the ``0/`` field configs
(``U``, ``p``) with BCs — built from ``configurations(incompressibleFluid)``.

The mesh is out of scope (presets describe configuration, not geometry); patch
names are exposed via :data:`CAVITY_PATCHES` so a test's ``blockMeshDict`` matches.
"""

from __future__ import annotations

from neofoam.framework.solver.configurations import configurations
from neofoam.io.base import BaseConfig
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

# Boundary patch names the cavity preset expects on the mesh.
CAVITY_PATCHES = ("movingWall", "fixedWalls", "frontAndBack")


def _uniform_vector(v: tuple[float, float, float]) -> str:
    return f"uniform ({v[0]} {v[1]} {v[2]})"


def lid_driven_cavity(
    *,
    nu: float = 0.01,
    lid_velocity: tuple[float, float, float] = (1.0, 0.0, 0.0),
    end_time: float = 0.005,
    delta_t: float = 0.001,
    write_interval: int = 1,
) -> list[BaseConfig]:
    """Return the configs + ``0/`` fields for a laminar lid-driven cavity.

    Set ``end_time == delta_t`` for a single solver step. Boundary patches are
    :data:`CAVITY_PATCHES`. Persist via :func:`neofoam.io.write_configs`.
    """
    cfgs = configurations(incompressibleFluid)

    control = cfgs["ControlDictConfig"].model_validate(
        {
            "application": "pimpleFoam",
            "startTime": 0.0,
            "endTime": end_time,
            "deltaT": delta_t,
            "writeControl": "timeStep",
            "writeInterval": write_interval,
        }
    )
    schemes = cfgs["Pimple_fvSchemes"].model_validate(
        {
            "ddtSchemes": {"default": "Euler", "ddt(U)": "Euler"},
            "divSchemes": {
                "default": "none",
                "div(phi,U)": "Gauss linear",
                "div((nuEff*dev2(T(grad(U)))))": "Gauss linear",
            },
            "gradSchemes": {
                "default": "Gauss linear",
                "grad(U)": "Gauss linear",
                "grad(p)": "Gauss linear",
                "grad(p_rgh)": "Gauss linear",
                "grad(rhok)": "Gauss linear",
            },
            "laplacianSchemes": {
                "default": "Gauss linear corrected",
                "laplacian(nuEff,U)": "Gauss linear corrected",
                "laplacian(rAU,p)": "Gauss linear corrected",
                "laplacian(rAUf,p_rgh)": "Gauss linear corrected",
            },
            "interpolationSchemes": {
                "default": "linear",
                "flux(HbyA)": "linear",
                "interpolate(rAU)": "linear",
                "dotInterpolate(S,U_0)": "linear",
                "flux(U)": "linear",
            },
            "snGradSchemes": {
                "default": "corrected",
                "snGrad(p)": "corrected",
                "snGrad(rhok)": "corrected",
                "snGrad(p_rgh)": "corrected",
            },
        }
    )
    solution = cfgs["Pimple_fvSolution"].model_validate(
        {
            "solvers": {
                "p": {
                    "solver": "PCG",
                    "preconditioner": "DIC",
                    "tolerance": 1e-7,
                    "relTol": 0.05,
                },
                "pFinal": {
                    "solver": "PCG",
                    "preconditioner": "DIC",
                    "tolerance": 1e-7,
                    "relTol": 0.0,
                },
                "p_rgh": {
                    "solver": "PCG",
                    "preconditioner": "DIC",
                    "tolerance": 1e-7,
                    "relTol": 0.05,
                },
                "U": {
                    "solver": "PBiCGStab",
                    "preconditioner": "DILU",
                    "tolerance": 1e-8,
                    "relTol": 0.0,
                },
            },
            # Closed cavity ⇒ pressure defined up to a constant; pin a reference.
            "PIMPLE": {
                "nCorrectors": 2,
                "nNonOrthogonalCorrectors": 0,
                "momentumPredictor": True,
                "pRefCell": 0,
                "pRefValue": 0,
            },
        }
    )
    transport = cfgs["TransportPropertiesConfig"].model_validate(
        {"transportModel": "Newtonian", "nu": nu}
    )
    turbulence = cfgs["TurbulencePropertiesConfig"].model_validate(
        {"simulationType": "laminar"}
    )
    u_field = cfgs["UFieldConfig"].model_validate(
        {
            "internalField": "uniform (0 0 0)",
            "boundaryField": {
                "movingWall": {
                    "type": "fixedValue",
                    "value": _uniform_vector(lid_velocity),
                },
                "fixedWalls": {"type": "noSlip"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    p_field = cfgs["pFieldConfig"].model_validate(
        {
            "internalField": "uniform 0",
            "boundaryField": {
                "movingWall": {"type": "zeroGradient"},
                "fixedWalls": {"type": "zeroGradient"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    return [control, schemes, solution, transport, turbulence, u_field, p_field]

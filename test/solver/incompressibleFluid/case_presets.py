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
                "p_rghFinal": {
                    "solver": "PCG",
                    "preconditioner": "DIC",
                    "tolerance": 1e-7,
                    "relTol": 0.0,
                },
                "U": {
                    "solver": "PBiCGStab",
                    "preconditioner": "DILU",
                    "tolerance": 1e-8,
                    "relTol": 0.0,
                },
                "UFinal": {
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


def buoyant_cavity(
    *,
    nu: float = 1e-3,
    beta: float = 3e-3,
    t_hot: float = 310.0,
    t_ref: float = 300.0,
    end_time: float = 0.001,
    delta_t: float = 0.001,
) -> list[BaseConfig]:
    """Return the configs + ``0/`` fields for a **laminar Boussinesq** cavity.

    A differentially-heated closed cavity (hot ``movingWall``, cold ``fixedWalls``,
    no lid) on :data:`CAVITY_PATCHES` — the smallest complete case that exercises
    the native ``laminar`` momentum-transport model together with the ``boussinesq``
    optional model (energy equation + buoyant pressure). It reuses the buoyancy-ready
    PIMPLE schemes/solution from :func:`lid_driven_cavity` and adds ``constant/g``,
    the Boussinesq transport params, the energy schemes/solver, and the
    ``T``/``p_rgh``/``alphat`` fields. ``beta``/``TRef`` in ``transportProperties``
    are what activate the boussinesq model (see its ``@detect``). Gravity acts in
    ``-y`` (in-plane; ``frontAndBack`` is the empty ``z`` direction). Set
    ``end_time == delta_t`` for a single solver step.
    """
    base = {
        type(c).__name__: c
        for c in lid_driven_cavity(nu=nu, end_time=end_time, delta_t=delta_t)
    }
    cfgs = configurations(incompressibleFluid)

    # No lid: buoyancy alone drives the flow, so all walls are no-slip.
    u_field = cfgs["UFieldConfig"].model_validate(
        {
            "internalField": "uniform (0 0 0)",
            "boundaryField": {
                "movingWall": {"type": "noSlip"},
                "fixedWalls": {"type": "noSlip"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    # p is derived (p = p_rgh + rhok*gh) ⇒ calculated, like buoyantBoussinesqPimpleFoam.
    p_field = cfgs["pFieldConfig"].model_validate(
        {
            "internalField": "uniform 0",
            "boundaryField": {
                "movingWall": {"type": "calculated", "value": "uniform 0"},
                "fixedWalls": {"type": "calculated", "value": "uniform 0"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    p_rgh_field = cfgs["p_rghFieldConfig"].model_validate(
        {
            "internalField": "uniform 0",
            "boundaryField": {
                "movingWall": {"type": "fixedFluxPressure", "value": "uniform 0"},
                "fixedWalls": {"type": "fixedFluxPressure", "value": "uniform 0"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    t_field = cfgs["TFieldConfig"].model_validate(
        {
            "internalField": f"uniform {t_ref}",
            "boundaryField": {
                "movingWall": {"type": "fixedValue", "value": f"uniform {t_hot}"},
                "fixedWalls": {"type": "fixedValue", "value": f"uniform {t_ref}"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    # Laminar ⇒ no eddy viscosity ⇒ alphat ≡ 0 (calculated, never a wall function).
    alphat_field = cfgs["alphatFieldConfig"].model_validate(
        {
            "internalField": "uniform 0",
            "boundaryField": {
                "movingWall": {"type": "calculated", "value": "uniform 0"},
                "fixedWalls": {"type": "calculated", "value": "uniform 0"},
                "frontAndBack": {"type": "empty"},
            },
        }
    )
    gravity = cfgs["GravityConfig"].model_validate({"value": [0.0, -9.81, 0.0]})
    boussinesq = cfgs["BoussinesqConfig"].model_validate(
        {"beta": beta, "TRef": t_ref, "Pr": 0.7, "Prt": 0.85}
    )
    bouss_schemes = cfgs["boussinesq_fvSchemes"].model_validate(
        {
            "ddtSchemes": {"default": "Euler"},
            "divSchemes": {"div(phi,T)": "Gauss upwind"},
            "gradSchemes": {"grad(T)": "Gauss linear"},
            "laplacianSchemes": {"default": "Gauss linear corrected"},
        }
    )
    bouss_solution = cfgs["boussinesq_fvSolution"].model_validate(
        {
            "solvers": {
                "T": {
                    "solver": "PBiCGStab",
                    "preconditioner": "DILU",
                    "tolerance": 1e-8,
                    "relTol": 0.0,
                },
                "TFinal": {
                    "solver": "PBiCGStab",
                    "preconditioner": "DILU",
                    "tolerance": 1e-8,
                    "relTol": 0.0,
                },
            }
        }
    )
    return [
        base["ControlDictConfig"],
        base["Pimple_fvSchemes"],
        base["Pimple_fvSolution"],
        base["TransportPropertiesConfig"],
        base["TurbulencePropertiesConfig"],
        u_field,
        p_field,
        p_rgh_field,
        t_field,
        alphat_field,
        gravity,
        boussinesq,
        bouss_schemes,
        bouss_solution,
    ]

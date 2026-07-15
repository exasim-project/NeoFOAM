# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Boussinesq optional model for the incompressibleFluid solver.

Adds buoyancy-driven flow on top of PIMPLE: temperature T, thermal
turbulent diffusivity alphat, density factor rhok, geopotential gh/ghf,
modified pressure p_rgh, plus the energy equation and rhok update ops.

Ported from ``feat/python_solvers`` and adapted to the ModelSpec API:
build receives the loaded config (no ``self``). The energy equation
declares its ``system/fvSchemes`` entries and ``system/fvSolution``
solver via per-spec ``@BoussinesqFvSchemes.add(...)`` /
``@BoussinesqFvSolution.add(...)`` slices, mirroring the PIMPLE
algorithm.
"""

from pathlib import Path
from typing import Annotated, Any, Protocol

import pybFoam as pyf
from pybFoam import fvm, surfaceScalarField, volScalarField
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from neofoam.fields import (
    AlphatWallFunctionBC,
    CalculatedBC,
    CyclicBC,
    EmptyBC,
    FixedFluxPressureBC,
    FixedValueBC,
    GenericBC,
    InletOutletBC,
    Scalar,
    SymmetryBC,
    SymmetryPlaneBC,
    ZeroGradientBC,
)
from neofoam.foam import fvSchemes, fvSolution
from neofoam.framework.context import FieldUpdates
from neofoam.framework.initialization import ConfigContext, field, lazy
from neofoam.io import OF, BaseConfig, IOStrategy

from .incompressibleFluidModel import Model, incompressibleFluidModel


class ThermalTurbulenceModel(Protocol):
    def has_nut(self) -> bool: ...

    def nut(self) -> Any: ...

    def nu(self) -> Any: ...


def _fmt_component(x: float) -> str:
    """Render a vector component with no gratuitous ``.0`` (``0.0`` -> ``0``)."""
    return str(int(x)) if x == int(x) else repr(x)


def _divide(numerator: Any, denominator: Any) -> Any:
    """``numerator / denominator``, rebuilding the quotient when pybFoam binds no
    ``dimensionedScalar / dimensionedScalar``.

    A field numerator (the eddy-viscosity fallback) divides directly; laminar ``nu``
    is a plain ``dimensionedScalar``, and scalar/scalar has no bound operator — so it
    is rebuilt as a ``dimensionedScalar``. Mirrors
    :func:`neofoam.turbulence.stress._add_viscosity`.
    """
    try:
        return numerator / denominator
    except TypeError:
        return pyf.dimensionedScalar(
            pyf.Word(f"({numerator.name()}|{denominator.name()})"),
            numerator.dimensions() / denominator.dimensions(),
            numerator.value() / denominator.value(),
        )


class _GravityHeader(BaseModel):
    """The ``FoamFile`` header that identifies ``constant/g`` to OpenFOAM.

    ``class`` is pinned to ``uniformDimensionedVectorField`` (not the generic
    ``dictionary`` the writer injects for headerless configs) so OpenFOAM reads
    the file as a gravity field.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    version: str = "2.0"
    format: str = "ascii"
    field_class: str = Field(default="uniformDimensionedVectorField", alias="class")
    object: str = "g"


@IOStrategy(OF("constant/g"))
class GravityConfig(BaseConfig):
    """``constant/g`` -- the uniform gravitational acceleration buoyancy needs.

    The Boussinesq ``@build`` reads ``g`` at init (see ``_gh_ref``) to form the
    geopotential ``gh``/``ghf``; without this file the solver aborts with
    *cannot find file "constant/g"*. Defaults are Earth gravity acting in -y:
    ``dimensions [0 1 -2 0 0 0 0]``, ``value (0 -9.81 0)``. ``dimensions`` and
    ``value`` round-trip OpenFOAM's bracket/paren tokens as Python lists.
    """

    FoamFile: _GravityHeader = Field(default_factory=_GravityHeader)
    dimensions: list[int] = Field(default_factory=lambda: [0, 1, -2, 0, 0, 0, 0])
    value: list[float] = Field(default_factory=lambda: [0.0, -9.81, 0.0])

    @field_validator("dimensions", mode="before")
    @classmethod
    def _parse_dimensions(cls, value: Any) -> list[int]:
        if isinstance(value, str):
            return [int(p) for p in value.strip().strip("[]").split()]
        return [int(v) for v in value]

    @field_validator("value", mode="before")
    @classmethod
    def _parse_value(cls, value: Any) -> list[float]:
        if isinstance(value, str):
            return [float(p) for p in value.strip().strip("()").split()]
        return [float(v) for v in value]

    @field_serializer("dimensions", when_used="always")
    def _serialize_dimensions(self, value: list[int]) -> str:
        return "[" + " ".join(str(int(v)) for v in value) + "]"

    @field_serializer("value", when_used="always")
    def _serialize_value(self, value: list[float]) -> str:
        return "(" + " ".join(_fmt_component(v) for v in value) + ")"


@IOStrategy(OF("constant/transportProperties"))
class BoussinesqConfig(BaseConfig):
    beta: float
    TRef: float
    Pr: float
    Prt: float
    hRef: float = 0.0


def _read_boussinesq_config() -> BoussinesqConfig:
    """Read transportProperties and construct a validated BoussinesqConfig.

    ``beta`` and ``TRef`` are required and read directly; ``Pr``, ``Prt``,
    ``hRef`` keep their schema defaults if not present in the dict.
    """
    props = pyf.dictionary.read("constant/transportProperties")
    values: dict[str, float] = {
        "beta": props.get[float]("beta"),
        "TRef": props.get[float]("TRef"),
    }
    if props.found("Pr"):
        values["Pr"] = props.get[float]("Pr")
    if props.found("Prt"):
        values["Prt"] = props.get[float]("Prt")
    if props.found("hRef"):
        values["hRef"] = props.get[float]("hRef")
    return BoussinesqConfig(**values)


boussinesq = (
    Model("boussinesq")
    .register_with(incompressibleFluidModel)
    .labeled("Buoyancy (Boussinesq)")
)

# Declare the configs this model owns. ``BoussinesqConfig`` is loaded via
# ``@boussinesq.load`` below; registering it here only makes it part of the
# declared schema set (``collect_config_classes``) — the ``@load`` path
# still drives instantiation, so this does not change loading behaviour.
boussinesq.config(BoussinesqConfig)
boussinesq.config(GravityConfig)

# Per-spec fvSchemes / fvSolution slices for the energy equation.
BoussinesqFvSchemes = boussinesq.config(fvSchemes)
BoussinesqFvSolution = boussinesq.config(fvSolution)

# 0/<name> declarations the boussinesq plugin adds on top of pimple's
# ``U``/``p``. The disk schemas land in ``configurations(solver).fields``
# alongside pimple's so the case author / agent fills the union without
# touching either model's source. ``rhok`` / ``gh`` / ``ghf`` are
# computed at init (no ``0/`` file) and stay on the legacy ``field()``
# helper below.
boussinesq.field(
    "p_rgh",
    dimensions=[0, 2, -2, 0, 0, 0, 0],
    value_type=Scalar,
    # ``fixedFluxPressure`` is now a typed arm — the dominant
    # buoyancy-pressure BC in the upstream tutorials (hotRoom et al.).
    # ``GenericBC`` is kept for ``prghPressure`` / ``totalPressure``
    # and the rarer adjoint pressure arms.
    allowed_bcs=[
        FixedValueBC,
        FixedFluxPressureBC,
        ZeroGradientBC,
        InletOutletBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    # ``p_rgh_flux_required`` runs ``mesh.setFluxRequired("p_rgh")``
    # before the field is read — see the ``lazy(...)`` step in
    # ``@boussinesq.build`` below.
    depends_on=(
        "mesh",
        "p_rgh_flux_required",
        "fields.p",
        "fields.rhok",
        "fields.gh",
    ),
    write=True,
)
boussinesq.field(
    "T",
    dimensions=[0, 0, 0, 1, 0, 0, 0],
    value_type=Scalar,
    allowed_bcs=[
        FixedValueBC,
        ZeroGradientBC,
        InletOutletBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
    write=True,
)
boussinesq.field(
    "alphat",
    dimensions=[0, 2, -1, 0, 0, 0, 0],
    value_type=Scalar,
    # ``alphat`` is the thermal turbulent diffusivity. In the buoyant-
    # PIMPLE tutorials it's almost always one of the alphat wall
    # functions (``compressible::alphatWallFunction`` /
    # ``alphatJayatillekeWallFunction`` — both arms of
    # :class:`AlphatWallFunctionBC`) plus ``calculated`` on inlet/outlet
    # patches.
    allowed_bcs=[
        AlphatWallFunctionBC,
        CalculatedBC,
        FixedValueBC,
        EmptyBC,
        SymmetryBC,
        SymmetryPlaneBC,
        CyclicBC,
        GenericBC,
    ],
)


@boussinesq.load
def load(_case_dir: Path, _instance_id: str) -> BoussinesqConfig:
    return _read_boussinesq_config()


@boussinesq.detect
def detect_model() -> bool:
    """Boussinesq is active when transportProperties has both beta and TRef."""
    try:
        props = pyf.dictionary.read("constant/transportProperties")
        return bool(props.found("beta") and props.found("TRef"))
    except Exception:
        return False


@boussinesq.resolve
def resolve(_config: BoussinesqConfig, ctx: ConfigContext) -> None:
    """Flip ``use_boussinesq`` on the pressure-velocity algorithm spec.

    The algorithm spec is registered in ConfigContext under its ``name``
    (e.g. ``"Pimple"``). Setting this flag steers
    :func:`pimpleAlgorithm.collected_operations` to dispatch the
    boussinesq variants and adds ``p_rgh`` to the pressure-reference
    dependency list.
    """
    for algo_name in ("Pimple", "Simple", "Piso"):
        pressure_model = ctx.get(algo_name) if ctx.contains(algo_name) else None
        if pressure_model is not None:
            pressure_model.use_boussinesq = True
            return


@boussinesq.build
def build(configs: BoussinesqConfig) -> list[object]:
    """Lazy initializers for boussinesq state (non-field bits only).

    ``T`` / ``alphat`` / ``p_rgh`` are auto-synthesized from the
    ``boussinesq.field(...)`` declarations at the top of this module.
    ``@build`` only carries what the framework cannot synthesize:
    ``rhok`` / ``gh`` / ``ghf`` (computed at init from gravity + mesh,
    no ``0/<name>`` file), and the ``setFluxRequired`` side-effect that
    ``p_rgh`` needs *before* the auto-synthesized read fires.
    """

    def create_rhok(context: dict[str, Any]) -> volScalarField:
        # rhok is the Boussinesq density factor: a *computed* field, not read
        # from disk. Mirror OpenFOAM buoyantBoussinesqPimpleFoam createFields.H
        # (rhok = 1 - beta*(T - TRef), NO_READ) so a case never needs a 0/rhok
        # file — it is constructed from T here and refreshed by ``update_rhok``.
        temperature = context["fields.T"]
        beta = pyf.dimensionedScalar(
            "beta", pyf.dimless / pyf.dimTemperature, configs.beta
        )
        t_ref = pyf.dimensionedScalar("TRef", pyf.dimTemperature, configs.TRef)
        one = pyf.dimensionedScalar("one", pyf.dimless, 1.0)
        return volScalarField(pyf.Word("rhok"), one - beta * (temperature - t_ref))

    def _gh_ref(context: dict[str, Any]) -> tuple[Any, Any]:
        mesh = context["mesh"]
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        gh_ref = g_mag * configs.hRef if g_mag > 1e-15 else 0.0
        gh_ref_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, gh_ref
        )
        return g, gh_ref_dim

    def create_gh(context: dict[str, Any]) -> volScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gh_ref(context)
        return volScalarField(pyf.Word("gh"), (g & mesh.C()) - gh_ref_dim)

    def create_ghf(context: dict[str, Any]) -> surfaceScalarField:
        mesh = context["mesh"]
        g, gh_ref_dim = _gh_ref(context)
        return surfaceScalarField(pyf.Word("ghf"), (g & mesh.Cf()) - gh_ref_dim)

    def mark_p_rgh_flux_required(context: dict[str, Any]) -> None:
        """Tell the mesh that p_rgh participates in flux assembly.

        Must run *before* ``fields.p_rgh`` reads from disk — the
        ``p_rgh`` :class:`FieldDecl`'s ``depends_on`` lists this step's
        name (``p_rgh_flux_required``) so the topological sort
        guarantees the ordering.
        """
        context["mesh"].setFluxRequired(pyf.Word("p_rgh"))
        return None

    return [
        # rhok / gh / ghf have no 0/ file (computed at init) so they
        # stay on the legacy ``field()`` helper.
        field("rhok", create_rhok, depends_on=["mesh", "fields.T"]),
        field("gh", create_gh, depends_on=["mesh"]),
        field("ghf", create_ghf, depends_on=["mesh"]),
        # The setFluxRequired side-effect p_rgh used to do inline now
        # lives in its own init step; ``boussinesq.field("p_rgh", ...)``
        # declares it as a dependency.
        lazy("p_rgh_flux_required", mark_p_rgh_flux_required, depends_on=["mesh"]),
    ]


@boussinesq.operation(operation_number="2.5", depends_on=["momentum"])
@BoussinesqFvSchemes.add(
    ddt="default", div="div(phi,T)", grad="grad(T)", laplacian="default"
)
@BoussinesqFvSolution.add("T")
def solve_energy(
    self: Any,
    T: volScalarField,
    phi: surfaceScalarField,
    turbulence: Annotated[ThermalTurbulenceModel, "models"],
    alphat: volScalarField,
) -> FieldUpdates:
    configs: BoussinesqConfig = self.config
    pr = pyf.dimensionedScalar("Pr", pyf.dimless, configs.Pr)
    prt = pyf.dimensionedScalar("Prt", pyf.dimless, configs.Prt)

    if turbulence.has_nut():
        alphat.assign(turbulence.nut() / prt)
    else:
        # Laminar: no eddy viscosity ⇒ alphat ≡ 0, so alpha_eff = nu/Pr (the
        # molecular thermal diffusivity). Zero alphat in place, keeping its dims.
        zero = pyf.dimensionedScalar("zero", pyf.dimless, 0.0)
        alphat.assign(alphat * zero)
    alphat.correctBoundaryConditions()

    alpha_eff = _divide(turbulence.nu(), pr) + alphat
    t_eqn = pyf.fvScalarMatrix(
        fvm.ddt(T) + fvm.div(phi, T) - fvm.laplacian(alpha_eff, T)
    )
    t_eqn.relax()
    t_eqn.solve()

    return FieldUpdates({"T": T, "alphat": alphat})


@boussinesq.operation(operation_number="2.7", depends_on=["solve_energy"])
def update_rhok(self: Any, T: volScalarField, rhok: volScalarField) -> FieldUpdates:
    configs: BoussinesqConfig = self.config
    beta = pyf.dimensionedScalar("beta", pyf.dimless / pyf.dimTemperature, configs.beta)
    t_ref = pyf.dimensionedScalar("TRef", pyf.dimTemperature, configs.TRef)
    one = pyf.dimensionedScalar("one", pyf.dimless, 1.0)

    rhok.assign(one - beta * (T - t_ref))
    return FieldUpdates({"rhok": rhok})

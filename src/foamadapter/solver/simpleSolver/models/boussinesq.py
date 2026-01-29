# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

"""
Boussinesq Buoyancy Model with Temperature Transport

Implements Boussinesq approximation for buoyancy-driven flows with:
- Temperature transport equation
- p_rgh pressure formulation
- Kinematic density correction: rhok = 1 - beta*(T - TRef)

This model signals to pressure-velocity algorithms to use Boussinesq
formulation by providing rhok, p_rgh, gh, and ghf fields.

Example Usage:
    from foamadapter.models.buoyancy import BoussinesqModel

    solver = IncompressibleFluid(
        argv=["-case", "hotRoom"],
        models=[BoussinesqModel()]
    )
    solver.run()
"""

from dataclasses import dataclass
from typing import Any

import pybFoam as pyf
from pybFoam import (
    volScalarField,
    surfaceScalarField,
    fvm,
)

from foamadapter.framework.context import (
    FieldUpdates,
    Model as ModelAnnotation,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.model import Model
from foamadapter.framework.operations import Operation, OperationCollection
from foamadapter.solver.simpleSolver.models.base import SimpleSolverModel


@SimpleSolverModel.register
@Model
@dataclass
class BoussinesqModel(SimpleSolverModel):
    """
    Boussinesq buoyancy model with temperature transport and p_rgh formulation.

    Physics:
    - Solves temperature transport equation
    - Computes rhok = 1 - beta*(T - TRef)
    - Creates p_rgh = p - rhok*gh fields
    - Signals algorithm to use Boussinesq momentum/continuity

    Configuration (read from constant/transportProperties):
        beta: Thermal expansion coefficient [1/K]
        TRef: Reference temperature [K]
        Pr: Prandtl number
        Prt: Turbulent Prandtl number
    """

    name: str = "boussinesq"

    # Properties (read from files in __post_init__)
    beta: float = 3e-3  # Thermal expansion coefficient [1/K]
    TRef: float = 300.0  # Reference temperature [K]
    Pr: float = 0.7  # Prandtl number
    Prt: float = 0.85  # Turbulent Prandtl number
    g: tuple[float, float, float] = (0, -9.81, 0)  # Gravity vector
    hRef: float = 0.0  # Reference height for hydrostatic pressure

    @staticmethod
    def detect() -> bool:
        """Return True if beta and TRef are in transportProperties."""
        try:
            props = pyf.dictionary.read("constant/transportProperties")
            toc = list(props.toc())
            return "beta" in toc and "TRef" in toc
        except Exception:
            return False

    def __post_init__(self) -> None:
        """
        Initialize Boussinesq model by reading properties and creating fields.

        Reads:
        - beta, TRef, Pr, Prt from transportProperties
        - gravity vector from g file
        - Temperature field T
        - Turbulent thermal diffusivity alphat
        - Kinematic density rhok
        - Hydrostatic pressure fields gh, ghf, p_rgh
        """
        from pybFoam import Info

        # Read transport properties
        props = pyf.dictionary.read("constant/transportProperties")
        self.beta = props.get_scalar("beta")
        self.TRef = props.get_scalar("TRef")
        self.Pr = props.get_scalar("Pr")
        self.Prt = props.get_scalar("Prt")

        Info(
            f"Boussinesq model: TRef={self.TRef} K, beta={self.beta} 1/K, "
            f"Pr={self.Pr}, Prt={self.Prt}"
        )

        # Read gravity
        try:
            g_dict = pyf.dictionary.read("constant/g")
            g_vec = g_dict.get("value")
            if hasattr(g_vec, "x"):
                self.g = (g_vec.x(), g_vec.y(), g_vec.z())
            else:
                # Handle tuple format
                g_tuple = tuple(float(x) for x in g_vec)
                if len(g_tuple) == 3:
                    self.g = g_tuple
        except Exception:
            pass  # Use default gravity

    def _create_fields(self, mesh: Any) -> dict[str, Any]:
        """
        Create Boussinesq fields (T, alphat, rhok, gh, ghf, p_rgh).

        Returns dict with field objects.
        """
        # Read temperature field
        T = volScalarField.read_field(mesh, "T")

        # Read turbulent thermal diffusivity
        alphat = volScalarField.read_field(mesh, "alphat")

        # Read or create rhok = 1 - beta*(T - TRef)
        rhok = volScalarField.read_field(mesh, "rhok")

        # Create gh field: gh = (g & C) - ghRef
        g = pyf.uniformDimensionedVectorField(mesh, "g")
        g_value = g.value()
        SMALL = 1e-15
        g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
        ghRef = g_mag * self.hRef if g_mag > SMALL else 0.0

        C = mesh.C()
        gh_temp = g & C
        ghRef_dim = pyf.dimensionedScalar(
            "ghRef", g.dimensions() * pyf.dimLength, ghRef
        )
        gh = volScalarField(pyf.Word("gh"), gh_temp - ghRef_dim)

        # Create ghf field: ghf = (g & Cf) - ghRef
        Cf = mesh.Cf()
        ghf_temp = g & Cf
        ghf = surfaceScalarField(pyf.Word("ghf"), ghf_temp - ghRef_dim)

        # Set flux requirement for p_rgh (needed for pEqn.flux())
        mesh.setFluxRequired(pyf.Word("p_rgh"))
        p_rgh = volScalarField.read_field(mesh, "p_rgh")

        return {
            "T": T,
            "alphat": alphat,
            "rhok": rhok,
            "gh": gh,
            "ghf": ghf,
            "p_rgh": p_rgh,
        }

    def build(self) -> list[Any]:
        """
        Build stage: Create lazy initializers for model fields.

        Returns list of LazyInit objects for field creation.
        """
        from foamadapter.framework.initialization.lazy_init import LazyInit

        # Create individual LazyInit objects for each field
        initializers = []

        # T field
        def create_T(ctx: dict[str, Any]) -> Any:
            mesh = ctx["mesh"]
            return volScalarField.read_field(mesh, "T")

        initializers.append(
            LazyInit(
                name="fields.T",
                depends_on=["mesh"],
                initializer=create_T,
                category="fields",
            )
        )

        # alphat field
        def create_alphat(ctx: dict[str, Any]) -> Any:
            mesh = ctx["mesh"]
            return volScalarField.read_field(mesh, "alphat")

        initializers.append(
            LazyInit(
                name="fields.alphat",
                depends_on=["mesh"],
                initializer=create_alphat,
                category="fields",
            )
        )

        # rhok field
        def create_rhok(ctx: dict[str, Any]) -> Any:
            mesh = ctx["mesh"]
            return volScalarField.read_field(mesh, "rhok")

        initializers.append(
            LazyInit(
                name="fields.rhok",
                depends_on=["mesh", "fields.T"],
                initializer=create_rhok,
                category="fields",
            )
        )

        # gh field
        def create_gh(ctx: dict[str, Any]) -> Any:
            mesh = ctx["mesh"]
            g = pyf.uniformDimensionedVectorField(mesh, "g")
            g_value = g.value()
            SMALL = 1e-15
            g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
            ghRef = g_mag * self.hRef if g_mag > SMALL else 0.0

            C = mesh.C()
            gh_temp = g & C
            ghRef_dim = pyf.dimensionedScalar(
                "ghRef", g.dimensions() * pyf.dimLength, ghRef
            )
            return volScalarField(pyf.Word("gh"), gh_temp - ghRef_dim)

        initializers.append(
            LazyInit(
                name="fields.gh",
                depends_on=["mesh"],
                initializer=create_gh,
                category="fields",
            )
        )

        # ghf field
        def create_ghf(ctx: dict[str, Any]) -> Any:
            mesh = ctx["mesh"]
            g = pyf.uniformDimensionedVectorField(mesh, "g")
            g_value = g.value()
            SMALL = 1e-15
            g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
            ghRef = g_mag * self.hRef if g_mag > SMALL else 0.0

            Cf = mesh.Cf()
            ghf_temp = g & Cf
            ghRef_dim = pyf.dimensionedScalar(
                "ghRef", g.dimensions() * pyf.dimLength, ghRef
            )
            return surfaceScalarField(pyf.Word("ghf"), ghf_temp - ghRef_dim)

        initializers.append(
            LazyInit(
                name="fields.ghf",
                depends_on=["mesh"],
                initializer=create_ghf,
                category="fields",
            )
        )

        # p_rgh field
        def create_p_rgh(ctx: dict[str, Any]) -> Any:
            mesh = ctx["mesh"]
            mesh.setFluxRequired(pyf.Word("p_rgh"))
            return volScalarField.read_field(mesh, "p_rgh")

        initializers.append(
            LazyInit(
                name="fields.p_rgh",
                depends_on=["mesh", "fields.p", "fields.rhok", "fields.gh"],
                initializer=create_p_rgh,
                category="fields",
            )
        )

        return initializers

    def configure_algorithm(self, algorithm: Any) -> None:
        """
        Configure pressure-velocity algorithm for Boussinesq mode.

        Sets the _use_boussinesq flag to trigger p_rgh formulation.
        """
        from pybFoam import Info

        algorithm._use_boussinesq = True
        Info("Configuring algorithm for Boussinesq formulation")
        Info(f"_use_boussinesq flag is now: {algorithm._use_boussinesq}")

    @Model.operation(operation_number="2.5", depends_on=["momentum"])
    def solve_energy(
        self,
        T: Any,
        phi: Any,
        turbulence: ModelAnnotation[Any],
        alphat: Any,
    ) -> FieldUpdates:
        """
        Solve temperature transport equation.

        Equation: ∂T/∂t + ∇·(φT) - ∇·(α_eff ∇T) = 0
        where α_eff = ν/Pr + ν_t/Pr_t
        """
        # Create dimensioned scalars
        Pr_dim = pyf.dimensionedScalar("Pr", pyf.dimless, self.Pr)
        Prt_dim = pyf.dimensionedScalar("Prt", pyf.dimless, self.Prt)

        # Update turbulent thermal diffusivity
        alphat.assign(turbulence.nut() / Prt_dim)
        alphat.correctBoundaryConditions()

        # Effective thermal diffusivity
        alphaEff = turbulence.nu() / Pr_dim + alphat

        # Assemble and solve energy equation
        TEqn = pyf.fvScalarMatrix(
            fvm.ddt(T) + fvm.div(phi, T) - fvm.laplacian(alphaEff, T)
        )

        TEqn.relax()
        TEqn.solve()

        return FieldUpdates({"T": T, "alphat": alphat})

    @Model.operation(operation_number="2.7", depends_on=["solve_energy"])
    def update_rhok(self, T: Any, rhok: Any) -> FieldUpdates:
        """
        Update rhok after energy equation.

        Computes: rhok = 1 - beta*(T - TRef)
        """
        # Create dimensioned scalars for computation
        beta_dim = pyf.dimensionedScalar(
            "beta", pyf.dimless / pyf.dimTemperature, self.beta
        )
        TRef_dim = pyf.dimensionedScalar("TRef", pyf.dimTemperature, self.TRef)

        # Update rhok: rhok = 1 - beta*(T - TRef)
        rhok.assign(1.0 - beta_dim * (T - TRef_dim))
        return FieldUpdates({"rhok": rhok})

    def operations(self) -> OperationCollection:
        """
        Return operations this model contributes.

        Automatically discovers all @Model.operation decorated methods.
        """
        funcs = decorated_member_functions(self)
        ops = OperationCollection()
        for func in funcs:
            op = Operation.create_SeqOp(func)
            ops.add(op)
        return ops

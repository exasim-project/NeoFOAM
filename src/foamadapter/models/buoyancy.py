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

from typing import Any, Literal

import pybFoam as pyf
from pybFoam import (
    volScalarField,
    surfaceScalarField,
    fvm,
)
from pydantic import BaseModel

from foamadapter.framework.context import (
    FieldUpdates,
    Model as ModelAnnotation,
)
from foamadapter.framework.decorator import decorated_member_functions
from foamadapter.framework.initialization import ConfigContext
from foamadapter.framework.model import Model
from foamadapter.framework.operations import Operation, OperationCollection
from foamadapter.models.incompressible_fluid_model import IncompressibleFluidModel


@IncompressibleFluidModel.register
@Model
class BoussinesqModel(BaseModel):
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

    model_config = {"arbitrary_types_allowed": True}

    model_type: Literal["boussinesq"] = "boussinesq"
    name: str = "boussinesq"

    # Properties (read from files)
    beta: float = 3e-3  # Thermal expansion coefficient [1/K]
    TRef: float = 300.0  # Reference temperature [K]
    Pr: float = 0.7  # Prandtl number
    Prt: float = 0.85  # Turbulent Prandtl number
    g: tuple[float, float, float] = (0, -9.81, 0)  # Gravity vector
    hRef: float = 0.0  # Reference height for hydrostatic pressure

    # Lifecycle state
    configured: bool = False
    _props: Any | None = None
    _g_dict: Any | None = None

    @staticmethod
    def detect() -> bool:
        """Return True if beta and TRef are in transportProperties."""
        try:
            props = pyf.dictionary.read("constant/transportProperties")
            toc = list(props.toc())
            return "beta" in toc and "TRef" in toc
        except Exception:
            return False

    @Model.load
    def load_boussinesq_properties(self) -> dict[str, Any]:
        """
        LOAD: Load Boussinesq properties from constant/transportProperties and g.

        Reads:
        - beta, TRef, Pr, Prt from transportProperties
        - gravity vector from g file
        """
        # Read transport properties
        self._props = pyf.dictionary.read("constant/transportProperties")

        self.beta = pyf.dimensionedScalar(
            "beta", pyf.dimless / pyf.dimTemperature, self._props.get_scalar("beta")
        )
        self.TRef = pyf.dimensionedScalar(
            "TRef", pyf.dimTemperature, self._props.get_scalar("TRef")
        )
        self.Pr = pyf.dimensionedScalar("Pr", pyf.dimless, self._props.get_scalar("Pr"))
        self.Prt = pyf.dimensionedScalar(
            "Prt", pyf.dimless, self._props.get_scalar("Prt")
        )

        # Read gravity
        try:
            self._g_dict = pyf.dictionary.read("constant/g")
            g_vec = self._g_dict.get("value")
            if hasattr(g_vec, "x"):
                self.g = (g_vec.x(), g_vec.y(), g_vec.z())
            else:
                # Handle tuple format
                g_tuple = tuple(float(x) for x in g_vec)
                if len(g_tuple) == 3:
                    self.g = (g_tuple[0], g_tuple[1], g_tuple[2])
        except Exception:
            pass  # Use default gravity

        # No configurable objects to register
        return {}

    @Model.resolve_dependencies
    def configure_boussinesq(self, config: ConfigContext) -> None:
        """
        RESOLVE_DEPENDENCIES: Configure pressure-velocity algorithm for Boussinesq mode.

        Accesses the algorithm via ConfigContext and sets the _use_boussinesq flag.
        This ensures the algorithm uses p_rgh formulation before operations are built.
        """
        # Access algorithm from config - must exist or initialization is broken
        if not hasattr(config, "algorithm") or config.algorithm is None:
            raise RuntimeError(
                "BoussinesqModel requires algorithm to be available in ConfigContext. "
                "The solver must create and register the algorithm during LOAD stage "
                "before models' RESOLVE_DEPENDENCIES stage runs."
            )

        config.algorithm._use_boussinesq = True
        self.configured = True

    @Model.build
    def setup_boussinesq_fields(self, mesh: Any) -> list[Any]:
        """
        BUILD: Create Boussinesq fields (T, alphat, rhok, gh, ghf, p_rgh).

        Returns field initializers for the framework.
        """
        from foamadapter.framework.initialization.helpers import field

        def create_T(ctx: dict[str, Any]) -> Any:
            """Read temperature field."""
            mesh = ctx["mesh"]
            return volScalarField.read_field(mesh, "T")

        def create_alphat(ctx: dict[str, Any]) -> Any:
            """Read turbulent thermal diffusivity."""
            mesh = ctx["mesh"]
            return volScalarField.read_field(mesh, "alphat")

        def create_rhok(ctx: dict[str, Any]) -> Any:
            """Compute rhok = 1 - beta*(T - TRef)."""
            mesh = ctx["mesh"]

            # For now, initialize rhok as uniform 1.0 (dimensionless)
            # TODO: Fix dimension handling for beta * (T - TRef) calculation
            rhok = volScalarField.read_field(mesh, "rhok")
            return rhok

        def create_gh(ctx: dict[str, Any]) -> Any:
            """Calculate field g.h

            OpenFOAM: volScalarField gh("gh", (g & mesh.C()) - ghRef);

            Implementation matches finiteVolume/cfdTools/general/include/gh.H
            """
            mesh = ctx["mesh"]

            # Read gravity from constant/g
            g = pyf.uniformDimensionedVectorField(mesh, "g")
            g_value = g.value()

            # Calculate ghRef
            # OpenFOAM: ghRef = mag(g) > SMALL ? g & (cmptMag(g)/mag(g))*hRef : 0
            # Simplifies to: mag(g) * hRef
            SMALL = 1e-15
            g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
            ghRef = g_mag * self.hRef if g_mag > SMALL else 0.0

            # Get cell centers
            # Get cell centers
            C = mesh.C()

            # Create gh field: gh = (g & C) - ghRef
            # Matches OpenFOAM: volScalarField gh("gh", (g & mesh.C()) - ghRef);
            gh_temp = g & C
            # Dimensions: g[m/s²] * length[m] = [m²/s²]
            ghRef_dim = pyf.dimensionedScalar(
                "ghRef", g.dimensions() * pyf.dimLength, ghRef
            )
            gh = volScalarField(pyf.Word("gh"), gh_temp - ghRef_dim)

            return gh

        def create_ghf(ctx: dict[str, Any]) -> Any:
            """Calculate field g.hf at face centers

            OpenFOAM: surfaceScalarField ghf("ghf", (g & mesh.Cf()) - ghRef);

            Implementation matches finiteVolume/cfdTools/general/include/gh.H
            """
            mesh = ctx["mesh"]

            # Read gravity from constant/g
            g = pyf.uniformDimensionedVectorField(mesh, "g")
            g_value = g.value()

            # Calculate ghRef (same as in gh)
            SMALL = 1e-15
            g_mag = (g_value[0] ** 2 + g_value[1] ** 2 + g_value[2] ** 2) ** 0.5
            ghRef = g_mag * self.hRef if g_mag > SMALL else 0.0

            # Get face centers
            Cf = mesh.Cf()

            # Create ghf field: ghf = (g & Cf) - ghRef
            # Matches OpenFOAM: surfaceScalarField ghf("ghf", (g & mesh.Cf()) - ghRef);
            ghf_temp = g & Cf
            # Dimensions: g[m/s²] * length[m] = [m²/s²]
            ghRef_dim = pyf.dimensionedScalar(
                "ghRef", g.dimensions() * pyf.dimLength, ghRef
            )
            ghf = surfaceScalarField(pyf.Word("ghf"), ghf_temp - ghRef_dim)

            return ghf

        def create_p_rgh(ctx: dict[str, Any]) -> Any:
            """Read or create p_rgh field."""
            mesh = ctx["mesh"]

            # Set flux requirement for p_rgh (needed for pEqn.flux())
            mesh.setFluxRequired(pyf.Word("p_rgh"))
            return volScalarField.read_field(mesh, "p_rgh")

        return [
            field("T", create=create_T, depends_on=["mesh"]),
            field("alphat", create=create_alphat, depends_on=["mesh"]),
            field("rhok", create=create_rhok, depends_on=["fields.T"]),
            field("gh", create=create_gh, depends_on=["mesh"]),
            field("ghf", create=create_ghf, depends_on=["mesh"]),
            field(
                "p_rgh",
                create=create_p_rgh,
                depends_on=["mesh", "fields.p", "fields.rhok", "fields.gh"],
            ),
        ]

    @Model.operation(operation_number="0.3", depends_on=["momentum"])
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
        # Update turbulent thermal diffusivity
        alphat.assign(turbulence.nut() / self.Prt)
        alphat.correctBoundaryConditions()

        # Effective thermal diffusivity
        alphaEff = turbulence.nu() / self.Pr + alphat

        # Assemble and solve energy equation
        TEqn = pyf.fvScalarMatrix(
            fvm.ddt(T) + fvm.div(phi, T) - fvm.laplacian(alphaEff, T)
        )

        TEqn.relax()
        TEqn.solve()

        return FieldUpdates({"T": T, "alphat": alphat})

    @Model.operation(operation_number="0.7", depends_on=["solve_energy"])
    def update_rhok(self, T: Any, rhok: Any) -> FieldUpdates:
        """
        Update rhok after energy equation.

        Computes: rhok = 1 - beta*(T - TRef)
        """
        # Update rhok: rhok = 1 - beta*(T - TRef)
        # beta is a scalar [1/K], (T - TRef_dim) is a field [K]
        rhok.assign(1.0 - self.beta * (T - self.TRef))
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

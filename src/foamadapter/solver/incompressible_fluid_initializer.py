# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from typing import Any

import pybFoam as pyf

from foamadapter.framework import (
    Initializer,
    BaseInitializer,
    ConfigContext,
)
from foamadapter.algorithms.pressure_velocity import (
    PressureVelocityAlgorithm,
)
from foamadapter.models.stability_criteria import CFLCondition
from foamadapter.models.transport_model import TransportModel
from foamadapter.models.turbulence import TurbulenceModel
from foamadapter.models.incompressible_fluid_model import IncompressibleFluidModel
from foamadapter.foam.initialization import create_time_mesh


@Initializer
class IncompressibleFluidInitializer(BaseInitializer):
    """3-stage initialization for IncompressibleFluid solver."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.algorithm: PressureVelocityAlgorithm | None = None
        self.fvSolution: Any | None = None
        self.cfl_condition: CFLCondition | None = None

        # Detect optional models early so SolverInitializer can process them
        self.optional_models: list[IncompressibleFluidModel] = (
            IncompressibleFluidModel.detect_models()
        )
        # SolverInitializer expects models in self.models
        self.models = self.optional_models

    def get_models(self) -> list[Any]:
        """Return optional models to be processed by SolverInitializer."""
        return self.models

    @Initializer.load
    def load_core_components(self) -> dict[str, Any]:
        """LOAD: Initialize algorithm and CFL condition."""
        self.cfl_condition = CFLCondition()
        self.fvSolution = pyf.dictionary.read("system/fvSolution")
        self.algorithm = PressureVelocityAlgorithm.from_fvSolution(self.fvSolution)

        # Return algorithm for registration in ConfigContext
        return {"algorithm": self.algorithm, "cfl_condition": self.cfl_condition}

    @Initializer.resolve_dependencies
    def configure_models(self, config: ConfigContext) -> None:
        """RESOLVE_DEPENDENCIES: Additional orchestration if needed."""
        pass

    @Initializer.build
    def build_runtime(self, mesh: Any = None) -> list[Any]:
        """BUILD: Create runtime, mesh and algorithm fields."""
        _ = mesh  # Unused, we create the mesh here
        from foamadapter.framework.initialization.helpers import lazy, model

        # 1. Create runtime and mesh
        initializers = create_time_mesh(self.argv)

        # 2. Add algorithm fields (p, U, etc.)
        assert self.algorithm is not None
        initializers.extend(self.algorithm.setup())

        # 3. Add transport and turbulence models
        initializers.append(
            model(
                "laminarTransport",
                depends_on=["fields.U", "fields.phi"],
                create=self._create_transport,
            )
        )
        initializers.append(
            model(
                "turbulence",
                depends_on=["fields.U", "fields.phi", "models.laminarTransport"],
                create=self._create_turbulence,
            )
        )

        # 4. Final algorithm build (sets pressure reference)
        initializers.append(
            model(
                "algorithm",
                depends_on=[
                    "fields.p",
                    "mesh",
                    "models.laminarTransport",
                    "models.turbulence",
                ],
                create=self._finalize_algorithm,
            )
        )

        # 5. Keep dictionaries alive to prevent segfaults
        initializers.append(lazy("fvSolution_dict", create=lambda _: self.fvSolution))

        return initializers

    def _create_transport(self, ctx: dict[str, Any]) -> Any:
        return TransportModel.from_type("singlePhase").create_instance(
            ctx["fields.U"], ctx["fields.phi"]
        )

    def _create_turbulence(self, ctx: dict[str, Any]) -> Any:
        return TurbulenceModel.from_type("openfoam_rts").create_instance(
            ctx["fields.U"], ctx["fields.phi"], ctx["models.laminarTransport"]
        )

    def _finalize_algorithm(self, context: dict[str, Any]) -> Any:
        p = context["fields.p"]
        mesh = context["mesh"]
        # Pass p_rgh if available (e.g. from Boussinesq model)
        p_rgh = context.get("fields.p_rgh", None)
        assert self.algorithm is not None
        self.algorithm.set_pressure_reference(p, mesh, self.fvSolution, p_rgh)
        return self.algorithm

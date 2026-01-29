# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2025 NeoFOAM authors

from dataclasses import dataclass, field
from typing import Any

import pybFoam as pyf

from foamadapter.framework import ConfigContext
from foamadapter.algorithms.pressure_velocity import PressureVelocityAlgorithm
from foamadapter.models.stability_criteria import CFLCondition
from foamadapter.models.transport_model import TransportModel
from foamadapter.models.turbulence import TurbulenceModel
from foamadapter.models.incompressible_fluid_model import IncompressibleFluidModel
from foamadapter.foam.initialization import create_time_mesh


@dataclass
class InitializationData:
    """Explicit state container for initialization stages."""

    # Core models - always present
    algorithm: Any = None  # PressureVelocityAlgorithm
    cfl_condition: Any = None  # CFLCondition

    # Internal state
    fvSolution: Any = None

    # Optional models - physics extensions
    optional_models: list = field(default_factory=list)

    # Validation state
    load_validated: bool = False
    resolve_validated: bool = False

    @property
    def core_models(self) -> list[Any]:
        """Core models that define solver structure."""
        return [m for m in [self.algorithm, self.cfl_condition] if m is not None]


@dataclass
class ValidationError:
    """Validation error with field and message."""

    field: str
    message: str
    severity: str = "error"  # "error" or "warning"


@dataclass
class IncompressibleFluidInitializer:
    """3-stage initialization for IncompressibleFluid solver."""

    def __init__(self, argv: list[str]) -> None:
        self.argv = argv
        self.data = InitializationData()
        self.data.optional_models = IncompressibleFluidModel.detect_models()

    def load(self) -> dict[str, Any]:
        """LOAD stage: Load config from files. Explicit method."""
        self.data.fvSolution = pyf.dictionary.read("system/fvSolution")
        self.data.algorithm = PressureVelocityAlgorithm.from_fvSolution(
            self.data.fvSolution
        )
        self.data.cfl_condition = CFLCondition()

        load_results = {
            "algorithm": self.data.algorithm,
            "cfl_condition": self.data.cfl_condition,
        }

        # Run LOAD on optional models
        for model in self.data.optional_models:
            if hasattr(model, "load"):
                res = model.load()
                if isinstance(res, dict):
                    load_results.update(res)

        return load_results

    def validate_load(self) -> list[ValidationError]:
        """Validate configuration after LOAD stage."""
        errors = []

        if self.data.algorithm is None:
            errors.append(ValidationError("algorithm", "No algorithm configured"))

        if self.data.fvSolution is None:
            errors.append(ValidationError("fvSolution", "Failed to read fvSolution"))

        self.data.load_validated = (
            len([e for e in errors if e.severity == "error"]) == 0
        )
        return errors

    def resolve(self, config: ConfigContext) -> None:
        """RESOLVE stage: Connect models. Explicit method."""
        for model in self.data.optional_models:
            if hasattr(model, "resolve"):
                model.resolve(config)

    def validate_resolve(self, config: ConfigContext) -> list[ValidationError]:
        """Validate model connections after RESOLVE stage."""
        errors = []

        # Check optional model connections
        for model in self.data.optional_models:
            if hasattr(model, "validate_stage"):
                model_errors = model.validate_stage(config)
                errors.extend(model_errors)

        self.data.resolve_validated = (
            len([e for e in errors if e.severity == "error"]) == 0
        )
        return errors

    def build(self) -> list[Any]:
        """BUILD stage: Create lazy initializers. Explicit method."""
        from foamadapter.framework.initialization.helpers import lazy, model

        initializers = create_time_mesh(self.argv)

        # Algorithm fields
        assert self.data.algorithm is not None
        initializers.extend(self.data.algorithm.setup())

        # Transport model
        initializers.append(
            model(
                "laminarTransport",
                depends_on=["fields.U", "fields.phi"],
                create=self._create_transport,
            )
        )

        # Turbulence model
        initializers.append(
            model(
                "turbulence",
                depends_on=["fields.U", "fields.phi", "models.laminarTransport"],
                create=self._create_turbulence,
            )
        )

        # Final algorithm build
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

        # Keep dictionaries alive
        initializers.append(
            lazy("fvSolution_dict", create=lambda _: self.data.fvSolution)
        )

        # Build optional models
        for opt_model in self.data.optional_models:
            if hasattr(opt_model, "build"):
                initializers.extend(opt_model.build())

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
        assert self.data.algorithm is not None
        self.data.algorithm.set_pressure_reference(p, mesh, self.data.fvSolution, p_rgh)
        return self.data.algorithm

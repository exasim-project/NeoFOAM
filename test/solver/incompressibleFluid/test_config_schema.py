# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free config schema for the incompressibleFluid solver.

``config_classes()`` enumerates every config class the solver may consume
without a case directory — the schema an agent fills to scaffold a case.
The loaded *instances* for a concrete case come from
``create_init(case_dir).run_load().configs`` instead (see
``test_input_validation``).
"""

from neofoam.io import BaseConfig
from neofoam.solver.incompressibleFluid import config_classes


def test_config_classes_is_case_free() -> None:
    """No case directory is touched, yet the full schema set is returned."""
    classes = config_classes()

    assert all(isinstance(c, type) and issubclass(c, BaseConfig) for c in classes)
    assert len(classes) == len(set(classes))  # deduped

    names = {c.__name__ for c in classes}
    # Solver-core configs.
    assert {"ControlDictConfig", "TransportPropertiesConfig"} <= names
    # The fluid-property models are part of the schema: viscosity model
    # selection lives in constant/transportProperties (TransportPropertiesConfig,
    # asserted above), turbulence model selection in constant/turbulenceProperties.
    assert "TurbulencePropertiesConfig" in names
    # PIMPLE fvSchemes / fvSolution slices (per-spec subclass names).
    assert any("fvSchemes" in n for n in names)
    assert any("fvSolution" in n for n in names)
    # Optional boussinesq model is listed without running detection.
    assert "BoussinesqConfig" in names
    # Optional time-step models are part of the case-free catalog too:
    # registration makes them discoverable without running a case.
    assert "CourantConfig" in names
    assert "MaxDeltaTConfig" in names

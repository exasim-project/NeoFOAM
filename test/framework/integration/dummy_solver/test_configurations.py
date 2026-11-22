# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Case-free ``configurations`` schema on the (pybFoam-free) DummySolver.

Exercises the solver-agnostic API end-to-end on a real solver spec:

- ``SolverSpec.config(...)`` declares the four solver-core configs.
- ``SolverSpec.models(DummyModelInterface)`` binds the optional
  family; its registered members (model1-4) contribute their configs.
- ``configurations(solver)`` unions both, case-free (no case directory and
  no detection), and the ``Configurations`` view builds / validates / dumps
  schema from values alone.

Complements ``test_config_collection.py`` (the *with-a-case* instance view
via ``LoadResult``); here nothing is loaded from a case.
"""

from pathlib import Path

from neofoam import Configurations, configurations

from .dummy_init import CoreModel2, DummyAlgorithm, MeshConfig, SolverConfig
from .dummy_solver import dummy_solver_spec
from .models.dummy_model import DummyModelInterface

_CORE = ["SolverConfig", "MeshConfig", "CoreModel2", "DummyAlgorithm"]
_OPTIONAL = {
    "Model1Config",
    "Model1StepConfig",
    "Model2Config",
    "Model3Config",
    "Model4Config",
}


def test_optional_family_bound_on_spec() -> None:
    assert dummy_solver_spec.optional_model_specs == [DummyModelInterface]
    assert dummy_solver_spec.required_model_specs == []  # no required dispatcher family


def test_model_specs_lists_every_registered_member() -> None:
    names = {spec.name for spec in dummy_solver_spec.model_specs}
    assert names == {"DummyModel1", "DummyModel2", "CoupledModel", "MultiModel"}


def test_configurations_unions_core_and_optional_configs() -> None:
    cfg = configurations(dummy_solver_spec)
    assert isinstance(cfg, Configurations)
    # Solver-core configs come first, in declaration order.
    assert cfg.names[:4] == _CORE
    # Every optional member's declared configs are present.
    assert _OPTIONAL <= set(cfg.names)
    # Deduped.
    assert len(cfg.names) == len(set(cfg.names))


def test_view_lookup_build_and_validate() -> None:
    cfg = configurations(dummy_solver_spec)

    assert cfg["MeshConfig"] is MeshConfig
    mesh = cfg.new("MeshConfig", nPoints=100)
    assert mesh.nPoints == 100

    # json_schema covers every class; output model has one field per class.
    assert set(cfg.json_schema()) == set(cfg.names)
    out = cfg.as_output_model()
    assert len(out.model_fields) == len(cfg.names)


def test_configurations_is_case_free() -> None:
    """No case directory is touched: a bare spec yields the full schema."""
    cfg = configurations(dummy_solver_spec)
    # The four core classes are exactly the declared solver configs.
    assert {SolverConfig, MeshConfig, CoreModel2, DummyAlgorithm} <= set(cfg)


def test_detect_optional_models_instantiates_runtimes() -> None:
    case_dir = Path(__file__).parent / "configs"
    runtimes = dummy_solver_spec.detect_optional_models(case_dir)
    # All four members detect() True, so all instantiate.
    assert {rt.spec.name for rt in runtimes} == {
        "DummyModel1",
        "DummyModel2",
        "CoupledModel",
        "MultiModel",
    }
